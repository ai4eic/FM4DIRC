import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

import os
import json
import argparse
import random
import numpy as np
import pkbar
from datetime import datetime

from transformers import get_cosine_schedule_with_warmup

from dataloader.dataset import DIRC_Dataset_Classification
from dataloader.tokenizer import TimeTokenizer
from dataloader.dataloader import CreateLoadersClassification

from models.GPT import Cherenkov_GPT

def main(config,resume,distributed):

	# Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

	# Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    print(exp_name)

	# Create directory structure
    output_folder = config['output']['dir']
    os.makedirs(os.path.join(output_folder,exp_name),exist_ok=True)
    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)


    print('Creating Loaders.')
    pion_path = config['dataset']['training']['pion_data_path']
    kaon_path = config['dataset']['training']['kaon_data_path']
    val_pion_path = config['dataset']['validation']['pion_data_path']
    val_kaon_path = config['dataset']['validation']['kaon_data_path']

    # Model params.
    vocab_size = config['model']['vocab_size']
    time_vocab = config['model']['time_vocab']
    embed_dim = config['model']['embed_dim']
    drop_rate = config['model']['drop_rate']
    attn_heads = config['model']['attn_heads']
    num_blocks = config['model']['num_blocks']
    kin_size = config['model']['kin_size']
    hidden_units = config['model']['hidden_units']
    mlp_scale = config['model']['mlp_scale']
    msl = config['model']['max_seq_length']

    # Time tokenization
    digitize_time = bool(config['digitize_time'])
    if digitize_time:
        print("Digitizing time - classification over adjacent vocabulary.")
        print("Time vocab: ",time_vocab)
        time_res = config['stats']['time_res']
        t_max = config['stats']['time_max']
        t_min = config['stats']['time_min']
        print("T_Max: ",t_max," T_Min: ",t_min, "T_Res: ",time_res)
        time_digitizer = TimeTokenizer(t_max=t_max,t_min=t_min,resolution=time_res)

    else:
        print("Using regression over time domain.")
        time_digitizer = None


    train_dataset = DIRC_Dataset_Classification(pion_path=pion_path,kaon_path=kaon_path,max_seq_length=msl,time_digitizer=time_digitizer,stats=config['stats'])
    val_dataset = DIRC_Dataset_Classification(pion_path=val_pion_path,kaon_path=val_kaon_path,max_seq_length=msl,time_digitizer=time_digitizer,stats=config['stats'])
    pad_token = train_dataset.pad_token
    EOS_token = train_dataset.EOS_token
    SOS_token = train_dataset.SOS_token
    time_pad_token = train_dataset.time_pad_token
    time_EOS_token = train_dataset.time_EOS_token
    print("Time vocab: ",time_vocab)
    history = {'train_loss':[],'val_loss':[],'lr':[]}
    run_val = True
    train_loader,val_loader = CreateLoadersClassification(train_dataset,val_dataset,config)

    print("========= Special Tokens ============")
    print(f"Pixels - Pad: {pad_token}, SOS: {SOS_token}, EOS: {EOS_token}")
    print(f"Time   - Pad: {time_pad_token}, SOS: {SOS_token}, EOS: {time_EOS_token}")
    print("=====================================")


    if not distributed:
        print("Using single GPU.")
        net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,time_vocab=time_vocab,classification=True)
    else:
        print("Using {0} GPUs.".format(torch.cuda.device_count()))
        print(" ")
        net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,time_vocab=time_vocab,classification=True)
        net = DataParallel(net)

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    net.to('cuda')

	# Optimizer
    num_epochs = int(config['num_epochs'])
    lr = float(config['optimizer']['lr_cls'])

    use_warmup = bool(config['optimizer']['use_warmup'])

    if use_warmup:
        print("Using warmup.")
        batch_size = config['dataloader']['train']['batch_size']
        num_samples = len(train_dataset)
        steps_per_epoch = math.ceil(num_samples / batch_size)

        total_steps = steps_per_epoch * num_epochs

        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters())), lr=lr)
        warmup_steps = int(config['optimizer']['warmup_percent'] * total_steps)  

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)
    else:
        print("Using fixed LR with RAdam.")
        optimizer = torch.optim.RAdam(list(filter(lambda p: p.requires_grad, net.parameters())), lr=lr)
    
    
    startEpoch = 0
    global_step = 0

    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        if use_warmup:
            scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)
    else:
        print("========= Starting Training ================:")

    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      num_epochs:', num_epochs)
    print('')

    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')


    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

		###################
		## Training loop ##
		###################
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            tokens  = data[0].to('cuda').long()
            y = data[-1].to('cuda').float()
            
            if not digitize_time:
                times = data[1].to('cuda').float()
            else:
                times = data[1].to('cuda').long()


            #print(times.min(),times.max())

            k  = data[2].to('cuda').float()
            #weights_ = torch.clip(k[:,0] + 1.0,0.5,2.0) # Shift from -1,1 to 0.5,2

            padding_mask = (tokens == pad_token).to('cuda',dtype=torch.bool)
        
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                logits = net(tokens,times,k,padding_mask=padding_mask)

            #loss = (loss_fn(logits,y) * weights_).mean()
            loss = loss_fn(logits,y)
            train_acc = (torch.sum(torch.round(F.sigmoid(logits)) == y)) / len(y)
            loss.backward()
            optimizer.step()
            if use_warmup:
                scheduler.step()
    

			# statistics
            running_loss += loss.item() * tokens.shape[0]

            kbar.update(i, values=[("loss", loss.item()),("acc",train_acc.item())])

            global_step += 1

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        if use_warmup:
            history['lr'].append(scheduler.get_last_lr()[0])


		######################
		## validation phase ##
		######################
        if run_val:
            net.eval()
            val_loss = 0
            val_acc = 0
            for i, data in enumerate(val_loader):
                tokens  = data[0].to('cuda').long()
                y = data[-1].to('cuda').float()

                if not digitize_time:
                    times = data[1].to('cuda').float()
                else:
                    times = data[1].to('cuda').long()
                

                k  = data[2].to('cuda').float()
                #weights_ = torch.clip(k[:,0] + 1.0,0.5,2.0) # Shift from -1,1 to 0.5,2

                padding_mask = (tokens == pad_token).to('cuda',dtype=torch.bool)
                
                with torch.no_grad():
                    logits = net(tokens,times,k,padding_mask=padding_mask)

                #val_loss += (loss_fn(logits,y) * weights_).mean()
                val_loss += loss_fn(logits,y)
                val_acc += (torch.sum(torch.round(F.sigmoid(logits)) == y)) / len(y)

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            kbar.add(1, values=[("Val_loss", val_loss.item()),("Val_acc",val_acc.item())])

            name_output_file = config['name']+'_epoch{:02d}_val_loss_{:.6f}.pth'.format(epoch, val_loss)

        else:
            kbar.add(1,values=[('val_loss',0.)])
            name_output_file = config['name']+'_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))

        filename = os.path.join(output_folder , exp_name , name_output_file)

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        if use_warmup:
            checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        print('')


if __name__=='__main__':
	# PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Swin Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--distributed', default=0, type=int,
	                    help='Training on multiple GPUs.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    os.makedirs("Trained_Models",exist_ok=True)

    main(config,args.resume,bool(args.distributed))