import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from utils.utils import time_loss_fn

import os
import json
import argparse
import random
import numpy as np
import pkbar
import math
import warnings
from datetime import datetime

from dataloader.dataset import DIRC_Dataset
from dataloader.tokenizer import TimeTokenizer
from dataloader.dataloader import CreateLoaders

from models.GPT import Cherenkov_GPT

warnings.filterwarnings("ignore", message=".*weights_only.*")

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
    data_type = config['data_type']
    if data_type == "Kaons":
        data_path = config['dataset']['training']['kaon_data_path']
        val_data_path = config['dataset']['validation']['kaon_data_path']
    elif data_type == "Pions":
        data_path = config['dataset']['training']['pion_data_path']
        val_data_path = config['dataset']['validation']['pion_data_path']
    else:
        raise ValueError("Data type: {0} is not supported.".format(data_type))

    # Model params.
    vocab_size = config['model']['vocab_size']
    time_vocab = config['model']['time_vocab']
    embed_dim = config['model']['embed_dim']
    attn_heads = config['model']['attn_heads']
    num_blocks = config['model']['num_blocks']
    kin_size = config['model']['kin_size']
    hidden_units = config['model']['hidden_units']
    mlp_scale = config['model']['mlp_scale']
    msl = config['model']['max_seq_length']
    drop_rates = config['model']['drop_rates']

    # Time tokenization
    digitize_time = bool(config['digitize_time'])
    if digitize_time:
        print("Digitizing time - classification over adjacent vocabulary.")
        print("Time vocab: ",config['model']['time_vocab'])
        time_res = config['stats']['time_res']
        t_max = config['stats']['time_max']
        t_min = config['stats']['time_min']
        print("T_Max: ",t_max," T_Min: ",t_min, "T_Res: ",time_res)
        time_digitizer = TimeTokenizer(t_max=t_max,t_min=t_min,resolution=time_res)

    else:
        print("Using regression over time domain.")
        time_digitizer = None


    train_dataset = DIRC_Dataset(data_path=data_path,data_type=data_type,max_seq_length=msl,time_digitizer=time_digitizer,stats=config['stats'])
    val_dataset = DIRC_Dataset(data_path=val_data_path,data_type=data_type,max_seq_length=msl,time_digitizer=time_digitizer,stats=config['stats'])
    pad_token = train_dataset.pad_token
    EOS_token = train_dataset.EOS_token
    SOS_token = train_dataset.SOS_token

    time_pad_token = train_dataset.time_pad_token
    time_EOS_token = train_dataset.time_EOS_token

    print("========= Special Tokens ============")
    print(f"Pixels - Pad: {pad_token}, SOS: {SOS_token}, EOS: {EOS_token}")
    print(f"Time   - Pad: {time_pad_token}, SOS: {SOS_token}, EOS: {time_EOS_token}")
    print("=====================================")

    history = {'train_loss':[],'val_loss':[],'lr':[]}
    run_val = True
    train_loader,val_loader = CreateLoaders(train_dataset,val_dataset,config)


    if not distributed:
        print("Using single GPU.")
        net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,time_vocab=time_vocab,drop_rates=drop_rates)
    else:
        print("Using {0} GPUs.".format(torch.cuda.device_count()))
        print(" ")
        net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,time_vocab=time_vocab,drop_rates=drop_rates)
        net = DataParallel(net)

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    net.to('cuda')

	# Optimizer
    num_epochs = int(config['num_epochs'])
    lr = float(config['optimizer']['lr'])

    # No need for warmup
    optimizer = torch.optim.RAdam(list(filter(lambda p: p.requires_grad, net.parameters())), lr=lr)


    startEpoch = 0
    global_step = 0

    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
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

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)
    if digitize_time:
        print("Time vocab: ",time_pad_token+1)
        time_ce = nn.CrossEntropyLoss(ignore_index=time_pad_token)


    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

		###################
		## Training loop ##
		###################
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):

            tokens  = data[0].to('cuda').long()
            next_tokens = tokens[:, 1:].clone()
            tokens = tokens[:, :-1]  
            
            if not digitize_time:
                times = data[1].to('cuda').float()
            else:
                times = data[1].to('cuda').long()

            next_times = times[:, 1:].clone()   
            times = times[:, :-1]
            
            k  = data[2].to('cuda').float()

            padding_mask = (tokens == pad_token).to('cuda',dtype=torch.bool)
        
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                logits,t = net(tokens,times,k,padding_mask=padding_mask)


            logits = logits[:,k.shape[1]:,:]
            t = t[:,k.shape[1]:,:]

            pixel_loss = loss_fn(logits.reshape(-1, logits.size(-1)), next_tokens.reshape(-1))

            if not digitize_time:
                regression_mask = ~torch.isin(next_tokens,torch.tensor([pad_token, SOS_token,EOS_token], device=next_tokens.device))
                time_loss = time_loss_fn(next_times,t,regression_mask)
            else:
                time_loss = time_ce(t.reshape(-1, t.size(-1)), next_times.reshape(-1))

            loss = pixel_loss + time_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
			# statistics
            running_loss += loss.item() * tokens.shape[0]

            kbar.update(i, values=[("loss", loss.item()),("pix",pixel_loss.item()),("time",time_loss.item())])

            global_step += 1
        
        history['train_loss'].append(running_loss / len(train_loader.dataset))

		######################
		## validation phase ##
		######################
        if run_val:
            net.eval()
            val_time_loss = 0.0
            val_pixel_loss = 0.0
            for i, data in enumerate(val_loader):
                tokens  = data[0].to('cuda').long()
                next_tokens = tokens[:, 1:].clone()
                tokens = tokens[:, :-1]  
                
                if not digitize_time:
                    times = data[1].to('cuda').float()
                else:
                    times = data[1].to('cuda').long()
                
                next_times = times[:, 1:].clone()   
                times = times[:, :-1]

                k  = data[2].to('cuda').float()

                padding_mask = (tokens == pad_token).to('cuda',dtype=torch.bool)
                
                with torch.no_grad():
                    logits,t = net(tokens,times,k,padding_mask=padding_mask)

                logits = logits[:,k.shape[1]:,:]
                t = t[:,k.shape[1]:,:]

                if not digitize_time:
                    regression_mask = ~torch.isin(next_tokens,torch.tensor([pad_token, SOS_token,EOS_token], device=next_tokens.device))
                    val_time_loss += time_loss_fn(next_times,t,regression_mask)
                else:
                    val_time_loss += time_ce(t.reshape(-1, t.size(-1)), next_times.reshape(-1))

                val_pixel_loss += loss_fn(logits.reshape(-1, logits.size(-1)), next_tokens.reshape(-1))

            val_time_loss /= len(val_loader)
            val_pixel_loss /= len(val_loader)
            val_loss = val_pixel_loss + val_time_loss

            kbar.add(1, values=[("Val_loss", val_loss.item()),("val_pix",val_pixel_loss.item()),("val_time",val_time_loss.item())])

            name_output_file = config['name']+'_epoch{:02d}_val_loss_{:.6f}.pth'.format(epoch, val_loss)

        else:
            kbar.add(1,values=[('val_loss',0.)])
            name_output_file = config['name']+'_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))

        filename = os.path.join(output_folder , exp_name , name_output_file)

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        print('')


if __name__=='__main__':
	# PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Generative Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--distributed', default=0, type=int,
	                    help='Training on multiple GPUs.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    #os.makedirs("Trained_Models",exist_ok=True)

    main(config,args.resume,bool(args.distributed))