import os
import json
import argparse
import random
import pkbar
import time
import pickle
import warnings
import numpy as np
import itertools
from datetime import datetime

import torch
import torch.nn.functional as F

from dataloader.dataset import DIRC_Dataset_Classification
from dataloader.tokenizer import TimeTokenizer
from dataloader.dataloader import InferenceLoader

from utils.classification_utils import *

from models.GPT import Cherenkov_GPT

warnings.filterwarnings("ignore", message=".*weights_only.*")

def run_inference(net,test_loader,digitize_time,pad_token):
    kbar = pkbar.Kbar(target=len(test_loader), width=20, always_stateful=False)
    net.eval()
    predictions = []
    truth = []
    conditions = []
    dll_geom = []
    start = time.time()
    tts = []
    acc = 0
    for i, data in enumerate(test_loader):
        tokens  = data[0].to('cuda').long()
        y = data[-1].to('cuda').float()
        conditions.append(data[-2].numpy())

        if not digitize_time:
            times = data[1].to('cuda').float()
        else:
            times = data[1].to('cuda').long()
        

        k  = data[2].to('cuda').float()

        padding_mask = (tokens == pad_token).to('cuda',dtype=torch.bool)
        
        with torch.no_grad():
            tt = time.time()
            logits = net(tokens,times,k,padding_mask=padding_mask)

        acc += (torch.sum(torch.round(F.sigmoid(logits)) == y)) / len(y)
        tts.append((time.time() - tt)/len(logits))
        predictions.append(logits.detach().cpu().numpy())
        truth.append(y.detach().cpu().numpy())
        kbar.update(i)

    print(" ")
    print("Average GPU time: ",np.average(tts))

    end = time.time()

    print(" ")
    print("Elapsed Time: ", end - start)

    predictions = np.concatenate(predictions).astype('float32')
    truth = np.concatenate(truth).astype('float32')
    # Eventually compare to geometric methods - keep as zeros placeholder
    dll_geom = np.zeros_like(predictions)
    print("Is NaN" ,np.isnan(dll_geom))
    print(dll_geom.max(),dll_geom.min())
    conditions = np.concatenate(conditions)
    print("Time / event: ",(end - start) / len(predictions))
    print(" ")
    print("Accuracy: ",100 * acc.item() / len(test_loader))
    print(" ")

    k_idx = np.where(truth == 1.0)[0]
    LL_Kaon = {"z_value":predictions[k_idx],"Truth":truth[k_idx],"Kins":conditions[k_idx],"hyp_pion_geom":dll_geom[k_idx],"hyp_kaon_geom":dll_geom[k_idx]}
    p_idx = np.where(truth == 0.0)[0]
    LL_Pion = {"z_value":predictions[p_idx],"Truth":truth[p_idx],"Kins":conditions[p_idx],"hyp_pion_geom":dll_geom[p_idx],"hyp_kaon_geom":dll_geom[p_idx]}

    return LL_Pion,LL_Kaon


def main(config,args):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    print("Running inference")
    stats = config['stats']

    if os.path.exists(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Kaon_DLL_Results.pkl")) and os.path.exists(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Pion_DLL_Results.pkl")):
        print("Found existing inference files. Skipping inference and only plotting.")
        LL_Kaon = np.load(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Kaon_DLL_Results.pkl"),allow_pickle=True)
        LL_Pion = np.load(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Pion_DLL_Results.pkl"),allow_pickle=True)
        out_folder = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum))
        run_plotting(out_folder,args.momentum,args.model_type)

    else:
        if args.momentum == 9.0:
            test_pion_path = config['dataset']['fixed_point']["pion_data_path_9GeV"]
            test_kaon_path = config['dataset']['fixed_point']["kaon_data_path_9GeV"]
        elif args.momentum == 6.0:
            test_pion_path = config['dataset']['fixed_point']["pion_data_path_6GeV"]
            test_kaon_path = config['dataset']['fixed_point']["kaon_data_path_6GeV"]
        elif args.momentum == 3.0:
            test_pion_path = config['dataset']['fixed_point']["pion_data_path_3GeV"]
            test_kaon_path = config['dataset']['fixed_point']["kaon_data_path_3GeV"]
        else:
            raise ValueError("Momentum value not found.")
        

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
            print("Time vocab:",time_vocab)
            time_res = config['stats']['time_res']
            t_max = config['stats']['time_max']
            t_min = config['stats']['time_min']
            time_digitizer = TimeTokenizer(t_max=t_max,t_min=t_min,resolution=time_res)

        else:
            print("Using regression over time domain.")
            time_digitizer = None


        test_dataset = DIRC_Dataset_Classification(pion_path=test_pion_path,kaon_path=test_kaon_path,max_seq_length=msl,time_digitizer=time_digitizer,stats=config['stats'])
        EOS_token = test_dataset.EOS_token
        SOS_token = test_dataset.SOS_token
        pad_token = test_dataset.pad_token
        time_pad_token = test_dataset.time_pad_token
        time_EOS_token = test_dataset.time_EOS_token

        print("========= Special Tokens ============")
        print(f"Pixels - Pad: {pad_token}, SOS: {SOS_token}, EOS: {EOS_token}")
        print(f"Time   - Pad: {time_pad_token}, SOS: {SOS_token}, EOS: {time_EOS_token}")
        print("=====================================")


        history = {'train_loss':[],'val_loss':[],'lr':[]}
        run_val = True
        test_loader = InferenceLoader(test_dataset,config)


        net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                            num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,
                            mlp_scale=mlp_scale,classification=True,time_vocab=time_vocab,drop_rates=drop_rates)

        net.to('cuda')
        model_path = config['Inference']['classifier_path']
        dicte = torch.load(model_path)
        net.load_state_dict(dicte['net_state_dict'])
        net.eval()
        

        LL_Pion,LL_Kaon = run_inference(net,test_loader,digitize_time,pad_token)

        
        print('Inference plots can be found in: ' + config['Inference']['out_dir_fixed'])
        os.makedirs(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum)),exist_ok=True)

        pion_path = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Pion_DLL_Results.pkl")
        with open(pion_path,"wb") as file:
            pickle.dump(LL_Pion,file)

        kaon_path = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Kaon_DLL_Results.pkl")
        with open(kaon_path,"wb") as file:
            pickle.dump(LL_Kaon,file)

        out_folder = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum))
        run_plotting(out_folder,args.momentum,args.model_type)



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='DLL at fixed kinematics.')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p','--momentum',default=6.0,type=float,help='Momentum value.')
    parser.add_argument('-mt','--model_type',default="Transformer",type=str,help="Model type.")
    args = parser.parse_args()

    config = json.load(open(args.config))

    if not os.path.exists("Inference"):
        print("Making Inference Directory.")
        os.makedirs("Inference",exist_ok=True)

    main(config,args)
