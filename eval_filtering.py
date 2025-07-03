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

from dataloader.dataset import DIRC_Dataset_SequenceLevel
from dataloader.tokenizer import TimeTokenizer
from dataloader.dataloader import InferenceLoaderSequence

from utils.filtering_utils import filtering_plots
from utils.classification_utils import compute_metrics

#from models.GPT_Original import Cherenkov_GPT
from models.GPT import Cherenkov_GPT

warnings.filterwarnings("ignore", message=".*weights_only.*")

def run_inference(net,test_loader,digitize_time,pad_token,EOS_token,SOS_token):
    kbar = pkbar.Kbar(target=len(test_loader), width=20, always_stateful=False)
    net.eval()

    predictions = []
    truth = []
    PIDs = []
    conditions = []
    time_bins = []
    
    precision,recall,f1 = 0.,0.,0.

    start = time.time()
    for i, data in enumerate(test_loader):
        tokens  = data[0].to('cuda').long()
        y = data[-1].to('cuda').float()
        PIDs.append(data[-2].detach().cpu().numpy())
        conditions.append(data[2].numpy())
        
        if not digitize_time:
            times = data[1].to('cuda').float()
        else:
            times = data[1].to('cuda').long()

        

        k  = data[2].to('cuda').float()

        padding_mask = (tokens == pad_token).to('cuda',dtype=torch.bool)
        valid_mask = ~((tokens == pad_token) | (tokens == SOS_token) |  (tokens == EOS_token)).to('cuda',dtype=torch.bool)

        time_bins.append(times[valid_mask].detach().cpu().numpy())

        y = y[valid_mask]

        with torch.set_grad_enabled(False):
            logits = net(tokens,times,k,padding_mask=padding_mask)

        logits = logits[:,k.shape[1]:]

        pred = F.sigmoid(logits[valid_mask]).float()

        p,r,f = compute_metrics(pred.round(),y)

        precision += p
        recall += r
        f1 += f

        predictions.append(pred.detach().cpu().numpy())
        truth.append(y.detach().cpu().numpy())
        kbar.update(i)

    end = time.time()

    print(" ")
    print("Elapsed Time: ", end - start)

    predictions = np.concatenate(predictions).astype('float32')
    truth = np.concatenate(truth).astype('float32')
    conditions = np.concatenate(conditions).astype('float32')
    time_bins = np.concatenate(time_bins).astype('int32')
    PIDs = np.concatenate(PIDs).astype('float32')
    print("Time / event: ",(end - start) / len(predictions))
    print(" ")
    print("Precision: ",100 * precision / len(test_loader))
    print("Recall: ",100 * recall / len(test_loader))
    print("F1: ",100 * f1 / len(test_loader))
    print(" ")

    k_idx = np.where(PIDs == 1.0)[0]
    results_kaon = {"prob":predictions[k_idx],"Truth":truth[k_idx],"Kins":conditions[k_idx],"time_bins":time_bins[k_idx]}
    p_idx = np.where(PIDs == 0.0)[0]
    results_pion = {"prob":predictions[p_idx],"Truth":truth[p_idx],"Kins":conditions[p_idx],"time_bins":time_bins[p_idx]}

    return results_pion,results_kaon


def main(config,args):

    print("Running inference")
    stats = config['stats']

    if os.path.exists(os.path.join(config['Inference']['out_dir_filtering'],str(args.momentum),"Kaon_Results.pkl")) and os.path.exists(os.path.join(config['Inference']['out_dir_filtering'],str(args.momentum),"Pion_Results.pkl")):
        print(f"Found existing inference files at {args.momentum} GeV. Skipping inference and only plotting.")
        kaon_results = np.load(os.path.join(config['Inference']['out_dir_filtering'],str(args.momentum),"Kaon_Results.pkl"),allow_pickle=True)
        pion_results = np.load(os.path.join(config['Inference']['out_dir_filtering'],str(args.momentum),"Pion_Results.pkl"),allow_pickle=True)
        out_folder = os.path.join(config['Inference']['out_dir_filtering'],str(args.momentum))
        filtering_plots(kaon_results,pion_results,out_folder,momentum=args.momentum)

    else:
        if args.momentum == 9.0:
            print("Loading 9 GeV Datasets.")
            test_pion_path = config['dataset']['fixed_point']["pion_data_path_9GeV"]
            test_kaon_path = config['dataset']['fixed_point']["kaon_data_path_9GeV"]
        elif args.momentum == 6.0:
            print("Loading 6 GeV Datasets.")
            test_pion_path = config['dataset']['fixed_point']["pion_data_path_6GeV"]
            test_kaon_path = config['dataset']['fixed_point']["kaon_data_path_6GeV"]
        elif args.momentum == 3.0:
            print("Loading 3 GeV Datasets.")
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
        # Set MoE False here explicitly - don't need it.
        use_MoE = False

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


        net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                            num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,
                            mlp_scale=mlp_scale,classification=True,sequence_level=True,time_vocab=time_vocab,
                            drop_rates=drop_rates,use_MoE=use_MoE)

        net.to('cuda')
        model_path = config['Inference']['filter_path']
        dicte = torch.load(model_path)
        net.load_state_dict(dicte['net_state_dict'])
        net.eval()

        kaon_results = []
        pion_results = []

        seeds = [829345, 473829, 190284, 675902, 385729, 912384, 508271, 760294, 643891, 138204,
                298173, 582903, 760238, 182739, 943820, 385920, 128394, 576092, 720384, 238472,
                183920, 658392, 920384, 743820, 384920, 2653, 22132, 24021, 31036, 85951, 127882,
                306155, 355463, 418136, 428305, 447956, 492507, 539531, 594425, 598474, 665440,
                790990, 808845, 96623, 752022, 682024, 7111996, 28101994, 4121970, 24101958]

        print(f"Running evaluation with {int(len(seeds))} different seeds.")

        for i,seed in enumerate(seeds):
            print("Running evaluation with seed: ",seed)
            # Setup random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)

            test_dataset = DIRC_Dataset_SequenceLevel(pion_path=test_pion_path,kaon_path=test_kaon_path,max_seq_length=msl,time_digitizer=time_digitizer,stats=config['stats'])
            EOS_token = test_dataset.EOS_token
            SOS_token = test_dataset.SOS_token
            pad_token = test_dataset.pad_token
            time_pad_token = test_dataset.time_pad_token
            time_EOS_token = test_dataset.time_EOS_token
            test_loader = InferenceLoaderSequence(test_dataset,config)
        
            Pion,Kaon = run_inference(net,test_loader,digitize_time,pad_token=pad_token,EOS_token=EOS_token,SOS_token=SOS_token)
            pion_results.append(Pion)
            kaon_results.append(Kaon)
            print("--------------------------------------- ")

        print('Inference plots can be found in: ' + config['Inference']['out_dir_filtering'])
        os.makedirs(os.path.join(config['Inference']['out_dir_filtering'],str(args.momentum)),exist_ok=True)

        pion_path = os.path.join(config['Inference']['out_dir_filtering'],str(args.momentum),"Pion_Results.pkl")
        with open(pion_path,"wb") as file:
            pickle.dump(pion_results,file)

        kaon_path = os.path.join(config['Inference']['out_dir_filtering'],str(args.momentum),"Kaon_Results.pkl")
        with open(kaon_path,"wb") as file:
            pickle.dump(kaon_results,file)

        out_folder = os.path.join(config['Inference']['out_dir_filtering'],str(args.momentum))
        filtering_plots(kaon_results,pion_results,out_folder,momentum=args.momentum)



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Filtering at fixed kinematics.')
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
