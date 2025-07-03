import os
import json
import pkbar
import argparse
import pickle
import warnings
import time

import torch
import random
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors

from utils.utils import convert_indices,convert_indices_gt,dynamic_batch

from dataloader.tokenizer import TimeTokenizer
from models.GPT import Cherenkov_GPT

warnings.filterwarnings("ignore", message=".*weights_only.*")


def main(config,args):
    # Remove seeding, make it random.
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    gpu_mem, _ = torch.cuda.mem_get_info()
    gpu_mem = gpu_mem / (1024 ** 3) # GB

    config['method'] = args.method

    if config['method'] == "Pion":
        print("Generating for pions.")
        dicte = torch.load(config['Inference']['pion_model_path'])
        print(config['Inference']['pion_model_path'])
        PID = 211
    elif config['method'] == 'Kaon':
        print("Generation for kaons.")
        dicte = torch.load(config['Inference']['kaon_model_path'])
        print(config['Inference']['kaon_model_path'])
        PID = 321
    else:
        print("Specify particle to generate in config file")
        exit()


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
    use_MoE = bool(config['model']['use_MoE'])
    num_experts = config['model']['num_experts']
    num_classes = config['model']['num_classes']
    # data params
    inference_batch = config['Inference']['batch_size']
    stats = config['stats']
    conditional_maxes = np.array([stats['P_max'],stats['theta_max']])
    conditional_mins = np.array([stats['P_min'],stats['theta_min']])

    # Time tokenization
    digitize_time = bool(config['digitize_time'])
    if digitize_time:
        print("Digitizing time - classification over adjacent vocabulary.")
        time_res = config['stats']['time_res']
        t_max = config['stats']['time_max']
        t_min = config['stats']['time_min']
        print("Time Res: ",time_res)
        print("Time vocab: ",time_vocab)
        time_digitizer = TimeTokenizer(t_max=t_max,t_min=t_min,resolution=time_res)
        de_tokenize_func = time_digitizer.de_tokenize

    else:
        print("Using regression over time domain.")
        time_digitizer = None
        de_tokenize_func = None

    net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
        num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,
        time_vocab=time_vocab,detokenize_func=de_tokenize_func,drop_rates=drop_rates,use_MoE=use_MoE,num_experts=num_experts,num_classes=num_classes)

    if args.distributed:
        net = DataParallel(net)

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    device = torch.device('cuda')
    net.to('cuda')
    net.load_state_dict(dicte['net_state_dict'])
    net.eval()

    net = torch.compile(model=net,mode="max-autotune")

    for layer_ in net.layers:
        if hasattr(layer_.attn, "g_scale"):
            print(layer_.attn.__class__.__name__, layer_.attn.g_scale)

    if config['method'] == 'Pion':
        print("Generating pions with momentum of {0} GeV/c".format(args.momentum))
        if args.momentum == 1.0:
            datapoints = np.load(config['dataset']['fixed_point']["pion_data_path_1GeV"],allow_pickle=True)
        elif args.momentum == 3.0:
            datapoints = np.load(config['dataset']['fixed_point']["pion_data_path_3GeV"],allow_pickle=True)
        elif args.momentum == 6.0:
            datapoints = np.load(config['dataset']['fixed_point']["pion_data_path_6GeV"],allow_pickle=True)
        elif args.momentum == 9.0:
            datapoints = np.load(config['dataset']['fixed_point']["pion_data_path_9GeV"],allow_pickle=True)
        else:
            raise ValueError("Value of momentum does correspond to a dataset. Check if the path is correct, or simulate and processes.")

    elif config['method'] == 'Kaon':
        print("Generating kaons with momentum of {0} GeV/c".format(args.momentum))
        if args.momentum == 1.0:
            datapoints = np.load(config['dataset']['fixed_point']["kaon_data_path_1GeV"],allow_pickle=True)
        elif args.momentum == 3.0:
            datapoints = np.load(config['dataset']['fixed_point']["kaon_data_path_3GeV"],allow_pickle=True)
        elif args.momentum == 6.0:
            datapoints = np.load(config['dataset']['fixed_point']["kaon_data_path_6GeV"],allow_pickle=True)
        elif args.momentum == 9.0:
            datapoints = np.load(config['dataset']['fixed_point']["kaon_data_path_9GeV"],allow_pickle=True)
        else:
            raise ValueError("Value of momentum does correspond to a dataset. Check if the path is correct, or simulate and processes.")       
        
    else:
        raise ValueError("Method not found.")


    numTracks = 0
    true_xs = []
    true_ys = []
    true_times = []
    ground_truth = []

    for i in range(len(datapoints)):
        if (datapoints[i]['Theta'] == args.theta) and (datapoints[i]['P'] == args.momentum) and (datapoints[i]['Phi'] == 0.0) and (datapoints[i]['NHits'] < 250) and (datapoints[i]['NHits'] > 5):
            numTracks += 1
            ground_truth.append(datapoints[i])
         
    
    
    k = np.array([args.momentum,args.theta])
    k_unscaled = np.tile(k.copy(), (inference_batch, 1))
    k = 2*(k - conditional_mins) / (conditional_maxes - conditional_mins) - 1.0
    k = torch.tensor(k).to('cuda').float().unsqueeze(0).repeat(inference_batch, 1)
    
    if config['method'] == "Pion" and use_MoE:
        class_label = torch.zeros((inference_batch,),dtype=torch.float32,device=k.device)
    elif config['method'] == "Kaon" and use_MoE:
        class_label = torch.ones((inference_batch,),dtype=torch.float32,device=k.device)
    else:
        class_label = None

    num_itter = numTracks // inference_batch
    last_batch = numTracks % inference_batch

    print("Generating {0} tracks, with p={1} and theta={2}.".format(numTracks,args.momentum,args.theta))
    print("Using temperature: ",args.temperature)
    print("Using",args.sampling,"sampling.")
    if args.sampling == "TopK":
        print("TopK value: ",args.topK)
    elif args.sampling == "Nucleus":
        print("Nucleus P: ",args.nucleus_p)
    else:
        pass

    if args.dynamic_temp:
        print("Using dynamic temperature scaling - exponential decay. See class reference.")

    kbar = pkbar.Kbar(target=num_itter + 1, width=20, always_stateful=False)

    fast_sim = []
    torch.cuda.empty_cache()
    start = time.time()
    for i in range(num_itter):

        with torch.no_grad():
            tracks = net.generate(k,unscaled_k=k_unscaled,class_label=class_label,method=args.sampling,
                                  temperature=args.temperature,topK=args.topK,nucleus_p=args.nucleus_p,
                                  dynamic_temp=args.dynamic_temp)

        fast_sim += tracks

        kbar.update(i)
        
    end = time.time()    
    torch.cuda.empty_cache()

    # Generate the last batch of tracks if any
    if last_batch > 0:
        k = np.array([args.momentum,args.theta])
        k_unscaled = k_unscaled = np.tile(k.copy(), (last_batch, 1))
        k = 2*(k - conditional_mins) / (conditional_maxes - conditional_mins) - 1.0
        k = torch.tensor(k).to('cuda').float().unsqueeze(0).repeat(last_batch, 1)  

        if config['method'] == "Pion" and use_MoE:
            class_label = torch.zeros((last_batch,),dtype=torch.float32,device=k.device)
        elif config['method'] == "Kaon" and use_MoE:
            class_label = torch.ones((last_batch,),dtype=torch.float32,device=k.device)
        else:
            class_label = None 

        with torch.no_grad():
            tracks = net.generate(k,unscaled_k=k_unscaled,class_label=class_label,method=args.sampling,
                                  temperature=args.temperature,topK=args.topK,nucleus_p=args.nucleus_p,
                                  dynamic_temp=args.dynamic_temp)  

        fast_sim += tracks

        kbar.add(1)
    torch.cuda.empty_cache()

    n_photons = 0
    n_gamma = 0

    for i in range(len(ground_truth)):
        n_photons += ground_truth[i]['NHits']

    for i in range(len(fast_sim)):
        n_gamma += fast_sim[i]['NHits']

    print(" ")
    print("Number of tracks generated: ",numTracks)
    print("Elapsed Time: ", end - start)
    print("Average time / track: ",(end - start) / (numTracks - last_batch))
    print("True photon yield: ",n_photons," Generated photon yield: ",n_gamma)
    print(" ")

    gen_dict = {}
    gen_dict['FastSimPhotons'] = n_gamma
    gen_dict['TruePhotons'] = n_photons
    gen_dict['fast_sim'] = fast_sim
    gen_dict['truth'] = ground_truth

    os.makedirs("Generations",exist_ok=True)
    out_folder = os.path.join("Generations",config['Inference']['fixed_point_dir'])
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

    out_path_ = os.path.join(out_folder,str(config['method'])+f"_p_{args.momentum}_theta_{args.theta}_PID_{config['method']}_ntracks_{numTracks}.pkl")
    
    with open(out_path_,"wb") as file:
        pickle.dump(gen_dict,file)


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p', '--momentum', default=6.0,type=float,help='Particle Momentum.')
    parser.add_argument('-t','--theta',default=30.0,type=float,help='Particle theta.')
    parser.add_argument('-m', '--method',default="Kaon",type=str,help='Generated particle type, Kaon or Pion.')
    parser.add_argument('-d','--distributed',action='store_true',help='Trained with multiple GPUs - DDP.')
    parser.add_argument('-tmp','--temperature',default=1.0,type=float,help='Generation temperature.')
    parser.add_argument('-s','--sampling',default="Nucleus",type=str,help='Default,TopK,Nucleus')
    parser.add_argument('-tk','--topK',default=300,type=int,help="TopK - only used if sampling = TopK")
    parser.add_argument('-np','--nucleus_p',default=0.95,type=float,help="Nucleus P value - only used if sampling = Nucleus")
    parser.add_argument('-dt','--dynamic_temp',action='store_true',help='Dynamic temperature scaling - exponential decay.')
    args = parser.parse_args()

    #args.context_len = None

    config = json.load(open(args.config))

    main(config,args)
