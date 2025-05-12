import os
import re
import json
import time
import copy
import pkbar
import random
import pickle
import warnings
import argparse
import numpy as np

import torch

from models.GPT import Cherenkov_GPT
from dataloader.tokenizer import TimeTokenizer

warnings.filterwarnings("ignore", message=".*weights_only.*")


def main(config,args):

    # Remove seeding, make it random.
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    assert args.method in ["MixPiK","Pion","Kaon"], "PID not found. Please choose from: MixPiK,Pion,Kaon"

    inference_batch = config["Inference"]['batch_size']
    num_itter = args.n_tracks // inference_batch

    print('------------------------ Setup ------------------------')
    print("Generating",args.n_tracks, args.method,"in batches of:", inference_batch)
    print(f"Config File: {args.config}")
    print(f"Number of Particles to Dump per .pkl File: {args.n_dump}")
    print(f"Generated Particle Type: {args.method}")
    print(f"Trained with Multiple GPUs (DP): {'Yes' if args.distributed else 'No'}")
    print(f"Momentum Value or Range: {args.momentum}")
    print(f"Theta Value or Range: {args.theta}")
    print(f"Include Hits from Dark Noise: {'Yes' if args.dark_noise else 'No'}")
    print(f"Generation Temperature: {args.temperature}")
    print(f"Use Dynamic Temperature: {'Yes' if args.dynamic_temperature else 'No'}")
    print(f"Sampling Method: {args.sampling}")
    print(f"TopK Value: {args.topK}")
    print(f"Nucleus P Value: {args.nucleus_p}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("No GPU was found! Exiting the program.")
    
    if not args.n_dump:
        print('No value found for n_dump. Setting it equal to {}'.format(int(args.n_tracks)))
        print("Consider dumping simulation to disk in batches.")
        args.n_dump = args.n_tracks
    print('-------------------------------------------------------')

    assert args.n_dump <= args.n_tracks, "total n_tracks must be at least n_dump, the number of tracks to dump per .pkl file. Got n_tracks of {} and n_dump of {} instead.".format(args.n_tracks, args.n_dump)

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

    if not args.distributed:
        net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                    num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,detokenize_func=de_tokenize_func,drop_rates=drop_rates)
    else:
        net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                num_blocks=num_blocks,hidden_units=hidden_units,use_kinematics=use_kinematics,mlp_scale=mlp_scale,drop_rates=drop_rates)
        net = DataParallel(net)

    pion_net = copy.deepcopy(net)
    kaon_net = copy.deepcopy(net)
    del net
    
    if args.method in ['Kaon','MixPiK']:
        Kdicte = torch.load(config['Inference']['kaon_model_path'])
        kaon_net.to('cuda')
        kaon_net.load_state_dict(Kdicte['net_state_dict'])
        kaon_net.eval()
        kaon_net = torch.compile(model=kaon_net,mode="max-autotune")
    if args.method in ['Pion','MixPiK']:
        Pdicte = torch.load(config['Inference']['pion_model_path'])
        pion_net.to('cuda')
        pion_net.load_state_dict(Pdicte['net_state_dict'])
        pion_net.eval()
        pion_net = torch.compile(model=pion_net,mode="max-autotune")

    if re.fullmatch(r"\d+(?:\.\d+)?", args.momentum):
        p_low = float(args.momentum)
        p_high = float(args.momentum)
    elif re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", args.momentum):
        match = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", args.momentum)
        p_low = float(match.group(1))
        p_high = float(match.group(2))
    else:
        raise ValueError('Momentum format is p1-p2, where p1, p2 are between 1 and 10 (e.g. 3.5-10).')
    
    if re.fullmatch(r"\d+(?:\.\d+)?", args.theta):
        theta_low = int(args.theta)
        theta_high = int(args.theta)
    elif re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", args.theta):
        match = re.fullmatch(r"(\d+)-(\d+)", args.theta)
        theta_low = int(match.group(1))
        theta_high = int(match.group(2))
    else:
        raise ValueError('Theta format is theta1-theta2, where theta1, theta2 are between 25 and 155 (e.g. 70.5-80)')
    
    assert 1 <= p_low <= 10 and 1 <= p_high <= 10, f"momentum range must be between 0 and 10, got {p_low}, {p_high} instead."
    assert 25 <= theta_low <= 160 and 25 <= theta_high <= 160, f"theta range must be between 25 and 160, got {theta_low}, {theta_high} instead."

    os.makedirs("Simulation",exist_ok=True)
    out_folder = os.path.join("Simulation",config['Inference']['simulation_dir'])
    os.makedirs(out_folder,exist_ok=True)

    print("Simulation can be found in: ",out_folder)
    running_gen = 0
    file_count = 1
    generations = []
    kbar = pkbar.Kbar(target=num_itter + 1, width=20, always_stateful=False)
    start = time.time()

    while running_gen < args.n_tracks:
        if args.method in ['Kaon','MixPiK']:
            with torch.set_grad_enabled(False):
                p = np.random.uniform(low = p_low, high = p_high, size = inference_batch)
                theta = np.random.uniform(low = theta_low, high = theta_high, size = inference_batch)
                k = np.stack([p, theta], axis=1)
                k_unscaled = k.copy()
                k = 2*(k - conditional_mins) / (conditional_maxes - conditional_mins) - 1.0
                k = torch.tensor(k).to('cuda').float()
                gen = kaon_net.generate(k,unscaled_k=k_unscaled,method=args.sampling,temperature=args.temperature,topK=args.topK,nucleus_p=args.nucleus_p,
                                        dynamic_temp=args.dynamic_temperature,add_dark_noise=args.dark_noise,PID=321)

            generations += gen
            running_gen += len(gen)
            kbar.add(1)

            if running_gen >= args.n_tracks:
                break

            if len(generations) >= args.n_dump:
                out_path_ = os.path.join(out_folder,str(args.method)+f"_p_{args.momentum}_theta_{args.theta}_PID_{args.method}_ntracks_{len(generations)}_{file_count}.pkl")
                random.shuffle(generations)
                with open(out_path_,"wb") as file:
                    pickle.dump(generations,file)

                # reset lists
                generations = []
                file_count += 1

        if args.method in ['Pion','MixPiK']:
            with torch.set_grad_enabled(False):
                p = np.random.uniform(low = p_low, high = p_high, size = inference_batch)
                theta = np.random.uniform(low = theta_low, high = theta_high, size = inference_batch)
                k = np.stack([p, theta], axis=1)
                k_unscaled = k.copy()
                k = 2*(k - conditional_mins) / (conditional_maxes - conditional_mins) - 1.0
                k = torch.tensor(k).to('cuda').float()
                gen = pion_net.generate(k,unscaled_k=k_unscaled,method=args.sampling,temperature=args.temperature,topK=args.topK,nucleus_p=args.nucleus_p,
                                        dynamic_temp=args.dynamic_temperature,add_dark_noise=args.dark_noise,PID=211)

            generations += gen
            running_gen += len(gen)
            kbar.add(1)

            if running_gen >= args.n_tracks:
                break

            if len(generations) >= args.n_dump:
                out_path_ = os.path.join(out_folder,str(args.method)+f"_p_{args.momentum}_theta_{args.theta}_PID_{args.method}_ntracks_{len(generations)}_{file_count}.pkl")
                random.shuffle(generations)
                with open(out_path_,"wb") as file:
                    pickle.dump(generations,file)

                # reset lists
                generations = []
                file_count += 1
        
    end = time.time()

    time_per_track = (end - start) / len(generations) if len(generations) > 0 else r'N/A'
    
    print(" ")
    print("Sampling statistics:")
    print("Elapsed Time: ", end - start)
    print("Average time / track: ",time_per_track)
    print(" ")
    
    if len(generations) > 0:
        print(f'Writing final samples...')
        out_path_ = os.path.join(out_folder, str(args.method) + f"_p_{args.momentum}_theta_{args.theta}_PID_{args.method}_ntracks_{len(generations)}_{file_count}.pkl")
        with open(out_path_, "wb") as file:
            pickle.dump(generations, file)
    

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Dataset creation.')
    parser.add_argument('-c', '--config', default='CA_config.json',type=str,
                        help='Path to the config file (default: CA_config.json)')
    parser.add_argument('-nt', '--n_tracks', default=1e5,type=int,help='Number of particles to generate. Take the first n_tracks.')
    parser.add_argument('-nd', '--n_dump', default=None, type=int, help='Number of particles to dump per .pkl file.')
    parser.add_argument('-m', '--method',default="MixPiK",type=str,help='Generated particle type, Kaon, Pion, or MixPiK.')
    parser.add_argument('-d','--distributed',action='store_true',help='Trained with multiple GPUs - DDP.')
    parser.add_argument('-p','--momentum',default="6",type=str,help='Momentum value, or range.')
    parser.add_argument('-th','--theta',default="30",type=str,help='Theta value, or range.')
    parser.add_argument('-dn','--dark_noise',action='store_true',help='Included hits from dark noise with predefined rate. See source code for more details.')
    parser.add_argument('-tmp','--temperature',default=1.05,type=float,help='Generation temperature.')
    parser.add_argument('-dt', '--dynamic_temperature',action='store_true',help='Use dynamic temperature with predefined values. See source code for more details.')
    parser.add_argument('-s','--sampling',default="Nucleus",type=str,help='Default,TopK,Nucleus')
    parser.add_argument('-tk','--topK',default=300,type=int,help="TopK")
    parser.add_argument('-np','--nucleus_p',default=0.995,type=float,help="Nucleus P value - only used if sampling = Nucleus")
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)