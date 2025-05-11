import os
import json
import time
import copy
import random
import warnings
import argparse
import numpy as np

import torch

from models.GPT import Cherenkov_GPT
from make_plots import make_PDFs
from dataloader.tokenizer import TimeTokenizer

warnings.filterwarnings("ignore", message=".*weights_only.*")


def main(config,args):

    # Remove seeding, make it random.
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    inference_batch = config["Inference"]['batch_size']

    os.makedirs("PDFs",exist_ok=True)
    out_folder = os.path.join("PDFs",config['Inference']['pdf_dir'])
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

    print('------------------------ Setup ------------------------')
    print("Generating",args.fs_support,"photons, in batches of:", inference_batch,"tracks.")
    print(f"Trained with Multiple GPUs (DP): {'Yes' if args.distributed else 'No'}")
    print(f"Momentum Value: {args.momentum}")
    print(f"Include Hits from Dark Noise: {'Yes' if args.dark_noise else 'No'}")
    print(f"Generation Temperature: {args.temperature}")
    print(f"Use Dynamic Temperature: {'Yes' if args.dynamic_temperature else 'No'}")
    print(f"Sampling Method: {args.sampling}")
    print(f"TopK Value: {args.topK}")
    print(f"Nucleus P Value: {args.nucleus_p}")
    print('-------------------------------------------------------')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("No GPU was found! Exiting the program.")
    
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

    Kdicte = torch.load(config['Inference']['kaon_model_path'])
    kaon_net.to('cuda')
    kaon_net.load_state_dict(Kdicte['net_state_dict'])
    kaon_net.eval()
    kaon_net = torch.compile(model=kaon_net,mode="max-autotune")

    Pdicte = torch.load(config['Inference']['pion_model_path'])
    pion_net.to('cuda')
    pion_net.load_state_dict(Pdicte['net_state_dict'])
    pion_net.eval()
    pion_net = torch.compile(model=pion_net,mode="max-autotune")

    thetas =  [25,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150.,155.] 
    for theta_ in thetas:
        print(" ")
        torch.cuda.empty_cache()
        print("--------------- Fast PDF Simulation -----------------")
        print("Genearting pion and kaon PDFs for momentum = {0}, theta = {1}, of size {2}.".format(args.momentum,theta_,args.fs_support))

        with torch.set_grad_enabled(False):
            k = np.array([args.momentum,theta_])
            k_unscaled = np.tile(k.copy(), (inference_batch, 1))
            k = 2*(k - conditional_mins) / (conditional_maxes - conditional_mins) - 1.0
            k = torch.tensor(k).to('cuda').float().unsqueeze(0).repeat(inference_batch, 1)
            
            start = time.time()
            support_pions = pion_net.generate_PDF(k,k_unscaled,numPhotons=args.fs_support)
            support_kaons = kaon_net.generate_PDF(k,k_unscaled,numPhotons=args.fs_support)
            end = time.time()
          
        print("Time to create both PDFs: ",end - start)
        print("Time / photon: {0}.\n".format((end - start)/ (2*args.fs_support)))

        outpath = os.path.join(out_folder,f"Example_PDFs_theta_{theta_}_p_{args.momentum}.pdf")
        make_PDFs(support_pions, support_kaons, outpath, momentum=args.momentum, theta=theta_, log_norm=True)
 


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Dataset creation.')
    parser.add_argument('-c', '--config', default='CA_config.json',type=str,
                        help='Path to the config file (default: CA_config.json)')
    parser.add_argument('-fs','--fs_support', default=10e4,type=float,help='Number of Fast Simulated support photons.')
    parser.add_argument('-d','--distributed',action='store_true',help='Trained with multiple GPUs - DDP.')
    parser.add_argument('-p','--momentum',default=6.0,type=float,help='Momentum value, or range.')
    parser.add_argument('-th','--theta',default="30",type=str,help='Theta value, or range.')
    parser.add_argument('-dn','--dark_noise',action='store_true',help='Included hits from dark noise with predefined rate. See source code for more details.')
    parser.add_argument('-tmp','--temperature',default=1.05,type=float,help='Generation temperature.')
    parser.add_argument('-dt', '--dynamic_temperature',action='store_true',help='Use dynamic temperature with predefined values. See source code for more details.')
    parser.add_argument('-s','--sampling',default="Nucleus",type=str,help='Default,TopK,Nucleus')
    parser.add_argument('-tk','--topK',default=300,type=int,help="TopK")
    parser.add_argument('-np','--nucleus_p',default=0.995,type=float,help="Nucleus P value - only used if sampling = Nucleus")
    args = parser.parse_args()

    args.fs_support = int(args.fs_support)
    config = json.load(open(args.config))

    main(config,args)