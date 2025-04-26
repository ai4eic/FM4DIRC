import os
import json
import argparse
import torch
import random
import time
import math
import pkbar
import pickle
import warnings
import itertools
import numpy as np
from datetime import datetime

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataloader.dataloader import CreateLoaders

from dataloader.tokenizer import TimeTokenizer
from models.GPT import Cherenkov_GPT
#from models.CA_GPT import Cherenkov_GPT

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors

from cuml.neighbors import KernelDensity
import cupy as cp
from utils.utils import convert_pmt_pix
from utils.KDE_utils import perform_fit_KDE,gaussian_normalized,gaussian_unnormalized,plot,FastDIRC
from utils.utils_hpDIRC import gapx,gapy,pixel_width,pixel_height,npix,npmt

warnings.filterwarnings("ignore", message=".*weights_only.*")
# This is a bandwith warning. Occurs with low number of support photons.
warnings.filterwarnings("ignore", message=".*Grid size 1 will likely result in GPU under-utilization.*")

def add_dark_noise(hits,dark_noise_pmt=28000):
    # probability to have a noise hit in 100 ns window
    prob = dark_noise_pmt * 100 / 1e9
    new_hits = []
    for p in range(npmt):
        for i in range(int(prob) + 1):
            if(i == 0) and (prob - int(prob) < np.random.uniform()):
                continue

            dn_time = 100 * np.random.uniform() # [1,100] ns
            dn_pix = int(npix * np.random.uniform())
            row = (p//6) * 16 + dn_pix//16 
            col = (p%6) * 16 + dn_pix%16
            x = 2 + col * pixel_width + (p % 6) * gapx + (pixel_width) / 2. # Center at middle
            y = 2 + row * pixel_height + (p // 6) * gapy + (pixel_height) / 2. # Center at middle
            h = [x,y,dn_time]
            new_hits.append(h)

    if new_hits:
        new_hits = np.array(new_hits)
        hits = np.vstack([hits,new_hits])

    return hits


def create_supports_geant(pions,kaons):
    pmtID = np.concatenate([pion['pmtID'] for pion in pions])
    pixelID = np.concatenate([pion['pixelID'] for pion in pions])
    t = np.concatenate([pion['leadTime'] for pion in pions])

    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    support_pions = np.vstack((x, y,t)).T

    pmtID = np.concatenate([kaon['pmtID'] for kaon in kaons])
    pixelID = np.concatenate([kaon['pixelID'] for kaon in kaons])
    t = np.concatenate([kaon['leadTime'] for kaon in kaons])

    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    support_kaons = np.vstack((x, y,t)).T  

    return support_pions[np.where((support_pions[:,2] < 100.0) & (support_pions[:,2] > 10.0))[0]],support_kaons[np.where((support_kaons[:,2] < 100.0) & (support_kaons[:,2] > 10.0))[0]]


def create_supports_fs(pions,kaons):
    pmtID = pions['pmtID']
    pixelID = pions['pixelID']
    t = pions['leadTime']

    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    support_pions = np.vstack((x, y,t)).T

    pmtID = kaons['pmtID']
    pixelID = kaons['pixelID']
    t = kaons['leadTime']

    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    support_kaons = np.vstack((x, y,t)).T  

    return support_pions[np.where((support_pions[:,2] < 100.0) & (support_pions[:,2] > 10.0))[0]],support_kaons[np.where((support_kaons[:,2] < 100.0) & (support_kaons[:,2] > 10.0))[0]]


def inference(tracks,dirc_obj,support_kaons,support_pions,add_dn=False):
    DLLs = []
    tprobs_k = []
    tprobs_p = []
    kbar = pkbar.Kbar(len(tracks))
    start = time.time()
    for i,track in enumerate(tracks):
        
        pixelID = np.array(track['pixelID'])
        pmtID = np.array(track['pmtID'])
        row = (pmtID//6) * 16 + pixelID//16 
        col = (pmtID%6) * 16 + pixelID%16
        
        x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
        y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
        t = track['leadTime']
        
        hits = np.vstack([x,y,t]).T

        if add_dn:
            hits = add_dark_noise(hits)

        ll_k,tprob_k = dirc_obj.get_log_likelihood(hits.astype('float16'),support_kaons.astype('float16'))
        ll_p,tprob_p = dirc_obj.get_log_likelihood(hits.astype('float16'),support_pions.astype("float16"))
        
        DLLs.append(ll_k - ll_p)
        tprobs_k.append({"rvalue": tprob_k,"coords":hits})
        tprobs_p.append({"rvalue": tprob_p,"coords":hits})
        kbar.update(i)
    end = time.time()

    print(" - Time/track: ",(end - start)/len(DLLs))

    return np.array(DLLs),tprobs_k,tprobs_p


def main(config,args):

    # Setup random seed
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    gpu_mem, _ = torch.cuda.mem_get_info()
    gpu_mem = gpu_mem / (1024 ** 3) # GB

    print("---------------- PDF Stats ------------------")
    print("Fast Simulated Support Tracks: ",args.fs_support)
    print("Geant4 Support Tracks: ",args.geant_support)
    print("---------------------------------------------")
    device = torch.device('cuda')

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
        time_digitizer = TimeTokenizer(t_max=t_max,t_min=t_min,resolution=time_res)
        de_tokenize_func = time_digitizer.de_tokenize

    else:
        print("Using regression over time domain.")
        time_digitizer = None
        de_tokenize_func = None


    dicte = torch.load(config['Inference']['pion_model_path'])
    pion_net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                    num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,detokenize_func=de_tokenize_func)
    pion_net.to('cuda')
    pion_net.load_state_dict(dicte['net_state_dict'])
    pion_net.eval()
    print("Loaded pion network.")

    dicte = torch.load(config['Inference']['kaon_model_path'])
    kaon_net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
                    num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,detokenize_func=de_tokenize_func)
    kaon_net.to('cuda')
    kaon_net.load_state_dict(dicte['net_state_dict'])
    kaon_net.eval()
    print("Loaded kaon network.")

    if args.momentum == 3.0:
        print("Loading 3GeV datasets.")
        geant = np.load(config['dataset']['time_imaging']["data_path_3GeV"],allow_pickle=True)
        inference_datapoints = np.load(config['dataset']['fixed_point']['pion_data_path_3GeV'],allow_pickle=True) \
                             + np.load(config['dataset']['fixed_point']['kaon_data_path_3GeV'],allow_pickle=True)
    elif args.momentum == 6.0:
        print("Loading 6GeV datasets.")
        geant = np.load(config['dataset']['time_imaging']["data_path_6GeV"],allow_pickle=True)
        inference_datapoints = np.load(config['dataset']['fixed_point']['pion_data_path_6GeV'],allow_pickle=True) \
                             + np.load(config['dataset']['fixed_point']['kaon_data_path_6GeV'],allow_pickle=True)
    elif args.momentum == 9.0:
        print("Loading 9GeV datasets.")
        geant = np.load(config['dataset']['time_imaging']["data_path_9GeV"],allow_pickle=True)
        inference_datapoints = np.load(config['dataset']['fixed_point']['pion_data_path_9GeV'],allow_pickle=True)  \
                             + np.load(config['dataset']['fixed_point']['kaon_data_path_9GeV'],allow_pickle=True)
    else:
        raise ValueError("Value of momentum does correspond to a dataset. Check if the path is correct, or simulate and processes.")


    os.makedirs("KDE_Fits",exist_ok=True)
    out_folder = config['Inference']['KDE_dir']
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

    if args.fs_inference:
        print("Using fast simulated data as inference to KDE. There is a slight overhead here in terms of time.")
    else:
        print("Using Geant4 data as inference to KDE.")

    if args.dark_noise:
        print("Adding dark noise to inference tracks.")
    else:
        pass

    fastDIRC = FastDIRC(device='cuda')
    sigma_dict_geant = {}
    sigma_dict_fs = {}
    thetas =  [30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150.] 
    #thetas = [140.]
    for theta_ in thetas:
        geant_support_pions = []
        geant_support_kaons = []
        inference_pions = []
        inference_kaons = []


        for i in range(len(geant)):
            if (geant[i]['Theta'] == theta_) and (geant[i]['P'] == args.momentum) and (geant[i]['Phi'] == 0.0) and (geant[i]['NHits'] < 250) and (geant[i]['NHits'] > 5):
                PDG = geant[i]['PDG']
                if PDG == 321 and len(geant_support_kaons) < args.geant_support:
                    geant_support_kaons.append(geant[i])
                elif PDG == 211 and len(geant_support_pions) < args.geant_support:
                    geant_support_pions.append(geant[i])
                else:
                    pass

                if len(geant_support_pions) == args.geant_support and len(geant_support_kaons) == args.geant_support:
                    break

        for i in range(len(inference_datapoints)):
            if (inference_datapoints[i]['Theta'] == theta_) and (inference_datapoints[i]['P'] == args.momentum) and (inference_datapoints[i]['Phi'] == 0.0) and (inference_datapoints[i]['NHits'] < 250) and (inference_datapoints[i]['NHits'] > 5):
                PDG = inference_datapoints[i]['PDG']
                if PDG == 321:
                    if not args.fs_inference:
                        inference_kaons.append(inference_datapoints[i])
                    else:
                        with torch.no_grad():
                            p = 2*(inference_datapoints[i]['P'] - stats['P_min'])  / (stats['P_max'] - stats['P_min']) - 1.0
                            theta = 2*(inference_datapoints[i]['Theta'] - stats['theta_min']) / (stats['theta_max'] - stats['theta_min']) - 1.0
                            k = torch.tensor(np.array([p,theta])).to('cuda').float().unsqueeze(0)
                            nhits = inference_datapoints[i]['NHits']
                            inf_k_ = kaon_net.create_tracks(num_samples=nhits,context=k,fine_grained_prior=args.fine_grained_prior)
                        inference_kaons.append(inf_k_)
                elif PDG == 211:
                    if not args.fs_inference:
                        inference_pions.append(inference_datapoints[i])
                    else:
                        with torch.no_grad():
                            p = 2*(inference_datapoints[i]['P'] - stats['P_min'])  / (stats['P_max'] - stats['P_min']) - 1.0
                            theta = 2*(inference_datapoints[i]['Theta'] - stats['theta_min']) / (stats['theta_max'] - stats['theta_min']) - 1.0
                            k = torch.tensor(np.array([p,theta])).to('cuda').float().unsqueeze(0)
                            nhits = inference_datapoints[i]['NHits']
                            inf_p_ = pion_net.create_tracks(num_samples=nhits,context=k,fine_grained_prior=args.fine_grained_prior)
                        inference_pions.append(inf_p_)
                else:
                    pass


        torch.cuda.empty_cache()
        print("Running inference for momentum = {0}, theta = {1}, for {2} pions and {3} kaons.".format(args.momentum,theta_,len(inference_pions),len(inference_kaons)))

        print("--------------- Fast Simulation -----------------")
        with torch.set_grad_enabled(False):
            k = np.array([args.momentum,theta_])
            k = 2*(k - conditional_mins) / (conditional_maxes - conditional_mins) - 1.0
            k = torch.tensor(k).to('cuda').float().unsqueeze(0).repeat(inference_batch, 1)
            
            fs_kaons = []
            fs_pions = []
            start = time.time()

            pion_pix,pion_t = pion_net.generate_PDF(k,numTracks=args.fs_support,temperature=1.)
            fs_pions = convert_pmt_pix(pion_pix,pion_t)
            del pion_pix,pion_t
            kaon_pix,kaon_t = kaon_net.generate_PDF(k,numTracks=args.fs_support,temperature=1.)
            fs_kaons = convert_pmt_pix(kaon_pix,kaon_t)
            del kaon_pix,kaon_t

        end = time.time()

        print(" ")
        support_pions,support_kaons = create_supports_fs(fs_pions,fs_kaons)
        print("Number of pion photons: ",len(support_pions)," Number of Kaon photons: ",len(support_kaons))
        del fs_kaons,fs_pions

        np.save(os.path.join(out_folder,f"FastSim_SupportPions_{theta_}.npy"),support_pions)
        np.save(os.path.join(out_folder,f"FastSim_SupportKaons_{theta_}.npy"),support_kaons)

        print("Time to create both PDFs: ",end - start)
        print("Time / photon: {0}.\n".format((end - start)/ (len(support_pions) + len(support_kaons))))
        print("Running inference for pions with Fast Simulated PDF.")
 
        DLL_p,tprobs_p_k,tprobs_p_p = inference(inference_pions,fastDIRC,support_kaons,support_pions,add_dn=args.dark_noise)
        torch.cuda.empty_cache()

        # See how likelihood is distributed, leaving for others.
        # with open(os.path.join(out_folder,f"FastSim_Pion_given_Kaon_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_p_k,file)

        # with open(os.path.join(out_folder,f"FastSim_Pion_given_Pion_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_p_p,file)  

        print(" ")
        print("Running inference for kaons with Fast Simulated PDF.")

        DLL_k,tprobs_k_k,tprobs_k_p = inference(inference_kaons,fastDIRC,support_kaons,support_pions,add_dn=args.dark_noise)
        torch.cuda.empty_cache()

        # with open(os.path.join(out_folder,f"FastSim_Kaon_given_Kaon_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_k_k,file)

        # with open(os.path.join(out_folder,f"FastSim_Kaon_given_Pion_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_k_p,file)  

        print("\n")
        DLL_k = DLL_k[~np.isnan(DLL_k)].astype('float32')
        DLL_p = DLL_p[~np.isnan(DLL_p)].astype('float32')
        DLL_k = DLL_k[~np.isinf(DLL_k)].astype('float32')
        DLL_p = DLL_p[~np.isinf(DLL_p)].astype('float32')
        fit_params = perform_fit_KDE(DLL_k,DLL_p,bins=200,normalized=False,momentum=args.momentum)
        plot(fit_params,DLL_p,DLL_k,"Fast Sim.",out_folder,theta_,pdf_method="Fast Sim.",bins=200,momentum=args.momentum)
        del support_pions,support_kaons
        sigma_dict_fs[theta_] = [fit_params[2],fit_params[-2]]


        # with open(os.path.join(out_folder,f"DLL_Pion_FastSim_theta_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(DLL_p,file)

        # with open(os.path.join(out_folder,f"DLL_Kaon_FastSim_theta_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(DLL_k,file)    

        

        print("------------------- Geant4 ---------------------")
        support_pions,support_kaons = create_supports_geant(geant_support_pions,geant_support_kaons)
        print("Number of pions: ",len(support_pions)," Number of Kaons: ",len(support_kaons))
        np.save(os.path.join(out_folder,f"Geant_SupportPions_{theta_}.npy"),support_pions)
        np.save(os.path.join(out_folder,f"Geant_SupportKaons_{theta_}.npy"),support_kaons)
        
        print("Running inference for pions with Geant4 PDF.")
        DLL_p,tprobs_p_k,tprobs_p_p = inference(inference_pions,fastDIRC,support_kaons,support_pions,add_dn=args.dark_noise)
        torch.cuda.empty_cache()

        # with open(os.path.join(out_folder,f"Geant_Pion_given_Kaon_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_p_k,file)

        # with open(os.path.join(out_folder,f"Geant_Pion_given_Pion_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_p_p,file)  
        print(" ")

        print("Running inference for kaons with Geant4 PDF.")
        
        DLL_k,tprobs_k_k,tprobs_k_p = inference(inference_kaons,fastDIRC,support_kaons,support_pions,add_dn=args.dark_noise)
        torch.cuda.empty_cache()

        # with open(os.path.join(out_folder,f"Geant_Kaon_given_Kaon_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_k_k,file)

        # with open(os.path.join(out_folder,f"Geant_Kaon_given_Pion_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_k_p,file)  

        print("\n")
        DLL_k = DLL_k[~np.isnan(DLL_k)].astype('float32')
        DLL_p = DLL_p[~np.isnan(DLL_p)].astype('float32')
        DLL_k = DLL_k[~np.isinf(DLL_k)].astype('float32')
        DLL_p = DLL_p[~np.isinf(DLL_p)].astype('float32')
        fit_params = perform_fit_KDE(DLL_k,DLL_p,bins=200,normalized=False,momentum=args.momentum)
        plot(fit_params,DLL_p,DLL_k,"Geant4",out_folder,theta_,pdf_method="Geant4",bins=200,momentum=args.momentum)
        print("\n")
        del support_pions,support_kaons
        sigma_dict_geant[theta_] = [fit_params[2],fit_params[-2]]

        # with open(os.path.join(out_folder,f"DLL_Pion_Geant_theta_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(DLL_p,file)

        # with open(os.path.join(out_folder,f"DLL_Kaon_Geant_theta_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(DLL_k,file)    



    with open(os.path.join(out_folder,"FastSim_Sigmas.pkl"),"wb") as file:
            pickle.dump(sigma_dict_fs,file)

    with open(os.path.join(out_folder,"Geant_Sigmas.pkl"),"wb") as file:
        pickle.dump(sigma_dict_geant,file)


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p', '--momentum', default=6.0,type=float,help='Particle Momentum.')
    parser.add_argument('-fs','--fs_support', default=1000,type=int,help='Number of Fast Simulated support tracks.')
    parser.add_argument('-fg','--geant_support', default=1000,type=int,help='Number of Geant4 support tracks.')
    parser.add_argument('-fsi','--fs_inference',action='store_true',help="Use Fast Simulated tracks for inference as opposed to Geant4 (default).")
    parser.add_argument('-dn', '--dark_noise',action='store_true',help="Add dark noise to inference tracks.")
    args = parser.parse_args()

    os.makedirs("KDE_Fits",exist_ok=True)
    os.makedirs(f"KDE_Fits/{args.momentum}",exist_ok=True)

    config = json.load(open(args.config))

    main(config,args)
