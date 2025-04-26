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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import metrics
from scipy.optimize import curve_fit
import glob
from PyPDF2 import PdfWriter
from scipy.stats import norm
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.special import expit

import torch
import torch.nn.functional as F

from dataloader.dataset import DIRC_Dataset_Classification
from dataloader.tokenizer import TimeTokenizer
from dataloader.dataloader import InferenceLoader

from models.GPT import Cherenkov_GPT

warnings.filterwarnings("ignore", message=".*weights_only.*")

def sigmoid(x):
    x = np.float64(x)
    return expit(x)


def compute_efficiency_rejection(delta_log_likelihood, true_labels):
    thresholds = np.linspace(-40.0, 40.0, 20000)
    thresholds_broadcasted = np.expand_dims(thresholds, axis=1)
    predicted_labels = delta_log_likelihood > thresholds_broadcasted

    TP = np.sum((predicted_labels == 1) & (true_labels == 1), axis=1)
    FP = np.sum((predicted_labels == 1) & (true_labels == 0), axis=1)
    TN = np.sum((predicted_labels == 0) & (true_labels == 0), axis=1)
    FN = np.sum((predicted_labels == 0) & (true_labels == 1), axis=1)

    efficiencies = TP / (TP + FN)  # Efficiency (True Positive Rate)
    rejections = TN / (TN + FP)  # Rejection (True Negative Rate)
    auc = np.trapz(y=np.flip(rejections),x=np.flip(efficiencies))

    return efficiencies,rejections,auc


def perform_fit(dll_k,dll_p,bins=200,normalized=False):
    if normalized:
        gaussian = gaussian_normalized
    else:
        gaussian = gaussian_unnormalized

    hist_k, bin_edges_k = np.histogram(dll_k, bins=bins, density=normalized)
    bin_centers_k = (bin_edges_k[:-1] + bin_edges_k[1:]) / 2
    try:
        popt_k, pcov_k = curve_fit(gaussian, bin_centers_k, hist_k, p0=[1, np.mean(dll_k), np.std(dll_k)],maxfev=1000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
        amplitude_k, mean_k, stddev_k = popt_k
        perr_k = np.sqrt(np.diag(pcov_k))
    except RuntimeError as e:
        print('Kaon error, exiting.')
        print(e)
        exit()
        

    hist_p, bin_edges_p = np.histogram(dll_p, bins=bins, density=normalized)
    bin_centers_p = (bin_edges_p[:-1] + bin_edges_p[1:]) / 2
    try:
        popt_p, pcov_p = curve_fit(gaussian, bin_centers_p, hist_p, p0=[1, np.mean(dll_p), np.std(dll_p)],maxfev=1000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
        amplitude_p, mean_p, stddev_p = popt_p
        perr_p = np.sqrt(np.diag(pcov_p))
    except RuntimeError as e:
        print('Pion error, exiting.')
        print(e)
        exit()
    
    sigma_sep = (mean_k - mean_p) / ((stddev_k + stddev_p)/2.) #np.sqrt(stddev_k**2 + stddev_p**2)
    sigma_err = (2*perr_k[1]/(stddev_k + stddev_p))** 2 + (2*perr_p[1]/(stddev_k + stddev_p))** 2 + (-2*(mean_k - mean_p) * perr_k[2] / (stddev_k + stddev_p)**2)**2 + (-2*(mean_k - mean_p) * perr_p[2] / (stddev_k + stddev_p)**2)**2
    return popt_k,popt_p,sigma_sep,bin_centers_k,bin_centers_p,np.sqrt(sigma_err), normalized

def gaussian_normalized(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) / (np.sqrt(2 * np.pi) * stddev)

def gaussian_unnormalized(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def extract_values(file_path):
    results = np.load(file_path,allow_pickle=True)
    sigmas = []
    thetas = []
    for theta, gr_value in results.items():
        if theta == 25.0:
            continue

        thetas.append(float(theta))
        sigmas.append(float(gr_value))
        
    sorted_thetas, sorted_sigmas = zip(*sorted(zip(thetas, sigmas)))

    return list(sorted_sigmas), list(sorted_thetas)


def run_plotting(out_folder,momentum,model_type='Swin'):

    LL_Kaon = np.load(os.path.join(out_folder,"Kaon_DLL_Results.pkl"),allow_pickle=True)
    LL_Pion = np.load(os.path.join(out_folder,"Pion_DLL_Results.pkl"),allow_pickle=True)

    kin_p = LL_Pion['Kins']
    kin_k = LL_Kaon['Kins']
    dll_p = LL_Pion['z_value']
    dll_k = LL_Kaon['z_value']
    print("NaN Checks: ",np.isnan(dll_k).sum())
    print("NaN Checks: ",np.isnan(dll_p).sum())
    dll_k = np.clip(dll_k[~np.isnan(dll_k)],-99999,99999)
    dll_p = np.clip(dll_p[~np.isnan(dll_p)],-99999,99999)
    kin_k =  kin_k[~np.isnan(dll_k)]
    kin_p = kin_p[~np.isnan(dll_p)]

    idx = np.where(kin_k[:,0] == momentum)[0]
    dll_k = dll_k[idx]
    kin_k = kin_k[idx]

    idx = np.where(kin_p[:,0] == momentum)[0]
    dll_p = dll_p[idx]
    kin_p = kin_p[idx]

    print("Pion max/min: ", dll_p.max(),dll_p.min())
    print("Kaon max/min: ",dll_k.max(),dll_k.min())

    
    if momentum == 6.0:
        bins = np.linspace(-20,20,400) 
    else:
        bins = np.linspace(-20,20,400) 

    ### Raw DLL
    plt.hist(dll_k,bins=bins,density=True,alpha=1.0,label=r'$\mathcal{K} - $'+str(model_type),color='red',histtype='step',lw=2)
    plt.hist(dll_p,bins=bins,density=True,alpha=1.0,label=r'$\pi - $'+str(model_type),color='blue',histtype='step',lw=2)
    plt.xlabel('Loglikelihood Difference',fontsize=25)
    plt.ylabel('A.U.',fontsize=25)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(r'$ \Delta \mathcal{L}_{\mathcal{K} \pi}$',fontsize=30)
    out_path_DLL = os.path.join(out_folder,"DLL_piK.pdf")
    plt.savefig(out_path_DLL,bbox_inches='tight')
    plt.close()



    thetas = [30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150.,155.]
    seps = []
    sep_err = []
    seps_cnf = []
    sep_err_cnf = []

    for theta in thetas:
        k_idx = np.where(kin_k[:,1] == theta)[0]
        p_idx = np.where(kin_p[:,1] == theta)[0]
        print("Theta: ",theta, "Pions: ",len(p_idx)," Kaons: ",len(k_idx))
        popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF,se,normalized = perform_fit(dll_k[k_idx],dll_p[p_idx],bins)
        seps.append(abs(sep_NF))
        sep_err.append(se)

        if normalized:
            gaussian = gaussian_normalized
        else:
            gaussian = gaussian_unnormalized

        fig = plt.figure(figsize=(6,4))
        plt.plot(bin_centers_k_NF, gaussian(bin_centers_k_NF, *popt_k_NF),color='blue', label=r"$\mathcal{K}$")
        plt.plot(bin_centers_p_NF, gaussian(bin_centers_p_NF, *popt_p_NF),color='red', label=r"$\pi$")
        plt.hist(dll_p[p_idx],bins=bins,density=normalized,color='red',histtype='step',lw=3)
        plt.hist(dll_k[k_idx],bins=bins,density=normalized,color='blue',histtype='step',lw=3)
        plt.legend(fontsize=18) 
        plt.title(r"$\theta = $ {0}".format(theta)+ r", $\sigma = $ {0:.2f}".format(sep_NF),fontsize=18)
        plt.xlabel(r"$Ln \, L(\mathcal{K}) - Ln \, L(\pi)$",fontsize=18)
        plt.ylabel("entries [#]",fontsize=18)
        plt.savefig(os.path.join(out_folder,"Gauss_fit_theta_{0}.pdf".format(theta)),bbox_inches="tight")
        plt.close()


    if momentum == 6.0:
        path_ = "LUT_Stats/6GeV/sigma_sep.pkl"
        sigma_10mill,theta_10mill = extract_values(path_)
    elif momentum == 3.0:
        path_ = "LUT_Stats/3GeV/sigma_sep.pkl"
        sigma_10mill,theta_10mill = extract_values(path_)
    elif momentum == 9.0:
        path_ = "LUT_Stats/9GeV/sigma_sep.pkl"
        sigma_10mill,theta_10mill = extract_values(path_)
    else:
        raise ValueError("Momentum value not found.")

    fig = plt.figure(figsize=(12,6))

    plt.errorbar(thetas, seps, yerr=sep_err, color='k', lw=2, 
                label=str(model_type)+r'- DLL - $\bar{\sigma} = $' + "{0:.2f}".format(np.average(seps)), capsize=5, linestyle='--', 
                fmt='o', markersize=4)
    plt.plot(theta_10mill,sigma_10mill,color='magenta',lw=2,linestyle='--',
            label=r'LUT - $\bar{\sigma} = $' + "{0:.2f}".format(np.average(sigma_10mill)),markersize=4,marker='o')
    plt.legend(fontsize=22,ncol=2)
    plt.xlabel("Polar Angle [deg.]",fontsize=25,labelpad=15)
    plt.ylabel("Separation [s.d.]",fontsize=25,labelpad=15)
    plt.ylim(0,None)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(r"$|\vec{p}| = "+ r" {0} \; GeV$".format(momentum),fontsize=28)
    plt.savefig(os.path.join(out_folder,"Seperation_{0}_LUT_{1}GeV.pdf".format(str(model_type),int(momentum))),bbox_inches="tight")
    plt.close()

    print(" ")
    print(model_type)
    print("Average sigma: ",np.average(seps)," +- ",np.std(seps) / np.sqrt(len(seps)))
    print("LUT")
    print("Average sigma: ",np.average(sigma_10mill)," +- ",np.std(sigma_10mill) / np.sqrt(len(sigma_10mill)))
    print(" ")


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
    #dll_geom = np.concatenate(dll_geom).astype('float32')
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
        LL_Kaon = np.load(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Kaon_DLL_Results.pkl"),allow_pickle=True)#[:10000]
        LL_Pion = np.load(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Pion_DLL_Results.pkl"),allow_pickle=True)#[:10000]
        print('Stats:',len(LL_Kaon),len(LL_Pion))
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
                            mlp_scale=mlp_scale,classification=True,time_vocab=time_vocab)

        net.to('cuda')
        model_path = config['Inference']['classifier_path']
        dicte = torch.load(model_path)
        net.load_state_dict(dicte['net_state_dict'])
        net.eval()
        

        LL_Pion,LL_Kaon = run_inference(net,test_loader,digitize_time,pad_token)

        
        print('Inference plots can be found in: ' + config['Inference']['out_dir_fixed'])
        os.makedirs(config['Inference']['out_dir_fixed'],exist_ok=True)
        os.makedirs(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum)+f"_{args.model_type}"),exist_ok=True)

        pion_path = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum)+f"_{args.model_type}","Pion_DLL_Results.pkl")
        with open(pion_path,"wb") as file:
            pickle.dump(LL_Pion,file)

        kaon_path = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum)+f"_{args.model_type}","Kaon_DLL_Results.pkl")
        with open(kaon_path,"wb") as file:
            pickle.dump(LL_Kaon,file)

        out_folder = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum)+f"_{args.model_type}")
        run_plotting(out_folder,args.momentum,args.model_type)



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='DLL at fixed kinematics.')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p','--momentum',default=6.0,type=float,help='Momentum value.')
    parser.add_argument('-mt','--model_type',default="GPT",type=str,help="Model type.")
    args = parser.parse_args()

    config = json.load(open(args.config))

    if not os.path.exists("Inference"):
        print("Making Inference Directory.")
        os.makedirs("Inference",exist_ok=True)

    main(config,args)
