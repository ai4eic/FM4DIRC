import os
import glob
import pickle
import numpy as np
from PyPDF2 import PdfWriter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn import metrics
from sklearn.metrics import roc_curve, auc,roc_auc_score

from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skewnorm

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

    efficiencies = TP / (TP + FN)  
    rejections = TN / (TN + FP)  
    auc = np.trapz(y=np.flip(rejections),x=np.flip(efficiencies))

    return efficiencies,rejections,auc

def fit_skewnorm(dll_k, dll_p, bins=200, normalized=True, n_bootstrap=100):
    if not normalized:
        raise ValueError("Skew-normal fit requires normalized histogram (PDF). Set normalized=True.")

    def fit_one(dll, bins):
        hist, bin_edges = np.histogram(dll, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        a_guess, loc_guess, scale_guess = 0, np.mean(dll), np.std(dll)

        popt, pcov = curve_fit(
            lambda x, a, loc, scale: skewnorm.pdf(x, a, loc, scale),
            bin_centers, hist,
            p0=[a_guess, loc_guess, scale_guess],
            maxfev=10000
        )
        a, loc, scale = popt
        delta = a / np.sqrt(1 + a**2)
        mean = loc + scale * delta * np.sqrt(2 / np.pi)
        stddev = scale * np.sqrt(1 - (2 * delta**2) / np.pi)
        return popt, mean, stddev, bin_centers

    try:
        popt_k, mean_k, stddev_k, bin_centers_k = fit_one(dll_k, bins)
        popt_p, mean_p, stddev_p, bin_centers_p = fit_one(dll_p, bins)
    except RuntimeError as e:
        print("Skewnorm fit error:", e)
        exit()

    sigma_sep = (mean_k - mean_p) / ((stddev_k + stddev_p) / 2.)

    # Bootstrap for uncertainty
    sigma_samples = []
    for _ in range(n_bootstrap):
        resample_k = np.random.choice(dll_k, size=len(dll_k), replace=True)
        resample_p = np.random.choice(dll_p, size=len(dll_p), replace=True)
        try:
            _, mk, sk, _ = fit_one(resample_k, bins)
            _, mp, sp, _ = fit_one(resample_p, bins)
            sigma_boot = (mk - mp) / ((sk + sp) / 2.)
            sigma_samples.append(sigma_boot)
        except RuntimeError:
            continue  # skip failed fits for now - doesn't occur

    sigma_err = np.std(sigma_samples)

    return popt_k, popt_p, sigma_sep, bin_centers_k, bin_centers_p, sigma_err, normalized

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
    
    sigma_sep = (mean_k - mean_p) / ((stddev_k + stddev_p)/2.)
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

def plot_skewnorm(popt, bin_centers, min_=-15,max_=15):
    a, loc, scale = popt

    x_vals = np.linspace(min_,max_, 1000)
    pdf_vals = skewnorm.pdf(x_vals, a, loc, scale)

    return x_vals,pdf_vals


def run_plotting(out_folder,momentum,model_type='Swin',skewnorm=False):

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


    thetas = [30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150]
    seps = []
    sep_err = []
    seps_cnf = []
    sep_err_cnf = []
    fig = plt.figure(figsize=(6,4))

    for theta in thetas:
        k_idx = np.where(kin_k[:,1] == theta)[0]
        p_idx = np.where(kin_p[:,1] == theta)[0]
        print("Theta: ",theta, "Pions: ",len(p_idx)," Kaons: ",len(k_idx))
        if not skewnorm:
            popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF,se,normalized = perform_fit(dll_k[k_idx],dll_p[p_idx],bins)
        else:
            popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF,se,normalized = fit_skewnorm(dll_k[k_idx],dll_p[p_idx],bins)
        seps.append(abs(sep_NF))
        sep_err.append(se)

        

        if not skewnorm:
            if normalized:
                gaussian = gaussian_normalized
            else:
                gaussian = gaussian_unnormalized
            plt.plot(bin_centers_k_NF, gaussian(bin_centers_k_NF, *popt_k_NF),color='blue', label=r"$\mathcal{K}$")
            plt.plot(bin_centers_p_NF, gaussian(bin_centers_p_NF, *popt_p_NF),color='red', label=r"$\pi$")

        else:
            x_pion,pdf_pion = plot_skewnorm(popt_p_NF, bin_centers_p_NF)
            x_kaon,pdf_kaon = plot_skewnorm(popt_k_NF, bin_centers_k_NF)
            plt.plot(x_kaon, pdf_kaon,color='blue', label=r"$\mathcal{K}$")
            plt.plot(x_pion, pdf_pion,color='red', label=r"$\pi$")

        plt.hist(dll_p[p_idx],bins=bins,density=normalized,color='red',histtype='step',lw=3)
        plt.hist(dll_k[k_idx],bins=bins,density=normalized,color='blue',histtype='step',lw=3)
        plt.legend(fontsize=18) 
        plt.title(r"$\theta = $ {0}".format(theta)+ r", $\sigma = $ {0:.2f}".format(sep_NF),fontsize=18)
        plt.xlabel(r"$Ln \, L(\mathcal{K}) - Ln \, L(\pi)$",fontsize=18)
        plt.ylabel("entries [#]",fontsize=18)
        plt.savefig(os.path.join(out_folder,"Gauss_fit_theta_{0}.pdf".format(theta)),bbox_inches="tight")
        plt.close()

    seps_NF = np.load(f"../Cherenkov_FastSim/Inference/NF_Comparison/{momentum}/Separation_NF_{int(momentum)}.pkl",allow_pickle=True)
    NF_err = np.load(f"../Cherenkov_FastSim/Inference/NF_Comparison/{momentum}/Errors_NF_{int(momentum)}.pkl",allow_pickle=True)
    
    results_dict = {"sigmas": seps,"errors": sep_err}

    with open(os.path.join(out_folder,"Results.pkl"),"wb") as file:
        pickle.dump(results_dict,file)
    
    seps = gaussian_filter1d(seps, sigma=1.25)

    seps_NF = gaussian_filter1d(seps_NF,sigma=1.25)

    fig = plt.figure(figsize=(12,6))

    plt.errorbar(thetas, seps, yerr=sep_err, color='black', lw=2, 
                label=str(model_type)+r' - $\bar{\sigma} = $' + "{0:.2f}".format(np.average(seps)), capsize=5, linestyle='--', 
                fmt='o', markersize=4)
    plt.errorbar(thetas, seps_NF, yerr=NF_err, color='blue', lw=2, 
                label=r'NF-DLL - $\bar{\sigma} = $' + "{0:.2f}".format(np.average(seps_NF)), capsize=5, linestyle='--', 
                fmt='o', markersize=4)
    plt.legend(fontsize=22,ncol=2,loc="lower left")
    plt.xlabel("Polar Angle [deg.]",fontsize=25,labelpad=15)
    plt.ylabel("Separation [s.d.]",fontsize=25,labelpad=15)
    plt.ylim(0,None)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if momentum == 6.0:
        plt.ylim(0,5)
    plt.title(r"$ {0} \; GeV/c$".format(int(momentum)),fontsize=28)
    plt.savefig(os.path.join(out_folder,"Seperation_{0}_NF_{1}GeV.pdf".format(str(model_type),int(momentum))),bbox_inches="tight")
    plt.close()

    print(" ")
    print(model_type)
    print("Average sigma: ",np.average(seps)," +- ",np.std(seps) / np.sqrt(len(seps)))
    print("NF")
    print("Average sigma: ",np.average(seps_NF)," +- ",np.std(seps_NF) / np.sqrt(len(seps_NF)))
    print(" ")