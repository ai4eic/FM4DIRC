import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def compute_pr_curve(probs, labels):
    steps = 100
    thresholds = np.linspace(0, 1, steps, endpoint=False)
    precisions = []
    recalls = []

    labels = labels.astype(bool)

    for t in thresholds:
        preds = probs >= t

        TP = np.sum((preds == 1) & (labels == 1)) # Predicted Noise is Noise
        FP = np.sum((preds == 1) & (labels == 0)) # Predicted Noise is Signal
        FN = np.sum((preds == 0) & (labels == 1)) # Predicted Signal is Noise

        if TP + FP == 0:
            precision = 1.0  # 
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            recall = 0.0  
        else:
            recall = TP / (TP + FN)

        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def compute_sig_eff_back_rej(probs, labels):
    steps = 100
    thresholds = np.linspace(0, 1, steps, endpoint=True)
    signal_retention = []  # TP / (TP + FN)
    background_rejection = []  # TN / (TN + FP)

    labels = labels.astype(bool)

    for t in thresholds:
        preds = (probs >= t).astype(int)

        TP = np.sum((preds == 1) & (labels == 1))  # Noise is predicted noise
        TN = np.sum((preds == 0) & (labels == 0))  # Signal is predicted signal
        FP = np.sum((preds == 1) & (labels == 0))  # Signal is predicted noise
        FN = np.sum((preds == 0) & (labels == 1))  # Noise is predicted signal

        sig_ret = TN / (TN + FP)
        back_rej = TP / (TP + FN)

        signal_retention.append(sig_ret)
        background_rejection.append(back_rej)

    signal_retention = np.array(signal_retention)
    background_rejection = np.array(background_rejection)

    sorted_idx = np.argsort(signal_retention)
    auc = np.trapz(background_rejection[sorted_idx], signal_retention[sorted_idx])

    return signal_retention, background_rejection, auc

def bootstrap_eff_rej(runs):
    sig_rets, back_rejs,aucs = [], [], []

    for run in runs:
        probs, labels = run['prob'], run['Truth']
        sig_ret, back_rej,auc = compute_sig_eff_back_rej(probs, labels)
        aucs.append(auc)
        sig_rets.append(sig_ret.reshape(1, -1))
        back_rejs.append(back_rej.reshape(1, -1))

    sig_rets = np.concatenate(sig_rets)
    back_rejs = np.concatenate(back_rejs)
    aucs = np.array(aucs)

    mean_sig_ret = sig_rets.mean(axis=0)
    mean_back_rej = back_rejs.mean(axis=0)
    mean_auc = np.mean(aucs)

    # 99% CI
    lower_sig_ret = np.quantile(sig_rets, 0.005, axis=0)
    upper_sig_ret = np.quantile(sig_rets, 0.995, axis=0)

    lower_back_rej = np.quantile(back_rejs, 0.005, axis=0)
    upper_back_rej = np.quantile(back_rejs, 0.995, axis=0)

    lower_auc = np.quantile(aucs,0.005)
    upper_auc = np.quantile(aucs,0.995)

    return (mean_sig_ret,lower_sig_ret,upper_sig_ret),(mean_back_rej,lower_back_rej,upper_back_rej),(mean_auc,lower_auc,upper_auc)

def bootstrap_pr(runs):
    aps = []
    precisions,recalls = [],[]

    for run in runs:
        probs,labels = run['prob'],run['Truth']
        precision,recall = compute_pr_curve(probs, labels)
        ap = average_precision_score(labels, probs, pos_label=1)
        precision = np.insert(precision,0,0)
        recall = np.insert(recall,0,1)
        precision = np.append(precision,1)
        recall = np.append(recall,0)
        precisions.append(precision.reshape(1,-1))
        recalls.append(recall.reshape(1,-1))
        aps.append(ap)


    precisions = np.concatenate(precisions)
    recalls = np.concatenate(recalls)

    mean_precision = precisions.mean(axis=0)
    mean_recall = recalls.mean(axis=0)

    # Compute 99% CI bounds
    lower_precision = np.quantile(precisions, 0.005, axis=0)
    upper_precision = np.quantile(precisions, 0.995, axis=0)

    lower_recall = np.quantile(recalls, 0.005, axis=0)
    upper_recall = np.quantile(recalls, 0.995, axis=0)

    # Mean and 99% CI for AP
    ap = np.mean(aps)
    ap_lower = np.quantile(aps, 0.005)
    ap_upper = np.quantile(aps, 0.995)

    return (mean_precision,lower_precision,upper_precision),(mean_recall,lower_recall,upper_recall),(ap,ap_lower,ap_upper)


def filtering_plots(Kaons, Pions, out_folder, momentum=6):
    # Helper to stylize legend
    def stylize_legend():
        legend = plt.legend(fontsize=22, loc='lower left')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_boxstyle('round,pad=0.3')

    # ===== Pions: Sig Eff vs. Background Rej =====
    signal, noise,metrics = bootstrap_eff_rej(Pions)
    mean_sig_ret, lower_sig_ret, upper_sig_ret = signal
    mean_back_rej, lower_back_rej, upper_back_rej = noise
    mean_auc,auc_lower,auc_upper = metrics

    fig = plt.figure(figsize=(8, 8))
    plt.plot(mean_sig_ret, mean_back_rej, color="k", label=f"AUC={mean_auc:.3f} [{auc_lower:.3f}, {auc_upper:.3f}]", lw=2, linestyle='--')
    plt.fill_between(mean_sig_ret, lower_back_rej, upper_back_rej, color="blue", alpha=0.25, label="99% CI Rejection",linewidth=0)
    plt.fill_betweenx(mean_back_rej, lower_sig_ret, upper_sig_ret, color="red", alpha=0.25, label="99% CI Retention",linewidth=0)

    plt.xlabel("Signal Retention", fontsize=30,labelpad=15)
    plt.ylabel("Noise Rejection", fontsize=30,labelpad=15)
    #plt.title("Signal Retention vs. Background Rejection", fontsize=25)
    plt.text(0.025,0.4, r"$\pi^{+-}$" "\n" rf"{int(momentum)} GeV/c ",
             fontsize=22, verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.tick_params(axis='x', labelsize=22, pad=10) 
    plt.tick_params(axis='y', labelsize=22, pad=10)
    plt.ylim(0, 1.025)
    plt.xlim(0, 1.025)
    plt.grid(True)
    stylize_legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "SigEff_BackRej_Pions.pdf"), bbox_inches="tight")
    plt.close()

    # ===== Kaons: Sig Eff vs. Background Rej =====
    signal, noise, metrics = bootstrap_eff_rej(Kaons)
    mean_sig_ret, lower_sig_ret, upper_sig_ret = signal
    mean_back_rej, lower_back_rej, upper_back_rej = noise
    mean_auc,auc_lower,auc_upper = metrics

    fig = plt.figure(figsize=(8, 8))
    plt.plot(mean_sig_ret, mean_back_rej, color="k", label=f"AUC={mean_auc:.3f} [{auc_lower:.3f}, {auc_upper:.3f}]", lw=2, linestyle='--')
    plt.fill_between(mean_sig_ret, lower_back_rej, upper_back_rej, color="blue", alpha=0.25, label="99% CI Rejection",linewidth=0)
    plt.fill_betweenx(mean_back_rej, lower_sig_ret, upper_sig_ret, color="red", alpha=0.25, label="99% CI Retention",linewidth=0)

    plt.xlabel("Signal Retention", fontsize=30,labelpad=15)
    plt.ylabel("Noise Rejection", fontsize=30,labelpad=15)
    #plt.title("Signal Retention vs. Background Rejection", fontsize=25)
    plt.text(0.025,0.4, r"$K^{+-}$" "\n" rf"{int(momentum)} GeV/c ",
             fontsize=22, verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.tick_params(axis='x', labelsize=22, pad=10) 
    plt.tick_params(axis='y', labelsize=22, pad=10)
    plt.ylim(0, 1.025)
    plt.xlim(0, 1.025)
    plt.grid(True)
    stylize_legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "SigEff_BackRej_Kaons.pdf"), bbox_inches="tight")
    plt.close()

    # ===== Pions: Precision-Recall Curve =====
    precisions, recalls, aps = bootstrap_pr(Pions)
    mean_precision, lower_precision, upper_precision = precisions
    mean_recall, lower_recall, upper_recall = recalls
    ap, ap_lower, ap_upper = aps

    fig = plt.figure(figsize=(8, 8))
    plt.plot(mean_recall, mean_precision,
             color="k", label=f"AP={ap:.3f} [{ap_lower:.3f}, {ap_upper:.3f}]",
             lw=2, linestyle='--')
    plt.fill_between(mean_recall, lower_precision, upper_precision,
                     color="blue", alpha=0.25, label="99% CI Precision",interpolate=True, linewidth=0)
    plt.fill_betweenx(mean_precision, lower_recall, upper_recall,
                      color="red", alpha=0.25,label="99% CI Recall",interpolate=True, linewidth=0)

    plt.xlabel("Recall (Noise Detection Rate)", fontsize=30,labelpad=15)
    plt.ylabel("Precision (Noise Purity)", fontsize=30,labelpad=15)
    #plt.title("Precision-Recall Curve", fontsize=25)
    #0.025,0.325
    plt.text(0.025,0.4, r"$\pi^{+-}$" "\n" rf"{int(momentum)} GeV/c ",
             fontsize=22, verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.ylim(0, 1.025)
    plt.xlim(0, 1.025)
    plt.tick_params(axis='x', labelsize=22, pad=10) 
    plt.tick_params(axis='y', labelsize=22, pad=10)
    plt.grid(True)
    stylize_legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "PR_Curve_Pions.pdf"), bbox_inches="tight")
    plt.close()

    # ===== Kaons: Precision-Recall Curve =====
    precisions, recalls, aps = bootstrap_pr(Kaons)
    mean_precision, lower_precision, upper_precision = precisions
    mean_recall, lower_recall, upper_recall = recalls
    ap, ap_lower, ap_upper = aps

    fig = plt.figure(figsize=(8, 8))
    plt.plot(mean_recall, mean_precision,
             color="k", label=f"AP={ap:.3f} [{ap_lower:.3f}, {ap_upper:.3f}]",
             lw=2, linestyle='--')
    plt.fill_between(mean_recall, lower_precision, upper_precision,
                     color="blue", alpha=0.25, label="99% CI Precision",interpolate=True, linewidth=0)
    plt.fill_betweenx(mean_precision, lower_recall, upper_recall,
                      color="red", alpha=0.25,label="99% CI Recall",interpolate=True, linewidth=0)
    

    plt.xlabel("Recall (Noise Detection Rate)", fontsize=30,labelpad=15)
    plt.ylabel("Precision (Noise Purity)", fontsize=30,labelpad=15)
    #plt.title("Precision-Recall Curve", fontsize=25)
    # 0.025,0.325
    plt.text(0.025,0.4, r"$K^{+-}$" "\n" rf"{int(momentum)} GeV/c ",
             fontsize=22, verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.ylim(0, 1.025)
    plt.xlim(0, 1.025)
    plt.tick_params(axis='x', labelsize=22, pad=10) 
    plt.tick_params(axis='y', labelsize=22, pad=10)
    plt.grid(True)
    stylize_legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "PR_Curve_Kaons.pdf"), bbox_inches="tight")
    plt.close()
