import math
import numpy as np
from scipy.stats import cosine

import torch
import torch.nn as nn

from utils.utils_hpDIRC import gapx,gapy,pixel_width,pixel_height,npmt,npix


def time_loss_fn(true_times, pred_times, padding_mask):
    true_times = true_times.view(-1)
    pred_times = pred_times.view(-1)
    

    mask = padding_mask.view(-1) == 0 
    
    loss = nn.SmoothL1Loss(reduction='none')(pred_times, true_times)  
    loss = loss * mask  
    loss = loss.sum() / mask.sum()  

    return loss


def convert_indices_gt(pmtID,pixelID): 
    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    
    return x,y

def convert_indices(p,t):
    pmtID = p // npix
    pixelID = p % npix

    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle

    return np.concatenate([np.c_[x],np.c_[y],np.c_[t]],axis=1)

    
def convert_pmt_pix(p,t):
    pmtID = p // npix
    pixelID = p % npix

    return {"pmtID":pmtID,"pixelID":pixelID,"leadTime":t}


def scaled_cosine(theta):
    rv = cosine(loc=0, scale=0.3)  
    x = np.linspace(-1, 1, 100)
    y = rv.pdf(x)
    y_min = y.min()
    y_max = y.max()
    return 1 * (rv.pdf(theta) - y_min) / (y_max - y_min) + 1

def dynamic_batch(gpu_mem, total_samples,theta_value,verbose=True):
    # when doing fixed point generation, we can dynamically batch quite easily
    # largest VRAM overhead occurs towards the tails of theta ~ batch of 85 / 24GB
    # we can essentially tripple this at middle region
    # follow cosine distribution
    default_known_mem = 24
    mem_scale = math.ceil(gpu_mem / default_known_mem) # likely to have 48,96,etc
    events_min = 85 * mem_scale
    inference_batch = math.ceil(events_min * scaled_cosine(theta_value))

    num_itter = total_samples // inference_batch
    last_batch = total_samples % inference_batch

    if verbose:
        print("----------- Dynamic Batching --------------")
        print("Batch size: ",inference_batch, " numItter: ",num_itter," last_batch: ",last_batch)
        print("-------------------------------------------")

    return inference_batch,num_itter,last_batch

def log_vram_usage(tag=""):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print(f"VRAM_LOG,{tag},{allocated:.2f},{reserved:.2f},{peak_allocated:.2f}")
    return allocated, reserved, peak_allocated