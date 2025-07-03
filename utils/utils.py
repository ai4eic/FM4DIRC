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

def init_from_MoE(net_state_dict,net):
    keys = list(net_state_dict.keys())
    layers_ = [s for s in keys if 'layers' in s]
    num_layers = layers_[-1].split(".")[1]
    avg_weights = {}
    for i in range(int(num_layers)+1):
        sub_keys = [s for s in keys if f"layers.{i}.FF.experts" in s]
        num_experts = sub_keys[-1].split(".")[-4]
        expert_W1,expert_W2,expert_B1,expert_B2 = [],[],[],[]
        
        for j in range(int(num_experts)+1):
            expert_W1 += [s for s in keys if f"layers.{i}.FF.experts.{j}.nn.0.weight" in s]
            expert_W2 += [s for s in keys if f"layers.{i}.FF.experts.{j}.nn.2.weight" in s]
            expert_B1 += [s for s in keys if f"layers.{i}.FF.experts.{j}.nn.0.bias" in s]
            expert_B2 += [s for s in keys if f"layers.{i}.FF.experts.{j}.nn.2.bias" in s]

        expert_W1 = torch.concat([net_state_dict[s].unsqueeze(0) for s in expert_W1],dim=0)
        expert_W2 = torch.concat([net_state_dict[s].unsqueeze(0) for s in expert_W2],dim=0)
        avg_weight1 = torch.mean(expert_W1,dim=0)
        avg_weight2 = torch.mean(expert_W2,dim=0)

        expert_B1 = torch.concat([net_state_dict[s].unsqueeze(0) for s in expert_B1],dim=0)
        expert_B2 = torch.concat([net_state_dict[s].unsqueeze(0) for s in expert_B2],dim=0)
        avg_bias1 = torch.mean(expert_B1,dim=0)
        avg_bias2 = torch.mean(expert_B2,dim=0)

        
        try:
            if hasattr(net.layers[i],"FF"):
                net.layers[i].FF.nn[0].weight = torch.nn.Parameter(avg_weight1,requires_grad=True).to(net.device)
                net.layers[i].FF.nn[2].weight = torch.nn.Parameter(avg_weight2,requires_grad=True).to(net.device)
                net.layers[i].FF.nn[0].bias = torch.nn.Parameter(avg_bias1,requires_grad=True).to(net.device)
                net.layers[i].FF.nn[2].bias = torch.nn.Parameter(avg_bias2,requires_grad=True).to(net.device)
            print(f"Succesfully initalized FF block {i} from average of {int(num_experts)+1} expert weights")
            
        except (AttributeError, IndexError) as e:
            print(f"Skipping layer {i} due to missing FF structure: {e}")
            
    return net