import torch.nn as nn
import numpy as np

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
