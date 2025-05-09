import torch.nn as nn
import numpy as np

class TimeTokenizer():
    def __init__(self,t_max=157.0,t_min=9.0,resolution=0.025):
        super().__init__()
        self.t_max = t_max
        self.t_min = t_min 
        self.time_res = resolution
        self.t_bins = np.arange(self.t_min,self.t_max + self.time_res,self.time_res)

    def tokenize(self,times):
        return np.digitize(times,self.t_bins)

    def de_tokenize(self,tokens):
        z = self.t_min + (tokens + 0.5) * self.time_res
        # sample with time resolution
        #z = z + np.random.normal(loc=0,scale=self.time_res * 0.5,size=tokens.shape)
        z = z + np.random.uniform(-0.5 * self.time_res, 0.5 * self.time_res, size=tokens.shape)
        return np.clip(z,self.t_min,self.t_max)


        
