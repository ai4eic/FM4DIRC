
import math
import pkbar
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from utils.utils_hpDIRC import gapx,gapy,pixel_width,pixel_height,npmt,npix

class ResNetBlock(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.activation = nn.ReLU()

    def forward(self, x):
        inputs = x
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x) + inputs)
        return x


def resnet_subnet(c_in, c_out):
    layers = [nn.Linear(c_in,hidden_units)]
    
    # Stack residual blocks
    for _ in range(num_blocks):
        layers.append(ResNetBlock(hidden_units))
    
    layers += [nn.Linear(hidden_units, c_out)]
    return nn.Sequential(*layers)


class TimeRegression(nn.Module):
    def __init__(self,num_blocks,hidden_units,embed_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_units = hidden_units
        self.embed_dim = embed_dim

        layers = [nn.Linear(self.embed_dim,self.hidden_units)]

        for _ in range(self.num_blocks):
            layers.append(ResNetBlock(self.hidden_units))

        layers += [nn.Linear(self.hidden_units,1),nn.ReLU()]

        self.nn = nn.Sequential(*layers)

    def forward(self,x,k=None):
            return self.nn(x)


class FF(nn.Module):
    def __init__(self,embed_dim, mlp_scale : int = 2, drop_rate: float = 0.0):
        super().__init__()
        self.nn = nn.Sequential(*[nn.Linear(embed_dim,embed_dim * mlp_scale),nn.GELU(),nn.Linear(embed_dim * mlp_scale,embed_dim),nn.Dropout(drop_rate)])

    def forward(self,x):
        return self.nn(x)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len=250, dropout=0.2,device='cuda',qk_norm=True):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.seq_length = seq_len
        self.head_dim = embed_dim // num_heads
        self.d_k = self.head_dim ** -0.5
        self.device = device
        self.qk_norm = qk_norm
        self.g_scale = nn.Parameter(torch.tensor(1.0/self.d_k,dtype=torch.float,device=self.device),requires_grad=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,t_embed,attn_mask=None,key_padding_mask=None,need_weights=False):
        batch_size, seq_len, embed_dim = x.shape

        # Treat soft(qk)v - query from time to space (key) - extract values from space (value)
        q, k, v = self.q_proj(t_embed),self.k_proj(x),self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # normalize QK matrices with L2 norm over head dim, learnable scale clamped to d_k 
        if self.qk_norm:
            k = F.normalize(k,p=2,dim=-1)
            q = F.normalize(q,p=2,dim=-1)
            attn_scores = self.g_scale * q @ k.transpose(2,3)

        else:
            attn_scores = self.d_k * q @ k.transpose(2,3) 

        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask,-torch.inf)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_scores.masked_fill_(key_padding_mask,-torch.inf)

        attn_scores = F.softmax(attn_scores,dim=-1) # consider forcing to float32
        attn_scores = self.dropout(attn_scores)

        attn_output = (attn_scores @ v).transpose(1,2)

        attn_output = attn_output.contiguous().view(batch_size,seq_len,embed_dim)

        if need_weights:
            return attn_output,attn_scores
        else:
            return attn_output,None


class CATransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads, mlp_scale : int = 2,drop_rate: float = 0.2, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.mlp_scale = mlp_scale
        self.drop_rate = drop_rate 
        self.xN = nn.LayerNorm(self.embed_dim)
        self.tN = nn.LayerNorm(self.embed_dim)
        self.attn = CrossAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate,device=self.device)
        self.c_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.LN2 = nn.LayerNorm(self.embed_dim)
        self.FF = FF(self.embed_dim,mlp_scale=self.mlp_scale)


    def generate_mask(self,seq_len):
        return torch.triu(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool), diagonal=1)

    def forward(self,x,t_embed,padding_mask=None,need_weights=False,classification=False):
        B,N_t,t_dim = x.shape
        x_norm = self.xN(x)
        t_norm = self.tN(t_embed)

        if not classification:
            mask_ = self.generate_mask(N_t)
            attn,attn_weights = self.attn(x_norm, t_norm, attn_mask=mask_,key_padding_mask=padding_mask,need_weights=False)
        else:
            attn,attn_weights = self.attn(x_norm, t_norm,key_padding_mask=padding_mask,need_weights=False)

        attn = self.c_proj(attn)
        x = x + attn
        x = x + self.FF(self.LN2(x))
        return x


class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len=250, dropout=0.2,device='cuda',qk_norm=True):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.seq_length = seq_len
        self.head_dim = embed_dim // num_heads
        self.d_k = self.head_dim ** -0.5
        self.device = device
        self.qk_norm = qk_norm
        self.g_scale = nn.Parameter(torch.tensor(1.0/self.d_k,dtype=torch.float,device=self.device),requires_grad=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x,attn_mask=None,key_padding_mask=None,need_weights=False):
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.q_proj(x),self.k_proj(x),self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # normalize QK matrices with L2 norm over head dim, learnable scale init to d_k 
        if self.qk_norm:
            k = F.normalize(k,p=2,dim=-1)
            q = F.normalize(q,p=2,dim=-1)
            attn_scores = self.g_scale * q @ k.transpose(2,3)

        else:
            attn_scores = self.d_k * q @ k.transpose(2,3) 
            

        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask,-torch.inf)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_scores.masked_fill_(key_padding_mask,-torch.inf)

        attn_scores = F.softmax(attn_scores,dim=-1) # consider forcing to float32
        attn_scores = self.dropout(attn_scores)

        attn_output = (attn_scores @ v).transpose(1,2)

        attn_output = attn_output.contiguous().view(batch_size,seq_len,embed_dim)

        if need_weights:
            return attn_output,attn_scores
        else:
            return attn_output,None


class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads, mlp_scale : int = 2,drop_rate: float = 0.2, device='cuda'):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.mlp_scale = mlp_scale 
        self.drop_rate = drop_rate
        self.LN1 = nn.LayerNorm(self.embed_dim)
        self.attn = MHSA(self.embed_dim, self.num_heads, dropout=self.drop_rate,device=self.device)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.LN2 = nn.LayerNorm(self.embed_dim)
        self.FF = FF(self.embed_dim,mlp_scale=self.mlp_scale)


    def generate_mask(self,seq_len):
        return torch.triu(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool), diagonal=1)

    def forward(self, x,padding_mask=None,need_weights=False,classification=False):
        B,N_t,t_dim = x.shape
        x_norm = self.LN1(x)

        if not classification:
            mask_ = self.generate_mask(N_t)
            attn,attn_weights = self.attn(x_norm,attn_mask=mask_,key_padding_mask=padding_mask, need_weights=need_weights)
        else:
            attn,attn_weights = self.attn(x_norm,key_padding_mask=padding_mask, need_weights=need_weights)

        attn = self.c_proj(attn)

        x = x + attn
        x = x + self.FF(self.LN2(x))
        return x



class Cherenkov_GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim,attn_heads = [2,4,2],kin_size=2,
                num_blocks=2,hidden_units=128, digitize_time=True, mlp_scale : int = 2,
                time_vocab : int = 5923,drop_rates=[0.0,0.0,0.0],
                detokenize_func = None,
                classification=False,
                device='cuda'):
        super().__init__()
        self.classification = classification
        self.digitize_time = digitize_time
        self.detokenize_func = detokenize_func
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.time_pos_embedding = nn.Embedding(seq_len,embed_dim)
        self.momentum_embedding = nn.Linear(1,embed_dim) 
        self.theta_embedding = nn.Linear(1,embed_dim)
        # Can refactor this - fine for now
        layers_ = [CATransformerBlock(embed_dim, attn_heads[0], mlp_scale,drop_rate=drop_rates[0])]  
        layers_ += [TransformerBlock(embed_dim, attn_heads[i], mlp_scale,drop_rate=drop_rates[i]) for i in range(1, len(attn_heads))] 
        self.layers = nn.ModuleList(layers_)
        self.LN = nn.LayerNorm(embed_dim)

        if not self.classification:
            if self.digitize_time: # Multiclass
                self.time_embedding = nn.Embedding(time_vocab,embed_dim)
                self.time_head = nn.Linear(embed_dim,time_vocab) 
            else: # Regression
                self.time_embedding = nn.Linear(1,embed_dim)
                self.time_head = TimeRegression(num_blocks,hidden_units,embed_dim)

            self.logits_head = nn.Linear(embed_dim, vocab_size)

        else:
            if self.digitize_time: # Time resolution based tokenization
                self.time_embedding = nn.Embedding(time_vocab,embed_dim)
            else: # Fully continuous
                self.time_embedding = nn.Linear(1,embed_dim)

            self.classification_head = nn.Linear(embed_dim,1)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.device = device
        self.SOS_token = 0
        self.EOS_token = 6145
        self.pad_token = 6146
        self.time_pad_token = time_vocab - 1
        self.EOS_time_token = time_vocab - 2

    def forward(self, x,t,k,padding_mask=None):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        if not self.digitize_time:
            t = t.reshape(-1, 1)  # [batch_size * seq_len, 1]
            t_embed_flat = self.time_embedding(t)  # 
            t_embed = t_embed_flat.view(batch_size, seq_len, t_embed_flat.shape[-1]) # [batch_size, seq_len,embed_dim]
            t_embed = t_embed + self.time_pos_embedding(pos)
        else:
            t_embed = self.time_embedding(t) + self.time_pos_embedding(pos)
            
        momentum = k[:,0].unsqueeze(-1)
        theta = k[:,1].unsqueeze(-1)
        p_embed = self.momentum_embedding(momentum).unsqueeze(1)  
        theta_embed = self.theta_embedding(theta).unsqueeze(1)  

        x = self.token_embedding(x) + self.pos_embedding(pos) 
        t_embed = torch.cat((p_embed,theta_embed,t_embed),dim=1)
        x = torch.cat((p_embed, theta_embed, x), dim=1) 

        # Instead of adding time and position embeddings, combine through Cross attention
        # Query from time space, given space (key,value) 

        if padding_mask is not None:
            kinematic_mask = torch.zeros(batch_size, k.shape[-1], dtype=torch.bool, device=x.device)  # No masking for kinematic tokens
            padding_mask = torch.cat((kinematic_mask, padding_mask), dim=1) 

        if self.classification:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            cls_mask = torch.zeros(batch_size,1,dtype=torch.bool,device=x.device) # no mask for cls token
            padding_mask = torch.cat((cls_mask,padding_mask),dim=1)
            x = torch.cat((cls_tokens,x),dim=1)
            t_embed = torch.cat((cls_tokens,t_embed),dim=1)

        for layer in self.layers:
            if layer.__class__.__name__ == "CATransformerBlock":
                x = layer(x, t_embed,padding_mask=padding_mask,classification=self.classification)
            else:
                x = layer(x,padding_mask=padding_mask,classification=self.classification)
        
        x = self.LN(x)
 
        if not self.classification: # Generations - next hit prediction
            if not self.digitize_time:
                t_out = self.time_head(x).squeeze(-1) # direct regression of time 
            else:
                t_out = self.time_head(x) # logits over time 

            pixel = self.logits_head(x)

            return pixel,t_out

        else:
            return self.classification_head(x[:, 0]).squeeze(-1)

    def __topK(self,logits,topK=50):
        topk_logits, topk_indices = torch.topk(logits, k=topK, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        idx_next = topk_indices.gather(-1, sampled)
        return idx_next

    def __min_p(self,logits, min_p=0.05, min_tokens_to_keep=50, return_logits=False):
        assert 0 <= min_p <= 1, "min_p must be between 0 and 1"

        probs = torch.softmax(logits, dim=-1)
        p_max = torch.max(probs, dim=-1,keepdim=True).values
        p_scaled = min_p * p_max
        min_p_mask = probs < p_scaled

        sorted_indices = torch.argsort(logits, descending=True, dim=-1)
        sorted_indices_to_remove = min_p_mask.gather(-1, sorted_indices)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)

        min_p_logits = logits.masked_fill(indices_to_remove, float('-inf'))
        min_p_probs = torch.softmax(min_p_logits, dim=-1)

        sample_token = torch.multinomial(min_p_probs, num_samples=1)

        if return_logits:
            return sample_token, min_p_logits
        return sample_token

    def __nucleus(self,logits,p=0.9):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(probs, dim=-1)

        mask = cumsum_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        sorted_logits[mask] = float('-inf')
        filtered_logits = torch.gather(sorted_logits, -1, torch.argsort(sorted_indices, dim=-1))

        probs = torch.softmax(filtered_logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1) 

        return idx_next

    def __linear_dynamic_temp(self,step,max_length,max_temp=1.0,min_temp=0.95):
        return   max(max_temp - step / max_length,min_temp)
    
    def __exp_dynamic_temp(self,step, max_length, max_temp=1.05, min_temp=0.95):
        alpha = (max_temp - min_temp)
        decay_rate = -math.log(1e-2) / max_length  
        temperature = min_temp + alpha * math.exp(-decay_rate * step)
        return temperature

    def make_track(self,p,t,k,add_dark_noise=False,PID=None):
        num_samples = p.shape[0]
        pmtID = p // npix
        pixelID = p % npix
        channel = p

        if PID is None:
            PID = -1

        row = (pmtID//6) * 16 + pixelID//16 
        col = (pmtID%6) * 16 + pixelID%16
        
        x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
        y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle

        Phi = 0.0
        if add_dark_noise:
            x,y,t,pmtID,pixelID,channel,dn_hits = self.__add_dark_noise(np.concatenate([np.c_[x],np.c_[y],np.c_[t],np.c_[pmtID],np.c_[pixelID],np.c_[channel]],axis=1))
            num_samples += dn_hits
            return {"NHits":num_samples,"P":k[0],"Theta":k[1],"Phi":Phi,"x":x,"y":y,"leadTime":t,"pmtID":pmtID,"pixelID":pixelID,"channel":channel,"PID":PID}
        else:
            return {"NHits":num_samples,"P":k[0],"Theta":k[1],"Phi":Phi,"x":x,"y":y,"leadTime":t,"pmtID":pmtID,"pixelID":pixelID,"channel":p,"PID":PID}


    def post_process(self, pixels, times,k,add_dark_noise=False,PID=None):
        processed_tracks = []
        valid_pixels = []
        valid_times = []

        pixel_mask_tokens = torch.tensor([self.pad_token, self.SOS_token, self.EOS_token], device=pixels.device)
        time_mask_tokens = torch.tensor([self.time_pad_token, self.SOS_token, self.EOS_time_token], device=pixels.device)
        
        for i,(idx, t) in enumerate(zip(pixels, times)):
            pixel_mask = ~torch.isin(idx, pixel_mask_tokens)
            time_mask = ~torch.isin(t, time_mask_tokens)
            mask = pixel_mask & time_mask  # must be valid in both pixel and time

            if idx[mask].shape[0] < 5 or idx[mask].shape[0] > 250: # incase
                continue
            
            valid_pixels.append(idx[mask])
            valid_times.append(t[mask])
            processed_pixels = (idx[mask].clone() - 1).detach().cpu().numpy()
            if self.digitize_time and self.detokenize_func is not None:
                processed_times = self.detokenize_func(t[mask].clone().detach().cpu().numpy())
            else:
                processed_times = t[mask].detach().cpu().numpy()
            
            processed_tracks.append(self.make_track(processed_pixels,processed_times,k[i],add_dark_noise=add_dark_noise,PID=PID))

        return processed_tracks


    @torch.no_grad()
    def generate(self, k, unscaled_k, max_seq_len: int = 250, context_len = None,
                 temperature: float = 1.05, method="Default", topK=100, nucleus_p=0.98,
                 dynamic_temp=False,add_dark_noise=False,PID=None):

        assert method in ["Nucleus", "TopK", "Default","Greedy","Min_p"]
        batch_size = k.shape[0]

        # Start tokens
        idx = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)  # pixel token
        if self.digitize_time:
            t = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)  # time token
        else:
            t = torch.zeros(batch_size,1).to(self.device).float()

        is_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step_ in range(max_seq_len):
            if dynamic_temp:
                temperature = self.__exp_dynamic_temp(step_,max_seq_len)
            
            if context_len is None:
                idx_cond = idx
                t_cond = t
            else:
                idx_cond = idx[:, -context_len:]
                t_cond = t[:, -context_len:]

            logits, logits_time = self(idx_cond, t_cond, k, padding_mask=None)

            logits = logits[:, -1, :] / temperature

            # ---- Pixel sampling ----
            if method == "Default":
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            elif method == "TopK":
                idx_next = self.__topK(logits, topK)
            elif method == "Nucleus":
                idx_next = self.__nucleus(logits, nucleus_p)
            elif method == "Greedy":
                idx_next = torch.argmax(logits, dim=-1).unsqueeze(1)
            elif method == "Min_p":
                idx_next = self.__min_p(logits, return_logits=False)
            elif method == "Typical":
                idx_next = self.__typical_sampling(logits,return_logits=False)

            # ---- Time sampling ----
            if self.digitize_time:
                logits_time = logits_time[:, -1, :] / temperature
                if method == "Default":
                    probs_time = F.softmax(logits_time, dim=-1,dtype=torch.float32)
                    t_next = torch.multinomial(probs_time, num_samples=1)
                elif method == "TopK":
                    t_next = self.__topK(logits_time, topK)
                elif method == "Nucleus":
                    t_next = self.__nucleus(logits_time, nucleus_p)
                elif method == "Greedy":
                    t_next = torch.argmax(logits_time, dim=-1).unsqueeze(1)
                elif method == "Min_p":
                    t_next = self.__min_p(logits_time, return_logits=False)
                elif method == "Typical":
                    t_next = self.__typical_sampling(logits_time,return_logits=False)
            else:
                t_next = logits_time[:, -1].unsqueeze(1)

            # pixel, or time
            is_done |= (t_next.squeeze(1) == self.EOS_time_token) | (idx_next.squeeze(1) == self.EOS_token)

            idx_next[is_done] = self.EOS_token
            t_next[is_done] = self.EOS_time_token

            idx = torch.cat((idx, idx_next), dim=1)
            t = torch.cat((t, t_next), dim=1)

            if torch.all(is_done):
                break
        
        return self.post_process(idx,t,unscaled_k,add_dark_noise,PID)


    @torch.no_grad()
    def generate_PDF(self,kinematics,unscaled_k, numPhotons=2e5,max_seq_len: int = 250,
                 context_len = None, temperature: float = 1.05, method="Nucleus", topK=100,
                 nucleus_p=0.995, dynamic_temp=False,add_dark_noise=False):

        assert kinematics is not None

        batch_size = kinematics.shape[0]
        kbar = pkbar.Kbar(target=numPhotons, width=20, always_stateful=False)

        torch.cuda.empty_cache()
        tracks = []
        n_total = 0
        while n_total < numPhotons:

            with torch.no_grad():
                track = self.generate(kinematics,unscaled_k,method=method,temperature=temperature,
                                      topK=topK,nucleus_p=nucleus_p,dynamic_temp=dynamic_temp,add_dark_noise=add_dark_noise)

            tracks += track
            n_generated = self.__count_photons(track)
            n_total += n_generated

            kbar.add(n_generated)

        torch.cuda.empty_cache()


        xs,ys,times = [],[],[]

        for track_ in tracks:
            xs.append(track_['x'])
            ys.append(track_['y'])
            times.append(track_['leadTime'])

        xs = np.concatenate(xs)[:numPhotons]
        ys = np.concatenate(ys)[:numPhotons]
        times = np.concatenate(times)[:numPhotons]
        return {"x":xs,"y":ys,"leadTime":times}


    def __count_photons(self,tracks):
        counter = 0
        for track in tracks:
            counter += track["NHits"]

        return counter

    # Based off of: https://github.com/rdom/eicdirc/blob/996e031d40825ce14292d1379fc173c54594ec5f/src/PrtPixelSD.cxx#L212
    # Dark rate coincides with -c 2031 in standalone simulation
    def __add_dark_noise(self,hits,dark_noise_pmt=28000):
        # probability to have a noise hit in 100 ns window
        prob = dark_noise_pmt * 100 / 1e9
        new_hits = []
        for p in range(npmt):
            for i in range(int(prob) + 1):
                if(i == 0) and (prob - int(prob) < np.random.uniform()):
                    continue

                dn_time = 100 * np.random.uniform() # [1,100] ns
                dn_pix = int(npix * np.random.uniform())
                dn_channel = int(dn_pix * p)
                row = (p//6) * 16 + dn_pix//16 
                col = (p%6) * 16 + dn_pix%16
                x = 2 + col * pixel_width + (p % 6) * gapx + (pixel_width) / 2. # Center at middle
                y = 2 + row * pixel_height + (p // 6) * gapy + (pixel_height) / 2. # Center at middle
                # x,y,t,pmtID,pixelID,channel
                h = [x,y,dn_time,p,dn_pix,dn_channel]
                new_hits.append(h)

        if new_hits:
            new_hits = np.array(new_hits)
            hits = np.vstack([hits,new_hits])
            return hits[:,0],hits[:,1],hits[:,2],hits[:,3],hits[:,4],hits[:,5],hits.shape[0]
        else:
            return hits[:,0],hits[:,1],hits[:,2],hits[:,3],hits[:,4],hits[:,5],0


