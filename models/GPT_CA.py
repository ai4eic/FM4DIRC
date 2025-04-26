
import math
import pkbar
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

############################################################
# References: https://github.com/karpathy/minGPT/tree/master
############################################################

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

# Rascka 
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len=250, dropout=0.0,device='cuda'):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.seq_length = seq_len
        self.head_dim = embed_dim // num_heads
        self.device = device
        self.g_scale = nn.Parameter(torch.tensor(np.sqrt(self.head_dim),dtype=torch.float,device=self.device),requires_grad=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        #print("Cross attention scale: ",self.g_scale)

    # implement KV cache
    # CA is elegant guiding as a function of Kinematics - doesn't seem to work the best
    def forward(self, x,k_embed,attn_mask,key_padding_mask=None,need_weights=False):
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.q_proj(k_embed),self.k_proj(x),self.v_proj(x)

        # Treat soft(qk)v - query from kinematics to x (key) - extract values from x (value)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # normalize QK matrices with L2 norm over head dim, learnable scale clamped to d_k 

        k = F.normalize(k,p=2,dim=-1)
        q = F.normalize(q,p=2,dim=-1)

        attn_scores = self.g_scale * q @ k.transpose(2,3)

        attn_scores.masked_fill_(attn_mask,-torch.inf)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_scores.masked_fill_(key_padding_mask,-torch.inf)

        attn_scores = F.softmax(attn_scores,dim=-1)
        attn_scores = self.dropout(attn_scores)

        attn_output = (attn_scores @ v).transpose(1,2)

        attn_output = attn_output.contiguous().view(batch_size,seq_len,embed_dim)

        if need_weights:
            return attn_output,attn_scores
        else:
            return attn_output,None


class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads, mlp_scale : int = 2,drop_rate: float = 0.0, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.mlp_scale = mlp_scale
        self.drop_rate = drop_rate 
        print("Using Custom Cross Attention.")
        print("MLP Scale: ",self.mlp_scale)
        # top down layers
        self.xN = nn.LayerNorm(self.embed_dim)
        self.kN = nn.LayerNorm(self.embed_dim)
        self.attn = CrossAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate,device=self.device)
        self.c_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.LN2 = nn.LayerNorm(self.embed_dim)
        self.FF = FF(self.embed_dim,mlp_scale=self.mlp_scale)


    def generate_mask(self,seq_len):
        return torch.triu(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool), diagonal=1)

    def forward(self,x,k_embed,padding_mask=None,need_weights=False,classification=False):
        B,N_t,t_dim = x.shape
        x_norm = self.xN(x)
        k_norm = self.kN(k_embed)

        if not classification:
            mask_ = self.generate_mask(N_t)
            attn,attn_weights = self.attn(x_norm, k_norm, attn_mask=mask_,key_padding_mask=padding_mask,need_weights=False)
        else:
            attn,attn_weights = self.attn(x_norm, k_norm,key_padding_mask=padding_mask,need_weights=False)

        attn = self.c_proj(attn)
        x = x + attn
        x = x + self.FF(self.LN2(x))
        return x



class Cherenkov_GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim,attn_heads = [2,4,2],kin_size=2,
                num_blocks=2,hidden_units=128, digitize_time=True, mlp_scale : int = 2,
                time_vocab : int = 5923,
                detokenize_func = None,
                classification=False,
                device='cuda'):
        super().__init__()
        self.classification = classification
        self.digitize_time = digitize_time
        self.detokenize_func = detokenize_func
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.momentum_embedding = nn.Linear(1,embed_dim) 
        self.theta_embedding = nn.Linear(1,embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, attn_heads[i],mlp_scale) for i in range(len(attn_heads))])
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
            if self.digitize_time: # Time resolution tokenization
                self.time_embedding = nn.Embedding(time_vocab,embed_dim)
            else: # Fully continuous
                self.time_embedding = nn.Linear(1,embed_dim)

            self.classification_head = nn.Linear(embed_dim,1)
        
        self.device = device
        self.SOS_token = 0
        self.EOS_token = 6145
        self.pad_token = 6146
        self.EOS_time_token = time_vocab - 2

    def forward(self, x,t,k,padding_mask=None):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        if self.classification:
            classification_mask = ~torch.isin(x, torch.tensor([self.SOS_token,self.EOS_token,self.pad_token],dtype=torch.long,device=x.device))


        if not self.digitize_time:
            t = t.reshape(-1, 1)  # [batch_size * seq_len, 1]
            t_embed_flat = self.time_embedding(t)  # 
            t_embed = t_embed_flat.view(batch_size, seq_len, t_embed_flat.shape[-1]) # [batch_size, seq_len,embed_dim]

        else:
            t_embed = self.time_embedding(t)


        k_embed = self.momentum_embedding(k[:,0].unsqueeze(-1)) + self.theta_embedding(k[:,1].unsqueeze(-1))
        k_embed = k_embed.unsqueeze(1).repeat(1, seq_len, 1)
        x = self.token_embedding(x) + t_embed + self.pos_embedding(pos) 

            
        for layer in self.layers:
            x = layer(x,k_embed,padding_mask=padding_mask,classification=self.classification)
        
        x = self.LN(x)
 
        if not self.classification: # Generations - next hit prediction
            pixel = self.logits_head(x)

            if not self.digitize_time:
                t_out = self.time_head(x).squeeze(-1) # direct regression of time - subject to mode collapse
            else:
                t_out = self.time_head(x) # logits over time 

            return pixel,t_out

        else:
            masked_x = x * classification_mask.unsqueeze(-1).float() 
            masked_x_mean = masked_x.sum(dim=1) 
            return self.classification_head(masked_x_mean).squeeze(-1)


    def post_process(self, pixels, times):
        processed_pixels = []
        processed_times = []
        
        for idx, t in zip(pixels, times):
            idx = idx[2:] # remove kinematic tokens
            t = t[2:] # remove kinematic tokens
            mask = ~torch.isin(idx, torch.tensor([self.pad_token, self.SOS_token, self.EOS_token], device=pixels.device))
            processed_pixels.append((idx[mask] -1).detach().cpu().numpy())
            if self.digitize_time and self.detokenize_func is not None:
                processed_times.append(self.detokenize_func(t[mask].detach().cpu().numpy())) 
            else:
                processed_times.append(t[mask].detach().cpu().numpy()) 
        
        return processed_pixels, processed_times


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

    def __topK(self,logits,topK=50):
        topk_logits, topk_indices = torch.topk(logits, k=topK, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        idx_next = topk_indices.gather(-1, sampled)
        return idx_next


    @torch.no_grad()
    def generate(self, k, max_seq_len: int = 250,
                context_len: int = 250, temperature: float = 1.0, 
                method="Nucleus", topK=100, nucleus_p=0.98):

        assert method in ["Nucleus", "TopK", "Default"]
        batch_size = k.shape[0]

        # Start tokens
        idx = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)  # pixel token
        if self.digitize_time:
            t = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)  # time token
        else:
            t = torch.zeros(batch_size,1).to(self.device).float()

        is_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_seq_len):
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

            # ---- Time sampling ----
            if self.digitize_time:
                logits_time = logits_time[:, -1, :] / temperature
                if method == "Default":
                    probs_time = F.softmax(logits_time, dim=-1)
                    t_next = torch.multinomial(probs_time, num_samples=1)
                elif method == "TopK":
                    t_next = self.__topK(logits_time, topK)
                elif method == "Nucleus":
                    t_next = self.__nucleus(logits_time, nucleus_p)
            else:
                t_next = logits_time[:, -1].unsqueeze(1)

            if self.digitize_time:
                # ---- Stop sequences based on EOS_time_token ----
                is_done |= (t_next.squeeze(1) == self.EOS_time_token)
            else:
                # ---- Stop sequences based on EOS_time_token ----
                is_done |= (idx_next.squeeze(1) == self.EOS_token)

            # For done sequences, freeze output to last token
            idx_next[is_done] = self.EOS_token
            t_next[is_done] = self.EOS_time_token

            idx = torch.cat((idx, idx_next), dim=1)
            t = torch.cat((t, t_next), dim=1)

            if torch.all(is_done):
                break

        return self.post_process(idx,t)


    @torch.no_grad()
    def generate_PDF(self,kinematics, numTracks=2e4,max_seq_len: int = 250,
                 context_len: int = 250, temperature: float = 1.0, 
                 method="Default",topK=100,nucleus_p=0.99):

        assert kinematics is not None

        print("PDF Generation: ")
        print("Sampling Method: ",method)
        print("Temperature: ",temperature)
        if method == "Nucleus":
            print("Nucleus P:",nucleus_p)
        elif method == "TopK":
            print("TopK: ",topK)
        else:
            pass

        batch_size = kinematics.shape[0]
        num_itter = numTracks // batch_size
        last_batch = numTracks % batch_size

        kbar = pkbar.Kbar(target=num_itter + 1, width=20, always_stateful=False)

        torch.cuda.empty_cache()
        times = []
        pixels = []
        for i in range(int(num_itter)):

            with torch.no_grad():
                idx,t = self.generate(kinematics,method=method,temperature=temperature,
                                      topK=topK,nucleus_p=nucleus_p)

            pixels += idx
            times += t 

            kbar.update(i)

        torch.cuda.empty_cache()

        if last_batch > 0:
            with torch.no_grad():
                idx,t = self.generate(kinematics[:last_batch],method=method,temperature=temperature,
                                      topK=topK,nucleus_p=nucleus_p)
            pixels += idx
            times += t 

            kbar.add(1)

        torch.cuda.empty_cache()

        pixels,times = np.concatenate(pixels,0),np.concatenate(times,0)

        return pixels,times


