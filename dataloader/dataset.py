from torch.utils.data import Dataset
import numpy as np
import os
import torch
import random
import collections
import pickle


class DIRC_Dataset(Dataset):
    def __init__(self,data_path,data_type="Kaon",max_seq_length=250,time_digitizer = None,
        stats={"x_max": 350.0,"x_min":2.0,"y_max":230.1,"y_min":2.0,"time_max":157.00,"time_min":9.0,"P_max":10.0 ,"P_min":0.5 ,"theta_max": 160.0,"theta_min": 25.0}):
        self.stats = stats
        self.gapx =  1.89216111455965 + 4.
        self.gapy = 1.3571428571428572 + 4.
        self.pixel_width = 3.3125
        self.pixel_height = 3.3125
        self.time_digitizer = time_digitizer

        assert data_type in data_path

        

        data = np.load(data_path,allow_pickle=True)#[:int(1e5)]
        print("Initial {0}: ".format(data_type),len(data))

        cut_data = []
        print("Applying fiducial cuts.")
        for i in range(len(data)):
            theta__ = data[i]['Theta']
            p__ = data[i]['P']
            nh = data[i]['NHits']
            if ((theta__ > self.stats['theta_min']) and (theta__ < self.stats['theta_max']) and (p__ > self.stats['P_min']) 
                 and (p__ < self.stats['P_max']) and (nh > 5) and (nh < 250) and (np.min(data[i]['leadTime']) > self.stats['time_min']) and (np.max(data[i]['leadTime']) < self.stats['time_max'])):
                cut_data.append(data[i])

        print("Deleting original data copy.")
        del data
        print("Done.")
        print(" ")
        print("Number of {0}: ".format(data_type),len(cut_data))
        print(" ")
        self.data = cut_data
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min']])
        # Per pmt - 16x16
        self.num_pixels = 256
        self.max_seq_length = max_seq_length
        # Global token
        self.SOS_token = 0
        # Positional tokens
        self.EOS_token = 6145
        self.pad_token = 6146
        # Time tokens
        # self.time_EOS_token = 14801
        # self.time_pad_token = 14802
        self.time_EOS_token = 5921
        self.time_pad_token = 5922
        # self.time_EOS_token = 3141
        # self.time_pad_token = 3142
        print("Maximum seq length: ",self.max_seq_length)

    def __len__(self):
        return len(self.data)


    def scale_data(self,hits,stats):
        x = hits[:,0]
        y = hits[:,1]
        time = hits[:,2]
        x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
        y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
        time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0
        return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)

    def __getitem__(self, idx):

        particle = self.data[idx]
        pmtID = np.array(particle['pmtID'])
        time = np.array(particle['leadTime'])
        pixelID = np.array(particle['pixelID'])

        positional_token = pmtID * self.num_pixels + pixelID + 1 # SOS is 0
        
    
        pos_time = np.where((time > self.stats['time_min']) & (time < self.stats['time_max']))[0]
        time = time[pos_time]
        positional_token = positional_token[pos_time]

        sorted_indices = np.argsort(time)

        sorted_tokens = positional_token[sorted_indices]
        sorted_time = time[sorted_indices]

        # Create independent vocab for time, time relationship with pixels is one to many - fine assumption.
        if self.time_digitizer is not None:
            sorted_time = self.time_digitizer.tokenize(sorted_time)

        else:
            sorted_time = (sorted_time - self.stats['time_min'])/(self.stats['time_max'] - self.stats['time_min'])

        assert len(sorted_tokens) == len(sorted_time)

        sorted_tokens = np.insert(sorted_tokens, 0, self.SOS_token)   # Insert start token
        sorted_tokens = np.append(sorted_tokens, self.EOS_token)       # Append stop token
        sorted_time = np.insert(sorted_time,0,self.SOS_token)
        sorted_time = np.append(sorted_time,self.time_EOS_token)


        # Pad sequences
        pad_length = self.max_seq_length - len(sorted_tokens)
        if pad_length > 0:
            sorted_tokens = np.pad(sorted_tokens, (0, pad_length), 'constant', constant_values=self.pad_token)
            sorted_time = np.pad(sorted_time, (0, pad_length), 'constant', constant_values=self.time_pad_token) 
        elif pad_length < 0:
            sorted_tokens = sorted_tokens[:self.max_seq_length - 1]
            sorted_time = sorted_time[:self.max_seq_length - 1]
            sorted_tokens = np.append(sorted_tokens,self.EOS_token)
            sorted_time = np.append(sorted_time,self.time_EOS_token)
        else:
            pass

        kinematics = np.array([particle['P'],particle['Theta']])
        unscaled_kinematics = kinematics.copy()
        #kinematics = (kinematics - self.conditional_mins) / (self.conditional_maxes - self.conditional_mins)
        kinematics = 2*(kinematics - self.conditional_mins) / (self.conditional_maxes - self.conditional_mins) - 1.0

        return sorted_tokens,sorted_time,kinematics,unscaled_kinematics


class DIRC_Dataset_Classification(Dataset):
    def __init__(self,pion_path,kaon_path,max_seq_length=250,time_digitizer = None,
        stats={"x_max": 350.0,"x_min":2.0,"y_max":230.1,"y_min":2.0,"time_max":157.00,"time_min":9.0,"P_max":10.0 ,"P_min":0.5 ,"theta_max": 160.0,"theta_min": 25.0}):
        self.stats = stats
        self.gapx =  1.89216111455965 + 4.
        self.gapy = 1.3571428571428572 + 4.
        self.pixel_width = 3.3125
        self.pixel_height = 3.3125
        self.time_digitizer = time_digitizer

        assert "Pions" in pion_path
        assert "Kaons" in kaon_path

        data = np.load(pion_path,allow_pickle=True) + np.load(kaon_path,allow_pickle=True)
        random.shuffle(data)
        print("Initial Pions and Kaons: " ,len(data))

        cut_data = []
        print("Applying fiducial cuts.")
        for i in range(len(data)):
            theta__ = data[i]['Theta']
            p__ = data[i]['P']
            nh = data[i]['NHits']
            if ((theta__ > self.stats['theta_min']) and (theta__ < self.stats['theta_max']) and (p__ > self.stats['P_min']) 
                 and (p__ < self.stats['P_max']) and (nh > 5) and (nh < 250) and (np.min(data[i]['leadTime']) > self.stats['time_min']) and (np.max(data[i]['leadTime']) < self.stats['time_max'])):
                cut_data.append(data[i])

        print("Deleting original data copy.")
        del data
        print("Done.")
        print(" ")
        print("Number of Pions and Kaons: ",len(cut_data))
        print(" ")
        self.data = cut_data
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min']])
        # Per pmt - 16x16
        self.num_pixels = 256
        self.max_seq_length = max_seq_length
        # Global token
        self.SOS_token = 0
        # Positional tokens
        self.EOS_token = 6145
        self.pad_token = 6146
        # Time tokens
        self.time_EOS_token = 5921
        self.time_pad_token = 5922
        #self.time_EOS_token = 3141
        #self.time_pad_token = 3142
        print("Maximum seq length: ",self.max_seq_length)

    def __len__(self):
        return len(self.data)


    def scale_data(self,hits,stats):
        x = hits[:,0]
        y = hits[:,1]
        time = hits[:,2]
        x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
        y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
        time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0
        return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)

    def __getitem__(self, idx):

        particle = self.data[idx]
        pmtID = np.array(particle['pmtID'])
        time = np.array(particle['leadTime'])
        pixelID = np.array(particle['pixelID'])
        PID = np.array(particle['PDG'])

        positional_token = pmtID * self.num_pixels + pixelID + 1 # SOS is 0
        
    
        pos_time = np.where((time > self.stats['time_min']) & (time < self.stats['time_max']))[0]
        time = time[pos_time]
        positional_token = positional_token[pos_time]

        sorted_indices = np.argsort(time)

        sorted_tokens = positional_token[sorted_indices]
        sorted_time = time[sorted_indices]

        # Create independent vocab for time, time relationship with pixels is one to many - fine assumption.
        if self.time_digitizer is not None:
            sorted_time = self.time_digitizer.tokenize(sorted_time)

        else:
            sorted_time = (sorted_time - self.stats['time_min'])/(self.stats['time_max'] - self.stats['time_min'])

        assert len(sorted_tokens) == len(sorted_time)

        sorted_tokens = np.insert(sorted_tokens, 0, self.SOS_token)   # Insert start token
        sorted_tokens = np.append(sorted_tokens, self.EOS_token)       # Append stop token
        sorted_time = np.insert(sorted_time,0,self.SOS_token)
        sorted_time = np.append(sorted_time,self.time_EOS_token)


        # Pad sequences
        pad_length = self.max_seq_length - len(sorted_tokens)
        if pad_length > 0:
            sorted_tokens = np.pad(sorted_tokens, (0, pad_length), 'constant', constant_values=self.pad_token)
            sorted_time = np.pad(sorted_time, (0, pad_length), 'constant', constant_values=self.time_pad_token) 
        elif pad_length < 0:
            sorted_tokens = sorted_tokens[:self.max_seq_length - 1]
            sorted_time = sorted_time[:self.max_seq_length - 1]
            sorted_tokens = np.append(sorted_tokens,self.EOS_token)
            sorted_time = np.append(sorted_time,self.time_EOS_token)
        else:
            pass

        kinematics = np.array([particle['P'],particle['Theta']])
        unscaled_kinematics = kinematics.copy()
        #kinematics = (kinematics - self.conditional_mins) / (self.conditional_maxes - self.conditional_mins)
        kinematics = 2*(kinematics - self.conditional_mins) / (self.conditional_maxes - self.conditional_mins) - 1.0

        if abs(PID) == 211: # 211 is Pion 
            PID = 0
        elif abs(PID) == 321: # 321 is Kaon 
            PID = 1
        else:
            print("Unknown PID!")

        return sorted_tokens,sorted_time,kinematics,unscaled_kinematics,PID


class hpDIRC_DLL_Dataset(Dataset):
    def __init__(self,path_,stats={"x_max": 350.0,"x_min":2.0,"y_max":230.1,"y_min":2.0,"time_max":157.00,"time_min":0.0,"P_max":10.0 ,"P_min":0.5 ,"theta_max": 160.0,"theta_min": 25.0},time_cuts=None,n_photons=300,n_particles=200000,fast_sim_comp=False,fast_sim_type=None,geant=True):
        self.stats = stats
        self.geant = geant
        if not fast_sim_comp: # path_ is a .pkl file here
            data = np.load(path_,allow_pickle=True)[:n_particles] # Useful for testing
            self.data = []
            print(len(data))
            for i in range(len(data)):
                theta__ = data[i]['Theta']
                p__ = data[i]['P']
                n_hits = data[i]['NHits']
                if ((theta__ > self.stats['theta_min']) and (theta__ < self.stats['theta_max']) and (p__ > self.stats['P_min']) and (p__ < self.stats['P_max']) and (n_hits > 0)):
                    self.data.append(data[i])

        elif fast_sim_comp and fast_sim_type is not None and not geant: # Use fast sim file structure, provide PID, dont use geant4 data
            data = []
            files = os.listdir(path_) # path_ is a directory here
            for file in files:
                if ".pkl" in file:
                    if fast_sim_type in file:
                        print("Loading file: ",file)
                        d_ = np.load(os.path.join(path_,file),allow_pickle=True)
                        itter = len(d_['fast_sim']) if len(d_['fast_sim']) < len(d_['truth']) else len(d_['truth'])
                        for i in range(itter):
                            d_['fast_sim'][i]['PDG'] = d_['truth'][i]['PDG']
            
                        data += d_['fast_sim']
                else:
                    continue
            
            self.data = data[:n_particles]
            del data
            print(len(self.data))

        elif fast_sim_comp and fast_sim_type is not None and geant: # Use fast sim file structure, provide PID, use geant
            data = []
            files = os.listdir(path_) # path_ is a directory here
            for file in files:
                if ".pkl" in file:
                    if fast_sim_type in file:
                        print("Loading file: ",file)
                        d_ = np.load(os.path.join(path_,file),allow_pickle=True)
                        data += d_['truth']
                else:
                    continue
            
            self.data = data[:n_particles]
            del data
            print(len(self.data))      

        else:
            raise ValueError("Fast Sim Comp is set true, but have not provided the type. Set as either Pion or Kaon. Check geant arg.")


        self.n_photons = n_photons
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min']])
        self.time_cuts = time_cuts
        self.gapx =  1.89216111455965 + 4.
        self.gapy = 1.3571428571428572 + 4.
        self.pixel_width = 3.3125
        self.pixel_height = 3.3125
        if self.time_cuts is not None:
            print('Rejecting photons with time > {0}'.format(self.time_cuts))

    def __len__(self):
        return len(self.data)

    def scale_data(self,hits,stats):
        x = hits[:,0]
        y = hits[:,1]
        time = hits[:,2]
        x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
        y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
        time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0
        return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)


    def __getitem__(self, idx):

        data = self.data[idx]
        try:
            PID = data['PDG']
        except:
            PID = 211
        LL_k = 0.0
        LL_pi = 0.0
        pmtID = np.array(data['pmtID'])
        if data['NHits'] == 0:
            print("Stop.",NHits)

        if self.geant:

            pixelID = np.array(data['pixelID'])

            row = (pmtID//6) * 16 + pixelID//16 
            col = (pmtID%6) * 16 + pixelID%16
            
            x = 2 + col * self.pixel_width + (pmtID % 6) * self.gapx + (self.pixel_width) / 2. # Center at middle
            y = 2 + row * self.pixel_height + (pmtID // 6) * self.gapy + (self.pixel_height) / 2. # Center at middle

        else:
            x = data['x']
            y = data['y']

        time = np.array(data['leadTime'])      

        pos_time = np.where((time > 0) & (time < self.stats['time_max']))[0]
        time = time[pos_time]
        x = x[pos_time]
        y = y[pos_time]

        if len(time) == 0:
            print("Hey, wrong.",idx)
        
        if len(time) > self.n_photons:

            time_idx = np.argsort(time)[:self.n_photons]
            time = time[time_idx]
            x = x[time_idx]
            y = y[time_idx]

        assert len(x) == len(time)
        assert len(y) == len(time)

        hits = np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)
        hits = self.scale_data(hits,self.stats)
        conds = np.array([data['P'],data['Theta']])
        #print(print("------------------Start----------------------"))
        #print(conds)
        conds = conds.reshape(1,-1).repeat(len(x),0)
        unscaled_conds = conds.copy()
        n_hits = len(hits)


        #print(self.conditional_maxes,self.conditional_mins)

        conds = (conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins)

        #print(conds)
        #print("--------------------------------------------")
        if len(hits) > self.n_photons:
            #usually argsort in time
            hits = hits[np.argsort(time)]
            hits = hits[:self.n_photons]
            conds = conds[:self.n_photons]
            unscaled_conds = unscaled_conds[:self.n_photons]
            time = time[np.argsort(time)]
            time = time[:self.n_photons]

        elif len(hits) < self.n_photons:
            n_needed = self.n_photons - len(hits)
            hits = np.pad(hits,((0,n_needed),(0,0)),mode='constant',constant_values=-np.inf)
            conds = np.pad(conds,((0,n_needed),(0,0)),mode='constant',constant_values=-np.inf)
            unscaled_conds = np.pad(unscaled_conds,((0,n_needed),(0,0)),mode='constant',constant_values=-np.inf)
        else: # Already taken care of
            pass

        #print(hits)

        return hits,conds,PID,n_hits,unscaled_conds,LL_k,LL_pi