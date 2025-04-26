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

