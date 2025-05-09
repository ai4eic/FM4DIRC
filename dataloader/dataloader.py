import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch


def Inference_collate(batch):
    hits = []
    conditions = []
    PIDs = []
    unscaled = []
    n_hits = []
    LL_ks = []
    LL_pis = []
    for h,cond,PID,nh,u,LL_k,LL_pi in batch:
        hits.append(torch.tensor(h))
        conditions.append(torch.tensor(cond))
        PIDs.append(torch.tensor(PID))
        n_hits.append(torch.tensor(nh))
        unscaled.append(torch.tensor(u))
        LL_ks.append(torch.tensor(LL_k))
        LL_pis.append(torch.tensor(LL_pi))

    return torch.stack(hits),torch.stack(conditions),torch.tensor(PIDs),torch.tensor(n_hits),torch.stack(unscaled),torch.tensor(LL_ks),torch.tensor(LL_pis)


def DIRC_collate(batch):
    spatial_tokens = []
    time_values = []
    kinematics = []
    unscaled_kinematics = []

    for st,tv,k,uk in batch:
        spatial_tokens.append(torch.tensor(st))
        time_values.append(torch.tensor(tv))
        kinematics.append(torch.tensor(k))
        unscaled_kinematics.append(torch.tensor(uk))
    
    return torch.stack(spatial_tokens),torch.stack(time_values),torch.stack(kinematics),torch.stack(unscaled_kinematics)

def DIRC_collate_classification(batch):
    spatial_tokens = []
    time_values = []
    kinematics = []
    unscaled_kinematics = []
    PIDs = []

    for st,tv,k,uk,PID in batch:
        spatial_tokens.append(torch.tensor(st))
        time_values.append(torch.tensor(tv))
        kinematics.append(torch.tensor(k))
        unscaled_kinematics.append(torch.tensor(uk))
        PIDs.append(torch.tensor(PID))
    
    return torch.stack(spatial_tokens),torch.stack(time_values),torch.stack(kinematics),torch.stack(unscaled_kinematics),torch.tensor(PIDs)

# Create dataloaders to iterate.
def CreateLoaders(train_dataset,val_dataset,config):
    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size'],
                            shuffle=True,collate_fn=DIRC_collate,num_workers=config['dataloader']['train']['num_workers'],
                            pin_memory=False)
    val_loader =  DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size'],
                            shuffle=False,collate_fn=DIRC_collate,num_workers=config['dataloader']['val']['num_workers'],
                            pin_memory=False)

    return train_loader,val_loader


def CreateLoadersClassification(train_dataset,val_dataset,config):
    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size'],
                            shuffle=True,collate_fn=DIRC_collate_classification,num_workers=config['dataloader']['train']['num_workers'],
                            pin_memory=False)
    val_loader =  DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size'],
                            shuffle=False,collate_fn=DIRC_collate_classification,num_workers=config['dataloader']['val']['num_workers'],
                            pin_memory=False)

    return train_loader,val_loader


def InferenceLoader(test_dataset,config):
    return DataLoader(test_dataset,
                            batch_size=config['dataloader']['test']['batch_size'],
                            shuffle=False,collate_fn=DIRC_collate_classification,num_workers=0,
                            pin_memory=False)


def CreateInferenceLoaderDLL(test_dataset,config):
    test_loader =  DataLoader(test_dataset,
                            batch_size=config['dataloader']['test']['batch_size_DLL'],
                            shuffle=False,collate_fn=DLL_Inference_collate,num_workers=0)
    return test_loader    

# Create dataloaders to iterate.
def CreateInferenceLoader(test_dataset,config):
    test_loader =  DataLoader(test_dataset,
                            batch_size=config['dataloader']['test']['batch_size'],
                            shuffle=False,collate_fn=Inference_collate,num_workers=0)
    return test_loader                     