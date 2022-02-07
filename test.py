import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from scipy.io import loadmat
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from Fastonn import SelfONNTranspose1d as SelfONNTranspose1dlayer
from Fastonn import SelfONN1d as SelfONN1dlayer
from utils import ECGDataset, ECGDataModule,init_weights,TECGDataset,TECGDataModule
from GAN_Arch_details import Upsample,Downsample,CycleGAN_Unet_Generator,CycleGAN_Discriminator




G_basestyle = CycleGAN_Unet_Generator()
checkpoint =torch.load("model_weights_16NQ3.pth")

G_basestyle.load_state_dict(checkpoint)

G_basestyle.eval()




for i in range(1,11):
    print(i)
    # Sanity Check
    data_dir = "all_patients/"+str(i)+"/"
    batch_size = 8    
    dm = TECGDataModule(data_dir, batch_size, phase='test')
    dm.prepare_data()
    dataloader = dm.train_dataloader()
    base, style = next(iter(dataloader))
    print('Input Shape {}, {}'.format(base.size(), style.size())) 
    net = G_basestyle
    net.eval()
    predicted = []
    predicted=pd.DataFrame(data=predicted)
    actual = []
    actual=pd.DataFrame(data=actual)
    with torch.no_grad():
        for base, style in (dataloader):     
            output = net(net(base)).squeeze()
            
            ganoutput=output.detach().numpy()
            ganoutput=pd.DataFrame(data=ganoutput)
            predicted=pd.concat([predicted,ganoutput])
            ganacc=base.detach().numpy().squeeze()
            ganacc=pd.DataFrame(data=ganacc)
            actual=pd.concat([actual,ganacc])
    gan_outputs=predicted.values.reshape(len(predicted)*4000,1)
    real_outputs=actual.values.reshape(len(actual)*4000,1)
    import scipy.io as sio
    sio.savemat('all_test_outputs/'+str(i)+'/gan_outputs.mat', {'gan_outputs':gan_outputs})    
    sio.savemat('all_test_outputs/'+str(i)+'/real_sig.mat', {'real_sig':real_outputs})    

