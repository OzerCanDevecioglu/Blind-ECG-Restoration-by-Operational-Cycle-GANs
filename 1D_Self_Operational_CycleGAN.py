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




class CycleGAN_LightningSystem(pl.LightningModule):
    def __init__(self, G_basestyle, G_stylebase, D_base, D_style, lr, reconstr_w=10, id_w=2):
        super(CycleGAN_LightningSystem, self).__init__()
        self.G_basestyle = G_basestyle
        self.G_stylebase = G_stylebase
        self.D_base = D_base
        self.D_style = D_style
        self.lr = lr        
        self.reconstr_w = reconstr_w
        self.id_w = id_w
        self.cnt_train_step = 0
        self.step = 0
        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()
        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []
        self.validity = []
        self.reconstr = []
        self.identity = []

    def configure_optimizers(self):
        self.g_basestyle_optimizer = optim.Adam(self.G_basestyle.parameters(), lr=self.lr['G'], betas=(0.5, 0.999))
        self.g_stylebase_optimizer = optim.Adam(self.G_stylebase.parameters(), lr=self.lr['G'], betas=(0.5, 0.999))
        self.d_base_optimizer = optim.Adam(self.D_base.parameters(), lr=self.lr['D'], betas=(0.5, 0.999))
        self.d_style_optimizer = optim.Adam(self.D_style.parameters(), lr=self.lr['D'], betas=(0.5, 0.999))

        return [self.g_basestyle_optimizer, self.g_stylebase_optimizer, self.d_base_optimizer, self.d_style_optimizer], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        base_img, style_img = batch
        base_img, style_img= base_img.cuda(), style_img.cuda()
        b = base_img.size()[0]

        valid = torch.ones(b, 1,  30).cuda()
        fake = torch.zeros(b, 1,  30).cuda()
        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            # Validity
            # MSELoss
            val_base = self.generator_loss(self.D_base(self.G_stylebase(style_img)), valid)
            val_style = self.generator_loss(self.D_style(self.G_basestyle(base_img)), valid)
            val_loss = (val_base + val_style) / 2

            # Reconstruction
            reconstr_base = self.mae(self.G_stylebase(self.G_basestyle(base_img)), base_img)
            reconstr_style = self.mae(self.G_basestyle(self.G_stylebase(style_img)), style_img)
            reconstr_loss = (reconstr_base + reconstr_style) / 2

            # Identity
            id_base = self.mae(self.G_stylebase(base_img), base_img)
            id_style = self.mae(self.G_basestyle(style_img), style_img)
            id_loss = (id_base + id_style) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss

            return {'loss': G_loss, 'validity': val_loss, 'reconstr': reconstr_loss, 'identity': id_loss}

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:
            # MSELoss
            D_base_gen_loss = self.discriminator_loss(self.D_base(self.G_stylebase(style_img)), fake)
            D_style_gen_loss = self.discriminator_loss(self.D_style(self.G_basestyle(base_img)), fake)
            D_base_valid_loss = self.discriminator_loss(self.D_base(base_img), valid)
            D_style_valid_loss = self.discriminator_loss(self.D_style(style_img), valid)
            
            D_gen_loss = (D_base_gen_loss + D_style_gen_loss) / 2
            
            # Loss Weight
            D_loss = (D_gen_loss + D_base_valid_loss + D_style_valid_loss) / 3

            # Count up
            self.cnt_train_step += 1

            return {'loss': D_loss}

    def training_epoch_end(self, outputs):
        self.step += 1
        print(self.step)
        avg_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 4 for i in range(4)])
        G_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        D_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2 for i in [2, 3]])
        validity = sum([torch.stack([x['validity'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        reconstr = sum([torch.stack([x['reconstr'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        identity = sum([torch.stack([x['identity'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
            
        self.losses.append(avg_loss)
        self.G_mean_losses.append(G_mean_loss)
        self.D_mean_losses.append(D_mean_loss)
        self.validity.append(validity)
        self.reconstr.append(reconstr)
        self.identity.append(identity)
        
      

        return None

# Config  -----------------------------------------------------------------
data_dir = 'input/gan-getting-started'
batch_size = 8
lr = {
    'G': 0.00001,
    'D': 0.00001
}
epoch = 1000
seed = 42
reconstr_w = 10
id_w = 5
# DataModule  -----------------------------------------------------------------    
dm = ECGDataModule(data_dir, batch_size, phase='test')
G_basestyle = CycleGAN_Unet_Generator().cuda()
G_stylebase = CycleGAN_Unet_Generator().cuda()
D_base = CycleGAN_Discriminator().cuda()
D_style = CycleGAN_Discriminator().cuda()

# Init Weight  --------------------------------------------------------------
for net in [G_basestyle, G_stylebase, D_base, D_style]:
    init_weights(net, init_type='normal')

# LightningModule  --------------------------------------------------------------
model = CycleGAN_LightningSystem(G_basestyle, G_stylebase, D_base, D_style, 
                                 lr, reconstr_w, id_w)

# Trainer  --------------------------------------------------------------
trainer = Trainer(
    logger=False,
    max_epochs=epoch,
    gpus=1,
    checkpoint_callback=False,
    reload_dataloaders_every_epoch=True,
    num_sanity_val_steps=0,  # Skip Sanity Check
)

# Train
trainer.fit(model, datamodule=dm)
os.makedirs('test_outputs', exist_ok=True)
net = model.G_basestyle
net.eval()
dm = ECGDataModule(data_dir, batch_size, phase='test')
dm.prepare_data()

dataloader = dm.train_dataloader()
base, style = next(iter(dataloader))

gen_img = net(base)
gen_img = gen_img.detach().cpu().numpy()
style = style.detach().cpu().numpy()


torch.save(net.state_dict(), 'model_weights.pth')
   