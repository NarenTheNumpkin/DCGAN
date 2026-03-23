import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Animefaces
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
from train import Trainer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

nz = 100          
batch_size = 128  
lr = 0.0002       
beta1 = 0.5       
epochs = 20       
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda")
gen = Generator().to(device)
dis = Discriminator().to(device)

gen.apply(weights_init)
dis.apply(weights_init)

dataset = Animefaces("GAN")
dataloader = DataLoader(dataset, 128, True)

optimizerD = optim.Adam(dis.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.BCELoss()

trainer = Trainer(
    generator=gen,
    discriminator=dis,
    gen_optimizer=optimizerG,
    dis_optimizer=optimizerD,
    trainer_loader=dataloader,
    device=device,
    save_dir=save_dir
)

trainer.train_epochs(epochs)