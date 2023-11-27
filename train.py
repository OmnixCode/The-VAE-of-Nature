#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:40:59 2023

@author: filipk
"""
import os

os.chdir("/home/filipk/Desktop/VAE_FINAL_v2/")

import encoder
import decoder
import torch
import torchvision
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
import math

class VAE(nn.Module):
    def __init__(self, VAE_encoder, VAE_decoder):
        super().__init__()
        self.encoder = VAE_encoder(args.image_size, args.lat_size)
        self.decoder = VAE_decoder(args.image_size, args.lat_size)
        
    def forward(self, x, noise):
        x=self.encoder(x, noise)
        x=self.decoder(x)
        return x
    
    def loss_function(self,input_image, reconstructed_image, epoch, kld_mult, ssim_metrics = True, alpha=0.5, beta=0.5, lambd = 1):
        #kld_weight=1
        kld_weight=(epoch / 100) - int(epoch / 100) #bilo0.03
        kld_weight=100/1000 #bilo 10000
        #bilo10
        recons_loss = 200*F.mse_loss(reconstructed_image, input_image)#*128*128*3 #change to 10, 20 gave interesting stuff#last=5
        if ssim_metrics == True:
            ssim_loss = SSIM(reconstructed_image, input_image)
        else:
            alpha=1
        
        sparse_loss = lambd *torch.sum(torch.abs(self.encoder.mu))
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.encoder.log_var - self.encoder.mu ** 2 -  self.encoder.log_var.exp(), dim = (1,2)), dim = 0)
        b,c,ld = self.encoder.log_var.size()
        kld_loss= kld_loss/(c*ld)
        sparse_loss= sparse_loss/(b*c*ld)
        
        loss = alpha*recons_loss + beta*(1-ssim_loss) + kld_weight * kld_loss + 0*sparse_loss #0.3 gore
        #return{'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        return{'loss': loss, 'Reconstruction_Loss':(alpha*recons_loss.detach()),'SSIM_Loss':(beta*(1-ssim_loss.detach())),'Sparse_Loss':sparse_loss, 'KLD':(kld_weight * kld_loss).detach()}
    
    def loss_function_correct9():
        pass
    
    def sample(self, n_samples, lat_size):
        noise = torch.mul(torch.randn((n_samples, 4, lat_size)).to(torch.device('cuda:0')), 1)
        x = self.decoder(noise)
        return(x)
    
    
def plot_images(images): ##boljiploter napravi
    plt.figure(figsize=(args.image_size, args.image_size))
    plt.imshow(torch.cat([
        torch.cat([i for i in images], dim=-1),
    ], dim=-2).permute(1, 2, 0))
    plt.show()  

def save_images(images, path):
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)
    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
# =============================================================================
#     ndarr = ndarr** 255
#     ndarr = ndarr.astype(np.uint8)
# =============================================================================
    im = Image.fromarray(ndarr)
    im.save(path)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
         
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
     
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(args.image_size + 1/4 *args.image_size)),  # args.image_size + 1/4 *args.image_size
        #torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        #AddGaussianNoise(0.1, 0.08),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def load_image(image_path):
    # Define a transformation to be applied to the image
    transform = transforms.Compose([
        torchvision.transforms.Resize(int(args.image_size + 1/4 *args.image_size)),  # args.image_size + 1/4 *args.image_size
        #torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_path).convert('RGB')  # Ensure that the image is in RGB format

    # Apply the transformation to the image
    tensor_image = transform(image)

    # Add an extra dimension to the tensor (batch dimension)
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def save_model_checkpoint(model, optimizer, loss, epoch, img_size, lat_size, kld_mult, args):
    param_string = str(img_size) + '_to_' + str(lat_size) + '_kld_mult_' + str(kld_mult) + '_epoch_' + str(epoch) + f"ckpt.pt"
    PATH = os.path.join("models", args.run_name, param_string)
    torch.save({
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss,
        'epoch' : epoch,
        'image_size' : img_size,
        'latent_size' : lat_size,
        'kld_mult' : kld_mult
        }, PATH)

    param_string = str(img_size) + '_to_' + str(lat_size) + '_kld_mult_' + str(kld_mult) + '_epoch_' + str(epoch-1) + f"ckpt.pt"
    PATH = os.path.join("models", args.run_name, param_string)
    if os.path.exists(PATH):
        os.remove(PATH)

def check_nan_inf(value, name="Value"):
    if isinstance(value, torch.Tensor):
        # Check for NaN values
        if torch.isnan(value).any():
            print(f"{name} (tensor) contains NaN values.")
            return 'Error'

        # Check for Inf values
        if torch.isinf(value).any():
            print(f"{name} (tensor) contains Inf values.")
            return 'Error'
        
    elif isinstance(value, (float, int)):
        # Check for NaN values
        if torch.isnan(torch.tensor(value)).item():
            print(f"{name} (float) is NaN.")
            return 'Error'
        # Check for Inf values
        if torch.isinf(torch.tensor(value)).item():
            print(f"{name} (float) is Inf.")
            return 'Error'
    else:
        print(f"Unsupported type for {name}: {type(value)}")
# =============================================================================
# def train(args):
#     setup_logging(args.run_name)
#     device=args.device
#     dataloader = get_data(args)
#     model = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr = args.lr)
#     logger = SummaryWriter(os.path.join("runs", args.run_name))
#     l = len(dataloader)
#     for epoch in range(args.epochs):
#         logging.info(f"Starting epoch {epoch}:")
#         pbar = tqdm(dataloader)    
#         for i, (images, _) in enumerate(pbar):
#             images = images.to(device)
#             noise = torch.randn((images.size(0), 4, 4, 4)).to(torch.device('cuda:0'))
#             predicted_image = model(images , noise)
#             loss = model.loss_function(images, predicted_image, epoch)['loss']
#             total = model.loss_function(images, predicted_image, epoch)['loss'].item()
#             rec_loss=model.loss_function(images, predicted_image, epoch)['Reconstruction_Loss'].item()
#             kld_loss=model.loss_function(images, predicted_image, epoch)['KLD'].item()
#             
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
# 
#             #pbar.set_postfix(MSE=loss.item())
#             pbar.set_postfix({'Total_loss':total,'rec_loss':rec_loss,'KLD_loss':kld_loss})
#             logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
#         save_images(images.detach(), os.path.join("results", args.run_name, f"{epoch}_orig.jpg"))
#         save_images(predicted_image.detach(), os.path.join("results", args.run_name, f"{epoch}.jpg"))
#         torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
# =============================================================================

def load_model_checkpoint(PATH):
    pass
      

def train(args):
    setup_logging(args.run_name)
    device=args.device
    dataloader = get_data(args)
    model = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    start_epoch = 0
    kld_mult=0.01
    if args.resume == True:
        ckpt = torch.load(args.resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        loss = ckpt['loss']
        start_epoch = ckpt['epoch']
        kld_mult = ckpt['kld_mult']
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)    
        total=0
        rec_loss=0
        kld_loss=0
        sparse_loss=0
        simm_loss=0
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            noise = torch.randn((images.size(0), 4, args.lat_size)).to(torch.device('cuda:0'))
            predicted_image = model(images , noise)
            loss_params = model.loss_function(images, predicted_image, epoch, kld_mult)
            loss = loss_params['loss']
            
            total += model.loss_function(images, predicted_image, epoch, kld_mult)['loss'].item() 
            rec_loss += model.loss_function(images, predicted_image, epoch, kld_mult)['Reconstruction_Loss'].item() 
            kld_loss += model.loss_function(images, predicted_image, epoch, kld_mult)['KLD'].item() 
            sparse_loss += model.loss_function(images, predicted_image, epoch, kld_mult)['Sparse_Loss'].item() 
            simm_loss += model.loss_function(images, predicted_image, epoch, kld_mult)['SSIM_Loss'].item() 
            
            if check_nan_inf(total)=='Error':
                print(batch_idx)
                print(model.loss_function(images, predicted_image, epoch, kld_mult)['loss'].item())
                print('total je problem')
            if check_nan_inf(rec_loss)=='Error':
                print(model.loss_function(images, predicted_image, epoch, kld_mult)['Reconstruction_Loss'].item())
                print(batch_idx)
                print('rec je problem')
            if check_nan_inf(kld_loss)=='Error':
                print(model.loss_function(images, predicted_image, epoch, kld_mult)['KLD'].item() )
                print(batch_idx)
                print('kld je problem')
            if check_nan_inf(sparse_loss)=='Error':
                print(model.loss_function(images, predicted_image, epoch, kld_mult)['Sparse_Loss'].item())
                print(batch_idx)
                print('sparse je problem')
            if check_nan_inf(simm_loss)=='Error':
                print(model.loss_function(images, predicted_image, epoch, kld_mult)['SSIM_Loss'].item())
                print(batch_idx)
                print('simm je problem')                
            

            
# =============================================================================
#             print(batch_idx)
# =============================================================================
            
            if ((batch_idx // args.batch_accum)+1)* args.batch_accum <= l: #izmenio na <= sa <
                if check_nan_inf(loss/args.batch_accum)=='Error':
                    print(loss_params['loss'])
                    print(batch_idx)
                    print('lossdiv1 je problem')  
                loss=loss/args.batch_accum
            else:
                if check_nan_inf(loss/(l%args.batch_accum))=='Error':
                    print(loss_params['loss'])
                    print(batch_idx)
                    print('lossdiv2 je problem')                  
                loss=loss/(l%args.batch_accum)
            
            
            
            loss.backward()
            
            div= batch_idx+1
            if ((batch_idx + 1) % args.batch_accum == 0) or (batch_idx + 1 == l): 
                optimizer.step()
                optimizer.zero_grad()
                logger.add_scalar("Total_loss", total/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
                logger.add_scalar("Reconstruction_loss", rec_loss/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
                logger.add_scalar("KLD_loss", kld_loss/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
                logger.add_scalar("Sparse_loss", sparse_loss/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
                logger.add_scalar("SSIM_loss", simm_loss/div, global_step=epoch * math.ceil(l/args.batch_accum) + (batch_idx+1)//args.batch_accum + ((batch_idx + 1) % args.batch_accum != 0)*int(batch_idx + 1 == l) )
            
           #pbar.set_postfix(MSE=loss.item())
            
            pbar.set_postfix({'Epoch' : epoch, 'Total_loss':total/div,'rec_loss':rec_loss/div,'KLD_loss':kld_loss/div,'Sparse':sparse_loss/div,'SIMM_loss':simm_loss/div})
            
        save_images(images.detach(), os.path.join("results", args.run_name, f"{epoch}_orig.jpg"))
        save_images(predicted_image.detach(), os.path.join("results", args.run_name, f"{epoch}.jpg"))
        #torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        save_model_checkpoint(model, optimizer, loss, epoch, images.size()[-1], noise.size()[-1], kld_mult, args)




parser = argparse.ArgumentParser()
args = parser.parse_args()
args.run_name = "VAE_returnto128_nodrop"
args.lat_size=256
args.epochs = 1000 #change to 4 and train on all dataset #740
args.batch_size = 10
args.batch_accum = 10#8
args.image_size = 128
args.dataset_path = r"/home/filipk/Desktop/TRAIN"
args.device = "cuda"
args.lr =1e-4  #3e-4    

args.resume = True
args.resume_path =  "/home/filipk/Desktop/VAE_FINAL_v2/models/VAE_returnto128_nodrop/128_to_256_kld_mult_0.01_epoch_6ckpt.pt"

# =============================================================================
# input_image = torch.rand((1, 3, 256, 256)).to(torch.device('cuda:0'))
# noise = torch.rand((1, 1, 64, 64)).to(torch.device('cuda:0'))
# 
# vae = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder).cuda()
# 
# output = vae(input_image, noise).cpu().detach()
# plt.imshow(output[0].permute(1, 2, 0))
# =============================================================================
infer =0
if infer ==1:
    device = "cuda"
    model = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder).to(device)
    model.eval()   
    with torch.no_grad():
        #script_module = torch.jit.load("/home/filipk/Desktop/VAE_SD_LOWRES/models/VAE/ckpt.pt")
        ckpt = torch.load("/home/filipk/Desktop/VAE_FINAL_v2/models/VAE_trained_subset/256_to_256_kld_mult_0.01_epoch_999ckpt.pt")
        model.load_state_dict(ckpt['model_state_dict'])
        for i in range(10):
            images =model.sample(20,256)
            epoch='test_noise_90img_0.06w'+str(i)
            save_images(images.detach(), os.path.join("results", args.run_name, f"{epoch}_orig.jpg"))
        
    dataloader = get_data(args)      
    pbar = tqdm(dataloader)    
    with torch.no_grad():
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            noise = torch.randn((2, 4, 8, 8)).to(torch.device('cuda:0'))
            predicted_image = model(images , noise)
        epoch='test3'
        save_images(predicted_image.detach(), os.path.join("results", args.run_name, f"{epoch}.jpg"))
        
def interpolate(img1, img2):
    device = "cuda"
    model = VAE(encoder.VAE_Encoder, decoder.VAE_Decoder).to(device)
    model.eval()   
    with torch.no_grad():
        #script_module = torch.jit.load("/home/filipk/Desktop/VAE_SD_LOWRES/models/VAE/ckpt.pt")
        ckpt = torch.load("/home/filipk/Desktop/VAE_FINAL/models/VAE/256_to_256_kld_mult_0.01_epoch_999ckpt.pt")
        model.load_state_dict(ckpt['model_state_dict'])
        pic1=load_image("/home/filipk/Desktop/nature_pics/sea/"+str(img1)).to(torch.device('cuda:0'))
        pic2=load_image("/home/filipk/Desktop/nature_pics/sea/"+str(img2)).to(torch.device('cuda:0'))
        pic_in= (pic1+pic2)/2
        noise = torch.randn((pic1.size(0), 4, 16)).to(torch.device('cuda:0'))
        pic_res=model(pic_in,noise)
        save_images(pic_res.detach(), "/home/filipk/Desktop/res.jpg")
       
# =============================================================================
# pic1=load_image("/home/filipk/Desktop/nature_pics/sea_selection/00000000_(2).jpg")
# 
# save_images(pic1.detach(), "/home/filipk/Desktop/res.jpg")
# 
# =============================================================================
