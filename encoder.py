#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:01:41 2023

@author: filipk
"""

import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self, img_size, lat_size, reduction=[4,4], dropout_early=True, der=0.2, dropout_late=True, dlr=0.1):
        super(VAE_Encoder, self).__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height, Width)
            VAE_ResidualBlock(128,128),
            
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height, Width)
            VAE_ResidualBlock(128,128),
            
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size= 3, stride=2, padding=0 ),
            

            #nn.Dropout2d(p=der),
            
            # (Batch_Size, Channel, Height / 2, Width / 2) -> (Batch_Size, Channel, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256),
            
            # (Batch_Size, Channel, Height / 2, Width / 2) -> (Batch_Size, Channel, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),
            
# =============================================================================
#             # (Batch_Size, Channel, Height / 2, Width / 2) -> (Batch_Size, Channel, Height / 4, Width / 4)
#             nn.Conv2d(256, 256, kernel_size=3, stride = 2, padding = 0),
#             
#             # (Batch_Size, Channel, Height / 4, Width / 4) -> (Batch_Size, Channel, Height / 4, Width / 4)
#             VAE_ResidualBlock(256, 512),
#             
#             # (Batch_Size, Channel, Height / 4, Width / 4) -> (Batch_Size, Channel, Height / 4, Width / 4)
#             VAE_ResidualBlock(512, 512),
# =============================================================================
            
            # (Batch_Size, Channel, Height / 4, Width / 4) -> (Batch_Size, Channel, Height / 8, Width / 8)
            nn.Conv2d(256, 256, kernel_size=3, stride = 2, padding = 0),

            #nn.Dropout(p=dlr),
            # (Batch_Size, Channel, Height / 8, Width / 8) -> (Batch_Size, Channel, Height / 8, Width / 8)
            VAE_ResidualBlock(256, 256),
            
            # (Batch_Size, Channel, Height / 8, Width / 8) -> (Batch_Size, Channel, Height / 8, Width / 8)
            VAE_ResidualBlock(256, 256),
            
            # (Batch_Size, Channel, Height / 8, Width / 8) -> (Batch_Size, Channel, Height / 8, Width / 8)
            VAE_ResidualBlock(256, 256),
            
            # (Batch_Size, Channel, Height / 8, Width / 8) -> (Batch_Size, Channel, Height / 8, Width / 8)
            VAE_AttentionBlock(256),
            
            # (Batch_Size, Channel, Height / 8, Width / 8) -> (Batch_Size, Channel, Height / 8, Width / 8)
            VAE_ResidualBlock(256, 256),
            
            # (Batch_Size, Channel, Height / 8, Width / 8) -> (Batch_Size, Channel, Height / 8, Width / 8)
            nn.GroupNorm(32, 256),
            
            # (Batch_Size, Channel, Height / 8, Width / 8) -> (Batch_Size, Channel, Height / 8, Width / 8)
            nn.SiLU(),
            
            # (Batch_Size, Channel, Height / 8, Width / 8) -> (Batch_Size, Channel, Height / 8, Width / 8)
            nn.Conv2d(256, 8, kernel_size=3, padding = 1),

            # (Batch_Size, Channel, Height / 8, Width / 8) -> (Batch_Size, Channel, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding = 0),            
            )
        
        self.mu_layer = nn.Linear(int((img_size * img_size)/16), lat_size)
        self.log_var_layer = nn.Linear(int((img_size * img_size)/16), lat_size)
        self.mu= None
        self.log_var= None
        self.reduction = reduction
        self.dropout_early=dropout_early
        self.dropout_late=dropout_late
    
    def reparametrize(self, mean, log_variance, noise=None):
        variance = log_variance.exp()
        stdev = variance.sqrt()
        if noise == None:
            noise = torch.randn_like(variance) 
        z = mean + stdev * noise
        return z
    
    def forward(self, x: torch.tensor, noise: torch.Tensor = None) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channel, Height / 8, Width / 8)
        
        x = x + torch.randn_like(x) * 0.1 
        
        for module in self:
# =============================================================================
#             if ((getattr(module, 'stride', None) == (self.reduction[0],self.reduction[0]))
#             or (getattr(module, 'stride', None) == (self.reduction[1],self.reduction[1]))) :
#                 # (Padding_left, Padding_Right, Padding_Top, Padding_Bottom)
#                 x = F.pad(x,(0,1,0,1))  #WHY????
# =============================================================================
            if getattr(module, 'stride', None) == (2,2):
                # (Padding_left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x,(0,1,0,1))  #WHY????                
            if type(module) not in [torch.nn.modules.linear.Linear,torch.nn.modules.dropout.Dropout2d,torch.nn.modules.dropout.Dropout]:
                x = module(x)
            
            if (type(module) == torch.nn.modules.dropout.Dropout2d and self.dropout_early==True ) :
                x = module(x)
                
            if (type(module) == torch.nn.modules.dropout.Dropout2d and self.dropout_early==True ) :
                x = module(x)
                
        
        #assert n == 32, str(bch)+' '+str(chan)+' '+str(n)+' '+str(m)
        mean_mat, log_variance_mat = torch.chunk(x, chunks=2, dim=1)
        bch, chan, n, m = mean_mat.size()
        #assert bch == 32, str(bch)+' '+str(chan)+' '+str(n)+' '+str(m)
        mean_mat = mean_mat.view(bch, chan, n * m)    
        log_variance_mat = log_variance_mat.view(bch, chan, n * m)    
        self.mu = self.mu_layer(mean_mat)
        self.log_var = self.log_var_layer(log_variance_mat)
        
        self.log_var = torch.clamp(self.log_var, -30, 20)
                
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
# =============================================================================
#         mean, log_variance = torch.chunk(x, chunks=2, dim=1)
#         self.mu = mean
#         self.log_var= log_variance
# =============================================================================
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        #log_variance = torch.clamp(log_variance, -30, 20)
        
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)       
        #variance = log_variance.exp()
        
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)       
        #stdev = variance.sqrt()
        
        # N(0, 1) -> N(mean, variance)=X?
        # X = mean + stdev * Z
        #x = mean + stdev * noise
        x = self.reparametrize(self.mu, self.log_var, noise)
        
        # Scale the output by a constant (why? found in the original work)
        x*= 0.18215
        bch, chan, m = x.size()
        #assert m==16
        return (x)

#old implementation
# =============================================================================
#     def forward(self, x: torch.tensor, noise: torch.Tensor) -> torch.Tensor:
#         # x: (Batch_Size, Channel, Height, Width)
#         # noise: (Batch_Size, Out_Channel, Height / 8, Width / 8)
#         
#         for module in self:
#             if getattr(module, 'stride', None) == (4,4):
#                 # (Padding_left, Padding_Right, Padding_Top, Padding_Bottom)
#                 x = F.pad(x,(0,1,0,1))  #WHY????
#             x = module(x)
#                 
#         # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
#         mean, log_variance = torch.chunk(x, chunks=2, dim=1)
#         self.mu = mean
#         self.log_var= log_variance
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
#         log_variance = torch.clamp(log_variance, -30, 20)
#         
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)       
#         variance = log_variance.exp()
#         
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)       
#         stdev = variance.sqrt()
#         
#         # N(0, 1) -> N(mean, variance)=X?
#         # X = mean + stdev * Z
#         x = mean + stdev * noise
#         
#         # Scale the output by a constant (why? found in the original work)
#         x*= 0.18215
#         return (x)
# =============================================================================
   
    