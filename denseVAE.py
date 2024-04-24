#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:43:04 2024

@author: mandyana
"""

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math


class DenseLayer(nn.Module):
    """
    A dense layer, consisting of two convolutional filters with batch
    normalisation and ReLU activation functions. The input layer is
    concatenated with the output layer.
    """
    
    def __init__(self, in_channels, growth_rate, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(DenseLayer, self).__init__()
        self.BN1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, 1, 1, 0)
            self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(in_channels, 4*growth_rate, 1, 1, 0)
            self.conv2 = nn.ConvTranspose2d(4*growth_rate, growth_rate, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(4*growth_rate)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        bn1 = self.BN1(x)
        relu1 = self.relu1(bn1)
        conv1 = self.conv1(relu1)
        bn2 = self.BN2(conv1)
        relu2 = self.relu2(bn2)
        conv2 = self.conv2(relu2)
        return torch.cat([x, conv2], dim=1)

class DenseBlock(nn.Module):
    """
    A dense block, consisting of three dense layers.
    """
    
    def __init__(self, in_channels, growth_rate, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(DenseBlock, self).__init__()
        self.DL1 = DenseLayer(in_channels+(growth_rate*0), growth_rate, mode)
        self.DL2 = DenseLayer(in_channels+(growth_rate*1), growth_rate, mode)
        self.DL3 = DenseLayer(in_channels+(growth_rate*2), growth_rate, mode)
    
    def forward(self, x):
        DL1 = self.DL1(x)
        DL2 = self.DL2(DL1)
        DL3 = self.DL3(DL2)
        return DL3

class TransitionBlock(nn.Module):
    """
    A transition block, consisting of a convolutional layer followed by a
    resize layer (average pooling for downsampling, transpose convolutional
    layer for upsampling).
    """
    
    def __init__(self, in_channels, c_rate, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(TransitionBlock, self).__init__()
        out_channels = int(c_rate*in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        if mode == 'encode':
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            self.resize_layer = nn.AvgPool2d(2, 2)
        elif mode == 'decode':
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 1, 1, 0)
            self.resize_layer = nn.ConvTranspose2d(out_channels, out_channels, 2, 2, 0)
    
    def forward(self, x):
        bn = self.BN(x)
        relu = self.relu(bn)
        conv = self.conv(relu)
        output = self.resize_layer(conv)
        return output
    
    
class DenseVAEEncoder(nn.Module):
    def __init__(self, growthRate, z_dim, K):
        super(DenseVAEEncoder, self).__init__()
        
        self.init_conv = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.db1 = DenseBlock(64, growthRate, 'encode')
        self.tb1 = TransitionBlock(64+(growthRate*3), 0.5, 'encode')
        self.db2 = DenseBlock((64+(growthRate*3))//2, growthRate, 'encode')
        self.tb2 = TransitionBlock((64+(growthRate*3))//2+growthRate*3, 0.5, 'encode')
        self.db3 = DenseBlock(((64+(growthRate*3))//2+growthRate*3)//2, growthRate, 'encode')
        self.final_channels=((64+(growthRate*3))//2+growthRate*3)//2+growthRate*3
        
        # Calculate nChannels for VAE-specific layers
        self.fc_mu = nn.Linear(self.final_channels*9*9, z_dim * K)  # Assuming final spatial size is 2x2
        self.fc_logvar = nn.Linear(self.final_channels*9*9, z_dim * K)
        self.fc_pi = nn.Linear(self.final_channels*9*9, K)

    def forward(self, x):
        x = F.relu(self.init_conv(x))
        x = self.db1(x)
        x = self.tb1(x)
        x = self.db2(x)
        x = self.tb2(x)
        x = self.db3(x)
        x = torch.flatten(x, 1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        pi = self.fc_pi(x)
        return mu, logvar, pi

class DenseVAEDecoder(nn.Module):
    def __init__(self, growthRate, z_dim, final_channels):
        super(DenseVAEDecoder, self).__init__()
        self.final_channels=final_channels
        self.fc = nn.Linear(z_dim, final_channels*9*9)
        self.init_conv = nn.ConvTranspose2d(final_channels, 24, kernel_size=2, stride=2, padding=1)
        self.db1 = DenseBlock(24, 8, 'decode') # 48 4 4
        self.tb1 = TransitionBlock(48, 0.5, 'decode') # 24 8 8
        self.db2 = DenseBlock(24, 8, 'decode') # 48 8 8
        self.tb2 = TransitionBlock(48, 0.5, 'decode') # 24 16 16
        self.db3 = DenseBlock(24, 8, 'decode') # 48 16 16
        self.BN1 = nn.BatchNorm2d(48)
        self.de_conv = nn.ConvTranspose2d(48, 4, 2, 2, 1) # 24 32 32
        self.BN2 = nn.BatchNorm2d(4)
        

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.final_channels, 9, 9)
        x = F.relu(self.init_conv(x))
        x = self.db1(x)
        x = self.tb1(x)
        x = self.db2(x)
        x = self.tb2(x)
        x = self.db3(x)
        x = torch.sigmoid(self.BN2(self.de_conv(x)))
        # Use interpolation to adjust the final size precisely
        x = F.interpolate(x, size=(75, 75), mode='bilinear', align_corners=False)
        return x


class denseGMVAE(nn.Module):
    
    def __init__(self, z_dim=8, beta=5, K=7, growthrate=4):
        super(denseGMVAE, self).__init__()
        self.latent_size = z_dim
        self.K = K
        self.beta = beta
        self.encoder = DenseVAEEncoder(growthRate=growthrate, z_dim=self.latent_size,K=self.K)
        self.decoder = DenseVAEDecoder(growthRate=growthrate, z_dim=self.latent_size,final_channels=self.encoder.final_channels)
        self.register_buffer('pi_prior', torch.full((K,), fill_value=1.0/K))
        
    def forward(self, x, temperature=0.5):
        mean, logvar, pi_logits = self.encoder(x)
        y = denseGMVAE.sample_concrete(pi_logits, temperature)
        z = denseGMVAE.sample(mean, logvar, y, self.K, self.latent_size)  # Note the additional arguments here
        #z_prime = self.attention_grab(z)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar, pi_logits, z
    
    def pairwise_cosine_similarity(self, x):
        x_normalized = x / x.norm(dim=1, keepdim=True)
        similarity_matrix = torch.mm(x_normalized, x_normalized.t())
        return similarity_matrix
    
    def attention_grab(self, z):
        cosine_sim = self.pairwise_cosine_similarity(z)
        z_prime = torch.mm(cosine_sim, z)
        return z_prime
    
    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -torch.log(-torch.log(U+eps)+eps)
    
    @staticmethod
    def sample_concrete(logits, temperature):
        gumbel_noise = denseGMVAE.sample_gumbel(logits.size())
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)
    
    @staticmethod
    def sample(mean, logvar, y, K, latent_size):
        batch_size = mean.size(0)
        
        # Reshape mean and logvar to [batch_size, K, latent_size] to separate components
        mean = mean.view(batch_size, K, latent_size)
        logvar = logvar.view(batch_size, K, latent_size)
        
        # Compute standard deviation
        std = torch.exp(0.5 * logvar)
        
        # Sample epsilon for each Gaussian component
        eps = torch.randn_like(std)
        
        # Reparameterize each component
        z_components = mean + eps * std  # Shape: [batch_size, K, latent_size]
        
        # Weight samples by responsibilities y
        # First, ensure y is correctly shaped for weighting the components
        y = y.unsqueeze(-1)  # Shape: [batch_size, K, 1] to broadcast over latent_size
        z_weighted = torch.sum(z_components * y, dim=1)  # Shape: [batch_size, latent_size]
        
        return z_weighted
    
    '''
    def compute_kl_concrete(self,logits, pi_prior, temperature, wt):
        q_y = F.softmax(logits / temperature, dim=-1)
        log_q_y = torch.log(q_y + 1e-20)  # Adding a small constant to prevent log(0)
        log_pi_prior = torch.log(pi_prior + 1e-20)
        kl_diverge_y = torch.sum(q_y * (log_q_y - log_pi_prior), dim=-1).mean()
        return kl_diverge_y*wt
    '''
        
    def loss(self, recon_x, x, mu, pi_logits, temperature, logvar, current_beta, current_alpha):
        l1_loss = current_alpha*self.calculate_l1_loss(recon_x, x)
        kl_loss = current_beta*self.calculate_gaussian_kl_loss(mu, logvar)
        cat_loss = self.compute_categorical_loss(pi_logits, temperature)
        total_loss = l1_loss + kl_loss + cat_loss
        return total_loss, l1_loss, kl_loss, cat_loss
    
    def calculate_l1_loss(self, recon_x, x):
        batch_size = x.size(0)
        return F.l1_loss(recon_x, x, reduction="sum") / batch_size

    def calculate_gaussian_kl_loss(self, mu, logvar):
        batch_size = mu.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        return kl_div

    def compute_categorical_loss(self, pi_logits, temperature=1.0):
        batch_size, num_classes = pi_logits.shape
        
        # Define the target distribution as uniform across the 8 classes
        targets = torch.full_like(pi_logits, fill_value=1.0/num_classes)
        
        # Apply temperature scaling on logits and compute the log softmax
        log_q = F.log_softmax(pi_logits / temperature, dim=-1)
        
        # Compute the categorical loss
        categorical_loss = -torch.mean(torch.sum(targets * log_q, dim=-1))
        
        return categorical_loss


    
    
if __name__ == '__main__':
    img = torch.randn(64, 4, 75, 75).to("cuda")
    model = denseGMVAE(z_dim=4, beta=5).to("cuda")
    out = model(img)