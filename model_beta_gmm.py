#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:09:15 2024

@author: mandyana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:13:14 2023

@author: administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

        

class DeanEncoder(nn.Module):
    def __init__(self, z_dim=8,dropout=0.3):
        super(DeanEncoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1) # Output: [32, 38, 38]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Output: [64, 19, 19]
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # Output: [128, 10, 10]
        self.conv4 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1) # Output: [256, 4, 4]
        self.dropout = nn.Dropout(p=dropout) # Dropout layer
        self.fc_mu = nn.Linear(256*4*4, z_dim)
        self.fc_logvar = nn.Linear(256*4*4, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x) # Applying dropout
        x = F.relu(self.conv4(x))
        x = self.dropout(x) # Applying dropout
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
   
class DeanDec(nn.Module):
    
    def __init__(self, z_dim=8):
        super(DeanDec, self).__init__()
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # Adjusted the stride and padding to maintain size
        self.deconv4 = nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        # Use interpolation to adjust the final size precisely
        x = F.interpolate(x, size=(75, 75), mode='bilinear', align_corners=False)
        return x

class DeanEncoderGMM(nn.Module):
    def __init__(self, original_encoder, z_dim=8, K=8, dropout=0.3):
        super(DeanEncoderGMM, self).__init__()
        self.K = K
        self.features = nn.Sequential(
            *list(original_encoder.children())[:-2],  # Example to exclude the last two layers
        )
        self.fc_mu = nn.Linear(256*4*4, z_dim * K)
        self.fc_logvar = nn.Linear(256*4*4, z_dim * K)
        self.fc_pi = nn.Linear(256*4*4, K)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        pi_logits = self.fc_pi(x)
        return mu, logvar, pi_logits
    
class DeanDecGMM(nn.Module):
    
    def __init__(self, original_decoder, z_dim=8):
        super(DeanDecGMM, self).__init__()
        self.features = nn.Sequential(original_decoder)
        
    def forward(self, z):
        x = self.features(z)
        x = F.interpolate(x, size=(75, 75), mode='bilinear', align_corners=False)
        return x


class deanGMVAE(nn.Module):
    
    def __init__(self, original_model, z_dim=8, beta=5, K=8, dropout=0.2):
        super(deanGMVAE, self).__init__()
        self.latent_size = z_dim
        self.K = K
        self.beta = beta
        self.dropout = dropout
        self.encoder = DeanEncoderGMM(original_model.encoder,z_dim=self.latent_size,K=self.K, dropout=self.dropout)
        self.decoder = DeanDecGMM(original_model.decoder, z_dim=self.latent_size)
        self.register_buffer('pi_prior', torch.full((K,), fill_value=1.0/K))
        
    def forward(self, x, temperature=0.5):
        mean, logvar, pi_logits = self.encoder(x)
        y = deanGMVAE.sample_concrete(pi_logits, temperature)
        z = deanGMVAE.sample(mean, logvar, y, self.K, self.latent_size)  # Note the additional arguments here
        recon_x = self.decoder(z)
        return recon_x, mean, logvar, pi_logits, z
    
    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -torch.log(-torch.log(U+eps)+eps)
    
    @staticmethod
    def sample_concrete(logits, temperature):
        gumbel_noise = deanGMVAE.sample_gumbel(logits.size())
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
        
    def loss(self, recon_x, x, mu, pi_logits, temperature, logvar, current_beta):
        l1_loss = self.calculate_l1_loss(recon_x, x)
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

class deanVAE(nn.Module):
    
    def __init__(self, z_dim=8, beta=5, K=8, dropout=0.2):
        super(deanVAE, self).__init__()
        self.K = K
        self.latent_size = z_dim
        self.beta = beta
        self.dropout = dropout
        self.encoder = DeanEncoder(z_dim=self.latent_size,dropout=self.dropout)
        self.decoder = DeanDec(z_dim=self.latent_size)
        
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sample(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar, z
    
    @staticmethod
    def sample(mean, logvar):
        std = torch.exp(0.5*logvar) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
    
    def loss(self, recon_x, x, mu, logvar, current_beta, z):
        l1_loss = F.l1_loss(recon_x, x, reduction='sum')
        kl_diverge = current_beta * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        loss_final = (l1_loss +kl_diverge) / x.shape[0]
        l1_loss = l1_loss / x.shape[0]
        kl_diverge = kl_diverge / x.shape[0]
        return loss_final, l1_loss, kl_diverge
    
if __name__ == '__main__':
    img = torch.randn(64, 4, 75, 75).to("cuda")
    VAE = deanVAE().to("cuda")
    print(VAE)
    gmmvae = deanGMVAE(VAE).to("cuda")
    print(gmmvae)
    one, two, three, four, five = gmmvae(img)
    