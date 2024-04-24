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

        

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 scale_factor, mode="bilinear"):
        super(ResizeConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x
    

class BasicBlockEnc(nn.Module):
    
    def __init__(self, in_planes, stride=1, dropout=0.2):
        super(BasicBlockEnc, self).__init__()
        planes = in_planes*stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Dropout(dropout),
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1, dropout=0.2):
        super(BasicBlockDec, self).__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                nn.Dropout(dropout),
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=3, nc=4):
        super(ResNet18Enc, self).__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar
    
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
    
class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=3, nc=4):
        super(ResNet18Dec, self).__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=(75/32))

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 4, 75 , 75)
        return x
    
    
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
    def __init__(self, z_dim=8, K=8, dropout=0.3):
        super(DeanEncoderGMM, self).__init__()
        self.K = K
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)  # Batch normalization
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)  # Batch normalization
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer
        
        # Fully connected layers
        self.fc_mu = nn.Linear(512*2*2, z_dim * K)
        self.fc_logvar = nn.Linear(512*2*2, z_dim * K)
        self.fc_pi = nn.Linear(512*2*2, K)
        

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x) # Applying dropout
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        pi_logits = self.fc_pi(x)
        return mu, logvar, pi_logits
    
class DeanDecGMM(nn.Module):
    
    def __init__(self, z_dim=8):
        super(DeanDecGMM, self).__init__()
        self.fc = nn.Linear(z_dim, 512 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # Adjusted the stride and padding to maintain size
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1)

        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))
        # Use interpolation to adjust the final size precisely
        x = F.interpolate(x, size=(75, 75), mode='bilinear', align_corners=False)
        return x



class deanGMVAE(nn.Module):
    
    def __init__(self, z_dim=8, beta=5, K=7, dropout=0.2):
        super(deanGMVAE, self).__init__()
        self.latent_size = z_dim
        self.K = K
        self.beta = beta
        self.dropout = dropout
        self.encoder = DeanEncoderGMM(z_dim=self.latent_size,K=self.K, dropout=self.dropout)
        self.decoder = DeanDecGMM(z_dim=self.latent_size)
        self.register_buffer('pi_prior', torch.full((K,), fill_value=1.0/K))
        
    def forward(self, x, temperature=0.5):
        mean, logvar, pi_logits = self.encoder(x)
        y = deanGMVAE.sample_concrete(pi_logits, temperature)
        z = deanGMVAE.sample(mean, logvar, y, self.K, self.latent_size)  # Note the additional arguments here
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


class deanVAE(nn.Module):
    
    def __init__(self, z_dim=8, beta=5, K=56, dropout=0.2):
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
    
class VAE(nn.Module):

    def __init__(self, z_dim=3, beta=5, dropout=0.2):
        super(VAE, self).__init__()
        self.latent_size = z_dim
        self.beta = beta
        self.dropout = dropout
        self.encoder = ResNet18Enc(z_dim=self.latent_size)
        self.decoder = ResNet18Dec(z_dim=self.latent_size)

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
    
    def loss(self, recon_x, x, mu, logvar):
        l1_loss = F.l1_loss(recon_x, x, reduction='sum')
        kl_diverge = self.beta * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        return (l1_loss +kl_diverge) / x.shape[0]
    
    
    
    
if __name__ == '__main__':
    img = torch.randn(64, 4, 75, 75).to("cuda")
    model = deanGMVAE(z_dim=4, beta=5, dropout=0.2).to("cuda")
    out = model(img)