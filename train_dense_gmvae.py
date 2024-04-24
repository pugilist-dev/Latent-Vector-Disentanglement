#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:02:36 2024

@author: mandyana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:39:09 2024

@author: administrator
"""

import argparse
import os

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np
from sklearn.model_selection import train_test_split
import wandb
import math
import pandas as pd
    
from data_loader import get_dataset
from denseVAE import denseGMVAE
from dynamic_prep_data import get_data
from config.sweep_config import get_sweep_config
import glob
from utils.plot_assist import draw_inference, draw_inference_gmmvae, Metrics
import plotly.express as px

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic=True
os.environ['PYTHONHASHSEED']=str(42)

def parse_args():
    parser = argparse.ArgumentParser(description='C8_Disentanglement')
    parser.add_argument('--model', type=str, default='Dense-GM-VAE',
                        help="Deep learning model name")
    parser.add_argument('--dataset', type=str, default='C8_Disentanglement',
                        help='dataset name int_events')
    parser.add_argument('--base_size', type=int, default=75,
                        help='Base image size running through the network')
    parser.add_argument('--resume', type=str, default="", 
                        help='put the path to resuming file if needed')
    parser.add_argument('--save_folder', default='./weights',
                        help='Directory for saving checkpoint models')
    parser.add_argument("--plot_check", action='store_true', default=True,
                        help="flag to check scatter plot disentanglement")
    args = parser.parse_args()
    device = torch.device("cuda")
    cudnn.benchmark = True
    args.device = device
    print(args)
    return args

class Trainer(object):
    def __init__(self, config=None):
        with wandb.init(config=config):
            '''
            if config is None:
                self.args = parse_args()
                self.build_dataset(batch_size = 496)
                self.build_model(z_dim = 5, beta = 1.4, dropout = .2)
                self.build_optimizer(optimizer = 'adam', learning_rate=0.0004749)
                self.best_pred = np.inf
                #self.train(epochs=100, batch_size=496, beta=1.4,
                #           dropout=0.2, lr=0.0004749,
                #           opt="adam", z_dim=5)
                self.build_gmmVAE(z_dim=5, beta=1.4, dropout=.2)
                self.build_gmm_opt(optimizer='adam', learning_rate=0.0004749)
                self.trainGMM(epochs=100, batch_size=496,
                              beta=1.4, dropout=.2, lr=0.0004749,
                              opt = 'adam', initial_temperature=3,
                              z_dim=5, gamma=8)
            
            '''
            config = wandb.config
            self.args = parse_args()
            self.build_dataset(config.batch_size)
            print("built data")
            self.best_pred = np.inf
            self.build_gmmVAE(z_dim=config.z_dim, beta=config.beta, K=48, growthrate=config.n)
            print("model initalized")
            self.build_gmm_opt(optimizer="adam", learning_rate=config.learning_rate)
            self.trainGMM(config.epochs, config.batch_size, config.beta, config.dropout, config.learning_rate,
                          config.optimizer, config.initial_temp, config.z_dim, config.n, config.alpha)
            
    def build_dataset(self, batch_size):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        #self.images, self.labels = get_data()
        #self.train_img, self.val_img, self.train_labels, self.val_labels = train_test_split(self.images,
                                                        #self.labels,
                                                        #test_size=0.2)
                                                        
        self.train_img, self.train_labels, self.val_img, self.val_labels, self.rest_img, self.rest_labels = get_data()
        train_data_kwargs = {'images': self.train_img,
                        'labels': self.train_labels,
                       'transform': self.transform,
                       'base_size': self.args.base_size,
                       'split':"train"}
        val_data_kwargs = {'images': self.val_img,
                        'labels': self.val_labels,
                       'transform': self.transform,
                       'base_size': self.args.base_size,
                       'split':"val"}
        
        self.train_data = get_dataset(self.args.dataset,  **train_data_kwargs)
        self.val_data = get_dataset(self.args.dataset, **val_data_kwargs)
        self.train_loader = data.DataLoader(dataset=self.train_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=8)
        
        self.val_loader = data.DataLoader(dataset = self.val_data,
                                          batch_size = batch_size,
                                          shuffle = False,
                                          num_workers=8)
        
        self.test_loader = data.DataLoader(dataset = self.val_data,
                                          batch_size = 1,
                                          shuffle = False,
                                          num_workers=8)
    
            
    def build_gmm_opt(self, optimizer, learning_rate):
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.gmmvae.parameters(),
                              lr=learning_rate, momentum=0.9)
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.96)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.gmmvae.parameters(),
                               lr=learning_rate)
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        
    def apply_initialization(self, model):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
        
    def build_gmmVAE(self, z_dim, beta, K, growthrate):
        self.gmmvae = denseGMVAE(z_dim= z_dim,beta = beta, K=K, growthrate= growthrate).to(self.args.device)
        #self.apply_initialization(self.gmmvae)

                
    def trainGMM(self, epochs, batch_size, beta, dropout, lr, opt, initial_temperature, z_dim, n, alpha):
        best_val_loss = np.inf
        best_val_l1_loss = np.inf
        best_val_kl = np.inf
        best_val_cat = np.inf
        best_epoch=0
        anneal_epochs = int(epochs/2)
        best_epoch_ari =-1
        best_epoch_nmi = 0
        
        beta_increment = beta / anneal_epochs  # Increment beta each epoch during annealing
        temp_increment = (initial_temperature-0.5) / anneal_epochs
        alpha_increment = (alpha-1) / anneal_epochs
        print("Started Training")
        for epoch in range(epochs):
            current_beta = min(beta, beta_increment * (epoch+1))
            temperature = max(0.5, initial_temperature-(temp_increment * (epoch+1)))
            current_alpha = max(1, alpha-(alpha_increment*(epoch+1)))
            self.gmmvae.train()
            train_loss=0
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.args.device)
                self.optimizer.zero_grad()
                
                recon_batch, mu, logvar, pi_logits, z = self.gmmvae(images, temperature=temperature)
                loss, l1_loss, kl_loss, cat_loss = self.gmmvae.loss(recon_batch, images, mu, pi_logits,temperature, logvar, current_beta,current_alpha)
                
                if math.isnan(loss) or math.isinf(loss):
                    best_val_loss = loss
                    wandb.log({"ari": best_epoch_ari})
                    wandb.log({"nmi": best_epoch_nmi})
                    wandb.log({"val_loss": best_val_loss})
                    return
                
                loss.backward()
                
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= i
            self.scheduler.step()
            val_loss, val_l1, val_kl_diverge, val_cat, ari, nmi = self.validationGMM(epoch, batch_size, current_beta, dropout, lr, opt, z_dim,n,temperature=temperature, current_alpha=current_alpha)
            print(f"epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_l1_loss = val_l1
                best_val_kl = val_kl_diverge
                best_val_cat = val_cat
                best_epoch=epoch
                best_epoch_ari = ari
                best_epoch_nmi = nmi
                wandb.log({"ari": best_epoch_ari})
                wandb.log({"nmi": best_epoch_nmi})
                wandb.log({"val_loss": best_val_loss})
            elif epoch >= best_epoch+20:
                wandb.log({"ari": best_epoch_ari})
                wandb.log({"nmi": best_epoch_nmi})
                wandb.log({"val_loss": best_val_loss})
                return
        wandb.log({"ari": best_epoch_ari})
        wandb.log({"nmi": best_epoch_nmi})
        wandb.log({"val_loss": best_val_loss})
    
                
    def validationGMM(self, epoch, batch_size, beta, dropout,lr, opt, z_dim, n, temperature, current_alpha):
        self.gmmvae.eval()
        val_loss = 0
        val_l1 = 0
        val_kl = 0
        val_cat = 0
        ari=-1
        nmi=0
        for i, (images, labels) in enumerate(self.val_loader):
            images = images.to(self.args.device)
            recon_batch, mu, logvar, pi_logits, z = self.gmmvae(images, temperature=temperature)
            loss, l1, kl, cat = self.gmmvae.loss(recon_batch, images, mu, pi_logits,temperature, logvar, beta, current_alpha)
            val_loss += loss.item()
            val_l1 += l1.item()
            val_kl += kl.item()
            val_cat += cat.item()
          
        val_loss = val_loss/i
        val_l1 = val_l1/len(self.val_loader)
        val_kl = val_kl/len(self.val_loader)
        val_cat = val_cat/len(self.val_loader)

        if val_loss < self.best_pred:
            self.best_pred = val_loss
            ari, nmi = save_checkpoint(model=self.gmmvae, args=self.args, epoch=epoch,
                            batch_size=batch_size, beta=beta, dropout=dropout,
                            lr=lr, opt=opt, z_dim = z_dim, v_loss=val_loss, l1_loss=val_l1, kl_loss = val_kl, kl_y=val_cat, n=n, loader = self.test_loader,is_best=True)
        return val_loss, val_l1, val_kl, val_cat, ari, nmi
            
def save_checkpoint(model, args, epoch, batch_size, beta, dropout, lr, opt,
                    z_dim, v_loss, l1_loss, kl_loss, kl_y, n, loader, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    previous_model = glob.glob("./weights/*.pth")
    if previous_model:
        previous_best = float(glob.glob("./weights/*.pth")[0].split("_")[-1].split(".")[0])
    else:
        previous_best = None
    if previous_best:
        if v_loss < previous_best:
            if args.plot_check:
                ari, nmi = plot_best_model(model, args, epoch, batch_size, beta, dropout, lr, opt,
                                    z_dim, v_loss, l1_loss, kl_loss, kl_y, n, loader)
                return ari, nmi
            
        else:
            return
    else:
        if is_best:
            if args.plot_check:
                ari, nmi = plot_best_model(model, args, epoch, batch_size, beta, dropout, lr, opt,
                                    z_dim, v_loss, l1_loss, kl_loss, kl_y, n, loader)
                return ari, nmi

def plot_best_model(model, args, epoch, batch_size, beta, dropout, lr, opt,
                    z_dim, v_loss, l1_loss, kl_loss, kl_y, n, loader):
    directory = os.path.expanduser(args.save_folder)

    inference = pd.DataFrame(draw_inference_gmmvae(model=model, loader=loader,
                                            device=args.device, z_dim=z_dim))
    metric = Metrics(inference)
    
    
    best_filename = f'{args.dataset}_{epoch}_{batch_size}_{beta:.5f}_{z_dim}_{metric.ari_score}_{metric.nmi}.pth'
    best_filename = os.path.join(directory, best_filename)
    torch.save(model.state_dict(), best_filename)
    inference.to_csv(f'./data/interim/{args.dataset}_{epoch}_{batch_size}_{beta:.5f}_{z_dim}_{metric.ari_score}_{metric.nmi}.csv')
    return metric.ari_score, metric.nmi


if __name__ == '__main__':
    tune = True
    if tune:
        sweep_config = get_sweep_config()
        wandb.login(key = "acc757e2b7b115ac775cc6f3d2c30fb3fa846f60")
        sweep_id = wandb.sweep(sweep_config, project='C8_Disentanglement')
        wandb.agent(sweep_id, Trainer, count=1000)
    else:
        trainer = Trainer()
