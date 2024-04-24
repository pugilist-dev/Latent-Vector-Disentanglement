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
#from model_beta_gmm import deanVAE, deanGMVAE
from model import deanVAE, deanGMVAE
#from prep_data import get_data
#from resnet_gm_vae import deanGMVAE
from dynamic_prep_data import get_data, get_all_data
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
    parser.add_argument('--model', type=str, default='GM-VAE',
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

            '''
            #self.build_model(config.z_dim, config.beta, config.dropout)
            #self.build_optimizer(config.optimizer, config.learning_rate)
            #self.train(config.epochs, config.batch_size, config.beta,
                      # config.dropout, config.learning_rate,
                       #config.optimizer,config.z_dim)
            self.build_pretrained_gmmVAE(config.n, config.beta, config.dropout)
            self.build_gmm_opt(optimizer='adam',learning_rate=config.learning_rate)
            z_dim = self.gmmvae.latent_size
            config.z_dim = z_dim
            self.trainGMM(config.epochs, config.batch_size, config.beta, config.dropout, config.learning_rate,
                          config.optimizer, config.initial_temp, config.z_dim, config.n)
            '''
            self.build_gmmVAE(z_dim=config.z_dim, beta=config.beta, K=8, dropout=config.dropout)
            print("model initalized")
            self.build_gmm_opt(optimizer="adam", learning_rate=config.learning_rate,gamma=config.n)
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
                                                        
        self.train_img, self.train_labels, self.val_img, self.val_labels, self.rest_images, self.rest_labels = get_data()
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
        
        rest_data_kwargs = {'images': self.rest_images,
                        'labels': self.rest_labels,
                       'transform': self.transform,
                       'base_size': self.args.base_size,
                       'split':"rest"}
        
        self.train_data = get_dataset(self.args.dataset,  **train_data_kwargs)
        self.val_data = get_dataset(self.args.dataset, **val_data_kwargs)
        self.rest_data = get_dataset(self.args.dataset, **rest_data_kwargs)
        self.train_loader = data.DataLoader(dataset=self.train_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=16)
        
        self.val_loader = data.DataLoader(dataset = self.val_data,
                                          batch_size = batch_size,
                                          shuffle = False,
                                          num_workers=16)
        
        self.rest_loader = data.DataLoader(dataset = self.rest_data,
                                          batch_size = 1,
                                          shuffle = False,
                                          num_workers=16)
        
        self.test_loader = data.DataLoader(dataset = self.val_data,
                                          batch_size = 1,
                                          shuffle = False,
                                          num_workers=16)
    
    def build_optimizer(self, optimizer, learning_rate):
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                              lr=learning_rate, momentum=0.9)
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                               lr=learning_rate)
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
            
    def build_gmm_opt(self, optimizer, learning_rate,gamma):
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.gmmvae.parameters(),
                              lr=learning_rate, momentum=0.9)
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.gmmvae.parameters(),
                               lr=learning_rate)
            self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        
    def build_model(self, z_dim, beta, dropout):
        self.model = deanVAE(z_dim=z_dim, beta=beta, dropout=dropout).to(self.args.device)
        self.apply_initialization(self.model)
        
    def apply_initialization(self, model):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def build_pretrained_gmmVAE(self, n, beta, dropout):
        #best_epoch = [int(a.split("_")[2]) for a in os.listdir("./weights/")]
        #pretrained_path = glob.glob("./weights/C8_Disentanglement_"+str(max(best_epoch))+"_"+"*.pth")[0]
        #print(pretrained_path)
        pretrained_path = os.listdir("./beta_weights/")[n]
        og_zdim = int(pretrained_path.split("_")[8])
        pretrained_path = glob.glob('./beta_weights/'+pretrained_path)[0]
        self.model = deanVAE(z_dim=og_zdim, beta=beta, dropout=dropout).to(self.args.device)
        self.model.load_state_dict(torch.load(pretrained_path))
        self.gmmvae = deanGMVAE(original_model=self.model, z_dim= og_zdim,
                                beta = beta, dropout = dropout).to(self.args.device)
        
    def build_gmmVAE(self, z_dim, beta, K, dropout):
        self.gmmvae = deanGMVAE(z_dim= z_dim,beta = beta, K=K, dropout = dropout).to(self.args.device)
        #self.apply_initialization(self.gmmvae)

    
    def train(self, epochs, batch_size, beta, dropout, lr, opt, z_dim):
        best_val_loss = np.inf
        best_val_l1_loss = np.inf
        best_val_kl = np.inf
        best_epoch=0
        train_loss = 0
        anneal_epochs = int(epochs/3)
        best_epoch_ari=-1
        
        beta_increment = beta / anneal_epochs  # Increment beta each epoch during annealing
        
        for epoch in range(epochs):
            current_beta = min(beta, beta_increment * (epoch+1))
            
            self.model.train()
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.args.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar, z = self.model(images)
                loss, l1_loss, kl_diverge = self.model.loss(recon_batch, images, mu, logvar, current_beta, z)
                if math.isnan(loss) or math.isinf(loss):
                    best_val_loss = loss
                    wandb.log({"ari": best_epoch_ari})
                    return
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            self.scheduler.step()
            val_loss, val_l1, val_kl_diverge, ari = self.validation(epoch, batch_size, current_beta, dropout, lr, opt, z_dim)
            print(f"epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_l1_loss = val_l1
                best_val_kl = val_kl_diverge
                best_epoch=epoch
                best_epoch_ari=ari
                wandb.log({"ari": best_epoch_ari})
            elif epoch >= best_epoch+10:
                wandb.log({"ari": best_epoch_ari})
                return
        wandb.log({"ari": best_epoch_ari})
                
    def validation(self, epoch, batch_size, beta, dropout,lr, opt, z_dim):
        self.model.eval()
        val_loss = 0
        val_l1 = 0
        val_kl = 0
        n=-1
        ari=-1
        for i, (images, labels) in enumerate(self.val_loader):
            images = images.to(self.args.device)
            recon_batch, mu, logvar, z = self.model(images)
            loss, l1, kl = self.model.loss(recon_batch, images, mu, logvar, beta, z)
            val_loss += loss.item()
            val_l1 += l1.item()
            val_kl += kl.item()
          
        val_loss = val_loss/len(self.val_loader)
        val_loss = val_loss/batch_size
        
        val_l1 = val_l1/len(self.val_loader)
        val_l1 = val_l1/batch_size
        
        val_kl = val_kl/len(self.val_loader)
        val_kl = val_kl/batch_size
        
       
        if val_loss < self.best_pred:
            self.best_pred = val_loss
            ari = save_checkpoint(model=self.model, args=self.args, epoch=epoch,
                            batch_size=batch_size, beta=beta, dropout=dropout,
                            lr=lr, opt=opt, z_dim = z_dim, v_loss=val_loss, l1_loss=val_l1, kl_loss = val_kl, kl_y=0, n=n, is_best=True)
        return val_loss, val_l1, val_kl, ari
                
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
            elif epoch >50 and epoch >= best_epoch+20:
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
        wandb.login(key = "f50e7404c274fd0240b3de443ad368d762b16643")
        sweep_id = wandb.sweep(sweep_config, project='C8_Disentanglement')
        wandb.agent(sweep_id, Trainer, count=1000)
    else:
        trainer = Trainer()
