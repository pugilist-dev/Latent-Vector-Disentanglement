#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:21:41 2024

@author: mandyana
"""

from utils.plot_assist import draw_inference_gmmvae, Metrics
from dynamic_prep_data import get_data
import torch
import torch.nn as nn
import torch.functional as F
import pandas as pd
from model import deanGMVAE
from data_loader import get_dataset
from torchvision import transforms
import torch.utils.data as data


def main(model, loader, device,z_dim):
    inference = pd.DataFrame(draw_inference_gmmvae(model=model, loader=loader,device=device, z_dim=z_dim))
    inference.to_csv(f'./best_model.csv')
    metric = Metrics(inference)
    return metric.ari_score, metric.nmi

if __name__ == '__main__':
    train_img, train_labels, val_img, val_labels, rest_images, rest_labels = get_data()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    val_data_kwargs = {'images': val_img,
                    'labels': val_labels,
                   'transform': transform,
                   'base_size': 75,
                   'split':"val"}

    
    val_data = get_dataset("C8_Disentanglement", **val_data_kwargs)
    
    val_loader = data.DataLoader(dataset = val_data,
                                      batch_size = 1,
                                      shuffle = False,
                                      num_workers=24)
    
    model_path = "./C8_Disentanglement_19_7_0.7200000000000001_0.25_0.0005359748770038437_adam_7_631.7508364257812_617.9018459472657_10.062434078216553_3.7865562229156495_5_0.49532368295268303.pth"
    gmmvae = deanGMVAE(z_dim= 7,beta = 0.720001, K=8, dropout = 0.25).to("cuda")
    gmmvae.load_state_dict(torch.load(model_path))
    gmmvae.eval()
    ari, nmi= main(gmmvae,val_loader,"cuda",7)
    print(ari, nmi)