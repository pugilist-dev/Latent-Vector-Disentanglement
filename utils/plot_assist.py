#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:36:14 2024

@author: administrator
"""
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, v_measure_score
from igraph import Graph
import networkx as nx
from community import community_louvain


def draw_inference(model, loader,  device, z_dim):
    model.eval()
    z_score = [str(a) for a in range(z_dim)]
    disent_z = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(device)
            output, mu, logvar, z = model(data)
            z = z.detach().cpu().numpy().reshape(z_dim)
            data = {z_score[i]: z[i] for i in range(len(z))}
            data["label"] = str(int(label))
            disent_z.append(data)
    return disent_z

def draw_inference_gmmvae(model, loader,  device, z_dim):
    model.eval()
    z_score = [str(a) for a in range(z_dim)]
    disent_z = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(device)
            output, mu, logvar, pi_logits, z = model(data)
            z = z.detach().cpu().numpy().reshape(z_dim)
            data = {z_score[i]: z[i] for i in range(len(z))}
            data["label"] = str(int(label))
            disent_z.append(data)
    return disent_z



class Metrics(object):
    def __init__(self, inference_df):
        self.z, self.labels = self.process(inference_df)
        self.adjacency_matrix = self.get_knn_adjacency(self.z)
        self.graph = Graph.Adjacency((self.adjacency_matrix > 0).tolist())
        self.partition = community_louvain.best_partition(nx.Graph(self.adjacency_matrix), resolution=0.7)
        self.predicted_labels = [self.partition.get(node) for node in range(len(self.z))]
        self.ari_score = adjusted_rand_score(self.labels, self.predicted_labels)
        self.nmi = v_measure_score(self.labels, self.predicted_labels)
        
    def process(self, df):
        data = df.iloc[:, 0:-1]  # Assuming the last column is labels
        labels = df.iloc[:, -1].to_list()
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data, labels
    
    def get_knn_adjacency(self, data, k=15):
        data = torch.from_numpy(data).to("cuda")
        dist_matrix = self.get_pw_dist(data)
        dist_matrix = dist_matrix.cpu().numpy()
        adjacency_matrix = np.zeros(dist_matrix.shape)
        for i in range(dist_matrix.shape[0]):
            sorted_indices = np.argsort(dist_matrix[i, :])
            adjacency_matrix[i, sorted_indices[0:k+1]] = 1  # Skipping the first one as it is the point itself
        return adjacency_matrix
    
    def get_pw_dist(self, z):
        distance_matrix = torch.cdist(z, z, p=2)
        return distance_matrix
        
