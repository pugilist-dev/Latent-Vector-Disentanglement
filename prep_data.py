#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:14:11 2024

@author: administrator
"""

import h5py
import numpy as np
import os
import glob

def augment_images(images):
    augmented_images = []
    augmented_images.append(images)
    
    for k in [1,2,3]:
        rotated_image = np.rot90(images, k=k, axes=(1,2))
        augmented_images.append(rotated_image)
    hflip_images = np.flip(images, axis=2)
    augmented_images.append(hflip_images)
    
    for k in [1,2,3]:
        rotated_mirrored_images = np.rot90(hflip_images, k=k, axes=(1,2))
        augmented_images.append(rotated_mirrored_images)
    merged_images = augmented_images[0]
    for i in range(1, len(augmented_images)):
        merged_images = np.concatenate((merged_images, augmented_images[i]), axis=0)
    return merged_images

def get_data(augment=True):
    root = './data/raw/'
    types = os.listdir(root)
    images = []
    labels = []
    for label, t in enumerate(types):
        current_type = glob.glob(root+t+"/*.hdf5")[0]
        print(t)
        with h5py.File(current_type, 'r') as f:
            imgs = np.array(f['images'][:])
            if t == "Common_FITC":
                imgs = imgs[np.random.choice(imgs.shape[0], 1000, replace=False)]
            elif t == "Common_CY5":
                imgs = imgs
            elif t == "Telocytes":
                imgs = imgs[np.random.choice(imgs.shape[0], 1000, replace=False)]
            elif augment == True:
                imgs = augment_images(imgs)
            images.append(imgs)
            labels.append(np.array([label]*imgs.shape[0]))
            print(imgs.shape[0])
            
    merged_data = images[0]
    merged_labels = labels[0]
    for i in range(1, len(images)):
        merged_data = np.concatenate((merged_data, images[i]), axis = 0)
        merged_labels = np.concatenate((merged_labels, labels[i]), axis = 0)
    return merged_data, merged_labels
