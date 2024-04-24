#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:50:53 2023

@author: administrator
"""

import random
import os
import numpy as np
import torch
import torch.utils.data as data
import glob

from PIL import Image, ImageOps, ImageFilter

__all__ = ['ContrastiveClustering', 'CafCcDisentanglement', 'C8Disentanglement']


class ContrastiveClustering(data.Dataset):
    
    BASE_DIR = "./data"
    
    def __init__(self, images, root='./data/processed/np/', split='train', transform=None,
                 base_size=32, **kwargs):
        super(ContrastiveClustering, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.base_size = base_size
        self.images = images
    
    
    def __getitem__(self, index):
        if self.split == "train":
            img = np.load(self.images[index])
            h,w,c = img.shape
            if h < 32 or w < 32:
                img = self.padding(img)
            neg_index = random.randint(1, 360)
            neg = np.load(self.images[index-neg_index])
            h,w,c = neg.shape
            if h < 32 or w < 32:
                neg = self.padding(neg)
            aug_flag = random.uniform(0, 1)
            if aug_flag < 0.25:
                pos = self.img_blur(img)
            elif aug_flag < 0.50:
                pos = self.vertical_flip(img)
            elif aug_flag < 0.75:
                pos = self.horizontal_flip(img)
            else:
                pos = self.rotation(img)
            img = self.transform(img.astype(np.float32))
            pos = self.transform(pos.astype(np.float32))
            neg = self.transform(neg.astype(np.float32))
        
            return (img, pos, neg)
        else:
            img = np.load(self.images[index])
            h,w,c = img.shape
            if h < 32 or w < 32:
                img = self.padding(img)
            img = self.transform(img.astype(np.float32))
            return img
    
    def img_blur(self, array):
        for i in range(0,4):
            image = Image.fromarray(array[:,:,i])
            image = image.convert("L")
            array[:,:,i] = np.asarray(image.filter(ImageFilter.BLUR))
        return array
    
    def vertical_flip(self, array):
        for i in range(0,4):
            image = Image.fromarray(array[:,:,i])
            array[:,:,i] = np.asarray(image.transpose(Image.FLIP_TOP_BOTTOM))
        return array
    
    def horizontal_flip(self, array):
        for i in range(0,4):
            image = Image.fromarray(array[:,:,i])
            array[:,:,i] = np.asarray(image.transpose(Image.FLIP_LEFT_RIGHT))
        return array
    
    def rotation(self, array):
        angle = random.randint(1, 360)
        for i in range(0,4):
            image = Image.fromarray(array[:,:,i])
            array[:,:,i] = np.asarray(image.rotate(angle, resample=Image.BICUBIC))
        return array
    
    def padding(self, img):
        channels = []
        h = img.shape[0]
        w = img.shape[1]
        xx = 32
        yy = 32
        
        a = (xx - h) // 2
        aa = xx - a - h
        
        b = (yy - w) // 2
        bb = yy - b - w
        
        for i in [0,1,2,3]:
            channels.append(np.pad(img[:,:,i], pad_width=((a, aa), (b, bb)),
                                   mode='constant'))
        event = np.stack([channels[0], channels[1], channels[2], channels[3]], axis=2)
        return event

    
    def __len__(self):
        return len(self.images)
    

class CafCcDisentanglement(data.Dataset):
    def __init__(self, images, transform, base_size, split, ):
        """
        Custom dataset that takes a tensor of images.
        Args:
        - images (Tensor): A tensor containing all images in the shape [num_images, channels, height, width]
        """
        self.images = images
        self.transform = transform
        self.base_size = base_size
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)/65535.0
        image = image.permute(0, 3, 1, 2)
        image = self.transform(image)
        return image
    
class C8Disentanglement(data.Dataset):
    def __init__(self, images, labels, transform, base_size, split, ):
        """
        Custom dataset that takes a tensor of images.
        Args:
        - images (Tensor): A tensor containing all images in the shape [num_images, channels, height, width]
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.base_size = base_size
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)/65535.0
        label = self.labels[idx]
        image = self.transform(image)
        return image, label
    