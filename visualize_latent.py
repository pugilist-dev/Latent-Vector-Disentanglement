#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:33:59 2024

@author: mandyana
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import deanGMVAE

path = "./weights//home/mandyana/Documents/C8_Disentanglement/data/interim/C8_Disentanglement_16_10_1.02_0.2_0.0005543121805853886_adam_7_634.6295870000666_615.7628964510831_14.736217617988586_4.130469799041748_6_0.2540073111431326.pth"

def explore_latent_dimension(model, dim_to_explore, num_steps=20, var_range=(-5, 5)):
    model.eval()  # Ensure model is in eval mode
    
    # Assuming a single Gaussian component for simplicity, in practice, you may need to
    # handle this according to your model's specific mixture model dynamics.
    
    linspace = np.linspace(var_range[0], var_range[1], num_steps)
    images = []
    
    with torch.no_grad():
        for val in linspace:
            # Generate a baseline z vector with the dimension of interest varied
            z = torch.zeros((1, model.latent_size))
            z[0, dim_to_explore] = val
            
            # Decode z to generate an image
            img = model.decoder(z).squeeze(0)
            images.append(img.cpu().numpy())
    
    # Plot the generated images
    fig, axs = plt.subplots(1, num_steps, figsize=(20, 2))
    
    for i, img in enumerate(images):
        # Initialize an empty RGB image
        rgb_image = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)  # Shape (H, W, 3)
        
        # Assign the first channel to blue
        rgb_image[:, :, 2] = img[0, :, :]+img[3, :, :]
        
        # Assign the second channel to red
        rgb_image[:, :, 0] = img[1, :, :]+img[3, :, :]
        
        # Assign the third channel to green
        rgb_image[:, :, 1] = img[2, :, :]+img[3, :, :]
        
        rgb_image[rgb_image > 1] = 1
        
        # Clip the values to be between 0 and 1 to ensure valid image data for display
        rgb_image = np.clip(rgb_image, 0, 1)
        
        axs[i].imshow(rgb_image)
        axs[i].axis('off')

plt.show()

# Load your model
model_trained = deanGMVAE(z_dim=9,beta=0.0007919304170535864,dropout=0.3, K=8)
model_trained.load_state_dict(torch.load(path))

# Example of using the function
explore_latent_dimension(model_trained, dim_to_explore=0)  # Modify as needed