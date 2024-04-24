#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:39:50 2024

@author: mandyana
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import deanGMVAE  # Make sure this matches your model's import
from statistics import mean
import pandas as pd
# Load your pre-trained model (adjust the path to your model's weights)
model_path = "./weights/C8_Disentanglement_15_13_0.7040000000000001_0.35_0.00043648534816818733_adam_10_624.5665314618279_607.3110293220071_13.365597220028148_3.889899965594797_3_0.5474126877635513.pth"  # Update this path
data_path = "./data/interim/C8_Disentanglement_15_13_0.7040000000000001_0.35_0.00043648534816818733_adam_10_624.5665314618279_607.3110293220071_13.365597220028148_3.889899965594797_3_0.5474126877635513.csv"
model = deanGMVAE(z_dim=10, beta=0.704, dropout=0.35, K=8)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

data = pd.read_csv(data_path)
data = data.iloc[:, 1:-1]


# Calculate min and max for each of the variables X0-X7
slider_ranges = []
for column in range(0,model.latent_size):
    min_value = float(data.iloc[:,column].min())
    max_value = float(data.iloc[:,column].max())
    slider_ranges.append((min_value, max_value))

# Print the slider ranges
print(slider_ranges)
# Function to generate an image from the model based on a latent vector
def generate_image(latent_vector):
    with torch.no_grad():
        img = model.decoder(latent_vector.unsqueeze(0)).squeeze(0)

        # Process the image data as before
        rgb_image = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)

        rgb_image[:, :, 0] = img[1, :, :] + img[3, :, :]  # Red
        rgb_image[:, :, 1] = img[2, :, :] + img[3, :, :]  # Green
        rgb_image[:, :, 2] = img[0, :, :] + img[3, :, :]  # Blue

        rgb_image = np.clip(rgb_image / rgb_image.max(), 0, 1)

        # Convert numpy array to PIL Image for display
        image_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
        
        return image_pil


# Initialize session state for sliders
latent_dim = model.latent_size
latent_vector = []
for i in range(latent_dim):
    min_val, max_val = slider_ranges[i]
    val = st.sidebar.slider(f"Dimension {i+1}", min_value=min_val, max_value=max_val, value=mean(slider_ranges[i]), step=0.05, key=f'slider_{i}')
    latent_vector.append(val)
    
latent_vector = torch.tensor(latent_vector, dtype=torch.float32)

# Generate and display the image automatically
img = generate_image(latent_vector)
st.image(img, caption='Generated Image', use_column_width=True)