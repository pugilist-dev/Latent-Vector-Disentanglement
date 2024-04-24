#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:55:01 2024

@author: mandyana
"""
import h5py
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import itertools

def dynamic_augment_images(images, target_count):
    # Initialize with original images
    augmented_images = [images]
    original_count = images.shape[0]
    
    if original_count >= target_count:
        # Downsample to target count if necessary
        selected_indices = np.random.choice(original_count, target_count, replace=False)
        return images[selected_indices]

    # Apply augmentations
    for k in [1, 2, 3]:
        if len(augmented_images) * original_count < target_count:
            rotated_image = np.rot90(images, k=k, axes=(1, 2))
            augmented_images.append(rotated_image)

    hflip_images = np.flip(images, axis=2)
    augmented_images.append(hflip_images)

    for k in [1, 2, 3]:
        if len(augmented_images) * original_count < target_count:
            rotated_mirrored_images = np.rot90(hflip_images, k=k, axes=(1, 2))
            augmented_images.append(rotated_mirrored_images)

    # Combine augmented images
    merged_images = np.concatenate(augmented_images, axis=0)
    
    if merged_images.shape[0] > target_count:
        # Downsample if over target after augmentation
        selected_indices = np.random.choice(merged_images.shape[0], target_count, replace=False)
        merged_images = merged_images[selected_indices]
    
    return merged_images

def shuffle_channels_and_return_permutations(images):
    # Assuming images are in the format [batch, height, width, channels]
    # Generate permutations for the 3 channels, keeping the first channel fixed
    channel_permutations = list(itertools.permutations([1, 2, 3]))
    permuted_images = []
    
    for image in images:
        # Include original image
        permuted_images.append(image)
        for perm in channel_permutations:
            # Apply each permutation
            new_channel_order = [0] + list(perm)
            permuted_image = image[..., new_channel_order]
            permuted_images.append(permuted_image)
            
    return np.array(permuted_images)

def get_data(augment=True, target_per_class=1000, test_size=0.2):
    root = './data/raw/'
    types = os.listdir(root)
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    rest_images = []
    rest_labels = []
    
    for label, t in enumerate(types):
        current_type = glob.glob(root+t+"/*.hdf5")[0]
        with h5py.File(current_type, 'r') as f:
            imgs = np.array(f['images'][:])
            original_count = imgs.shape[0]
        
            if original_count > target_per_class/(1-test_size):
                downsampled_indices = np.random.choice(imgs.shape[0],int(target_per_class/(1-test_size)),replace=False)
                train_val_imgs = imgs[downsampled_indices]
                # Assuming imgs is a numpy array and downsampled_indices is the array of selected indices for downsampling
                mask = np.ones(imgs.shape[0], dtype=bool)  # Create a mask of True values
                mask[downsampled_indices] = False  # Set the downsampled indices to False
                rest_imgs = imgs[mask]  # Select images where the mask is True, excluding downsampled indices
            else:
                train_val_imgs = imgs
                rest_imgs = np.empty((0,) + imgs.shape[1:])
            
            
            # Split into train and validation before augmentation or downsampling
            imgs_train, imgs_val = train_test_split(train_val_imgs, test_size=test_size, random_state=42)
            train_count = imgs_train.shape[0]
            # Apply augmentation or downsampling to training data
            if augment and train_count < target_per_class:
                imgs_train = dynamic_augment_images(imgs_train, int(target_per_class))
            
            
            train_images.append(imgs_train)
            train_labels.append(np.array([label] * imgs_train.shape[0]))
            val_images.append(imgs_val)
            val_labels.append(np.array([label] * imgs_val.shape[0]))
            rest_images.append(rest_imgs)
            rest_labels.append(np.array([label] * rest_imgs.shape[0]))
    
    # Concatenate all class data for training and validation
    train_data = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_data = np.concatenate(val_images, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    
    rest_data = np.concatenate(rest_images, axis=0)
    rest_labels = np.concatenate(rest_labels, axis=0)
    
    random_idx = np.random.choice(rest_data.shape[0],2000,replace=False)
    rest_data = rest_data[random_idx]
    rest_labels = rest_labels[random_idx]
    
    rest_data = np.concatenate([rest_data, val_data], axis=0)
    rest_labels = np.concatenate([rest_labels, val_labels], axis=0)

    return train_data, train_labels, val_data, val_labels, rest_data, rest_labels

def get_all_data():
    root = './data/raw/'
    types = os.listdir(root)
    images=[]
    labels=[]
    for label, t in enumerate(types):
        current_type = glob.glob(root+t+"/*.hdf5")[0]
        with h5py.File(current_type, 'r') as f:
            imgs = np.array(f['images'][:])
        images.append(imgs)
        labels.append(np.array([label] * imgs.shape[0]))
        
    final_images = np.concatenate(images, axis=0)
    final_labels = np.concatenate(labels, axis=0)
    
    num_samples = final_images.shape[0]  # This assumes the first dimension is the number of samples
    shuffled_indices = np.random.permutation(num_samples)
    
    # Step 3: Reorder both arrays using the shuffled indices
    final_images_shuffled = final_images[shuffled_indices]
    final_labels_shuffled = final_labels[shuffled_indices]
    
    # Now, final_images_shuffled and final_labels_shuffled are randomly shuffled,
    # but still correctly paired.
    
    # To sample, say, 100 pairs randomly:
    num_to_sample = 5000
    sampled_images = final_images_shuffled[:num_to_sample]
    sampled_labels = final_labels_shuffled[:num_to_sample]


    return sampled_images, sampled_labels



'''
def get_data(augment=True, target_per_class=1000, test_size=0.2):
    root = './data/raw/'
    types = os.listdir(root)
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    
    for label, t in enumerate(types):
        current_type = glob.glob(root+t+"/*.hdf5")[0]
        with h5py.File(current_type, 'r') as f:
            imgs = np.array(f['images'][:])
            original_count = imgs.shape[0]
            
            if original_count > target_per_class/(1-test_size):
                selected_indices = np.random.choice(original_count, int(target_per_class/(1-test_size)), replace=False)
                imgs = imgs[selected_indices]
                original_count = imgs.shape[0]
                
            # Shuffle channels and get permutations for each image
            imgs = shuffle_channels_and_return_permutations(imgs)
            
            # Adjusted loop for clarity: Calculate permutations factor based on the augmentation process
            permutations_factor = len(imgs) // original_count
            
            # For each permutation, treat as its own class
            for perm_index in range(permutations_factor):
                # Extract images for current permutation
                permuted_imgs = imgs[perm_index * original_count : (perm_index + 1) * original_count]
                # Split into train and validation
                imgs_train, imgs_val = train_test_split(permuted_imgs, test_size=test_size, random_state=42)
                
                # Apply augmentation or downsampling to training data
                if augment and imgs_train.shape[0] < target_per_class:
                    imgs_train = dynamic_augment_images(imgs_train, target_per_class)
                elif imgs_train.shape[0] > target_per_class:
                    selected_indices = np.random.choice(imgs_train.shape[0], target_per_class, replace=False)
                    imgs_train = imgs_train[selected_indices]
                
                # Adjust validation data if necessary
                if imgs_val.shape[0] > int(target_per_class * test_size):
                    selected_indices = np.random.choice(imgs_val.shape[0], int(target_per_class * test_size), replace=False)
                    imgs_val = imgs_val[selected_indices]
                
                # Append augmented/trained images and their labels
                train_images.append(imgs_train)
                train_labels.append(np.array([(label*permutations_factor) + perm_index] * imgs_train.shape[0]))
                val_images.append(imgs_val)
                val_labels.append(np.array([(label*permutations_factor) + perm_index] * imgs_val.shape[0]))
    
    # Concatenate all class data for training and validation
    train_data = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_data = np.concatenate(val_images, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    
    return train_data, train_labels, val_data, val_labels
'''

if __name__ == '__main__':
    get_data()