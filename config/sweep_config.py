#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:30:08 2024

@author: administrator
"""

def get_sweep_config():

    sweep_config = {
        'method': 'bayes'
        }
    metric = {
        'name': 'ari',
        'goal': 'maximize'   
        }

    sweep_config['metric'] = metric
    parameters_dict = {
        'optimizer': {
            'value' : 'adam'
            },
        'dropout': {
            'values': [0.25,0.3]
            },
        }

    sweep_config['parameters'] = parameters_dict

    parameters_dict.update({
        'epochs': {
            'value': 100}
        })

    parameters_dict.update({
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0005,
            'max': 0.0006
            #'values' : [0.0005359748770038437]
            },
        'batch_size': {
            # integers between 8 and 728
            # with evenly-distributed logarithms 
            #'distribution': 'q_log_uniform_values',
            #'q' : 64,
           #'min' : 512,
            #'max' : 4096,
            
            'values' : [a for a in range(4,32,1)]
            },
        'z_dim' : {
            'values' : [a for a in range(7, 64, 1)]
            },
        'alpha' : {
            'values': [1]
            },
        'initial_temp' : {
            'values' : [2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,6,7,8,9,10]
            },
        'beta' : {
            'values' : [1.7,1.8,1.9,2]
            },
        'n' : {
            'values' : [2,3,4,5,6,7,8,9,10]
            }
    })
    return sweep_config