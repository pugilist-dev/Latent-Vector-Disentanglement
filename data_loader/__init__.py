#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:47:34 2023

@author: administrator
"""

from .contrastiveClustering import ContrastiveClustering, CafCcDisentanglement, C8Disentanglement

datasets = {
    'clustering': ContrastiveClustering,
    'caf_cc_disentanglement': CafCcDisentanglement,
    'c8_disentanglement': C8Disentanglement
    }

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)