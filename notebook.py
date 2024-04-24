#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:26:46 2024

@author: mandyana
"""

from utils.plot_assist import Metrics
import pandas as pd

data=pd.read_csv("./data/interim/C8_Disentanglement_48_336_1.5_0.3_0.0003304127668098235_adam_7_2.151007000865832_2.1035097838378713_0.04740009623798516_9.717579544241963e-05_8_0.0.csv")
data = data.iloc[:,1:]

metric = Metrics(data)
print(metric.ari_score)