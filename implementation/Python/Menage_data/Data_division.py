"""
Script to divide the dataset into training and validation sets, and save them into a new compressed npz file.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "Dataset_mv/train_val_soh_processed_full.npz"
OUTPUT_FILE = "Dataset_mv/train_val_soh.npz"

print("Loading dataset from", INPUT_FILE)
data = np.load(INPUT_FILE)

X_all = data['X_train']
Y_all = data['Y_train']

# Dvision of the dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X_all, Y_all, 
    test_size=0.20,      # 20% left for validation
    random_state=42,     
    shuffle=True         
)
print("Dataset divided into training and validation sets.")

# saving the new dataset into a compressed npz file
np.savez_compressed(
    OUTPUT_FILE,
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    v_mean=data['v_mean'],
    v_std=data['v_std'],
    i_mean=data['i_mean'],
    i_std=data['i_std'],
)  

print("New dataset saved to", OUTPUT_FILE)
