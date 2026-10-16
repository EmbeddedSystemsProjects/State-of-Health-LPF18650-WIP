"""
Classes for creating PyTorch datasets and dataloaders for battery data, 
including both time-domain and FFT-transformed datasets.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BatteryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_input_shape(self):
        return self.X.shape[1:]

class FFTBatteryDataset(Dataset):
    def __init__(self, X, Y):
    
        # 1. Computing the real FFT along the temporal dimension (axis 1)
        fft_complex = np.fft.rfft(X, axis=1)
        
        # 2. Extraction of the magnitude spectrum
        fft_magnitude = np.abs(fft_complex).astype(np.float32)
        
        # 3. Normalization of the FFT spectrum using Z-Score
        mean_fft = np.mean(fft_magnitude, axis=(0, 1), keepdims=True)
        std_fft = np.std(fft_magnitude, axis=(0, 1), keepdims=True) + 1e-8
        fft_scaled = (fft_magnitude - mean_fft) / std_fft
        
        self.X = torch.tensor(fft_scaled, dtype=torch.float32).permute(0, 2, 1)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1) if Y.ndim == 1 else torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_input_shape(self):
        return self.X.shape[1:]

# Function to create a DataLoader for the BatteryDataset  
def loader_creator(X, Y, batch_size=64, num_workers=1):

    dataset = BatteryDataset(X, Y)
    data_loader = DataLoader(dataset, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=num_workers,
                             persistent_workers=True)

    return data_loader
