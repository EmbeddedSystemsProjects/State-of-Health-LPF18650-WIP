"""
1D Convolutional Neural Network (CNN) model for time series data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_1D(nn.Module):
    def __init__(self, input_shape, output_dim=1, **kwargs):
        super(CNN_1D, self).__init__()
        assert len(input_shape) == 2, "Input shape must be (channels, length)"
 
        self.input_shape = input_shape
        self.output_dim = output_dim
 
        self.filters = kwargs.get('filters', [4, 8, 16])
        self.kernel_size = kwargs.get('kernel_size', 3)
        self.hidden_neurons = kwargs.get('hidden_neurons', 64)
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)

        layers = []
        in_channels = input_shape[0]
        for i, f in enumerate(self.filters):
            layers.append(nn.Conv1d(in_channels, f, kernel_size=self.kernel_size, padding=(self.kernel_size-1) // 2))
            layers.append(nn.ReLU())
            if i < len(self.filters) - 1:
                layers.append(nn.AvgPool1d(kernel_size=2))
            in_channels = f
 
        self.conv = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Layer Fully Connected + Dropout
        self.fc1 = nn.Linear(self.filters[-1], self.hidden_neurons)
        self.dropout = nn.Dropout(p=self.dropout_rate) 
        self.fc2 = nn.Linear(self.hidden_neurons, self.output_dim)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)        
        x = self.fc2(x)
        return x
