"""
Temporal Convolutional Network (TCN) model for time series data.
"""
import torch
import torch.nn as nn


# Temporal Block: a single block of the TCN architecture
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))

        if out.shape[-1] != residual.shape[-1]:
            out = out[:, :, :residual.shape[-1]]

        return self.relu_out(out + residual)


# Temporal Convolutional Network (TCN) model
class TCN(nn.Module):
    def __init__(self, in_channels=2, num_channels=[16, 32, 64], kernel_size=3, dense_units=32, dropout=0.1):
        super(TCN, self).__init__()

        layers = []
        current_in = in_channels

        for i, out_ch in enumerate(num_channels):
            dilation_size = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=current_in,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    dropout=dropout
                )
            )
            current_in = out_ch

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(num_channels[-1], dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1)
        )

    def forward(self, x):
        features = self.network(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        soh_pred = self.regressor(flattened)
        return soh_pred
