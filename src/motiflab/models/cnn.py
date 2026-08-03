# src/motiflab/models/cnn.py

import torch
import torch.nn as nn
import numpy as np


class MotifCNN(nn.Module):
    def __init__(
        self, 
        in_channels: int = 4,   
        num_filters: int = 64,  # Количество потенциальных мотивов для поиска
        kernel_size: int = 15   # Длина биологического мотива
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        # Слой который будет искать мотивы 
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding="same" 
        )
        
        self.relu = nn.ReLU()
        
        # Global Max Pooling
        #  максимальный сигнал
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_filters, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cвертка (B, 4, L) - (B, num_filters, L)
        x = self.conv(x)
        x = self.relu(x)
        
        # пулинг (B, num_filters, L) - (B, num_filters, 1)
        x = self.pool(x)
        
        # выпрямлкение: (B, num_filters, 1) - (B, num_filters)
        x = torch.flatten(x, start_dim=1)
        
        # классификация: (B, num_filters) - (B, 1)
        logits = self.classifier(x)
        
        return logits

    def get_filters(self) -> np.ndarray: 
        #CPU и конвертация в NumPy
        weights = self.conv.weight.detach().cpu().numpy()
        return weights