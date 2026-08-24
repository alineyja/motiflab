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
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding="same" 
        )
        
        self.relu1 = nn.ReLU()
         # Локальный пулинг
        self.local_pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters * 2, 
            kernel_size=7,
            padding="same"
        )
        self.relu2 = nn.ReLU()

        #Global Pooling
        #самый сильный сигнал из всей последовательности
        self.global_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=num_filters * 2, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.local_pool(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Global Pool and Flatten
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        
        logits = self.classifier(x)
        return logits

    def get_filters(self) -> np.ndarray: 
        #CPU и конвертация в NumPy
        return self.conv1.weight.detach().cpu().numpy()