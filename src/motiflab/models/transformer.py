# src/motiflab/models/transformer.py

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class PositionalEncoding(nn.Module):
  
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        # Создаем матрицу позиционного кодирования
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        x = x + self.pe[:x.size(0), :]
        return x


class HybridMotifTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_filters: int = 64,   
        kernel_size: int = 15,  
        num_heads: int = 4,      
        num_layers: int = 1      
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        #Слой локального восприятия 
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding="same"
        )
        self.relu = nn.ReLU()
        #Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model=num_filters)
        
        #Блок трансформера
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=num_filters, 
            nhead=num_heads, 
            dim_feedforward=num_filters * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        
        #отдельный слой чисто для инференса/графиков
        self.raw_attention = nn.MultiheadAttention(embed_dim=num_filters, num_heads=num_heads, dropout=0.1)

        #Классификатор
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_filters, 1)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.relu(self.conv(x))
        features_t = features.permute(2, 0, 1) # (L, B, F)
        features_t = self.pos_encoder(features_t)
        if return_attention:
            attn_output, attn_weights = self.raw_attention(features_t, features_t, features_t)
            trans_out = features_t + attn_output
        else:
            trans_out = self.transformer_encoder(features_t)
            attn_weights = None
            
        out = trans_out.permute(1, 2, 0)
        
        #Глобальный пулинг
        out = self.global_pool(out) # (B, F, 1)
        out = torch.flatten(out, start_dim=1) # (B, F)
        
        logits = self.classifier(out) # (B, 1)
        
        if return_attention:
            return logits, attn_weights
        
        return logits

    def get_filters(self) -> np.ndarray:
        return self.conv.weight.detach().cpu().numpy()
