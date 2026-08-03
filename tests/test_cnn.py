# tests/models/test_cnn.py

import torch
from motiflab.models.cnn import MotifCNN

def test_motif_cnn_forward():
    batch_size = 16
    in_channels = 4
    seq_length = 200
    num_filters = 32
    kernel_size = 15

    model = MotifCNN(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=kernel_size
    )

    # фиктивный тензор
    dummy_input = torch.randn(batch_size, in_channels, seq_length)
    
    # Прогон через сеть
    logits = model(dummy_input)
    assert logits.shape == (batch_size, 1)
    assert logits.dtype == torch.float32

def test_motif_cnn_get_filters():
    num_filters = 16
    kernel_size = 9
    model = MotifCNN(num_filters=num_filters, kernel_size=kernel_size)
    
    filters = model.get_filters()
    
    assert filters.shape == (num_filters, 4, kernel_size)