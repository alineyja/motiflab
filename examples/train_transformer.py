# examples/train_transformer.py

import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Пути
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.bed import parse_bed
from motiflab.bioinformatics.fasta import FastaExtractor
from motiflab.preprocessing.encoder import OneHotEncoder
from motiflab.datasets.genomic_dataset import GenomicDataset
from motiflab.models.transformer import HybridMotifTransformer
from motiflab.bioinformatics.coordinates import GenomeCoordinates

torch.manual_seed(42)
np.random.seed(42)

def resize_coordinates(coords: GenomeCoordinates, target_length: int = 200) -> GenomeCoordinates:
    center = (coords.start + coords.end) // 2
    half_length = target_length // 2
    start = max(0, center - half_length)
    end = start + target_length
    return GenomeCoordinates(chromosome=coords.chromosome, start=start, end=end, strand=coords.strand)

def main():
    target_length = 200
    num_filters = 64  
    num_heads = 4     
    num_layers = 1   
    epochs = 12       
    batch_size = 64
    
    raw_dir = root_dir / "data" / "raw"
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    #Данные
    alphabet = DNAAlphabet.standard()
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    
    print("1. Parsing peaks...")
    all_peaks = list(parse_bed(raw_dir / "ctcf_peaks.bed"))
    train_peaks_raw = [p for p in all_peaks if p.chromosome == "chr21"]
    test_peaks_raw = [p for p in all_peaks if p.chromosome == "chr22"]
    
    print(f"   Train/Val peaks (chr21): {len(train_peaks_raw)}")
    print(f"   Test peaks (chr22):      {len(test_peaks_raw)}")
    
    train_peaks = [resize_coordinates(p, target_length) for p in train_peaks_raw]
    test_peaks = [resize_coordinates(p, target_length) for p in test_peaks_raw]

    print("\n2. Building Datasets...")
    train_extractor = FastaExtractor(raw_dir / "chr21.fa", alphabet)
    full_train_dataset = GenomicDataset(train_peaks, train_extractor, encoder)
    
    test_extractor = FastaExtractor(raw_dir / "chr22.fa", alphabet)
    test_dataset = GenomicDataset(test_peaks, test_extractor, encoder)
    
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #Модель
    device = torch.device("cpu")
    print(f"\nTraining HybridMotifTransformer on: {device}")
    
    model = HybridMotifTransformer(
        in_channels=4,
        num_filters=num_filters,
        kernel_size=15,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    #Обучение
    print("\n--- Training HybridMotifTransformer ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            # Внимание: возвращает только logits
            logits = model(x_batch, return_attention=False)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch, return_attention=False)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                
        val_acc = val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2%}")

    #Тестирование
    print("\nтест на 22 хромосоме")
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch, return_attention=False)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            test_correct += (preds == y_batch).sum().item()
            
    test_acc = test_correct / len(test_loader.dataset)
    print(f"последний тест acc {test_acc:.2%}")

    #ИЗВЛЕЧЕНИЕ МАТРИЦЫ ВНИМАНИЯ (Attention Map)
    print("\n извлечение Attention Map")
    test_sequences = [test_extractor.extract(p) for p in test_peaks]
    
    max_p = 0.0
    strongest_seq = None
    with torch.no_grad():
        for seq in test_sequences:
            matrix = encoder.encode(seq)
            tensor_x = torch.from_numpy(matrix).unsqueeze(0).to(device)
            p = torch.sigmoid(model(tensor_x, return_attention=False)).item()
            if p > max_p:
                max_p = p
                strongest_seq = seq

    print(f"коорд. сильнейшего пика {strongest_seq.coords} (Conf: {max_p:.2%})")

    matrix = encoder.encode(strongest_seq)
    tensor_x = torch.from_numpy(matrix).unsqueeze(0).to(device)
    _, attn_weights = model(tensor_x, return_attention=True)
    
    # attn_weights форма: (Batch=1, Length, Length) -> убираем батч
    attn_matrix = attn_weights[0].detach().cpu().numpy()
    
    #Визуализация
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, cmap="viridis", cbar=True)
    plt.title(f"Transformer Attention Matrix (Seq: {strongest_seq.coords})")
    plt.xlabel("Key Position (Source motif)")
    plt.ylabel("Query Position (Target motif)")
    
    plot_path = results_dir / "transformer_attention_map.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"   [Done] Attention Map saved to: {plot_path}")
    print("=============================================")

    train_extractor.close()
    test_extractor.close()

if __name__ == "__main__":
    main()