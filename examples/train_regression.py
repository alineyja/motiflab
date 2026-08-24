# examples/train_regression.py

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

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
    return GenomeCoordinates(
        chromosome=coords.chromosome, 
        start=max(0, center - half_length), 
        end=max(0, center - half_length) + target_length, 
        strand=coords.strand,
        signal_value=coords.signal_value 
    )

def main():
    target_length = 200
    batch_size = 256
    epochs = 10
    
    raw_dir = root_dir / "data" / "raw"
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    alphabet = DNAAlphabet.standard()
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    
    print("парсинг пиков")
    all_peaks = list(parse_bed(raw_dir / "ctcf_peaks.bed"))
    
    train_peaks_raw = [p for p in all_peaks if p.chromosome not in ["chr21", "chr22"]]
    val_peaks_raw = [p for p in all_peaks if p.chromosome == "chr21"]
    test_peaks_raw = [p for p in all_peaks if p.chromosome == "chr22"]
    
    print(f"   Train peaks: {len(train_peaks_raw):,}")
    print(f"   Val peaks:   {len(val_peaks_raw):,}")
    print(f"   Test peaks:  {len(test_peaks_raw):,}")
    
    train_peaks = [resize_coordinates(p, target_length) for p in train_peaks_raw]
    val_peaks = [resize_coordinates(p, target_length) for p in val_peaks_raw]
    test_peaks = [resize_coordinates(p, target_length) for p in test_peaks_raw]

    print("\nсоздание датасетов")
    extractor = FastaExtractor(raw_dir / "hg38.fa", alphabet)
    
    # task="regression" !
    train_dataset = GenomicDataset(train_peaks, extractor, encoder, task="regression")
    val_dataset = GenomicDataset(val_peaks, extractor, encoder, task="regression")
    test_dataset = GenomicDataset(test_peaks, extractor, encoder, task="regression")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n обучение на {device}")
    
    model = HybridMotifTransformer(
        in_channels=4, num_filters=64, kernel_size=15, num_heads=4, num_layers=1
    ).to(device)
    

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    print("\n обучение регрессивной модели")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            preds = model(x_batch, return_attention=False)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch, return_attention=False)
                val_loss += criterion(preds, y_batch).item() * x_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.4f} | Val (chr21) MSE: {val_loss:.4f}")

    print("\n тест на 22 хромосоме")
    model.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch, return_attention=False)
            
            all_true.extend(y_batch.cpu().numpy().flatten())
            all_pred.extend(preds.cpu().numpy().flatten())
            
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    # Считаем корреляцию Пирсона и Спирмена
    pearson_r, _ = pearsonr(all_true, all_pred)
    spearman_rho, _ = spearmanr(all_true, all_pred)
    

    print("количественные результаты сходства")
    print(f"Пирсона (R):   {pearson_r:.4f}")
    print(f"Спирмена (Rho): {spearman_rho:.4f}")

    
    print("график рассеяния")
    plt.figure(figsize=(7, 7))
    plt.scatter(all_true, all_pred, alpha=0.2, color='#2b83ba', s=10)
    
    # Идеальная линия тренда y = x
    max_val = max(all_true.max(), all_pred.max())
    plt.plot([0, max_val], [0, max_val], color='#d7191c', linestyle='--', label='Ideal (y = x)')
    
    plt.title(f"количественное сходство связывания (chr22 Test)\n пирсое R = {pearson_r:.3f}, спирман Rho = {spearman_rho:.3f}")
    plt.xlabel("экспериментальный сигнал log(1 + SignalValue)")
    plt.ylabel("прогнозируемая сходимость от ии log(1 + SignalValue)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_path = results_dir / "affinity_scatter_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"диаграмма рассеивания сохранена в {plot_path}")
    
    extractor.close()

if __name__ == "__main__":
    main()