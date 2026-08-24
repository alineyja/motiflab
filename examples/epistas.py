# examples/epistasis_analysis.py

import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.bed import parse_bed
from motiflab.bioinformatics.fasta import FastaExtractor
from motiflab.preprocessing.encoder import OneHotEncoder
from motiflab.models.cnn import MotifCNN
from motiflab.bioinformatics.coordinates import GenomeCoordinates
from motiflab.datasets.genomic_dataset import GenomicDataset


def resize_coordinates(coords: GenomeCoordinates, target_length: int = 200) -> GenomeCoordinates:
    center = (coords.start + coords.end) // 2
    half_length = target_length // 2
    start = max(0, center - half_length)
    end = start + target_length
    return GenomeCoordinates(chromosome=coords.chromosome, start=start, end=end, strand=coords.strand)


def main():
    target_length = 200
    kernel_size = 15
    device = torch.device("cpu")
    
    raw_dir = root_dir / "data" / "raw"
    results_dir = root_dir / "results"
    
    alphabet = DNAAlphabet.standard()
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    
    print("1. Training Model on chr21...")
    all_peaks = list(parse_bed(raw_dir / "ctcf_peaks.bed"))
    chr21_peaks = [resize_coordinates(p, target_length) for p in all_peaks if p.chromosome == "chr21"]
    
    with FastaExtractor(raw_dir / "chr21.fa", alphabet) as extractor:
        train_dataset = GenomicDataset(chr21_peaks, extractor, encoder)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        model = MotifCNN(num_filters=32, kernel_size=kernel_size).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        
        for epoch in range(5):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                
    model.eval()
    print("2. Finding the strongest peak on chr22...")
    chr22_peaks = [resize_coordinates(p, target_length) for p in all_peaks if p.chromosome == "chr22"]
    
    max_p = 0.0
    strongest_seq = None
    
    with FastaExtractor(raw_dir / "chr22.fa", alphabet) as extractor:
        test_sequences = [extractor.extract(p) for p in chr22_peaks[:500]]
        
        with torch.no_grad():
            for seq in test_sequences:
                tensor_x = torch.from_numpy(encoder.encode(seq)).unsqueeze(0).to(device)
                p = torch.sigmoid(model(tensor_x)).item()
                if p > max_p:
                    max_p = p
                    strongest_seq = seq

    print(f"Strongest Peak Confidence: {max_p:.4%}")
    
    # 3. Подготовка к попарному мутагенезу (Pairwise Mutagenesis)
    print("3. Running Pairwise Epistasis Analysis...")
    
    # Берем центральное окно, где ожидаемо сидит мотив
    center = target_length // 2
    window_start = center - (kernel_size // 2)
    window_end = window_start + kernel_size
    
    wt_matrix = encoder.encode(strongest_seq) # (4, 200)
    wt_tensor = torch.from_numpy(wt_matrix).unsqueeze(0).to(device)
    with torch.no_grad():
        wt_p = torch.sigmoid(model(wt_tensor)).item()

    # Матрица эпистаза 15x15 (сохраняем максимальный абсолютный эффект E_ij между позициями i и j)
    epistasis_matrix = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    # Для ускорения считаем все одиночные мутации заранее
    single_mut_probs = np.zeros((kernel_size, 4), dtype=np.float32)
    for pos_idx in range(kernel_size):
        actual_pos = window_start + pos_idx
        wt_base_idx = alphabet.encode_symbol(strongest_seq.sequence[actual_pos])
        for mut_base in range(4):
            if mut_base == wt_base_idx:
                single_mut_probs[pos_idx, mut_base] = wt_p
                continue
            
            mut_matrix = wt_matrix.copy()
            mut_matrix[:, actual_pos] = 0.0
            mut_matrix[mut_base, actual_pos] = 1.0
            
            with torch.no_grad():
                t_x = torch.from_numpy(mut_matrix).unsqueeze(0).to(device)
                single_mut_probs[pos_idx, mut_base] = torch.sigmoid(model(t_x)).item()

    # Считаем двойные мутации
    for i in range(kernel_size):
        actual_i = window_start + i
        wt_base_i = alphabet.encode_symbol(strongest_seq.sequence[actual_i])
        
        for j in range(i + 1, kernel_size):
            actual_j = window_start + j
            wt_base_j = alphabet.encode_symbol(strongest_seq.sequence[actual_j])
            
            max_abs_e = 0.0
            
            # Перебираем все возможные комбинации мутаций в этих двух позициях
            for mut_i in range(4):
                if mut_i == wt_base_i: continue
                for mut_j in range(4):
                    if mut_j == wt_base_j: continue
                    
                    # Дельта одиночных мутаций
                    dp_i = single_mut_probs[i, mut_i] - wt_p
                    dp_j = single_mut_probs[j, mut_j] - wt_p
                    
                    # Двойная мутация
                    mut_matrix = wt_matrix.copy()
                    mut_matrix[:, actual_i] = 0.0
                    mut_matrix[mut_i, actual_i] = 1.0
                    mut_matrix[:, actual_j] = 0.0
                    mut_matrix[mut_j, actual_j] = 1.0
                    
                    with torch.no_grad():
                        t_x = torch.from_numpy(mut_matrix).unsqueeze(0).to(device)
                        p_ij = torch.sigmoid(model(t_x)).item()
                        
                    dp_ij = p_ij - wt_p
                    
                    # Вычисляем Эпистаз (Неаддитивность)
                    e_ij = dp_ij - (dp_i + dp_j)
                    
                    if abs(e_ij) > abs(max_abs_e):
                        max_abs_e = e_ij
                        
            # Матрица симметрична
            epistasis_matrix[i, j] = max_abs_e
            epistasis_matrix[j, i] = max_abs_e

    # 4. Визуализация Эпистаза
    print("4. Plotting Epistasis Matrix...")
    motif_seq = strongest_seq.sequence[window_start:window_end]
    
    plt.figure(figsize=(8, 6))
    
    # Чтобы диагональ (сравнение позиции с самой собой) не мешала, ставим туда NaN
    np.fill_diagonal(epistasis_matrix, np.nan)
    
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap.set_bad(color='#aaaaaa')
    
    sns.heatmap(
        epistasis_matrix, 
        cmap=cmap, 
        center=0,
        annot=True,
        fmt=".3f",
        xticklabels=list(motif_seq),
        yticklabels=list(motif_seq)
    )
    plt.title(f"Pairwise Epistasis Matrix ($E_{{ij}}$) for Central 15bp")
    plt.xlabel("Position")
    plt.ylabel("Position")
    
    plot_path = results_dir / "epistasis_matrix.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("\n" + "=" * 50)
    print("EPISTASIS EXPERIMENT RESULTS")
    print("=" * 50)
    max_e = np.nanmax(np.abs(epistasis_matrix))
    print(f"Max Absolute Epistatic Interaction: {max_e:.5f}")
    if max_e > 0.05:
        print("CONCLUSION: Strong non-linear interactions detected! CNN is NOT a simple PWM.")
    else:
        print("CONCLUSION: Negligible epistasis. The model acts mostly linearly (like a PWM).")
    print(f"Plot saved to: {plot_path}")
    print("=============================================")

if __name__ == "__main__":
    main()