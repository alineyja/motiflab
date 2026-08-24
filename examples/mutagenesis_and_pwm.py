import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from motiflab.datasets.genomic_dataset import GenomicDataset

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))
from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.bed import parse_bed
from motiflab.bioinformatics.fasta import FastaExtractor
from motiflab.preprocessing.encoder import OneHotEncoder
from motiflab.models.cnn import MotifCNN
from motiflab.interpretation.motif import MotifInterpreter
from motiflab.interpretation.mutagenesis import MutagenesisAnalyzer
from motiflab.bioinformatics.coordinates import GenomeCoordinates

def main():
    target_length = 200
    kernel_size = 15
    num_peaks_to_test = 100
    
    raw_dir = root_dir / "data" / "raw"
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    #Загрузка компонентов
    alphabet = DNAAlphabet.standard()
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    print("загрузка и фильтрация пиков")
    device = torch.device("cpu")
    all_peaks = list(parse_bed(raw_dir / "ctcf_peaks.bed"))
    chr21_peaks = [resize_coordinates(p, target_length) for p in all_peaks if p.chromosome == "chr21"]
    chr22_peaks = [resize_coordinates(p, target_length) for p in all_peaks if p.chromosome == "chr22"]
    with FastaExtractor(raw_dir / "chr21.fa", alphabet) as extractor:
        train_dataset = GenomicDataset(chr21_peaks, extractor, encoder)
        train_Loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        model = MotifCNN(num_filters=32, kernel_size=15).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        print("обучение модели на пиках 21 хромосомы")
        for epoch in range(5):
            model.train()
            for x_batch, y_batch in train_Loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(x_batch)       
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
    print("обучение завершено, сохранение модели")


    interpreter = MotifInterpreter(alphabet)
    classifier_weights = model.classifier[1].weight.detach().cpu().numpy()
    best_filter_index = int(np.argmax(np.abs(classifier_weights)))
    filter_weights = model.get_filters()
    ppm = interpreter.weights_to_ppm(filter_weights[best_filter_index]) #4,15
    test_peaks = chr22_peaks[:500]
    test_sequences = []
    with FastaExtractor(raw_dir / "chr22.fa", alphabet) as extractor:
        for p in test_peaks:
            test_sequences.append(extractor.extract(p))
    scored_seqs = []
    analyzer = MutagenesisAnalyzer(model, encoder)
    print("анализ мутагенеза на 100 пиках 22 хромосомы")
    with torch.no_grad():
        for seq in test_sequences:
            tensor_x = torch.from_numpy(encoder.encode(seq)).unsqueeze(0).to(device)
            p = torch.sigmoid(model(tensor_x)).item()
            scored_seqs.append((seq, p))

    scored_seqs.sort(key=lambda x: x[1], reverse=True) 
    top_seqs = [seq for seq, score in scored_seqs[:num_peaks_to_test]]
    correlations = []

    print(f"прогон in silico мутагенеза на топ {num_peaks_to_test} пиках")
    for seq_idx, seq in enumerate(top_seqs):
        #deltaP матрица
        delta_p_matrix, baseline_p = analyzer.analyze(seq)
        best_window_idx = 0
        max_impact = -1e9
        
        for start_pos in range(200 - kernel_size + 1):
            #сумма абсолютных изменений в окне
            window = delta_p_matrix[:, start_pos : start_pos + kernel_size]
            impact = np.nansum(np.abs(window))
            if impact > max_impact:
                max_impact = impact
                best_window_idx = start_pos
                
        # матрица мутагенеза для этого окна
        actual_delta_p = delta_p_matrix[:, best_window_idx : best_window_idx + kernel_size] # (4, 15)
        
        #матрица deltaPPM для окна ДНК
        # DeltaPPM = PPM[mutated_base] - PPM[wild_type_base]
        theoretical_delta_ppm = np.zeros((4, kernel_size), dtype=np.float32)
        for pos in range(kernel_size):
            wt_base_char = seq.sequence[best_window_idx + pos]
            wt_base_idx = alphabet.encode_symbol(wt_base_char)
            
            for mut_base_idx in range(4):
                if mut_base_idx == wt_base_idx:
                    theoretical_delta_ppm[mut_base_idx, pos] = np.nan
                else:
                    theoretical_delta_ppm[mut_base_idx, pos] = ppm[mut_base_idx, pos] - ppm[wt_base_idx, pos]
                    
        #выпрямление матрицы в векторы, отбрасывая NaN
        mask = ~np.isnan(actual_delta_p)
        vector_actual = actual_delta_p[mask]
        vector_theoretical = theoretical_delta_ppm[mask]
        
        #корреляция Пирсона
        r = np.corrcoef(vector_actual, vector_theoretical)[0, 1]
        if not np.isnan(r):
            correlations.append(r)


    #   Строим график распределения корреляций

    plt.figure(figsize=(8, 5))
    sns.histplot(correlations, kde=True, color='purple', bins=15)
    plt.axvline(np.mean(correlations), color='red', linestyle='--', label=f'Среднее R: {np.mean(correlations):.3f}')
    plt.title("Распределение корреляций Пирсона между матрицами мутагенеза и PPM")
    plt.xlabel("Корреляция Пирсона (R)")
    plt.ylabel("Количество пиков")
    plt.legend()
    
    plot_path = results_dir / "ppm_vs_mutagenesis_correlation.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МАТРИЦ МУТАГЕНЕЗА И PPM")
    print(f"Средняя корреляция {np.mean(correlations):.4f}")
    print(f"Минимальная корреляция       {np.min(correlations):.4f}")
    print(f"Максимальная корреляция       {np.max(correlations):.4f}")
    print(f"График распределения сохранен в   {plot_path}")
    
from motiflab.datasets.genomic_dataset import GenomicDataset
import torch.nn as nn

def resize_coordinates(coords: GenomeCoordinates, target_length: int = 200) -> GenomeCoordinates:
    center = (coords.start + coords.end) // 2
    half_length = target_length // 2
    start = max(0, center - half_length)
    end = start + target_length
    return GenomeCoordinates(chromosome=coords.chromosome, start=start, end=end, strand=coords.strand)


if __name__ == "__main__":
    main()