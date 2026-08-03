# examples/sandbox_experiment.py

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

#src в пути поиска
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.bed import parse_bed
from motiflab.bioinformatics.fasta import FastaExtractor
from motiflab.preprocessing.encoder import OneHotEncoder
from motiflab.datasets.genomic_dataset import GenomicDataset
from motiflab.models.cnn import MotifCNN
from motiflab.interpretation.motif import MotifInterpreter

#случайность для воспроизводимости
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def generate_sandbox_data(output_dir: Path, motif: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fa_path = output_dir / "mock_genome.fa"
    bed_path = output_dir / "mock_peaks.bed"
    
    num_peaks = 1000
    peak_length = 100
    spacer_length = 50
    
    print(f"Генерация генома-пустышки с имплантацией мотива {motif}")
    
    #геном из блоков: [spacer (50bp) + peak (100bp)]
    full_genome_seq = []
    bed_lines = []
    
    current_pos = 0
    for i in range(num_peaks):
        #случайный спайсер(фон)
        spacer = "".join(random.choices(["A", "C", "G", "T"], k=spacer_length))
        full_genome_seq.append(spacer)
        current_pos += spacer_length
        
        #пик
        peak_seq = random.choices(["A", "C", "G", "T"], k=peak_length)
        
        #мотив в случайное место пика
        implant_pos = random.randint(0, peak_length - len(motif))
        peak_seq[implant_pos : implant_pos + len(motif)] = list(motif)
        
        full_genome_seq.append("".join(peak_seq))
        
        #координаты в BED (цепь случайная)
        strand = random.choice(["+", "-"])
        bed_lines.append(f"chr_mock\t{current_pos}\t{current_pos + peak_length}\tpeak_{i}\t0\t{strand}\n")
        
        current_pos += peak_length

    # Записываем FASTA
    with open(fa_path, "w") as f:
        f.write(">chr_mock\n")
        f.write("".join(full_genome_seq) + "\n")
        
    # Записываем BED
    with open(bed_path, "w") as f:
        f.writelines(bed_lines)
        
    print(f"Created: {fa_path}")
    print(f"Created: {bed_path}")


def main():
    secret_motif = "ATGCGATC" 
    kernel_size = 12          
    num_filters = 16
    epochs = 8
    batch_size = 64
    
    data_dir = root_dir / "data" / "sandbox"
    generate_sandbox_data(data_dir, secret_motif)
    
    #Инициализация конвейера данных
    alphabet = DNAAlphabet.standard()
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    peaks = list(parse_bed(data_dir / "mock_peaks.bed"))
    
    #Открываем FASTA и создаем DataLoader
    extractor = FastaExtractor(data_dir / "mock_genome.fa", alphabet)
    dataset = GenomicDataset(peaks, extractor, encoder)
    
    # Делим на Train/Val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #Инициализация модели, лосса и оптимизатора
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")
    
    model = MotifCNN(num_filters=num_filters, kernel_size=kernel_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    #Цикл обучения
    print("\n обучение MotifCNN на синтетических данных")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        #Валидация
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                
        val_acc = val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2%}")
    extractor.close()

    #ИНТЕРПРЕТАЦИЯ!!
    print("\n интерпретация модели для извлечения мотива")
    interpreter = MotifInterpreter(alphabet)
    
    #веса классификатора Linear слоя, чтобы найти самый важный фильтр
    classifier_weights = model.classifier[1].weight.detach().cpu().numpy()[0]
    best_filter_idx = int(np.argmax(classifier_weights))
    print(f"Best filter index: {best_filter_idx} (Linear weight: {classifier_weights[best_filter_idx]:.4f})")
    
    #PPM-матрица этого фильтра напрямую из весов свертки
    filters_weights = model.get_filters()  # (num_filters, 4, kernel_size)
    best_filter_weights = filters_weights[best_filter_idx]  # (4, kernel_size)
    
    ppm = interpreter.weights_to_ppm(filters_weights)[best_filter_idx]
    
    #консенсусная строка по PPM
    consensus_indices = np.argmax(ppm, axis=0)
    consensus_seq = "".join(alphabet.decode_symbol(idx) for idx in consensus_indices)
    
    #Information Content для каждой позиции
    ic = interpreter.calculate_information_content(ppm)
    

    print("MOCK EXPERIMENT RESULT")
    print(f"Original implanted motif:  {secret_motif}")
    print(f"Reverse complement of it:  {alphabet.complement(secret_motif)[::-1]}")
    print(f"Reconstructed by network:  {consensus_seq}")
    print("Position-by-position Information Content (bits, max 2.0):")
    for pos, bits in enumerate(ic):
        print(f"  Pos {pos}: {bits:.2f} bits  (Dominant letter: {consensus_seq[pos]})") 


if __name__ == "__main__":
    main()