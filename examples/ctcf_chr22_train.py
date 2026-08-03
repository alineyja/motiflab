# examples/ctcf_chr22_training.py

import os
import sys
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
from motiflab.bioinformatics.coordinates import GenomeCoordinates

#генератор случайных чисел
torch.manual_seed(42)
np.random.seed(42)


def resize_coordinates(coords: GenomeCoordinates, target_length: int = 200) -> GenomeCoordinates:
    center = (coords.start + coords.end) // 2
    half_length = target_length // 2
    
    # BED координаты должны быть строго положительными
    start = max(0, center - half_length)
    end = start + target_length
    
    return GenomeCoordinates(
        chromosome=coords.chromosome,
        start=start,
        end=end,
        strand=coords.strand
    )


def main():
    target_length = 200  
    kernel_size = 15     
    num_filters = 32    
    epochs = 10
    batch_size = 64
    
    raw_dir = root_dir / "data" / "raw"
    
    # 1. Загрузка и подготовка данных
    alphabet = DNAAlphabet.standard()
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    
    print("парсинг BED-файла CTCF")
    all_peaks = list(parse_bed(raw_dir / "ctcf_peaks.bed"))
    
    #только пики, лежащие на 22й хромосоме
    chr22_peaks = [p for p in all_peaks if p.chromosome == "chr22"]
    print(f"   Общее число пиков {len(all_peaks):,}")
    print(f"   пики на хромосоме 22 {len(chr22_peaks):,}")
    
    if not chr22_peaks:
        print("ошибка не найден файл")
        return

    #все пики к строго одинаковой длине 200 bp
    resized_peaks = [resize_coordinates(p, target_length) for p in chr22_peaks]

    #экстрактор ДНК и датасет
    print("инициализация экстрактора ДНК и датасета из FASTA")
    extractor = FastaExtractor(raw_dir / "chr22.fa", alphabet)
    dataset = GenomicDataset(resized_peaks, extractor, encoder)
    
    # Разбиваем на Train / Val (80 / 20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #Модель и обучение
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")
    
    model = MotifCNN(num_filters=num_filters, kernel_size=kernel_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("\n обучение модели...")
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

    #ИНТЕРПРЕТАЦИЯ
    print("\n извлечение мотивов из обученной модели")
    interpreter = MotifInterpreter(alphabet)
    
    #Ищем самый важный фильтр
    classifier_weights = model.classifier[1].weight.detach().cpu().numpy()[0]
    best_filter_idx = int(np.argmax(classifier_weights))
    print(f"Active filter: Index {best_filter_idx} (Linear weight: {classifier_weights[best_filter_idx]:.4f})")
    
    #Конвертируем веса в PPM
    filters_weights = model.get_filters()
    ppm = interpreter.weights_to_ppm(filters_weights)[best_filter_idx]
    
    #консенсус
    consensus_indices = np.argmax(ppm, axis=0)
    consensus_seq = "".join(alphabet.decode_symbol(idx) for idx in consensus_indices)
    
    #Information Content
    ic = interpreter.calculate_information_content(ppm)
    
    print("REAL BIOLOGY EXPERIMENT RESULT")
    print(f"CTCF Motif Consensus (JASPAR): CCACYAGGTGGCAG")
    print(f"Reconstructed by MotifCNN:     {consensus_seq}")
    print("Filter PPM Matrix (Probabilities):")
    print("Pos:   " + "  ".join(f"{i:2d}" for i in range(kernel_size)))
    for n_idx, char in enumerate(alphabet.symbols):
        row_str = "  ".join(f"{ppm[n_idx, col]*100:2.0f}" for col in range(kernel_size))
        print(f"  {char}:  {row_str}")
    print(f"Information Content sum (bits): {np.sum(ic):.2f} / {2.0 * kernel_size}")


if __name__ == "__main__":
    main()