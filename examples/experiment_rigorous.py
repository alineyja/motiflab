import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))
from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.bed import parse_bed
from motiflab.bioinformatics.fasta import FastaExtractor
from motiflab.preprocessing.encoder import OneHotEncoder
from motiflab.datasets.genomic_dataset import GenomicDataset
from motiflab.models.cnn import MotifCNN
from motiflab.interpretation.motif import MotifInterpreter
from motiflab.interpretation.mutagenesis import MutagenesisAnalyzer
from motiflab.visualization.logo import plot_motif_logo, plot_mutagenesis_heatmap
from motiflab.bioinformatics.coordinates import GenomeCoordinates

torch.manual_seed(42)
np.random.seed(42)

def resize_coordinates(coords: GenomeCoordinates, target_length: int = 200) -> GenomeCoordinates:
    center = (coords.start + coords.end) // 2
    half_length = target_length // 2
    start = max(0, center - half_length)
    end = start + target_length
    return GenomeCoordinates(
        chromosome=coords.chromosome,
        start=start,
        end=end,
        strand=coords.strand
    )



def main():
    target_length = 200  # Длина окон ДНК
    kernel_size = 15     # Длина мотива
    num_filters = 32
    epochs = 10
    batch_size = 64
    
    raw_dir = root_dir / "data" / "raw"
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    #1 инициализация ядра
    alphabet = DNAAlphabet.standard()
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    
    print("1. Loading and filtering peaks...")
    all_peaks = list(parse_bed(raw_dir / "ctcf_peaks.bed"))
    train_peaks_raw = [p for p in all_peaks if p.chromosome == "chr21"]
    test_peaks_raw = [p for p in all_peaks if p.chromosome == "chr22"]
    
    print(f"пики на 21 хромосоме {len(train_peaks_raw):,}")
    print(f"пики на 22 хромосоме {len(test_peaks_raw):,}")
    train_peaks = [resize_coordinates(p, target_length) for p in train_peaks_raw]
    test_peaks = [resize_coordinates(p, target_length) for p in test_peaks_raw]

    #2 инициализация экстракторов и датасетов
    print("\n2. Extracting DNA sequences and building PyTorch Datasets...")
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
    device = torch.device("cpu")
    print(f"\nTraining on: {device}")
    
    model = MotifCNN(num_filters=num_filters, kernel_size=kernel_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    #цикл обучения
    print("\n обучение модели на 21 хромосоме")
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
        
        # валидация
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                
        val_acc = val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch}/{epochs}. Train Loss {train_loss:.4f}. Val Accuracy {val_acc:.2%}")

    #слепое тестирование на 22 хромосоме
    print("\n слепое тестирование на 22 хромосоме")
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            test_correct += (preds == y_batch).sum().item()
            
    test_acc = test_correct / len(test_loader.dataset)
    print(f"последний тест {test_acc:.2%}")

    
    # sequence logo 
    print("\n интерпретация модели и построение логотипа последовательности")
    interpreter = MotifInterpreter(alphabet)
    
    classifier_weights = model.classifier[1].weight.detach().cpu().numpy()[0]
    best_filter_idx = int(np.argmax(classifier_weights))
    print(f"индекс активного фильтра {best_filter_idx}")
    
    # Сканируем независимый тест-сет на chr22 для поиска мест посадки
    test_sequences = [test_extractor.extract(p) for p in test_peaks]
    
    print("сканирование тестового набора для поиска мотивов")
    seqlets = interpreter.extract_activating_seqlets(
        model=model,
        sequences=test_sequences,
        filter_idx=best_filter_idx,
        threshold=0.75
    )
    print(f"найдено {len(seqlets)} сильных мотивов в геноме")
    
    if len(seqlets) > 0:
        sharp_ppm = interpreter.seqlets_to_ppm(seqlets)
        logo_path = results_dir / "ctcf_motif_logo.png"
        plot_motif_logo(sharp_ppm, alphabet.symbols, logo_path, title="Learned CTCF Motif")
        print(f"   Сохранено в {logo_path}")
    # объяснение модели с помощью In Silico Mutagenesis
    print("\n объяснение модели с помощью In Silico Mutagenesis")
    # Найдем пик на chr22, в котором максимальный скор
    max_p = 0.0
    strongest_seq = None
    
    with torch.no_grad():
        for seq in test_sequences:
            matrix = encoder.encode(seq)
            tensor_x = torch.from_numpy(matrix).unsqueeze(0).to(device)
            p = torch.sigmoid(model(tensor_x)).item()
            if p > max_p:
                max_p = p
                strongest_seq = seq

    print(f"сильнейший пик {strongest_seq.coords}")
    print(f"уверенность {max_p:.4%}")

    # запускаем мутагенез
    analyzer = MutagenesisAnalyzer(model, encoder)
    delta_p_matrix, baseline_p = analyzer.analyze(strongest_seq)
    
    # обрезаем тепловую карту до центральных 40 нуклеотидов, чтобы буквы были большими и читаемыми (мотив сидит в самом центре пика)
    center = target_length // 2
    crop_start = center - 20
    crop_end = center + 20
    
    cropped_matrix = delta_p_matrix[:, crop_start:crop_end]
    cropped_seq = strongest_seq.sequence[crop_start:crop_end]
    
    heatmap_path = results_dir / "ctcf_mutagenesis_heatmap.png"
    plot_mutagenesis_heatmap(
        cropped_matrix, 
        cropped_seq, 
        alphabet.symbols, 
        heatmap_path,
        title="CTCF Mutational Landscape"
    )
    print(f"   хитмап сохранен в  {heatmap_path}")
    print("эксперимент завершен успешно")

    # Закрываем экстракторы
    train_extractor.close()
    test_extractor.close()

if __name__ == "__main__":
    main()