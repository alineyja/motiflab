# examples/train_big_data.py

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.fasta import FastaExtractor
from motiflab.preprocessing.encoder import OneHotEncoder
from motiflab.datasets.genomic_dataset import GenomicDataset
from motiflab.models.transformer import HybridMotifTransformer
from motiflab.bioinformatics.coordinates import GenomeCoordinates
from motiflab.database.db import GenomicDatabase

torch.manual_seed(42)
np.random.seed(42)

def resize_coordinates(coords: GenomeCoordinates, target_length: int = 200) -> GenomeCoordinates:
    center = (coords.start + coords.end) // 2
    half_length = target_length // 2
    return GenomeCoordinates(
        chromosome=coords.chromosome, 
        start=max(0, center - half_length), 
        end=max(0, center - half_length) + target_length, 
        strand=coords.strand
    )

def main():
    target_length = 200
    batch_size = 256
    epochs = 10
    
    raw_dir = root_dir / "data" / "raw"
    db_path = root_dir / "data" / "motiflab.db"
    
    if not db_path.exists():
        print(f"база данных не найдена {db_path}")
        print("запусти python scripts/import_bed_to_sql.py .")
        return

    alphabet = DNAAlphabet.standard()
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    
    print("извлечение пиков из бд")
    db = GenomicDatabase(db_path)
    val_peaks_raw = db.get_peaks(chromosome="chr21", tf_name="CTCF")
    test_peaks_raw = db.get_peaks(chromosome="chr22", tf_name="CTCF")
    all_ctcf_peaks = db.get_peaks(tf_name="CTCF")
    train_peaks_raw = [p for p in all_ctcf_peaks if p.chromosome not in ["chr21", "chr22"]]
    
    print(f"Train peaks (Genome-wide): {len(train_peaks_raw):,}")
    print(f"Val peaks   (chr21):       {len(val_peaks_raw):,}")
    print(f"Test peaks  (chr22):       {len(test_peaks_raw):,}")

    train_peaks = [resize_coordinates(p, target_length) for p in train_peaks_raw]
    val_peaks = [resize_coordinates(p, target_length) for p in val_peaks_raw]
    test_peaks = [resize_coordinates(p, target_length) for p in test_peaks_raw]

    print("\nЗагрузка генома человека в индекс")
    extractor = FastaExtractor(raw_dir / "hg38.fa", alphabet)
    
    train_dataset = GenomicDataset(train_peaks, extractor, encoder)
    val_dataset = GenomicDataset(val_peaks, extractor, encoder)
    test_dataset = GenomicDataset(test_peaks, extractor, encoder)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nобучение на {device} (размер батча: {batch_size})")
    
    model = HybridMotifTransformer(
        in_channels=4, num_filters=64, kernel_size=15, num_heads=4, num_layers=1
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
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
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val (chr21) Accuracy: {val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), root_dir / "results" / "best_transformer.pth")

    print("\nтест на 22 хромосоме")
    model.load_state_dict(torch.load(root_dir / "results" / "best_transformer.pth"))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch, return_attention=False)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            test_correct += (preds == y_batch).sum().item()
            
    test_acc = test_correct / len(test_loader.dataset)
    print(f"последний тест на 22{test_acc:.2%}")

    print("\n логгин в бд эскпериментов")
    db.log_experiment(
        model_name="HybridMotifTransformer",
        tf_name="CTCF",
        train_size=len(train_loader.dataset),
        epochs=epochs,
        batch_size=batch_size,
        lr=5e-4, 
        best_val_acc=best_val_acc,
        test_acc=test_acc
    )
    print("сохранено")
   

    
    extractor.close()

if __name__ == "__main__":
    main()