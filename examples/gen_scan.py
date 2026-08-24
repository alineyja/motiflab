# examples/genome_scanner.py

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.bed import parse_bed
from motiflab.bioinformatics.fasta import FastaExtractor
from motiflab.preprocessing.encoder import OneHotEncoder
from motiflab.models.transformer import HybridMotifTransformer
from motiflab.bioinformatics.coordinates import GenomeCoordinates
from motiflab.bioinformatics.sequence import DNASequence

def find_hotspot(peaks: list, window_size: int = 200_000) -> tuple[int, int]:
    starts = [p.start for p in peaks]
    best_start = 0
    max_peaks = 0
    for i in range(len(starts)):
        current_start = starts[i]
        current_end = current_start + window_size
        count = sum(1 for s in starts if current_start <= s <= current_end)
        if count > max_peaks:
            max_peaks = count
            best_start = current_start
            
    return best_start, best_start + window_size


def main():
    target_chr = "chr22"
    scan_length = 200_000  
    window_size = 200      
    step_size = 20        
    batch_size = 512       
    
    raw_dir = root_dir / "data" / "raw"
    results_dir = root_dir / "results"
    model_path = results_dir / "best_transformer.pth"
    
    if not model_path.exists():
        print(f"не найден путь весов {model_path}")
        return
    alphabet = DNAAlphabet.standard()
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    
    #Hotspot на chr22
    all_peaks = list(parse_bed(raw_dir / "ctcf_peaks.bed"))
    chr22_peaks = [p for p in all_peaks if p.chromosome == target_chr]
    
    roi_start, roi_end = find_hotspot(chr22_peaks, scan_length)
    print(f"ROI {target_chr}:{roi_start}-{roi_end}")
    roi_peaks = [p for p in chr22_peaks if p.start >= roi_start and p.end <= roi_end]
    print(f"реал. пики {len(roi_peaks)}")
    print("извлечение днк")
    roi_coords = GenomeCoordinates(target_chr, roi_start, roi_end)
    with FastaExtractor(raw_dir / f"{target_chr}.fa", alphabet) as extractor:
        full_dna_seq = extractor.extract(roi_coords).sequence

    print("днк в окно и слайсинг")
    windows = []
    positions = [] 
    
    for i in range(0, len(full_dna_seq) - window_size + 1, step_size):
        window_seq = full_dna_seq[i : i + window_size]
        if "N" in window_seq:
            continue
            
        windows.append(window_seq)
        positions.append(roi_start + i + (window_size // 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"загрузка трансформатора на {device}...")
    model = HybridMotifTransformer(
        in_channels=4, num_filters=64, kernel_size=15, num_heads=4, num_layers=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #cканирование
    print(f"сканирование {len(windows)} используя ии")
    probabilities = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(windows), batch_size), desc="сканирование"):
            batch_seqs = windows[i : i + batch_size]
            
            #батч строк в матрицы NumPy
            encoded_list = [encoder.encode(DNASequence(s, alphabet)) for s in batch_seqs]
            tensor_x = torch.from_numpy(np.stack(encoded_list)).to(device)
            
            #Инференс
            logits = model(tensor_x, return_attention=False)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probabilities.extend(probs)

    #Отрисовка
    print("Отрисовка")
    plt.figure(figsize=(14, 4))
    
    #предсказания сети (Синяя линия)
    plt.plot(positions, probabilities, color='#2c7fb8', linewidth=1.5, label='Предикт вероятностей от ии')
    plt.fill_between(positions, probabilities, color='#2c7fb8', alpha=0.3)
    
    #пики(Красные)
    for p in roi_peaks:
        center = (p.start + p.end) / 2
        plt.axvline(x=center, color='#e34a33', linestyle='--', linewidth=2, alpha=0.8)
    
    # Хак для легенды: добавляем одну фейковую красную линию
    plt.axvline(x=-1, color='#e34a33', linestyle='--', linewidth=2, label='настоящие пики(encode)')
    
    plt.xlim(roi_start, roi_end)
    plt.ylim(0, 1.05)
    plt.title(f"сканирование генома {target_chr}:{roi_start:,}-{roi_end:,} (Hybrid Transformer)")
    plt.xlabel("координата генома")
    plt.ylabel("вероятность связыванния")
    plt.legend(loc="upper right")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    plot_path = results_dir / "genome_scan_track.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"   скан сохранен {plot_path}")

if __name__ == "__main__":
    main()