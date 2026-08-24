import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from motiflab.bioinformatics.bed import parse_bed
from motiflab.database.db import GenomicDatabase

def main():
    raw_dir = root_dir / "data" / "raw"
    db_path = root_dir / "data" / "motiflab.db"
    
    bed_file = raw_dir / "ctcf_peaks.bed"
    
    if not bed_file.exists():
        print(f"bed файл не найден {bed_file}")
        return
    print(f"путь к базе данных{db_path}")
    print(f"чтение из {bed_file.name}")
    
    db = GenomicDatabase(db_path)
    db.clear()
    peaks_generator = parse_bed(bed_file)
    t0 = time.perf_counter()
    db.insert_peaks(peaks_generator, tf_name="CTCF")
    t1 = time.perf_counter()
    
    
    loaded_peaks = db.get_peaks(tf_name="CTCF")
    print("импорт завершен")
    print(f"заняло {t1 - t0:.3f} секунд")
    print(f"в общем {len(loaded_peaks):,} строк")

if __name__ == "__main__":
    main()