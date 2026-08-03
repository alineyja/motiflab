# scripts/download_data.py

import os
import urllib.request
import gzip
import shutil
from pathlib import Path

#UCSC и ENCODE
URLS = {
    #Chromosome 22 
    "chr22": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz",
    
    #hg38
    "hg38_full": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    
    #ChIP-seq BED-файл CTCF (клеточная линия K562, сборка hg38)
    "ctcf_peaks": "https://raw.githubusercontent.com/stnshn/encode-ctcf-db/master/data/GM12878_ctcf_peaks.bed"
}


def download_and_extract(name: str, url: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    
    gz_path = target_dir / f"{name}.fa.gz" if "hg38" in name or "chr" in name else target_dir / f"{name}.bed.gz"
    extracted_path = target_dir / f"{name}.fa" if "hg38" in name or "chr" in name else target_dir / f"{name}.bed"
    
    if extracted_path.exists():
        print(f"   [+] {extracted_path.name} уже существует. пропуск.")
        return extracted_path

    print(f"   Скачивание {name} из ENCODE/UCSC")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response, open(gz_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    except Exception as e:
        print(f"   [Error] Failed to download from {url}. Reason: {e}")
        raise e
    print(f"   извлечение {gz_path.name}...")
    
    # Распаковка .gz
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extracted_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    os.remove(gz_path)
    print(f"извелчен в {extracted_path}\n")
    return extracted_path


def main():
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "raw"
    
    print("скачивание необходимых данных для экспериментов")
    print("скачивание данных из UCSC и ENCODE")
    download_and_extract("ctcf_peaks", URLS["ctcf_peaks"], raw_dir)
    
    # 2. Скачиваем Chromosome 22 для быстрого старта
    print("хромосома 22")
    download_and_extract("chr22", URLS["chr22"], raw_dir)
    
   
    # print("весь геном hg38 (для полного эксперимента)")
    # download_and_extract("hg38", URLS["hg38_full"], raw_dir)
    
    print("все данные готовы в 'data/raw/'!")


if __name__ == "__main__":
    main()