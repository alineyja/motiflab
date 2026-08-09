# scripts/download_data.py

import os
import urllib.request
import gzip
import shutil
import ssl
from pathlib import Path



FA_MIRRORS_CHR21 = [
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz"
]

FA_MIRRORS_CHR22 = [
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
]

def download_and_extract(name: str, urls: list[str], target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    
    is_gzip = urls[0].endswith(".gz")
    
    gz_path = target_dir / (f"{name}.fa.gz" if "chr" in name else f"{name}.bed.gz")
    extracted_path = target_dir / (f"{name}.fa" if "chr" in name else f"{name}.bed")
    
    if extracted_path.exists():
        print(f"   [+] {extracted_path.name} уже существует. Пропускаем.")
        return extracted_path

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    ssl_context = ssl._create_unverified_context()

    downloaded = False
    for url in urls:
        print(f"   Попытка скачать {name} из {url}")
        req = urllib.request.Request(url, headers=headers)
        try:
            if is_gzip:
                with urllib.request.urlopen(req, context=ssl_context) as response, open(gz_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                
                print(f"   Extracting {gz_path.name}...")
                with gzip.open(gz_path, 'rb') as f_in, open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
            else:

                with urllib.request.urlopen(req, context=ssl_context) as response, open(extracted_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            
            downloaded = True
            print(f"  Сохранено в {extracted_path}\n")
            break
        except Exception as e:
            print(f"   Ошибка по причиене {e}")
            print("   следущее зеркало\n")
            if gz_path.exists(): os.remove(gz_path)
            if extracted_path.exists(): os.remove(extracted_path)

    if not downloaded:
        raise RuntimeError(f"все зеркала не рабочие {name}.")
        
    return extracted_path


def main():
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "raw"
    
    print("скачивание необходимых данных для экспериментов")
    print("скачивание данных из UCSC и ENCODE")
    print("скачиваеик CTCF пиков (BED-файл)")
    download_and_extract("chr21", FA_MIRRORS_CHR21, raw_dir)
    

    print("скачивание Chromosome 22 (для быстрого старта)")
    download_and_extract("chr22", FA_MIRRORS_CHR22, raw_dir)
    
   
    # print("весь геном hg38 (для полного эксперимента)")
    # download_and_extract("hg38", URLS["hg38_full"], raw_dir)
    
    print("все данные готовы в 'data/raw/'!")


if __name__ == "__main__":
    main()