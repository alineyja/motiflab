import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from motiflab.database.db import GenomicDatabase

def main():
    db_path = root_dir / "data" / "motiflab.db"
    
    if not db_path.exists():
        print("бд не найдена")
        return
        
    db = GenomicDatabase(db_path)
    leaders = db.get_leaderboard(top_n=10)
    
    if not leaders:
        print("нету экспериментов")
        return
        
    print("\n лидер")
    print("=" * 85)
    print(f"{'ID':<5} | {'Date':<18} | {'Model':<24} | {'LR':<7} | {'Val Acc':<8} | {'Test Acc':<8}")
    print("-" * 85)
    
    for row in leaders:
        date = str(row['timestamp']).split('.')[0]
        model = row['model_name']
        lr = row['learning_rate']
        val = f"{row['best_val_acc']*100:.1f}%"
        test = f"{row['test_acc']*100:.1f}%"
        
        print(f"{row['id']:<5} | {date:<18} | {model:<24} | {lr:<7} | {val:<8} | {test:<8}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()