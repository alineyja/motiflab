import sqlite3
from typing import List, Iterator, Optional, Dict, Any
from pathlib import Path

from motiflab.bioinformatics.coordinates import GenomeCoordinates, Strand


class GenomicDatabase:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            #пики геномов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS peaks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chromosome TEXT NOT NULL,
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL,
                    strand TEXT NOT NULL,
                    transcription_factor TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chrom ON peaks (chromosome)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tf ON peaks (transcription_factor)")
            
            #Трекинг экспериментов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    tf_name TEXT NOT NULL,
                    train_size INTEGER,
                    epochs INTEGER,
                    batch_size INTEGER,
                    learning_rate REAL,
                    best_val_acc REAL,
                    test_acc REAL
                )
            """)
            conn.commit()

    def insert_peaks(self, peaks: Iterator[GenomeCoordinates], tf_name: str) -> None:
        data = [(p.chromosome, p.start, p.end, p.strand.value, tf_name) for p in peaks]
        with self._get_connection() as conn:
            conn.cursor().executemany("""
                INSERT INTO peaks (chromosome, start, end, strand, transcription_factor)
                VALUES (?, ?, ?, ?, ?)
            """, data)
            conn.commit()

    def get_peaks(self, chromosome: Optional[str] = None, tf_name: Optional[str] = None) -> List[GenomeCoordinates]:
        query = "SELECT chromosome, start, end, strand FROM peaks WHERE 1=1"
        params = []
        if chromosome:
            query += " AND chromosome = ?"
            params.append(chromosome)
        if tf_name:
            query += " AND transcription_factor = ?"
            params.append(tf_name)
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
        return [GenomeCoordinates(row[0], row[1], row[2], Strand(row[3])) for row in rows]

    def clear_peaks(self) -> None:
        with self._get_connection() as conn:
            conn.cursor().execute("DELETE FROM peaks")
            conn.commit()
    def log_experiment(
        self, model_name: str, tf_name: str, train_size: int, 
        epochs: int, batch_size: int, lr: float, best_val_acc: float, test_acc: float
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments 
                (model_name, tf_name, train_size, epochs, batch_size, learning_rate, best_val_acc, test_acc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_name, tf_name, train_size, epochs, batch_size, lr, best_val_acc, test_acc))
            conn.commit()

    def get_leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row 
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, model_name, epochs, learning_rate, best_val_acc, test_acc 
                FROM experiments 
                ORDER BY test_acc DESC LIMIT ?
            """, (top_n,))
            return [dict(row) for row in cursor.fetchall()]