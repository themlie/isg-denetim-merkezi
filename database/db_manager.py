from datetime import datetime 
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).parent / "isg_audit.db"

class DBManager:
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    facility_name TEXT NOT NULL,
                    no_helmet_count INTEGER,
                    no_vest_count INTEGER,
                    hazard_score REAL,
                    risk_level TEXT NOT NULL
                )
            ''')
            conn.commit()
        conn.close()

    def add_log(self, facility_name, no_helmet_count, no_vest_count, hazard_score, risk_level):
        timestamp = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_logs (timestamp, facility_name, no_helmet_count, no_vest_count, hazard_score, risk_level)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, facility_name, no_helmet_count, no_vest_count, hazard_score, risk_level))
            conn.commit()
            
        conn.close()
        return cursor.lastrowid

    def fetch_logs(self, limit=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM audit_logs ORDER BY timestamp DESC'
            params = ()  # Varsayılan olarak gönderilecek parametre yok (boş tuple)
            
            if limit is not None:
                query += ' LIMIT ?'
                params = (limit,)  # Sadece limit varsa parametre eklenecek
                
            cursor.execute(query, params)
            logs = cursor.fetchall()

        conn.close()      
        return logs 


if __name__ == "__main__":
    db = DBManager()
    print("Veritabanı hazır")
    log_id = db.add_log("Şantiye A", 3, 2, 8.5, "High")
    print(f"Log ID: {log_id} - Risk Level: High")

    log_id = db.add_log("Şantiye B", 1, 0, 6.0, "Medium")
    print(f"Log ID: {log_id} - Risk Level: Medium")
    
    for satir in db.fetch_logs():
        print(satir)