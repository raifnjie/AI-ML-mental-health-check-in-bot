import sqlite3 
import os
from typing import List, Dict

DATABASE_PATH = "mental_health_entries.db"

def init_db():
    """Initialize the database with the required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                sentiment_label TEXT NOT NULL
            )
        ''')
    conn.commit()
    conn.close()

def add_entry(timestamp: str, text: str, score:float, label: str):
    """Add a new entry to the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO entries (timestamp, text, sentiment_score, sentiment_label)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, text, score, label))
    conn.commit()
    conn.close()

def fetch_entries(limit: int = 50) -> List[Dict]:
    """Fetch recent journal entries from the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT timestamp, text, sentiment_score, sentiment_label
        FROM entries
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    entries = []
    for row in cursor.fetchall():
        entries.append({
            'timestamp': row[0],
            'text': row[1],
            'sentiment_score': row[2],
            'sentiment_label': row[3]
        })
    
    conn.close()
    return entries

def get_mood_stats() -> Dict:
    """Get basic mood statistics"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT sentiment_label, COUNT(*) FROM entries GROUP BY sentiment_label')
    label_counts = dict(cursor.fetchall())
    
    cursor.execute('SELECT AVG(sentiment_score) FROM entries WHERE date(timestamp) >= date("now", "-7 days")')
    avg_week = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT COUNT(*) FROM entries')
    total_entries = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'label_counts': label_counts,
        'weekly_average': round(avg_week, 3),
        'total_entries': total_entries
    }