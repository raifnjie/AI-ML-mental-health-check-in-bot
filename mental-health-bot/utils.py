import json
import os
from datetime import datetime
from pathlib import Path

DATA_FILE = Path("data/user_logs.json")

def load_entries():
    """Load journal entries from JSON file"""
    # Create data directory if it doesn't exist
    DATA_FILE.parent.mkdir(exist_ok=True)
    
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_entry(text, label, score):
    """Save a journal entry to JSON file"""
    # Create data directory if it doesn't exist
    DATA_FILE.parent.mkdir(exist_ok=True)
    
    entries = load_entries()
    entries.append({
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "sentiment_label": label,  # Match the template expectation
        "sentiment_score": score
    })
    
    with open(DATA_FILE, 'w') as file:
        json.dump(entries, file, indent=4)