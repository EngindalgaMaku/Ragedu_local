# src/analytics/analytics_logger.py

import csv
import os
from datetime import datetime

LOG_FILE_PATH = "data/analytics/query_logs.csv"

def log_query(query: str):
    """
    Logs a user query to a CSV file.

    Args:
        query (str): The user's query.
    """
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    
    file_exists = os.path.isfile(LOG_FILE_PATH)
    
    with open(LOG_FILE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query"])
        
        writer.writerow([datetime.now().isoformat(), query])
