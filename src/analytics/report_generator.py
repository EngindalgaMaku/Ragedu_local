# src/analytics/report_generator.py

import pandas as pd
from collections import Counter
import re

LOG_FILE_PATH = "data/analytics/query_logs.csv"

def generate_report():
    """
    Generates a report of the most frequently asked keywords.

    Returns:
        dict: A dictionary containing the top 5 keywords and their counts.
    """
    try:
        df = pd.read_csv(LOG_FILE_PATH)
    except FileNotFoundError:
        return {"error": "Log file not found. No queries have been logged yet."}

    # Basic keyword extraction: split by space and count words
    words = []
    for query in df["query"]:
        # Simple text cleaning: lowercase and remove non-alphanumeric characters
        cleaned_query = re.sub(r'\W+', ' ', str(query).lower())
        words.extend(cleaned_query.split())

    # Remove common stopwords (this list can be expanded)
    stopwords = set(["a", "an", "the", "is", "in", "it", "of", "for", "on", "with", "what", "who", "where", "when", "why", "how", "bu", "ve", "ile", "mi", "mu", "mı", "mü"])
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]

    word_counts = Counter(filtered_words)
    top_5_keywords = word_counts.most_common(5)
    
    return {"top_keywords": top_5_keywords}
