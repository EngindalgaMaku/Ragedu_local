#!/usr/bin/env python3
"""
Apply Emoji Feedback Migration (006)
Faz 4 - Eğitsel-KBRAG
"""

import sqlite3
import os
import sys

# Get database path
db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")

print(f"Applying emoji feedback migration to: {db_path}")

# Read migration file
migration_path = os.path.join(
    os.path.dirname(__file__),
    "database/migrations/006_add_emoji_feedback_columns.sql"
)

if not os.path.exists(migration_path):
    print(f"❌ Migration file not found: {migration_path}")
    sys.exit(1)

with open(migration_path, 'r', encoding='utf-8') as f:
    migration_sql = f.read()

# Apply migration
try:
    conn = sqlite3.connect(db_path)
    conn.executescript(migration_sql)
    conn.commit()
    conn.close()
    
    print("✅ Emoji feedback migration applied successfully!")
    print("\nNew columns added:")
    print("  - student_interactions.emoji_feedback")
    print("  - student_interactions.emoji_feedback_timestamp")
    print("  - student_interactions.emoji_comment")
    print("\nNew table created:")
    print("  - emoji_feedback_summary")
    
except Exception as e:
    print(f"❌ Migration failed: {e}")
    sys.exit(1)



