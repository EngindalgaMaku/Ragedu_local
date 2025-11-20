#!/usr/bin/env python3
"""
Manually add emoji feedback columns
"""
import sqlite3
import os

db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Adding emoji feedback columns...")

# Add columns one by one
try:
    cursor.execute("ALTER TABLE student_interactions ADD COLUMN emoji_feedback TEXT DEFAULT NULL")
    print("✅ Added emoji_feedback column")
except Exception as e:
    print(f"⚠️  emoji_feedback: {e}")

try:
    cursor.execute("ALTER TABLE student_interactions ADD COLUMN emoji_feedback_timestamp TIMESTAMP DEFAULT NULL")
    print("✅ Added emoji_feedback_timestamp column")
except Exception as e:
    print(f"⚠️  emoji_feedback_timestamp: {e}")

try:
    cursor.execute("ALTER TABLE student_interactions ADD COLUMN emoji_comment TEXT DEFAULT NULL")
    print("✅ Added emoji_comment column")
except Exception as e:
    print(f"⚠️  emoji_comment: {e}")

# Add feedback_score column (used by emoji feedback)
try:
    cursor.execute("ALTER TABLE student_interactions ADD COLUMN feedback_score REAL DEFAULT NULL")
    print("✅ Added feedback_score column")
except Exception as e:
    print(f"⚠️  feedback_score: {e}")

# Create index
try:
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emoji_feedback ON student_interactions(emoji_feedback, emoji_feedback_timestamp)")
    print("✅ Created index idx_emoji_feedback")
except Exception as e:
    print(f"⚠️  Index: {e}")

# Create emoji_feedback_summary table
try:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emoji_feedback_summary (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            emoji TEXT NOT NULL,
            emoji_count INTEGER DEFAULT 1,
            avg_score REAL DEFAULT 0.5,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, session_id, emoji)
        )
    """)
    print("✅ Created emoji_feedback_summary table")
except Exception as e:
    print(f"⚠️  Table: {e}")

# Create index for summary
try:
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emoji_summary_user_session ON emoji_feedback_summary(user_id, session_id)")
    print("✅ Created index idx_emoji_summary_user_session")
except Exception as e:
    print(f"⚠️  Summary Index: {e}")

conn.commit()
conn.close()

print("\n✅ Done! Verifying...")

# Verify
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(student_interactions)")
columns = [row[1] for row in cursor.fetchall()]
conn.close()

print("\nColumns in student_interactions:")
for col in columns:
    print(f"  - {col}")

if 'emoji_feedback' in columns:
    print("\n✅ SUCCESS: emoji_feedback column exists!")
else:
    print("\n❌ FAILED: emoji_feedback column not found!")







