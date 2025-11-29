#!/usr/bin/env python3
"""
Script to remove foreign key constraint from topic_progress to users table
This fixes the "no such table: main.users" error when deleting topics
"""

import sqlite3
import os
from pathlib import Path

# Determine database path
project_root = Path(__file__).resolve().parent
db_path = os.getenv(
    "APRAG_DB_PATH",
    str(project_root / "services" / "aprag_service" / "database" / "rag_assistant.db")
)

# Try Docker path if local doesn't exist
if not os.path.exists(db_path):
    db_path = "/app/data/rag_assistant.db"

print(f"🔧 Fixing topic_progress foreign key constraint...")
print(f"📁 Database: {db_path}")

if not os.path.exists(db_path):
    print(f"❌ Database not found at: {db_path}")
    print("💡 Trying alternative paths...")
    
    # Try other possible paths
    alt_paths = [
        "data/rag_assistant.db",
        "../data/rag_assistant.db",
        "../../data/rag_assistant.db",
    ]
    
    for alt_path in alt_paths:
        full_path = os.path.abspath(alt_path)
        if os.path.exists(full_path):
            db_path = full_path
            print(f"✅ Found database at: {db_path}")
            break
    else:
        print("❌ Database not found in any expected location")
        exit(1)

try:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Disable foreign keys
    conn.execute("PRAGMA foreign_keys = OFF")
    
    # Check if topic_progress table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topic_progress'")
    if not cursor.fetchone():
        print("⚠️  topic_progress table does not exist. Nothing to fix.")
        conn.close()
        exit(0)
    
    # Check current foreign keys
    cursor.execute("PRAGMA foreign_key_list(topic_progress)")
    fks = cursor.fetchall()
    has_users_fk = any(fk[2] == 'users' for fk in fks)  # fk[2] is the referenced table
    
    if not has_users_fk:
        print("✅ No foreign key to users table found. Migration already applied.")
        conn.close()
        exit(0)
    
    print(f"🔍 Found foreign key constraint to users table. Removing...")
    
    # Get table structure
    cursor.execute("PRAGMA table_info(topic_progress)")
    columns = cursor.fetchall()
    
    # Create new table without FK to users
    create_sql = """
    CREATE TABLE IF NOT EXISTS topic_progress_new (
        progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id VARCHAR(255) NOT NULL,
        session_id VARCHAR(255) NOT NULL,
        topic_id INTEGER NOT NULL,
        
        -- Progress metrics
        questions_asked INTEGER DEFAULT 0,
        average_understanding DECIMAL(3,2),
        average_satisfaction DECIMAL(3,2),
        last_question_timestamp TIMESTAMP,
        
        -- Mastery assessment
        mastery_level VARCHAR(20),
        mastery_score DECIMAL(3,2),
        
        -- Readiness for next topic
        is_ready_for_next BOOLEAN DEFAULT FALSE,
        readiness_score DECIMAL(3,2),
        
        -- Time tracking
        time_spent_minutes INTEGER DEFAULT 0,
        first_question_timestamp TIMESTAMP,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        UNIQUE(user_id, session_id, topic_id),
        FOREIGN KEY (topic_id) REFERENCES course_topics(topic_id) ON DELETE CASCADE
    )
    """
    
    cursor.execute(create_sql)
    print("✅ Created new topic_progress table without FK to users")
    
    # Copy existing data
    cursor.execute("SELECT COUNT(*) FROM topic_progress")
    count = cursor.fetchone()[0]
    print(f"📊 Found {count} existing records to migrate")
    
    if count > 0:
        # Get column names from old table
        old_columns = [col[1] for col in columns]
        new_columns = [
            'progress_id', 'user_id', 'session_id', 'topic_id',
            'questions_asked', 'average_understanding', 'average_satisfaction',
            'last_question_timestamp', 'mastery_level', 'mastery_score',
            'is_ready_for_next', 'readiness_score', 'time_spent_minutes',
            'first_question_timestamp', 'created_at', 'updated_at'
        ]
        
        # Find common columns
        common_cols = [col for col in old_columns if col in new_columns]
        cols_str = ', '.join(common_cols)
        
        cursor.execute(f"INSERT INTO topic_progress_new ({cols_str}) SELECT {cols_str} FROM topic_progress")
        print(f"✅ Migrated {count} records successfully")
    
    # Drop old table
    cursor.execute("DROP TABLE IF EXISTS topic_progress")
    print("🗑️  Dropped old topic_progress table")
    
    # Rename new table
    cursor.execute("ALTER TABLE topic_progress_new RENAME TO topic_progress")
    print("✅ Renamed new table to topic_progress")
    
    # Recreate indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_progress_user_topic ON topic_progress(user_id, topic_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_progress_session ON topic_progress(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_progress_topic ON topic_progress(topic_id)")
    print("✅ Recreated indexes")
    
    # Commit
    conn.commit()
    print("✅ Migration completed successfully!")
    
    # Verify
    cursor.execute("PRAGMA foreign_key_list(topic_progress)")
    fks_after = cursor.fetchall()
    has_users_fk_after = any(fk[2] == 'users' for fk in fks_after)
    
    if has_users_fk_after:
        print("⚠️  Warning: Foreign key to users still exists after migration")
    else:
        print("✅ Verification passed: No foreign key to users table")
    
    conn.close()
    print("\n🎉 Fix completed! You can now delete topics without errors.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    if 'conn' in locals():
        conn.rollback()
        conn.close()
    exit(1)




