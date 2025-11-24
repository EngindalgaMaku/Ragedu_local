#!/usr/bin/env python3
import sqlite3
import sys

db_path = '/app/data/rag_assistant.db'
conn = sqlite3.connect(db_path)

try:
    # Create feature_flags table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            flag_name TEXT,
            feature_name TEXT,
            is_enabled INTEGER NOT NULL DEFAULT 1,
            config_data TEXT,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_flags_session_feature 
        ON feature_flags(session_id, feature_name) WHERE session_id IS NOT NULL
    """)
    
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_flags_global_flag
        ON feature_flags(flag_name) WHERE session_id IS NULL
    """)
    
    # Insert default flags
    conn.execute("""
        INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description) 
        VALUES ('aprag_enabled', 1, 'Enable APRAG adaptive personalized system')
    """)
    
    conn.execute("""
        INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description) 
        VALUES ('topic_classification', 1, 'Enable automatic topic classification for questions')
    """)
    
    conn.commit()
    print("✓ feature_flags table created successfully")
    
    # Verify
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feature_flags'")
    if cursor.fetchone():
        print("✓ feature_flags table verified")
    else:
        print("✗ feature_flags table not found")
        sys.exit(1)
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    conn.close()

