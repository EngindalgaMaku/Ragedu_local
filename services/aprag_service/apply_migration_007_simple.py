#!/usr/bin/env python3
"""
Apply migration 007: Create session_settings table without FOREIGN KEY constraint
"""

import sqlite3
import os
import sys

def apply_migration(db_path: str):
    """Apply migration 007 - simple version"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_settings'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Table exists, need to recreate it
            print("Table exists, recreating without FOREIGN KEY...")
            
            # Create new table without FK
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_settings_new (
                    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    enable_progressive_assessment BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_personalized_responses BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_multi_dimensional_feedback BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_topic_analytics BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_cacs BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_zpd BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_bloom BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_cognitive_load BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_emoji_feedback BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id)
                )
            """)
            
            # Copy data
            cursor.execute("INSERT INTO session_settings_new SELECT * FROM session_settings")
            
            # Drop old table
            cursor.execute("DROP TABLE session_settings")
            
            # Rename
            cursor.execute("ALTER TABLE session_settings_new RENAME TO session_settings")
        else:
            # Table doesn't exist, create it without FK
            print("Table doesn't exist, creating without FOREIGN KEY...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_settings (
                    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    enable_progressive_assessment BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_personalized_responses BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_multi_dimensional_feedback BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_topic_analytics BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_cacs BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_zpd BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_bloom BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_cognitive_load BOOLEAN NOT NULL DEFAULT TRUE,
                    enable_emoji_feedback BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id)
                )
            """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_settings_session_id ON session_settings(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_settings_user_id ON session_settings(user_id)")
        
        # Create trigger
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_session_settings_updated_at
                AFTER UPDATE ON session_settings
                FOR EACH ROW
                WHEN NEW.updated_at <= OLD.updated_at
            BEGIN
                UPDATE session_settings SET updated_at = CURRENT_TIMESTAMP WHERE setting_id = NEW.setting_id;
            END
        """)
        
        conn.commit()
        print("✅ Migration 007 applied successfully")
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error applying migration: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        sys.exit(1)
    
    success = apply_migration(db_path)
    sys.exit(0 if success else 1)








