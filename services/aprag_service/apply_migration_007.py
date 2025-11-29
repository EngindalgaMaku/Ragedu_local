#!/usr/bin/env python3
"""
Apply migration 007: Remove FOREIGN KEY constraint from session_settings
"""

import sqlite3
import os
import sys

def apply_migration(db_path: str):
    """Apply migration 007"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Read migration file
        migration_path = "/app/database/migrations/007_remove_session_settings_fk.sql"
        if not os.path.exists(migration_path):
            migration_path = os.path.join(os.path.dirname(__file__), "database/migrations/007_remove_session_settings_fk.sql")
        
        if not os.path.exists(migration_path):
            print(f"Migration file not found: {migration_path}")
            return False
        
        with open(migration_path, 'r') as f:
            migration_sql = f.read()
        
        # Execute migration
        cursor.executescript(migration_sql)
        conn.commit()
        
        print("✅ Migration 007 applied successfully")
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error applying migration: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        sys.exit(1)
    
    success = apply_migration(db_path)
    sys.exit(0 if success else 1)








