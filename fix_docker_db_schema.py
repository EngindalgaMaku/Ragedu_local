#!/usr/bin/env python3
"""
Fix topic_progress table schema in Docker container database
This script connects to the Docker container and applies the migration
"""

import subprocess
import sys

# SQL commands to fix the schema
migration_sql = """
-- Add missing columns to topic_progress if they don't exist
-- SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we'll use a different approach

-- Check and add mastery_score
-- Note: We'll execute these one by one and ignore errors if column already exists

-- Add missing columns (will fail silently if they exist)
ALTER TABLE topic_progress ADD COLUMN mastery_score REAL DEFAULT 0.0;
ALTER TABLE topic_progress ADD COLUMN is_ready_for_next INTEGER DEFAULT 0;
ALTER TABLE topic_progress ADD COLUMN readiness_score REAL DEFAULT 0.0;
ALTER TABLE topic_progress ADD COLUMN time_spent_minutes INTEGER DEFAULT 0;

-- Create feature_flags table if not exists
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
);

-- Create indexes for feature_flags
CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_flags_session_feature 
ON feature_flags(session_id, feature_name) WHERE session_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_flags_global_flag
ON feature_flags(flag_name) WHERE session_id IS NULL;

-- Insert default flags
INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description, created_at) 
VALUES ('aprag_enabled', 1, 'Enable APRAG adaptive personalized system', CURRENT_TIMESTAMP);

INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description, created_at) 
VALUES ('topic_classification', 1, 'Enable automatic topic classification for questions', CURRENT_TIMESTAMP);
"""

print("Applying migration to Docker container database...")

# Execute SQL in the container
try:
    # Write SQL to a temporary file
    with open("/tmp/migration.sql", "w") as f:
        f.write(migration_sql)
    
    # Copy to container
    subprocess.run(
        ["docker", "cp", "/tmp/migration.sql", "aprag-service:/tmp/migration.sql"],
        check=True
    )
    
    # Execute in container
    result = subprocess.run(
        ["docker", "exec", "aprag-service", "python", "-c", """
import sqlite3
import sys

db_path = '/app/data/rag_assistant.db'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

try:
    with open('/tmp/migration.sql', 'r') as f:
        sql = f.read()
    
    # Split by semicolon and execute each statement
    statements = [s.strip() for s in sql.split(';') if s.strip()]
    
    for statement in statements:
        try:
            conn.execute(statement)
            print(f'✓ Executed: {statement[:50]}...')
        except sqlite3.OperationalError as e:
            if 'duplicate column' in str(e).lower() or 'already exists' in str(e).lower():
                print(f'  (Skipped - already exists: {statement[:50]}...)')
            else:
                print(f'✗ Error: {e}')
                print(f'  Statement: {statement[:100]}')
    
    conn.commit()
    
    # Verify
    cursor = conn.execute('PRAGMA table_info(topic_progress)')
    columns = [row[1] for row in cursor.fetchall()]
    print(f'\\nFinal columns: {columns}')
    
    if 'average_understanding' in columns:
        print('✓ average_understanding exists')
    if 'mastery_score' in columns:
        print('✓ mastery_score exists')
    if 'is_ready_for_next' in columns:
        print('✓ is_ready_for_next exists')
    
    # Check feature_flags
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feature_flags'")
    if cursor.fetchone():
        print('✓ feature_flags table exists')
    else:
        print('✗ feature_flags table missing')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    conn.close()
        """],
        check=True,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
except subprocess.CalledProcessError as e:
    print(f"Error executing migration: {e}")
    print(f"Output: {e.stdout}")
    print(f"Error: {e.stderr}")
    sys.exit(1)
except FileNotFoundError:
    print("Error: docker command not found. Make sure Docker is installed and running.")
    sys.exit(1)

print("\nMigration complete!")

