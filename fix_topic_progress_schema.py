#!/usr/bin/env python3
"""
Fix topic_progress table schema by adding missing columns
This script adds average_understanding and other missing columns to existing topic_progress table
"""

import sqlite3
import os
import sys

# Get database path
db_path = "data/rag_assistant.db"

# Try different paths
if not os.path.exists(db_path):
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "data", "rag_assistant.db"),
        os.path.join(os.path.dirname(__file__), "rag3_for_local", "data", "rag_assistant.db"),
        "rag3_for_local/data/rag_assistant.db",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            db_path = path
            break

if not os.path.exists(db_path):
    print(f"Database not found. Tried: {db_path}")
    print("Please run this script from the project root directory")
    sys.exit(1)

print(f"Connecting to database: {db_path}")

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

try:
    # Check if topic_progress table exists
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='topic_progress'
    """)
    
    if not cursor.fetchone():
        print("topic_progress table does not exist. Creating it...")
        # Create table with all columns
        conn.execute("""
            CREATE TABLE topic_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                topic_id INTEGER NOT NULL,
                questions_asked INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                mastery_level REAL DEFAULT 0.0,
                mastery_score REAL DEFAULT 0.0,
                average_understanding REAL DEFAULT 0.0,
                is_ready_for_next BOOLEAN DEFAULT FALSE,
                readiness_score REAL DEFAULT 0.0,
                time_spent_minutes INTEGER DEFAULT 0,
                first_interaction_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                last_question_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, session_id, topic_id)
            )
        """)
        print("✓ topic_progress table created")
    else:
        print("topic_progress table exists. Checking for missing columns...")
        
        # Get existing columns
        cursor = conn.execute("PRAGMA table_info(topic_progress)")
        existing_columns = {row[1]: row[2] for row in cursor.fetchall()}
        print(f"Existing columns: {list(existing_columns.keys())}")
        
        # Columns that should exist
        required_columns = {
            'average_understanding': 'REAL DEFAULT 0.0',
            'mastery_level': 'REAL DEFAULT 0.0',
            'mastery_score': 'REAL DEFAULT 0.0',
            'is_ready_for_next': 'BOOLEAN DEFAULT FALSE',
            'readiness_score': 'REAL DEFAULT 0.0',
            'time_spent_minutes': 'INTEGER DEFAULT 0',
        }
        
        # Add missing columns
        for col_name, col_def in required_columns.items():
            if col_name not in existing_columns:
                try:
                    # SQLite doesn't support all ALTER TABLE operations, so we need to be careful
                    # For REAL/INTEGER/BOOLEAN columns, we can use ALTER TABLE ADD COLUMN
                    if 'REAL' in col_def:
                        sql_type = 'REAL'
                        default = col_def.split('DEFAULT')[1].strip() if 'DEFAULT' in col_def else ''
                    elif 'INTEGER' in col_def:
                        sql_type = 'INTEGER'
                        default = col_def.split('DEFAULT')[1].strip() if 'DEFAULT' in col_def else ''
                    elif 'BOOLEAN' in col_def:
                        sql_type = 'INTEGER'  # SQLite uses INTEGER for BOOLEAN
                        default = '0'
                    else:
                        sql_type = 'TEXT'
                        default = ''
                    
                    alter_sql = f"ALTER TABLE topic_progress ADD COLUMN {col_name} {sql_type}"
                    if default:
                        alter_sql += f" DEFAULT {default}"
                    
                    conn.execute(alter_sql)
                    print(f"✓ Added column: {col_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        print(f"  Column {col_name} already exists")
                    else:
                        print(f"✗ Error adding column {col_name}: {e}")
            else:
                print(f"  Column {col_name} already exists")
    
    # Check and create feature_flags table
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='feature_flags'
    """)
    
    if not cursor.fetchone():
        print("\nCreating feature_flags table...")
        conn.execute("""
            CREATE TABLE feature_flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                flag_name TEXT,
                feature_name TEXT,
                is_enabled BOOLEAN NOT NULL DEFAULT 1,
                config_data TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description, created_at) 
            VALUES ('aprag_enabled', 1, 'Enable APRAG adaptive personalized system', CURRENT_TIMESTAMP)
        """)
        
        conn.execute("""
            INSERT OR IGNORE INTO feature_flags (flag_name, is_enabled, description, created_at) 
            VALUES ('topic_classification', 1, 'Enable automatic topic classification for questions', CURRENT_TIMESTAMP)
        """)
        
        print("✓ feature_flags table created with default flags")
    else:
        print("✓ feature_flags table already exists")
    
    # Commit changes
    conn.commit()
    print("\n✓ All changes committed successfully!")
    
    # Verify final state
    print("\nFinal verification:")
    cursor = conn.execute("PRAGMA table_info(topic_progress)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"topic_progress columns: {columns}")
    
    if 'average_understanding' in columns:
        print("✓ average_understanding column verified")
    else:
        print("✗ average_understanding column still missing!")
    
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
    import traceback
    traceback.print_exc()
    raise
finally:
    conn.close()

print("\nSchema fix complete!")

