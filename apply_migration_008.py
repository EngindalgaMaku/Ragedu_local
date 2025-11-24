#!/usr/bin/env python3
"""
Apply Migration 008: Fix topic classification system
This adds missing columns to topic_progress and creates feature_flags table
"""

import sqlite3
import os
import sys

# Get database path
db_path = "data/rag_assistant.db"

# If running from services/aprag_service directory
if not os.path.exists(db_path):
    db_path = os.path.join(os.path.dirname(__file__), "data/rag_assistant.db")

# If still not found, try from root
if not os.path.exists(db_path):
    db_path = os.path.join(os.path.dirname(__file__), "rag3_for_local", "data", "rag_assistant.db")

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    print("Please run this script from the project root directory")
    sys.exit(1)

print(f"Applying migration to: {db_path}")

# Read migration file
migration_path = os.path.join(
    os.path.dirname(__file__),
    "services/aprag_service/database/migrations/008_fix_topic_classification_system.sql"
)

if not os.path.exists(migration_path):
    # Try alternative path
    migration_path = os.path.join(
        os.path.dirname(__file__),
        "rag3_for_local/services/aprag_service/database/migrations/008_fix_topic_classification_system.sql"
    )

if not os.path.exists(migration_path):
    print(f"Migration file not found at: {migration_path}")
    sys.exit(1)

print(f"Reading migration from: {migration_path}")

# Connect to database
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

try:
    # Read and execute migration
    with open(migration_path, 'r', encoding='utf-8') as f:
        migration_sql = f.read()
    
    # Execute migration
    print("Executing migration...")
    conn.executescript(migration_sql)
    conn.commit()
    
    print("Migration applied successfully!")
    
    # Verify the changes
    print("\nVerifying changes...")
    
    # Check topic_progress table structure
    cursor = conn.execute("PRAGMA table_info(topic_progress)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"\ntopic_progress columns: {columns}")
    
    if 'average_understanding' in columns:
        print("✓ average_understanding column exists")
    else:
        print("✗ average_understanding column missing")
    
    # Check feature_flags table
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='feature_flags'
    """)
    if cursor.fetchone():
        print("✓ feature_flags table exists")
    else:
        print("✗ feature_flags table missing")
    
    # Check course_topics table
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='course_topics'
    """)
    if cursor.fetchone():
        print("✓ course_topics table exists")
    else:
        print("✗ course_topics table missing")
    
except Exception as e:
    print(f"Error applying migration: {e}")
    conn.rollback()
    raise
finally:
    conn.close()

print("\nMigration complete!")

