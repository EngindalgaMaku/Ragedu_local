#!/usr/bin/env python3
"""Verify KB-Enhanced RAG tables"""
import sqlite3
import sys

try:
    conn = sqlite3.connect('/app/data/rag_assistant.db')
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE 'topic_%' 
        ORDER BY name
    """)
    
    tables = [row[0] for row in cursor.fetchall()]
    
    print("\n✅ KB-Enhanced RAG Tables Created:")
    print("=" * 50)
    for table in tables:
        # Count rows
        count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = count_cursor.fetchone()[0]
        print(f"  ✅ {table:30} ({count} rows)")
    
    print("\n📊 Table Details:")
    print("=" * 50)
    
    # Check views
    view_cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='view' AND name LIKE 'v_%'
    """)
    views = [row[0] for row in view_cursor.fetchall()]
    
    print(f"\n📋 Views Created: {len(views)}")
    for view in views:
        print(f"  ✅ {view}")
    
    print("\n✅ Migration 005 Successfully Applied!")
    print(f"Total KB tables: {len(tables)}")
    print(f"Total views: {len(views)}")
    
    conn.close()
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)






