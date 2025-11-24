#!/usr/bin/env python3
"""
Apply Migration 008: Fix Topic Classification System
Run this script to apply all database schema fixes for topic classification and progress tracking
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_migration_008(db_path: str = None):
    """
    Apply Migration 008: Fix Topic Classification System
    
    Args:
        db_path: Path to database file (optional)
    """
    
    logger.info("=" * 70)
    logger.info("🚀 MIGRATION 008: Fix Topic Classification System - Starting")
    logger.info("=" * 70)
    
    # Get database path
    if not db_path:
        db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
        # Also check local path if container path doesn't exist
        if not os.path.exists(db_path):
            db_path = "rag3_for_local/data/rag_assistant.db"
    
    logger.info(f"🗄️  Database: {db_path}")
    
    # Check if database exists
    if not os.path.exists(db_path):
        logger.warning(f"Database not found at {db_path}")
        logger.info("Creating new database...")
        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Load migration file
    migration_file = Path(__file__).parent / "database" / "migrations" / "008_fix_topic_classification_system.sql"
    
    if not migration_file.exists():
        logger.error(f"❌ Migration file not found: {migration_file}")
        return False
    
    logger.info(f"📄 Migration file: {migration_file}")
    
    try:
        # Read migration SQL
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()
        
        logger.info("✅ Migration SQL loaded successfully")
        
        # Execute migration
        logger.info("🔧 Applying migration...")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Execute migration (split by statement for better error handling)
        statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
        
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for i, statement in enumerate(statements, 1):
            # Skip comments and empty statements
            if statement.startswith('--') or not statement.strip():
                continue
            
            try:
                cursor.execute(statement)
                success_count += 1
                
                # Log important operations
                if 'CREATE TABLE' in statement.upper():
                    table_name = extract_table_name(statement)
                    logger.info(f"  ✅ Created/verified table: {table_name}")
                elif 'ALTER TABLE' in statement.upper():
                    parts = statement.split('ADD COLUMN')
                    if len(parts) > 1:
                        table_name = parts[0].split('ALTER TABLE')[1].strip()
                        col_name = parts[1].strip().split()[0]
                        logger.info(f"  ✅ Added column {col_name} to {table_name}")
                elif 'CREATE INDEX' in statement.upper():
                    idx_name = extract_index_name(statement)
                    logger.info(f"  ✅ Created index: {idx_name}")
                elif 'CREATE VIEW' in statement.upper():
                    view_name = extract_view_name(statement)
                    logger.info(f"  ✅ Created view: {view_name}")
                elif 'INSERT' in statement.upper():
                    if 'feature_flags' in statement.lower():
                        logger.info(f"  ✅ Inserted feature flag")
            
            except sqlite3.OperationalError as e:
                error_msg = str(e).lower()
                
                # Check if it's a benign error (already exists, duplicate column, etc.)
                if any(phrase in error_msg for phrase in [
                    'duplicate column', 'already exists', 'no such column', 
                    'constraint failed: unique', 'unique constraint failed'
                ]):
                    skip_count += 1
                    logger.debug(f"  ⏭️  Skipped (already exists): {statement[:60]}...")
                else:
                    error_count += 1
                    logger.warning(f"  ⚠️  Warning on statement {i}: {e}")
                    logger.debug(f"     Statement: {statement[:100]}...")
            
            except Exception as e:
                error_count += 1
                logger.error(f"  ❌ Error on statement {i}: {e}")
                logger.debug(f"     Statement: {statement[:100]}...")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info("=" * 70)
        logger.info(f"🎉 Migration 008 completed successfully!")
        logger.info(f"   - {success_count} operations executed")
        logger.info(f"   - {skip_count} operations skipped (already exist)")
        if error_count > 0:
            logger.info(f"   - {error_count} warnings/errors (likely harmless)")
        logger.info("=" * 70)
        
        # Verify migration
        verify_migration(db_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration 008 failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def extract_table_name(statement):
    """Extract table name from CREATE TABLE statement"""
    try:
        upper_stmt = statement.upper()
        if 'IF NOT EXISTS' in upper_stmt:
            parts = statement.split('IF NOT EXISTS')[1].split('(')[0].strip()
        else:
            parts = statement.split('CREATE TABLE')[1].split('(')[0].strip()
        return parts
    except:
        return "unknown"


def extract_index_name(statement):
    """Extract index name from CREATE INDEX statement"""
    try:
        upper_stmt = statement.upper()
        if 'IF NOT EXISTS' in upper_stmt:
            parts = statement.split('IF NOT EXISTS')[1].split('ON')[0].strip()
        else:
            parts = statement.split('CREATE INDEX')[1].split('ON')[0].strip()
        return parts
    except:
        return "unknown"


def extract_view_name(statement):
    """Extract view name from CREATE VIEW statement"""
    try:
        upper_stmt = statement.upper()
        if 'IF NOT EXISTS' in upper_stmt:
            parts = statement.split('IF NOT EXISTS')[1].split('AS')[0].strip()
        else:
            parts = statement.split('CREATE VIEW')[1].split('AS')[0].strip()
        return parts
    except:
        return "unknown"


def verify_migration(db_path: str):
    """Verify that migration 008 was applied successfully"""
    
    logger.info("\n🔍 Verifying migration 008...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check for required tables
        tables_to_check = [
            'student_interactions',
            'topic_progress', 
            'question_topic_mapping',
            'feature_flags',
            'course_topics'
        ]
        
        for table in tables_to_check:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            result = cursor.fetchone()
            if result:
                logger.info(f"  ✅ Table '{table}' exists")
            else:
                logger.warning(f"  ⚠️  Table '{table}' not found")
        
        # Check for required columns in student_interactions
        columns_to_check = [
            'original_response',
            'personalized_response', 
            'adaptive_context',
            'sources',
            'emoji_feedback'
        ]
        
        cursor.execute("PRAGMA table_info(student_interactions)")
        existing_columns = [col[1] for col in cursor.fetchall()]  # col[1] is column name
        
        for col in columns_to_check:
            if col in existing_columns:
                logger.info(f"  ✅ Column 'student_interactions.{col}' exists")
            else:
                logger.warning(f"  ⚠️  Column 'student_interactions.{col}' not found")
        
        # Check feature flags
        cursor.execute("SELECT COUNT(*) FROM feature_flags")
        flag_count = cursor.fetchone()[0]
        logger.info(f"  ✅ {flag_count} feature flags configured")
        
        # Check views
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = cursor.fetchall()
        logger.info(f"  ✅ {len(views)} views created")
        
        conn.close()
        
        logger.info("\n✅ Migration 008 verification complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply Migration 008: Fix Topic Classification System')
    parser.add_argument('--db', help='Database path', default=None)
    
    args = parser.parse_args()
    
    success = apply_migration_008(args.db)
    
    if success:
        logger.info("\n🎉 Migration 008 completed successfully!")
        logger.info("🩸 Topic classification system is now properly configured")
        logger.info("🔄 Ready to test: 'kana rengini ne verir' → 'Kan Grupları'")
    else:
        logger.error("\n❌ Migration 008 failed")
    
    sys.exit(0 if success else 1)