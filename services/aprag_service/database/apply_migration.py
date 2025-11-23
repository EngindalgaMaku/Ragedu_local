#!/usr/bin/env python3
"""
Apply Eğitsel-KBRAG migration to APRAG database
Run this script to add new tables and columns
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.database import DatabaseManager
from config.feature_flags import FeatureFlags

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_egitsel_kbrag_migration(db_path: str = None):
    """
    Apply Eğitsel-KBRAG migration
    
    Args:
        db_path: Path to database file (optional)
    """
    
    # Check if APRAG is enabled
    if not FeatureFlags.is_aprag_enabled():
        logger.error("❌ APRAG is not enabled. Cannot apply Eğitsel-KBRAG migration.")
        logger.info("Enable APRAG first: export APRAG_ENABLED=true")
        return False
    
    logger.info("=" * 60)
    logger.info("Eğitsel-KBRAG Migration - Starting")
    logger.info("=" * 60)
    
    # Get database path
    if not db_path:
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
    
    logger.info(f"Database: {db_path}")
    
    # Check if database exists
    if not os.path.exists(db_path):
        logger.warning(f"Database not found at {db_path}")
        logger.info("Creating new database...")
    
    # Load migration file
    migration_file = Path(__file__).parent / "migrations" / "005_egitsel_kbrag_tables.sql"
    
    if not migration_file.exists():
        logger.error(f"❌ Migration file not found: {migration_file}")
        return False
    
    logger.info(f"Migration file: {migration_file}")
    
    try:
        # Read migration SQL
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()
        
        logger.info("Migration SQL loaded successfully")
        
        # Execute migration
        logger.info("Applying migration...")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute migration (split by statement for better error handling)
        statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
        
        success_count = 0
        skip_count = 0
        
        for i, statement in enumerate(statements, 1):
            # Skip comments and SELECT statements
            if statement.startswith('--') or statement.upper().startswith('SELECT'):
                continue
            
            try:
                cursor.execute(statement)
                success_count += 1
                
                # Log important operations
                if 'CREATE TABLE' in statement.upper():
                    table_name = statement.split('CREATE TABLE')[1].split('(')[0].strip().split()[0]
                    if 'IF NOT EXISTS' in statement.upper():
                        table_name = table_name.replace('IF NOT EXISTS', '').strip()
                    logger.info(f"  ✅ Created/verified table: {table_name}")
                elif 'ALTER TABLE' in statement.upper():
                    parts = statement.split('ADD COLUMN')
                    if len(parts) > 1:
                        table_name = parts[0].split('ALTER TABLE')[1].strip()
                        col_name = parts[1].strip().split()[0]
                        logger.info(f"  ✅ Added column {col_name} to {table_name}")
                elif 'CREATE INDEX' in statement.upper():
                    if 'IF NOT EXISTS' in statement.upper():
                        idx_name = statement.split('IF NOT EXISTS')[1].split('ON')[0].strip()
                    else:
                        idx_name = statement.split('CREATE INDEX')[1].split('ON')[0].strip()
                    logger.info(f"  ✅ Created index: {idx_name}")
            
            except sqlite3.OperationalError as e:
                error_msg = str(e).lower()
                
                # Check if it's a "duplicate column" or "already exists" error
                if 'duplicate' in error_msg or 'already exists' in error_msg:
                    skip_count += 1
                    logger.debug(f"  ⏭️  Skipped (already exists): {statement[:50]}...")
                else:
                    logger.warning(f"  ⚠️  Warning on statement {i}: {e}")
                    logger.debug(f"     Statement: {statement[:100]}...")
            
            except Exception as e:
                logger.error(f"  ❌ Error on statement {i}: {e}")
                logger.debug(f"     Statement: {statement[:100]}...")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info("=" * 60)
        logger.info(f"✅ Migration completed successfully!")
        logger.info(f"   - {success_count} operations executed")
        logger.info(f"   - {skip_count} operations skipped (already exist)")
        logger.info("=" * 60)
        
        # Verify tables
        verify_migration(db_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def verify_migration(db_path: str):
    """Verify that migration was applied successfully"""
    
    logger.info("\nVerifying migration...")
    
    try:
        db = DatabaseManager(db_path)
        
        # Check for new tables
        tables_to_check = [
            'document_global_scores',
            'cacs_scores',
            'pedagogical_analytics',
            'feature_flags'
        ]
        
        for table in tables_to_check:
            result = db.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            )
            if result:
                logger.info(f"  ✅ Table '{table}' exists")
            else:
                logger.warning(f"  ⚠️  Table '{table}' not found")
        
        # Check for new columns in student_interactions
        columns_to_check = [
            'bloom_level',
            'zpd_level',
            'cognitive_load_score',
            'cacs_score',
            'emoji_feedback'
        ]
        
        result = db.execute_query("PRAGMA table_info(student_interactions)")
        existing_columns = [col['name'] for col in result]
        
        for col in columns_to_check:
            if col in existing_columns:
                logger.info(f"  ✅ Column 'student_interactions.{col}' exists")
            else:
                logger.warning(f"  ⚠️  Column 'student_interactions.{col}' not found")
        
        # Check feature flags
        flags = db.execute_query("SELECT * FROM feature_flags")
        logger.info(f"  ✅ {len(flags)} feature flags configured")
        
        logger.info("\n✅ Verification complete")
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")


def rollback_migration(db_path: str = None):
    """
    Rollback Eğitsel-KBRAG migration (WARNING: This will delete data!)
    """
    
    logger.warning("=" * 60)
    logger.warning("⚠️  ROLLBACK WARNING")
    logger.warning("This will delete all Eğitsel-KBRAG data!")
    logger.warning("=" * 60)
    
    response = input("Are you sure you want to rollback? (type 'YES' to confirm): ")
    
    if response != 'YES':
        logger.info("Rollback cancelled")
        return False
    
    if not db_path:
        db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info("Dropping Eğitsel-KBRAG tables...")
        
        # Drop tables
        cursor.execute("DROP TABLE IF EXISTS feature_flags")
        cursor.execute("DROP TABLE IF EXISTS pedagogical_analytics")
        cursor.execute("DROP TABLE IF EXISTS cacs_scores")
        cursor.execute("DROP TABLE IF EXISTS document_global_scores")
        
        logger.info("Removing Eğitsel-KBRAG columns...")
        
        # Note: SQLite doesn't support DROP COLUMN easily
        # We'll just leave the columns (they won't hurt anything)
        logger.warning("Note: Columns in existing tables are preserved (SQLite limitation)")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Rollback complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply Eğitsel-KBRAG migration')
    parser.add_argument('--db', help='Database path', default=None)
    parser.add_argument('--rollback', action='store_true', help='Rollback migration')
    
    args = parser.parse_args()
    
    if args.rollback:
        success = rollback_migration(args.db)
    else:
        success = apply_egitsel_kbrag_migration(args.db)
    
    sys.exit(0 if success else 1)















