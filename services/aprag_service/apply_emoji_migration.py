#!/usr/bin/env python3
"""
Apply Emoji Feedback Migration (006)
"""
import os
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_emoji_migration():
    """Apply migration 006 for emoji feedback"""
    
    db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
    migration_file = "/app/database/migrations/006_add_emoji_feedback_columns.sql"
    
    logger.info(f"Applying emoji feedback migration to: {db_path}")
    logger.info(f"Migration file: {migration_file}")
    
    try:
        # Read migration SQL
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute migration statements
        statements = [s.strip() for s in migration_sql.split(';') if s.strip() and not s.strip().startswith('--')]
        
        for i, statement in enumerate(statements, 1):
            try:
                cursor.execute(statement)
                
                # Log what was done
                if 'ALTER TABLE' in statement.upper():
                    logger.info(f"  ✅ Executed ALTER TABLE statement {i}")
                elif 'CREATE TABLE' in statement.upper():
                    table_name = statement.split('CREATE TABLE')[1].split('(')[0].strip()
                    if 'IF NOT EXISTS' in statement.upper():
                        table_name = table_name.replace('IF NOT EXISTS', '').strip()
                    logger.info(f"  ✅ Created table: {table_name}")
                elif 'CREATE INDEX' in statement.upper():
                    logger.info(f"  ✅ Created index")
                else:
                    logger.info(f"  ✅ Executed statement {i}")
                    
            except sqlite3.OperationalError as e:
                error_msg = str(e).lower()
                if 'duplicate' in error_msg or 'already exists' in error_msg:
                    logger.info(f"  ⏭️  Skipped (already exists): {statement[:50]}...")
                else:
                    logger.warning(f"  ⚠️  Warning: {e}")
            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info("✅ Migration 006 applied successfully!")
        
        # Verify
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(student_interactions)")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        
        if 'emoji_feedback' in columns:
            logger.info("✅ Verified: emoji_feedback column exists")
        else:
            logger.error("❌ Verification failed: emoji_feedback column not found")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = apply_emoji_migration()
    exit(0 if success else 1)
