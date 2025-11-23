#!/usr/bin/env python3
"""
Apply Progressive Assessment Migration
Applies migration 007 - Progressive Assessment Tables
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_progressive_assessment_migration():
    """Apply progressive assessment database migration"""
    
    # Database path
    db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
    migration_file = "/app/database/migrations/007_add_progressive_assessment_tables.sql"
    
    # Check if running locally (adjust paths)
    if not os.path.exists(db_path):
        db_path = os.path.join(os.path.dirname(__file__), "data", "rag_assistant.db")
        migration_file = os.path.join(os.path.dirname(__file__), "database", "migrations", "007_add_progressive_assessment_tables.sql")
    
    logger.info(f"Using database: {db_path}")
    logger.info(f"Using migration file: {migration_file}")
    
    # Ensure database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Check if migration file exists
    if not os.path.exists(migration_file):
        logger.error(f"Migration file not found: {migration_file}")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info("Connected to database successfully")
        
        # Read migration SQL
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()
        
        logger.info("Migration SQL loaded successfully")
        
        # Split into individual statements
        statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
        
        logger.info(f"Executing {len(statements)} SQL statements...")
        
        # Execute each statement
        for i, statement in enumerate(statements, 1):
            try:
                cursor.execute(statement)
                logger.debug(f"✅ Statement {i}/{len(statements)} executed successfully")
            except sqlite3.Error as e:
                if "already exists" in str(e) or "duplicate column name" in str(e):
                    logger.info(f"⚠️  Statement {i}: {str(e)} (already exists, skipping)")
                else:
                    logger.error(f"❌ Statement {i} failed: {e}")
                    logger.error(f"Statement was: {statement[:100]}...")
                    raise e
        
        # Commit changes
        conn.commit()
        logger.info("✅ Migration committed successfully")
        
        # Verify tables were created
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('progressive_assessments', 'concept_confusion_log', 'requested_topics_log', 'progressive_assessment_summary')
        """)
        tables = cursor.fetchall()
        
        expected_tables = ['progressive_assessments', 'concept_confusion_log', 'requested_topics_log', 'progressive_assessment_summary']
        created_tables = [table[0] for table in tables]
        
        logger.info("Table verification:")
        for table in expected_tables:
            if table in created_tables:
                logger.info(f"  ✅ {table}")
            else:
                logger.warning(f"  ❌ {table} (not found)")
        
        # Check if columns were added to existing tables
        cursor.execute("PRAGMA table_info(student_interactions)")
        si_columns = [col[1] for col in cursor.fetchall()]
        
        progressive_columns = ['progressive_assessment_data', 'progressive_assessment_stage']
        logger.info("student_interactions columns verification:")
        for col in progressive_columns:
            if col in si_columns:
                logger.info(f"  ✅ {col}")
            else:
                logger.warning(f"  ❌ {col} (not found)")
        
        # Check student_profiles columns
        cursor.execute("PRAGMA table_info(student_profiles)")
        sp_columns = [col[1] for col in cursor.fetchall()]
        
        profile_columns = ['average_confidence', 'application_readiness', 'progressive_assessment_count', 'has_active_questions']
        logger.info("student_profiles columns verification:")
        for col in profile_columns:
            if col in sp_columns:
                logger.info(f"  ✅ {col}")
            else:
                logger.warning(f"  ❌ {col} (not found)")
        
        # Check feature flag
        cursor.execute("SELECT * FROM feature_flags WHERE flag_name = 'progressive_assessment'")
        flag_result = cursor.fetchone()
        
        if flag_result:
            logger.info("✅ Progressive assessment feature flag created")
        else:
            logger.warning("❌ Progressive assessment feature flag not found")
        
        conn.close()
        logger.info("✅ Progressive Assessment Migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False


if __name__ == "__main__":
    success = apply_progressive_assessment_migration()
    sys.exit(0 if success else 1)