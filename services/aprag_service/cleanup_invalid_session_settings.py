#!/usr/bin/env python3
"""
Cleanup script to remove invalid session_settings records
that have student user_ids instead of teacher user_ids.

These records were created by the migration that incorrectly
used student_interactions.user_id (which is a student ID).
"""

import os
import sys
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_invalid_session_settings(db_path: str):
    """
    Remove session_settings records that have user_ids from student_interactions.
    These are invalid because session_settings.user_id should be the teacher's ID,
    not the student's ID.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Find session_settings records where user_id exists in student_interactions
        # This indicates they were created from student_interactions (wrong!)
        cursor.execute("""
            SELECT ss.setting_id, ss.session_id, ss.user_id
            FROM session_settings ss
            WHERE EXISTS (
                SELECT 1 
                FROM student_interactions si 
                WHERE si.user_id = ss.user_id 
                AND si.session_id = ss.session_id
            )
        """)
        
        invalid_records = cursor.fetchall()
        
        if not invalid_records:
            logger.info("No invalid session_settings records found. Database is clean.")
            return 0
        
        logger.info(f"Found {len(invalid_records)} invalid session_settings records to delete:")
        for record in invalid_records:
            logger.info(f"  - Session: {record['session_id']}, User ID: {record['user_id']}, Setting ID: {record['setting_id']}")
        
        # Delete invalid records
        cursor.execute("""
            DELETE FROM session_settings
            WHERE EXISTS (
                SELECT 1 
                FROM student_interactions si 
                WHERE si.user_id = session_settings.user_id 
                AND si.session_id = session_settings.session_id
            )
        """)
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"✅ Successfully deleted {deleted_count} invalid session_settings records.")
        logger.info("These records will be recreated with correct teacher user_id when accessed.")
        
        conn.close()
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ Error cleaning up invalid session_settings: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return -1

if __name__ == "__main__":
    db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
    
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)
    
    logger.info(f"Cleaning up invalid session_settings in: {db_path}")
    result = cleanup_invalid_session_settings(db_path)
    
    if result >= 0:
        logger.info("✅ Cleanup completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Cleanup failed")
        sys.exit(1)








