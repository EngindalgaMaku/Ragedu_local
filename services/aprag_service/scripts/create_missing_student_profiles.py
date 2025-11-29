#!/usr/bin/env python3
"""
Script to create missing student profiles for existing students
This ensures all students who have interactions have profiles

Usage:
    docker-compose exec aprag-service python /app/scripts/create_missing_student_profiles.py
"""

import sys
import os

# Add /app to path (Docker container path)
sys.path.insert(0, "/app")

from database.database import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_missing_profiles():
    """Create profiles for all students who have interactions but no profile"""
    
    # Use environment variable or default path
    db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
    logger.info(f"Using database: {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return
    
    db = DatabaseManager(db_path)
    
    try:
        # Find all unique user_id, session_id pairs from interactions that don't have profiles
        query = """
            SELECT DISTINCT i.user_id, i.session_id
            FROM student_interactions i
            LEFT JOIN student_profiles p 
                ON i.user_id = p.user_id AND i.session_id = p.session_id
            WHERE p.profile_id IS NULL
        """
        
        missing_profiles = db.execute_query(query)
        
        if not missing_profiles:
            logger.info("✅ All students already have profiles!")
            return
        
        logger.info(f"Found {len(missing_profiles)} missing profiles to create")
        
        created_count = 0
        for row in missing_profiles:
            user_id = row.get("user_id") if isinstance(row, dict) else (row[0] if isinstance(row, (tuple, list)) else None)
            session_id = row.get("session_id") if isinstance(row, dict) else (row[1] if isinstance(row, (tuple, list)) and len(row) > 1 else None)
            
            if not user_id or not session_id:
                logger.warning(f"Skipping invalid row: {row}")
                continue
            
            try:
                # Check if profile already exists (race condition check)
                existing = db.execute_query(
                    "SELECT profile_id FROM student_profiles WHERE user_id = ? AND session_id = ?",
                    (user_id, session_id)
                )
                
                if existing:
                    logger.debug(f"Profile already exists for user {user_id}, session {session_id}")
                    continue
                
                # Create default profile
                db.execute_insert(
                    """
                    INSERT INTO student_profiles
                    (user_id, session_id, average_understanding, average_satisfaction,
                     total_interactions, total_feedback_count, last_updated, created_at)
                    VALUES (?, ?, 3.0, 3.0, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (user_id, session_id)
                )
                
                created_count += 1
                logger.info(f"✅ Created profile for user {user_id}, session {session_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create profile for user {user_id}, session {session_id}: {e}")
        
        logger.info(f"✅ Successfully created {created_count} profiles out of {len(missing_profiles)} missing profiles")
        
    except Exception as e:
        logger.error(f"❌ Error creating missing profiles: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    create_missing_profiles()



