#!/usr/bin/env python3
"""
Script to create missing student profiles for existing students
This ensures all students who have interactions have profiles

Usage:
    docker-compose exec aprag-service python /app/scripts/create_missing_student_profiles.py
"""

import sys
import os

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
# If running in Docker, use /app, otherwise use project root
if os.path.exists("/app"):
    sys.path.insert(0, "/app")
else:
    # Running locally - add rag3_for_local to path
    project_root = os.path.join(script_dir, '..')
    if os.path.basename(project_root) != 'rag3_for_local':
        project_root = os.path.join(project_root, 'rag3_for_local')
    sys.path.insert(0, project_root)

try:
    from services.aprag_service.database.database import DatabaseManager
except ImportError:
    # Try alternative import path
    try:
        from database.database import DatabaseManager
    except ImportError:
        print("ERROR: Could not import DatabaseManager")
        print(f"Python path: {sys.path}")
        sys.exit(1)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_missing_profiles():
    """Create profiles for all students who have interactions but no profile"""
    
    # Use environment variable or default path
    db_path = os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db")
    if not os.path.exists(db_path):
        # Try alternative paths
        alt_paths = [
            "data/rag_assistant.db",
            "/app/data/rag_assistant.db",
            os.path.join(script_dir, "..", "data", "rag_assistant.db"),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                db_path = alt_path
                break
        else:
            logger.error(f"Database not found at {db_path} or alternative paths")
            logger.info("Available paths tried:")
            for alt_path in alt_paths:
                logger.info(f"  - {alt_path}")
            return
    
    logger.info(f"Using database: {db_path}")
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
            user_id = row.get("user_id") or row[0] if isinstance(row, tuple) else None
            session_id = row.get("session_id") or row[1] if isinstance(row, tuple) else None
            
            if not user_id or not session_id:
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
        raise


if __name__ == "__main__":
    create_missing_profiles()

