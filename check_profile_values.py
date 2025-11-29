#!/usr/bin/env python3
"""
Check student profile values to see if understanding and satisfaction are the same
"""
import sqlite3
import os

db_path = os.getenv("APRAG_DB_PATH", "data/rag_assistant.db")
if not os.path.exists(db_path):
    # Try Docker path
    db_path = "/app/data/rag_assistant.db"

print(f"Connecting to: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check student profiles
    cursor.execute("""
        SELECT user_id, session_id, 
               average_understanding, 
               average_satisfaction,
               total_feedback_count
        FROM student_profiles
        WHERE average_understanding IS NOT NULL 
           OR average_satisfaction IS NOT NULL
        ORDER BY last_updated DESC
        LIMIT 20
    """)

    profiles = cursor.fetchall()
    print(f"\nFound {len(profiles)} profiles with data:")
    print("=" * 80)

    same_count = 0
    for profile in profiles:
        user_id = profile["user_id"]
        session_id = profile["session_id"]
        understanding = profile["average_understanding"]
        satisfaction = profile["average_satisfaction"]
        feedback_count = profile["total_feedback_count"]
        
        print(f"User: {user_id}, Session: {session_id}")
        print(f"  Understanding: {understanding}")
        print(f"  Satisfaction: {satisfaction}")
        print(f"  Feedback Count: {feedback_count}")
        
        # Check if they're the same
        if understanding is not None and satisfaction is not None:
            if abs(float(understanding) - float(satisfaction)) < 0.01:
                print("  ⚠️  WARNING: Understanding and Satisfaction are the SAME!")
                same_count += 1
        print()

    print(f"\n{'=' * 80}")
    print(f"Total profiles with same values: {same_count}/{len(profiles)}")
    
    # Check recent feedback entries
    print("\n" + "=" * 80)
    print("Checking recent feedback entries...")
    cursor.execute("""
        SELECT interaction_id, user_id, session_id,
               emoji_feedback, feedback_score,
               feedback_dimensions
        FROM student_interactions
        WHERE emoji_feedback IS NOT NULL OR feedback_dimensions IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 10
    """)
    
    interactions = cursor.fetchall()
    print(f"Found {len(interactions)} recent feedback entries:")
    for interaction in interactions:
        print(f"  Interaction {interaction['interaction_id']}: "
              f"emoji={interaction['emoji_feedback']}, "
              f"score={interaction['feedback_score']}, "
              f"dimensions={interaction['feedback_dimensions']}")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()



