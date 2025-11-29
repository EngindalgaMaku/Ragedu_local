import sqlite3
import os
import json
from pathlib import Path

def fix_rag_settings_column():
    # Define the database path
    db_path = Path("c:/Users/Engin/Documents/cursor projects/rag3_for_local/rag3_for_local/services/aprag_service/data/aprag.db")
    
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if the rag_settings column exists
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'rag_settings' not in columns:
            print("Adding missing 'rag_settings' column to 'sessions' table...")
            try:
                # Add the column with a default empty JSON object
                cursor.execute("""
                    ALTER TABLE sessions 
                    ADD COLUMN rag_settings TEXT DEFAULT '{}'
                """)
                conn.commit()
                print("Successfully added 'rag_settings' column to 'sessions' table.")
                return True
            except Exception as e:
                print(f"Error adding column: {e}")
                conn.rollback()
                return False
        else:
            print("'rag_settings' column already exists in 'sessions' table.")
            return True
            
    except Exception as e:
        print(f"Database error: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("Checking database schema...")
    if fix_rag_settings_column():
        print("Database check completed successfully.")
    else:
        print("There was an issue updating the database schema.")
        exit(1)
