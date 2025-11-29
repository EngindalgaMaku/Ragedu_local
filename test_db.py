import sqlite3
import os

def test_db_connection():
    db_path = "/app/data/analytics/sessions.db"
    
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if sessions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        if cursor.fetchone() is None:
            print("Error: 'sessions' table not found in the database.")
            return False
        
        # Try to query the sessions table
        cursor.execute("SELECT * FROM sessions LIMIT 1")
        print("Successfully connected to the database and queried the sessions table.")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_db_connection()
