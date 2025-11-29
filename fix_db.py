import sqlite3
import os

def fix_database():
    db_path = "/app/data/rag_assistant.db"
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if rag_settings column exists
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'rag_settings' not in columns:
            print("Adding missing 'rag_settings' column to 'sessions' table...")
            cursor.execute("""
                ALTER TABLE sessions 
                ADD COLUMN rag_settings TEXT DEFAULT '{}'
            """)
            conn.commit()
            print("Successfully added 'rag_settings' column.")
        else:
            print("'rag_settings' column already exists.")
            
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("Checking database schema...")
    if fix_database():
        print("Database check completed successfully.")
    else:
        print("There was an issue updating the database schema.")
        exit(1)
