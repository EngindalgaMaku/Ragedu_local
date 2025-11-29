import os
import sqlite3
import shutil
from datetime import datetime

def backup_database(db_path):
    """Create a timestamped backup of the database"""
    if not os.path.exists(db_path):
        print(f"⚠️ Database file not found at {db_path}")
        return None
        
    backup_dir = os.path.join(os.path.dirname(db_path), 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f'rag_assistant_{timestamp}.db')
    
    shutil.copy2(db_path, backup_path)
    print(f"✅ Database backed up to {backup_path}")
    return backup_path

def reset_database():
    # Define the database path - adjust this if your path is different
    db_path = os.path.join('data', 'rag_assistant.db')
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create backup
    print("🔄 Creating database backup...")
    backup_path = backup_database(db_path)
    
    try:
        # Connect to the database (this will create it if it doesn't exist)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        print("\n🔄 Applying migrations...")
        
        # 1. Create topic_qa_pairs table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS topic_qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 2. Create topic_progress table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS topic_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT,
            understanding_level INTEGER DEFAULT 0,
            confidence_level INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            mastery_score FLOAT DEFAULT 0.0,
            UNIQUE(topic_id, user_id, session_id) ON CONFLICT REPLACE
        )
        """)
        
        # 3. Add QA embeddings columns if they don't exist
        columns_to_add = [
            ("question_embedding", "TEXT"),
            ("embedding_model", "VARCHAR(100)"),
            ("embedding_dim", "INTEGER"),
            ("embedding_updated_at", "TIMESTAMP")
        ]
        
        cursor.execute("PRAGMA table_info(topic_qa_pairs)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        for col_name, col_type in columns_to_add:
            if col_name not in existing_columns:
                try:
                    cursor.execute(f"""
                        ALTER TABLE topic_qa_pairs 
                        ADD COLUMN {col_name} {col_type}
                    """)
                    print(f"✅ Added column {col_name} to topic_qa_pairs")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        print(f"ℹ️ Column {col_name} already exists")
                    else:
                        raise
        
        # 4. Create indexes if they don't exist
        indexes = [
            ("idx_qa_pairs_topic_active", 
             "CREATE INDEX IF NOT EXISTS idx_qa_pairs_topic_active ON topic_qa_pairs(topic_id, is_active)"),
            ("idx_qa_pairs_embedding_model", 
             "CREATE INDEX IF NOT EXISTS idx_qa_pairs_embedding_model ON topic_qa_pairs(embedding_model)")
        ]
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        existing_indexes = [row[0] for row in cursor.fetchall()]
        
        for idx_name, idx_sql in indexes:
            if idx_name not in existing_indexes:
                cursor.execute(idx_sql)
                print(f"✅ Created index {idx_name}")
            else:
                print(f"ℹ️ Index {idx_name} already exists")
        
        # 5. Verify schema
        print("\n🔍 Verifying database schema...")
        
        # Check topic_qa_pairs columns
        cursor.execute("PRAGMA table_info(topic_qa_pairs)")
        qa_columns = [col[1] for col in cursor.fetchall()]
        print("📋 topic_qa_pairs columns:", ", ".join(qa_columns))
        
        # Check topic_progress columns
        cursor.execute("PRAGMA table_info(topic_progress)")
        progress_columns = [col[1] for col in cursor.fetchall()]
        print("📋 topic_progress columns:", ", ".join(progress_columns))
        
        # Check indexes
        cursor.execute("""
            SELECT name, sql 
            FROM sqlite_master 
            WHERE type='index' 
            AND name LIKE 'idx_qa_pairs%'
        """)
        print("\n🔍 QA Pairs Indexes:")
        for idx in cursor.fetchall():
            print(f"  - {idx[0]}: {idx[1]}")
        
        conn.commit()
        print("\n✅ Database reset and migrations applied successfully!")
        
    except Exception as e:
        print(f"\n❌ Error resetting database: {e}")
        if backup_path and os.path.exists(backup_path):
            print(f"⚠️  To restore from backup, run:")
            print(f"   copy " + backup_path.replace("\\", "\\") + " " + db_path.replace("\\", "\\"))
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("🚀 Starting database reset...")
    print("📂 Working directory:", os.getcwd())
    reset_database()
    print("\n🔄 Database reset complete. Press any key to exit...")
    input()
