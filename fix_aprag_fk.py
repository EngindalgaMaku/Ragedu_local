import sqlite3
import sys

try:
    conn = sqlite3.connect('/app/data/aprag.db')
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_interactions'")
    if not cursor.fetchone():
        print('Table student_interactions does not exist yet, no migration needed')
        sys.exit(0)
    
    # Get table schema
    cursor.execute('PRAGMA table_info(student_interactions)')
    columns = cursor.fetchall()
    print(f'Current columns: {len(columns)}')
    
    # Create new table without foreign keys
    print('Creating new table without foreign keys...')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_interactions_new (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(255) NOT NULL,
            session_id VARCHAR(255) NOT NULL,
            query TEXT NOT NULL,
            original_response TEXT,
            personalized_response TEXT,
            processing_time_ms INTEGER,
            model_used VARCHAR(100),
            chain_type VARCHAR(50),
            sources TEXT,
            metadata TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Copy data
    print('Copying data...')
    cursor.execute('''
        INSERT INTO student_interactions_new 
        SELECT interaction_id, user_id, session_id, query, original_response, 
               personalized_response, processing_time_ms, model_used, chain_type, 
               sources, metadata, timestamp, created_at 
        FROM student_interactions
    ''')
    
    # Drop old table
    print('Dropping old table...')
    cursor.execute('DROP TABLE student_interactions')
    
    # Rename new table
    print('Renaming table...')
    cursor.execute('ALTER TABLE student_interactions_new RENAME TO student_interactions')
    
    # Recreate indexes
    print('Creating indexes...')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user_session ON student_interactions(user_id, session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON student_interactions(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_session_id ON student_interactions(session_id)')
    
    conn.commit()
    print('✅ Migration complete! Foreign key constraints removed.')
    conn.close()
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)












