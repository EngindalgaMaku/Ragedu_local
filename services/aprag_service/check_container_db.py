import sqlite3

def check_container_db():
    try:
        db_path = '/app/data/rag_assistant.db'
        db = sqlite3.connect(db_path)
        cursor = db.cursor()
        
        # Tabloları listele
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = [t[0] for t in cursor.fetchall()]
        print(f'Container DB Tables: {tables}')
        
        # student_interactions kontrol et
        if 'student_interactions' in tables:
            cursor.execute('PRAGMA table_info(student_interactions)')
            columns = [col[1] for col in cursor.fetchall()]
            print(f'student_interactions columns: {columns}')
            
        # Kritik tabloları kontrol et
        critical = ['topic_progress', 'question_topic_mapping', 'feature_flags', 'course_topics']
        for table in critical:
            exists = table in tables
            print(f'{table}: {"EXISTS" if exists else "MISSING"}')
            
        db.close()
        print('Container DB check completed')
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    check_container_db()