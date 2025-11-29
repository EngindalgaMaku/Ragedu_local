import sqlite3
import sys

def get_table_schema(db_path, table_name):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    # Get foreign keys
    cursor.execute(f"PRAGMA foreign_key_list({table_name})")
    foreign_keys = cursor.fetchall()
    
    # Get indexes
    cursor.execute(f"PRAGMA index_list({table_name})")
    indexes = cursor.fetchall()
    
    # Get table creation SQL
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")  
    create_sql = cursor.fetchone()
    
    conn.close()
    
    return {
        'columns': [dict(column) for column in columns],
        'foreign_keys': [dict(fk) for fk in foreign_keys],
        'indexes': [dict(index) for index in indexes],
        'create_sql': create_sql[0] if create_sql else None
    }

if __name__ == "__main__":
    db_path = "/app/data/analytics/sessions.db"
    table_name = "question_topic_mapping"
    
    try:
        schema = get_table_schema(db_path, table_name)
        print(f"Schema for table '{table_name}':")
        print("\nColumns:")
        for col in schema['columns']:
            print(f"  {col['name']} ({col['type']}) {'NOT NULL' if not col['notnull'] == 0 else ''} {'PRIMARY KEY' if col['pk'] == 1 else ''}")
        
        if schema['foreign_keys']:
            print("\nForeign Keys:")
            for fk in schema['foreign_keys']:
                print(f"  {fk['from']} -> {fk['table']}({fk['to']}) ON DELETE {fk.get('on_delete', 'NO ACTION')}")
        
        if schema['indexes']:
            print("\nIndexes:")
            for idx in schema['indexes']:
                cursor = sqlite3.connect(db_path).cursor()
                cursor.execute(f"PRAGMA index_info({idx['name']})")
                idx_columns = [col[2] for col in cursor.fetchall()]
                print(f"  {idx['name']} ({', '.join(idx_columns)}) {'UNIQUE' if idx.get('unique', 0) == 1 else ''}")
        
        if schema['create_sql']:
            print("\nCREATE TABLE SQL:")
            print(schema['create_sql'])
    
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
        print("\nAvailable tables in the database:")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for table in cursor.fetchall():
            print(f"- {table[0]}")
