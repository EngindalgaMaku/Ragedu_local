import sqlite3

db = sqlite3.connect('/app/data/rag_assistant.db')
db.execute('''
CREATE TABLE IF NOT EXISTS course_topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    topic_title TEXT NOT NULL,
    parent_topic_id INTEGER,
    topic_order INTEGER DEFAULT 0,
    description TEXT,
    keywords TEXT,
    estimated_difficulty TEXT DEFAULT 'intermediate',
    prerequisites TEXT,
    related_chunk_ids TEXT,
    extraction_method TEXT DEFAULT 'llm_analysis',
    extraction_confidence REAL DEFAULT 0.75,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_topic_id) REFERENCES course_topics(topic_id)
)''')
db.commit()
print('Container course_topics table created successfully')
db.close()