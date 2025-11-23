from database.database import DatabaseManager
import os

db = DatabaseManager(os.getenv("APRAG_DB_PATH", "/app/data/rag_assistant.db"))
result = db.execute_query("PRAGMA table_info(student_interactions)")
print("student_interactions columns:")
for r in result:
    print(f"  - {r['name']}: {r['type']}")












