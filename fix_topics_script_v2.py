import re

# Path to the topics.py file in the container
topics_file = "/app/api/topics.py"

# Read the file content
with open(topics_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the database connection to use the correct path
updated_content = re.sub(
    r'def get_db\(\).*?return DatabaseManager\(.*?\)',
    'def get_db():\n    return DatabaseManager("/app/data/analytics/sessions.db")',
    content,
    flags=re.DOTALL
)

# Write the updated content back to the file
with open(topics_file, 'w', encoding='utf-8') as f:
    f.write(updated_content)

print("Successfully updated the database path in topics.py")
