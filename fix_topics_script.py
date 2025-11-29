import re

# Path to the topics.py file in the container
topics_file = "/app/api/topics.py"

# Read the file content
with open(topics_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the table name from session_settings to sessions
updated_content = re.sub(
    r"SELECT rag_settings FROM session_settings WHERE session_id = \?",
    "SELECT rag_settings FROM sessions WHERE session_id = ?",
    content
)

# Write the updated content back to the file
with open(topics_file, 'w', encoding='utf-8') as f:
    f.write(updated_content)

print("Successfully updated topics.py to check sessions table instead of session_settings table")
