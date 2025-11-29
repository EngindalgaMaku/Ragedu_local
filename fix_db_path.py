import re

# Path to the session_manager.py file in the container
session_manager_file = "/app/src/services/session_manager.py"

# Read the file content
with open(session_manager_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Update the default db_path to use absolute path
updated_content = re.sub(
    r'def __init__\(self, db_path: str = "data/analytics/sessions.db"\):',
    'def __init__(self, db_path: str = "/app/data/analytics/sessions.db"):',
    content
)

# Write the updated content back to the file
with open(session_manager_file, 'w', encoding='utf-8') as f:
    f.write(updated_content)

print("Successfully updated the database path in session_manager.py")
