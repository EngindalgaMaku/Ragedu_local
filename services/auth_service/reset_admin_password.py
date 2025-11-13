#!/usr/bin/env python3
"""
Admin Password Reset Script for RAG Education Assistant
Forcefully resets the admin user's password to a known value.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

try:
    from database.database import DatabaseManager
    from database.models.user import User
except ImportError:
    print("Error: Could not import database modules. Ensure paths are correct.")
    sys.exit(1)

def force_reset_admin_password():
    """
    Finds the 'admin' user and resets their password to 'admin123'.
    """
    print("=" * 60)
    print("RAG Education Assistant - Force Admin Password Reset")
    print("=" * 60)

    db_path = os.getenv('DATABASE_PATH', '/app/data/rag_assistant.db')
    
    if not Path(db_path).is_file():
        print(f"✗ Database file not found at: {db_path}")
        return False

    try:
        db_manager = DatabaseManager(db_path)
        user_model = User(db_manager)
        print(f"✓ Database initialized: {db_path}")
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        return False

    admin_username = "admin"
    new_password = "admin123"

    try:
        admin_user = user_model.get_user_by_username(admin_username)

        if not admin_user:
            print(f"✗ Admin user '{admin_username}' not found in the database.")
            print("   Please ensure the database has been initialized correctly.")
            return False

        print(f"✓ Found admin user: {admin_username} (ID: {admin_user['id']})")

        # Forcefully reset the password
        success = user_model.reset_password(admin_user['id'], new_password)

        if success:
            print(f"✓ Admin password has been forcefully reset to: '{new_password}'")
            
            # Verify authentication with the new password
            print("\n🔐 Verifying new password...")
            auth_result = user_model.authenticate_user(admin_username, new_password)
            if auth_result:
                print("✓ Authentication test PASSED with the new password.")
            else:
                print("✗ Authentication test FAILED with the new password. Check hashing logic.")
            return True
        else:
            print("✗ Failed to reset admin password in the database.")
            return False

    except Exception as e:
        print(f"✗ An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    force_reset_admin_password()