#!/usr/bin/env python3
"""
Test User Creation Script for RAG Education Assistant
Creates a non-admin test user for debugging and testing purposes.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for module resolution
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# It seems the script is run from within the container, where /app is the root
# Let's try to adjust the path based on a potential /app structure
if '/app' not in str(project_root):
    # Assuming the script is run from the host, we need to find the 'src' dir
    # This is a bit of a guess, might need adjustment
    host_project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(host_project_root))
    # When running inside the container, the 'src' path is relative to /app
    # When running from host, it's relative to project root.
    # The import paths below (e.g., src.database.database) assume 'src' is in sys.path
    sys.path.insert(0, str(host_project_root / 'src'))


try:
    from database.database import DatabaseManager
    from database.models.user import User
    from database.models.role import Role
except ImportError:
    # Fallback for different execution context
    from src.database.database import DatabaseManager
    from src.database.models.user import User
    from src.database.models.role import Role


def create_test_user():
    """
    Creates a standard 'testuser' for testing login functionality.
    """
    print("=" * 60)
    print("RAG Education Assistant - Test User Creation")
    print("=" * 60)

    # In the Docker container, the DB_PATH is /app/data/rag_assistant.db
    # The volume 'database_data' is mounted at /app/data
    db_path = os.getenv('DATABASE_PATH', '/app/data/rag_assistant.db')
    
    # Check if the database file exists
    if not Path(db_path).is_file():
        print(f"✗ Database file not found at: {db_path}")
        print("Please ensure the service is running and the database has been initialized.")
        return False

    try:
        db_manager = DatabaseManager(db_path)
        user_model = User(db_manager)
        role_model = Role(db_manager)
        print(f"✓ Database initialized: {db_path}")
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        return False

    # Define test user details
    username = "testuser"
    email = "testuser@example.com"
    password = "password123"
    role_name = "user" # Standard user role

    # Ensure the 'user' role exists
    user_role = role_model.get_role_by_name(role_name)
    if not user_role:
        print(f"✗ Role '{role_name}' not found. Creating it...")
        try:
            # Define basic permissions for a standard user
            user_permissions = {
                'sessions': ['read'],
                'documents': ['read']
            }
            # Let's create a basic 'user' role if it doesn't exist
            user_role_id = role_model.create_role(
                name=role_name,
                description=f"Standard {role_name}",
                permissions=user_permissions
            )
            if user_role_id:
                user_role = role_model.get_role_by_id(user_role_id)
                print(f"✓ Role '{role_name}' created with ID: {user_role['id']}")
            else:
                print(f"✗ Failed to create role '{role_name}'.")
                return False
        except Exception as e:
            print(f"✗ Error creating role: {e}")
            return False

    # Check if user already exists
    existing_user = user_model.get_user_by_username(username)
    if existing_user:
        print(f"✓ User '{username}' already exists. Attempting to authenticate.")
        # Test authentication
        auth_result = user_model.authenticate_user(username, password)
        if auth_result:
            print("✓ Authentication test passed for existing user.")
        else:
            print("✗ Authentication test failed for existing user. Password might be different.")
        return True

    # Create the test user
    try:
        print(f"\nCreating new user '{username}'...")
        user_id = user_model.create_user(
            username=username,
            email=email,
            password=password,
            role_id=user_role['id'],
            first_name="Test",
            last_name="User",
            is_active=True
        )

        if user_id:
            print(f"✓ Test user created successfully!")
            print(f"  - Username: {username}")
            print(f"  - Password: {password}")

            # Test authentication
            print("\n🔐 Testing authentication...")
            auth_result = user_model.authenticate_user(username, password)
            if auth_result:
                print("✓ Authentication test passed for new user.")
            else:
                print("✗ Authentication test failed for new user.")
            return True
        else:
            print("✗ Failed to create test user.")
            return False

    except Exception as e:
        print(f"✗ Error creating test user: {e}")
        return False

if __name__ == "__main__":
    create_test_user()