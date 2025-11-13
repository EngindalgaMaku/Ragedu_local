#!/usr/bin/env python3
"""
Admin User Creation Script for RAG Education Assistant
Creates an admin user with secure password and proper role assignment
"""

import os
import sys
import getpass
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.database import DatabaseManager
from src.database.models.user import User
from src.database.models.role import Role


def create_admin_user():
    """
    Interactive script to create an admin user
    """
    print("=" * 60)
    print("RAG Education Assistant - Admin User Creation")
    print("=" * 60)
    
    # Initialize database
    try:
        db_path = os.getenv('DB_PATH', 'data/rag_assistant.db')
        db_manager = DatabaseManager(db_path)
        user_model = User(db_manager)
        role_model = Role(db_manager)
        
        print(f"✓ Database initialized: {db_path}")
        
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        return False
    
    # Check if admin role exists
    admin_role = role_model.get_role_by_name('admin')
    if not admin_role:
        print("✗ Admin role not found. Please run migrations first.")
        return False
    
    print(f"✓ Admin role found (ID: {admin_role['id']})")
    
    # Get admin user details
    print("\nEnter admin user details:")
    print("-" * 30)
    
    # Username
    while True:
        username = input("Username: ").strip()
        if not username:
            print("Username cannot be empty!")
            continue
        
        # Check if username already exists
        existing_user = user_model.get_user_by_username(username)
        if existing_user:
            overwrite = input(f"User '{username}' already exists. Overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                continue
            else:
                # Delete existing user
                user_model.delete_user(existing_user['id'])
                print(f"✓ Existing user '{username}' deleted")
        break
    
    # Email
    while True:
        email = input("Email: ").strip()
        if not email or '@' not in email:
            print("Please enter a valid email address!")
            continue
        
        # Check if email already exists
        existing_user = user_model.get_user_by_email(email)
        if existing_user and existing_user['username'] != username:
            print(f"Email '{email}' is already in use by another user!")
            continue
        break
    
    # Password
    while True:
        password = getpass.getpass("Password: ").strip()
        if len(password) < 8:
            print("Password must be at least 8 characters long!")
            continue
        
        confirm_password = getpass.getpass("Confirm Password: ").strip()
        if password != confirm_password:
            print("Passwords do not match!")
            continue
        break
    
    # Names
    first_name = input("First Name: ").strip() or "Admin"
    last_name = input("Last Name: ").strip() or "User"
    
    # Create admin user
    try:
        user_id = user_model.create_user(
            username=username,
            email=email,
            password=password,
            role_id=admin_role['id'],
            first_name=first_name,
            last_name=last_name,
            is_active=True
        )
        
        if user_id:
            print(f"\n✓ Admin user created successfully!")
            print(f"  User ID: {user_id}")
            print(f"  Username: {username}")
            print(f"  Email: {email}")
            print(f"  Role: admin")
            print(f"  Status: active")
            
            # Test authentication
            print("\n🔐 Testing authentication...")
            auth_result = user_model.authenticate_user(username, password)
            if auth_result:
                print("✓ Authentication test passed")
            else:
                print("✗ Authentication test failed")
            
            return True
        else:
            print("✗ Failed to create admin user")
            return False
            
    except Exception as e:
        print(f"✗ Error creating admin user: {e}")
        return False


def reset_admin_password():
    """
    Reset admin user password
    """
    print("=" * 60)
    print("RAG Education Assistant - Reset Admin Password")
    print("=" * 60)
    
    # Initialize database
    try:
        db_path = os.getenv('DB_PATH', 'data/rag_assistant.db')
        db_manager = DatabaseManager(db_path)
        user_model = User(db_manager)
        
        print(f"✓ Database initialized: {db_path}")
        
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        return False
    
    # Find admin user
    username = input("Enter admin username: ").strip()
    admin_user = user_model.get_user_by_username(username)
    
    if not admin_user:
        print(f"✗ User '{username}' not found")
        return False
    
    if admin_user['role_name'] != 'admin':
        print(f"✗ User '{username}' is not an admin")
        return False
    
    print(f"✓ Found admin user: {username}")
    
    # Get new password
    while True:
        new_password = getpass.getpass("New Password: ").strip()
        if len(new_password) < 8:
            print("Password must be at least 8 characters long!")
            continue
        
        confirm_password = getpass.getpass("Confirm New Password: ").strip()
        if new_password != confirm_password:
            print("Passwords do not match!")
            continue
        break
    
    # Reset password
    try:
        success = user_model.reset_password(admin_user['id'], new_password)
        if success:
            print("✓ Admin password reset successfully!")
            
            # Test authentication
            print("\n🔐 Testing authentication...")
            auth_result = user_model.authenticate_user(username, new_password)
            if auth_result:
                print("✓ Authentication test passed")
            else:
                print("✗ Authentication test failed")
            
            return True
        else:
            print("✗ Failed to reset password")
            return False
            
    except Exception as e:
        print(f"✗ Error resetting password: {e}")
        return False


def list_admin_users():
    """
    List all admin users
    """
    print("=" * 60)
    print("RAG Education Assistant - Admin Users")
    print("=" * 60)
    
    try:
        db_path = os.getenv('DB_PATH', 'data/rag_assistant.db')
        db_manager = DatabaseManager(db_path)
        user_model = User(db_manager)
        
        admin_users = user_model.list_users(role_name='admin')
        
        if not admin_users:
            print("No admin users found")
            return
        
        print(f"Found {len(admin_users)} admin user(s):")
        print("-" * 60)
        
        for user in admin_users:
            status = "✓ Active" if user['is_active'] else "✗ Inactive"
            print(f"ID: {user['id']}")
            print(f"Username: {user['username']}")
            print(f"Email: {user['email']}")
            print(f"Name: {user['first_name']} {user['last_name']}")
            print(f"Status: {status}")
            print(f"Created: {user['created_at']}")
            if user['last_login']:
                print(f"Last Login: {user['last_login']}")
            print("-" * 60)
            
    except Exception as e:
        print(f"✗ Error listing admin users: {e}")


def main():
    """
    Main function with menu system
    """
    while True:
        print("\nRAG Education Assistant - Admin Management")
        print("=" * 50)
        print("1. Create Admin User")
        print("2. Reset Admin Password")
        print("3. List Admin Users")
        print("4. Exit")
        print("=" * 50)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            create_admin_user()
        elif choice == '2':
            reset_admin_password()
        elif choice == '3':
            list_admin_users()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1, 2, 3, or 4.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")