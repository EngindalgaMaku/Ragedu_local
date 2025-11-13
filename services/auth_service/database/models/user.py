"""
User model for RAG Education Assistant
Handles user data management and authentication
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)


class User:
    """
    User model for managing user accounts and authentication
    """
    
    def __init__(self, db_manager):
        """
        Initialize User model
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    def create_user(self, username: str, email: str, password: str, 
                   role_id: int, first_name: str, last_name: str,
                   is_active: bool = True) -> Optional[int]:
        """
        Create a new user
        
        Args:
            username: Unique username
            email: Unique email address
            password: Plain text password (will be hashed)
            role_id: Role ID for the user
            first_name: User's first name
            last_name: User's last name
            is_active: Whether user account is active
            
        Returns:
            User ID if successful, None if failed
        """
        try:
            # Check if username or email already exists
            if self.get_user_by_username(username):
                logger.warning(f"Username {username} already exists")
                return None
                
            if self.get_user_by_email(email):
                logger.warning(f"Email {email} already exists")
                return None
            
            # Hash password
            password_hash = self.db.hash_password(password)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO users (username, email, password_hash, role_id, 
                                     first_name, last_name, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (username, email, password_hash, role_id, 
                      first_name, last_name, is_active))
                
                conn.commit()
                user_id = cursor.lastrowid
                
                logger.info(f"User {username} created with ID {user_id}")
                return user_id
                
        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User data dictionary or None
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT u.*, r.name as role_name, r.permissions
                    FROM users u
                    JOIN roles r ON u.role_id = r.id
                    WHERE u.id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    user_data = dict(row)
                    # Parse permissions JSON
                    user_data['permissions'] = json.loads(user_data['permissions'])
                    return user_data
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username
        
        Args:
            username: Username
            
        Returns:
            User data dictionary or None
        """
        return self.db.get_user_by_username(username)
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email
        
        Args:
            email: Email address
            
        Returns:
            User data dictionary or None
        """
        return self.db.get_user_by_email(email)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username/email and password
        
        Args:
            username: Username or email
            password: Plain text password
            
        Returns:
            User data if authentication successful, None otherwise
        """
        try:
            # Try to get user by username first, then by email
            user = self.get_user_by_username(username)
            if not user:
                user = self.get_user_by_email(username)
            
            if not user:
                logger.warning(f"User not found: {username}")
                return None
            
            if not user['is_active']:
                logger.warning(f"User account disabled: {username}")
                return None
            
            # Verify password
            if self.db.verify_password(password, user['password_hash']):
                # Update last login
                self.db.update_user_last_login(user['id'])
                
                # Remove password hash from returned data
                user_data = user.copy()
                del user_data['password_hash']
                
                # Parse permissions if it's a string
                if isinstance(user_data.get('permissions'), str):
                    user_data['permissions'] = json.loads(user_data['permissions'])
                
                logger.info(f"User {username} authenticated successfully")
                return user_data
            else:
                logger.warning(f"Invalid password for user: {username}")
                return None
                
        except Exception as e:
            logger.error(f"Authentication failed for {username}: {e}")
            return None
    
    def update_user(self, user_id: int, **kwargs) -> bool:
        """
        Update user information
        
        Args:
            user_id: User ID
            **kwargs: Fields to update (username, email, first_name, last_name, is_active, role_id)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter allowed fields
            allowed_fields = ['username', 'email', 'first_name', 'last_name', 'is_active', 'role_id']
            updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
            
            if not updates:
                return True  # No updates needed
            
            # Check if username is being updated and if it's already taken
            if 'username' in updates:
                new_username = updates['username'].lower() if updates['username'] else None
                if new_username:
                    existing_user = self.get_user_by_username(new_username)
                    if existing_user and existing_user['id'] != user_id:
                        logger.warning(f"Username {new_username} is already taken by user {existing_user['id']}")
                        raise ValueError(f"Username '{new_username}' is already taken")
                    # Update username to lowercase
                    updates['username'] = new_username
            
            # Check if email is being updated and if it's already taken
            if 'email' in updates and updates['email']:
                existing_user = self.get_user_by_email(updates['email'])
                if existing_user and existing_user['id'] != user_id:
                    logger.warning(f"Email {updates['email']} is already taken by user {existing_user['id']}")
                    raise ValueError(f"Email '{updates['email']}' is already taken")
            
            # Build dynamic update query
            set_clauses = [f"{field} = ?" for field in updates.keys()]
            values = list(updates.values())
            values.append(user_id)  # For WHERE clause
            
            query = f"""
                UPDATE users 
                SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            
            with self.db.get_connection() as conn:
                result = conn.execute(query, values)
                conn.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"User {user_id} updated successfully")
                else:
                    logger.warning(f"No user found with ID {user_id}")
                
                return success
                
        except ValueError as ve:
            # Re-raise validation errors
            raise ve
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            return False
    
    def update_password(self, user_id: int, new_password: str) -> bool:
        """
        Update user password (admin function - no old password required)
        
        Args:
            user_id: User ID
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Hash new password
            password_hash = self.db.hash_password(new_password)
            
            with self.db.get_connection() as conn:
                result = conn.execute("""
                    UPDATE users 
                    SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (password_hash, user_id))
                conn.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"Password updated for user {user_id}")
                else:
                    logger.warning(f"No user found with ID {user_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to update password for user {user_id}: {e}")
            return False
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> bool:
        """
        Change user password (requires old password verification)
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current user data
            user = self.get_user_by_id(user_id)
            if not user:
                logger.warning(f"User {user_id} not found")
                return False
            
            # Verify old password
            if not self.db.verify_password(old_password, user['password_hash']):
                logger.warning(f"Invalid old password for user {user_id}")
                return False
            
            # Hash new password
            new_password_hash = self.db.hash_password(new_password)
            
            with self.db.get_connection() as conn:
                result = conn.execute("""
                    UPDATE users 
                    SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_password_hash, user_id))
                
                conn.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"Password changed for user {user_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to change password for user {user_id}: {e}")
            return False
    
    def reset_password(self, user_id: int, new_password: str) -> bool:
        """
        Reset user password (admin function)
        
        Args:
            user_id: User ID
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Hash new password
            new_password_hash = self.db.hash_password(new_password)
            
            with self.db.get_connection() as conn:
                result = conn.execute("""
                    UPDATE users 
                    SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_password_hash, user_id))
                
                conn.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"Password reset for user {user_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to reset password for user {user_id}: {e}")
            return False
    
    def deactivate_user(self, user_id: int) -> bool:
        """
        Deactivate user account
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_user(user_id, is_active=False)
    
    def activate_user(self, user_id: int) -> bool:
        """
        Activate user account
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_user(user_id, is_active=True)
    
    def delete_user(self, user_id: int) -> bool:
        """
        Delete user (hard delete)
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db.get_connection() as conn:
                # Delete user sessions first (foreign key constraint)
                conn.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
                
                # Delete user
                result = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
                conn.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"User {user_id} deleted successfully")
                else:
                    logger.warning(f"No user found with ID {user_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    def list_users(self, role_name: str = None, is_active: bool = None, 
                   offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List users with optional filtering
        
        Args:
            role_name: Filter by role name
            is_active: Filter by active status
            offset: Pagination offset
            limit: Maximum number of results
            
        Returns:
            List of user dictionaries
        """
        try:
            # Build query with filters
            query = """
                SELECT u.id, u.username, u.email, u.first_name, u.last_name, 
                       u.is_active, u.created_at, u.updated_at, u.last_login,
                       r.name as role_name
                FROM users u
                JOIN roles r ON u.role_id = r.id
                WHERE 1=1
            """
            params = []
            
            if role_name:
                query += " AND r.name = ?"
                params.append(role_name)
            
            if is_active is not None:
                query += " AND u.is_active = ?"
                params.append(is_active)
            
            query += " ORDER BY u.created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(query, params)
                
                users = []
                for row in cursor.fetchall():
                    users.append(dict(row))
                
                return users
                
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []
    
    def get_user_count(self, role_name: str = None, is_active: bool = None) -> int:
        """
        Get total user count with optional filtering
        
        Args:
            role_name: Filter by role name
            is_active: Filter by active status
            
        Returns:
            Total user count
        """
        try:
            query = """
                SELECT COUNT(*) as count
                FROM users u
                JOIN roles r ON u.role_id = r.id
                WHERE 1=1
            """
            params = []
            
            if role_name:
                query += " AND r.name = ?"
                params.append(role_name)
            
            if is_active is not None:
                query += " AND u.is_active = ?"
                params.append(is_active)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(query, params)
                row = cursor.fetchone()
                return row['count'] if row else 0
                
        except Exception as e:
            logger.error(f"Failed to get user count: {e}")
            return 0
    
    def has_permission(self, user_id: int, resource: str, action: str) -> bool:
        """
        Check if user has specific permission
        
        Args:
            user_id: User ID
            resource: Resource name (e.g., 'users', 'documents')
            action: Action name (e.g., 'create', 'read', 'update', 'delete')
            
        Returns:
            True if user has permission, False otherwise
        """
        try:
            user = self.get_user_by_id(user_id)
            if not user or not user['is_active']:
                return False
            
            permissions = user.get('permissions', {})
            resource_permissions = permissions.get(resource, [])
            
            return action in resource_permissions
            
        except Exception as e:
            logger.error(f"Failed to check permission for user {user_id}: {e}")
            return False