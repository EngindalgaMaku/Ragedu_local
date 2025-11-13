"""
Role model for RAG Education Assistant
Handles role and permission management
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)


class Role:
    """
    Role model for managing user roles and permissions
    """
    
    def __init__(self, db_manager):
        """
        Initialize Role model
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    def create_role(self, name: str, description: str, permissions: Dict[str, List[str]]) -> Optional[int]:
        """
        Create a new role
        
        Args:
            name: Unique role name
            description: Role description
            permissions: Dictionary of permissions {resource: [actions]}
            
        Returns:
            Role ID if successful, None if failed
        """
        try:
            # Check if role name already exists
            if self.get_role_by_name(name):
                logger.warning(f"Role {name} already exists")
                return None
            
            # Convert permissions to JSON string
            permissions_json = json.dumps(permissions)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO roles (name, description, permissions)
                    VALUES (?, ?, ?)
                """, (name, description, permissions_json))
                
                conn.commit()
                role_id = cursor.lastrowid
                
                logger.info(f"Role {name} created with ID {role_id}")
                return role_id
                
        except Exception as e:
            logger.error(f"Failed to create role {name}: {e}")
            return None
    
    def get_role_by_id(self, role_id: int) -> Optional[Dict[str, Any]]:
        """
        Get role by ID
        
        Args:
            role_id: Role ID
            
        Returns:
            Role data dictionary or None
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM roles WHERE id = ?
                """, (role_id,))
                
                row = cursor.fetchone()
                if row:
                    role_data = dict(row)
                    # Parse permissions JSON
                    role_data['permissions'] = json.loads(role_data['permissions'])
                    return role_data
                return None
                
        except Exception as e:
            logger.error(f"Failed to get role by ID {role_id}: {e}")
            return None
    
    def get_role_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get role by name
        
        Args:
            name: Role name
            
        Returns:
            Role data dictionary or None
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM roles WHERE name = ?
                """, (name,))
                
                row = cursor.fetchone()
                if row:
                    role_data = dict(row)
                    # Parse permissions JSON
                    role_data['permissions'] = json.loads(role_data['permissions'])
                    return role_data
                return None
                
        except Exception as e:
            logger.error(f"Failed to get role by name {name}: {e}")
            return None
    
    def update_role(self, role_id: int, **kwargs) -> bool:
        """
        Update role information
        
        Args:
            role_id: Role ID
            **kwargs: Fields to update (name, description, permissions)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter allowed fields
            allowed_fields = ['name', 'description', 'permissions']
            updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
            
            if not updates:
                return True  # No updates needed
            
            # Convert permissions to JSON if provided
            if 'permissions' in updates:
                updates['permissions'] = json.dumps(updates['permissions'])
            
            # Build dynamic update query
            set_clauses = [f"{field} = ?" for field in updates.keys()]
            values = list(updates.values())
            values.append(role_id)  # For WHERE clause
            
            query = f"""
                UPDATE roles 
                SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            
            with self.db.get_connection() as conn:
                result = conn.execute(query, values)
                conn.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"Role {role_id} updated successfully")
                else:
                    logger.warning(f"No role found with ID {role_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to update role {role_id}: {e}")
            return False
    
    def delete_role(self, role_id: int) -> bool:
        """
        Delete role (only if no users are assigned to it)
        
        Args:
            role_id: Role ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db.get_connection() as conn:
                # Check if any users have this role
                cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM users WHERE role_id = ?
                """, (role_id,))
                
                row = cursor.fetchone()
                if row and row['count'] > 0:
                    logger.warning(f"Cannot delete role {role_id}: {row['count']} users assigned")
                    return False
                
                # Delete role
                result = conn.execute("DELETE FROM roles WHERE id = ?", (role_id,))
                conn.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"Role {role_id} deleted successfully")
                else:
                    logger.warning(f"No role found with ID {role_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to delete role {role_id}: {e}")
            return False
    
    def list_roles(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all roles
        
        Args:
            offset: Pagination offset
            limit: Maximum number of results
            
        Returns:
            List of role dictionaries
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM roles
                    ORDER BY created_at ASC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                roles = []
                for row in cursor.fetchall():
                    role_data = dict(row)
                    # Parse permissions JSON
                    role_data['permissions'] = json.loads(role_data['permissions'])
                    roles.append(role_data)
                
                return roles
                
        except Exception as e:
            logger.error(f"Failed to list roles: {e}")
            return []
    
    def get_role_count(self) -> int:
        """
        Get total role count
        
        Returns:
            Total role count
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) as count FROM roles")
                row = cursor.fetchone()
                return row['count'] if row else 0
                
        except Exception as e:
            logger.error(f"Failed to get role count: {e}")
            return 0
    
    def add_permission(self, role_id: int, resource: str, action: str) -> bool:
        """
        Add permission to role
        
        Args:
            role_id: Role ID
            resource: Resource name
            action: Action name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            role = self.get_role_by_id(role_id)
            if not role:
                logger.warning(f"Role {role_id} not found")
                return False
            
            permissions = role['permissions']
            
            if resource not in permissions:
                permissions[resource] = []
            
            if action not in permissions[resource]:
                permissions[resource].append(action)
                return self.update_role(role_id, permissions=permissions)
            
            return True  # Permission already exists
            
        except Exception as e:
            logger.error(f"Failed to add permission to role {role_id}: {e}")
            return False
    
    def remove_permission(self, role_id: int, resource: str, action: str) -> bool:
        """
        Remove permission from role
        
        Args:
            role_id: Role ID
            resource: Resource name
            action: Action name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            role = self.get_role_by_id(role_id)
            if not role:
                logger.warning(f"Role {role_id} not found")
                return False
            
            permissions = role['permissions']
            
            if resource in permissions and action in permissions[resource]:
                permissions[resource].remove(action)
                
                # Remove resource if no actions left
                if not permissions[resource]:
                    del permissions[resource]
                
                return self.update_role(role_id, permissions=permissions)
            
            return True  # Permission doesn't exist
            
        except Exception as e:
            logger.error(f"Failed to remove permission from role {role_id}: {e}")
            return False
    
    def has_permission(self, role_id: int, resource: str, action: str) -> bool:
        """
        Check if role has specific permission
        
        Args:
            role_id: Role ID
            resource: Resource name
            action: Action name
            
        Returns:
            True if role has permission, False otherwise
        """
        try:
            role = self.get_role_by_id(role_id)
            if not role:
                return False
            
            permissions = role.get('permissions', {})
            resource_permissions = permissions.get(resource, [])
            
            return action in resource_permissions
            
        except Exception as e:
            logger.error(f"Failed to check permission for role {role_id}: {e}")
            return False
    
    def get_users_by_role(self, role_id: int) -> List[Dict[str, Any]]:
        """
        Get all users assigned to a role
        
        Args:
            role_id: Role ID
            
        Returns:
            List of user dictionaries
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT u.id, u.username, u.email, u.first_name, u.last_name, 
                           u.is_active, u.created_at, u.last_login
                    FROM users u
                    WHERE u.role_id = ?
                    ORDER BY u.created_at DESC
                """, (role_id,))
                
                users = []
                for row in cursor.fetchall():
                    users.append(dict(row))
                
                return users
                
        except Exception as e:
            logger.error(f"Failed to get users for role {role_id}: {e}")
            return []
    
    def get_default_roles(self) -> Dict[str, Dict[str, Any]]:
        """
        Get predefined default roles
        
        Returns:
            Dictionary of default roles
        """
        return {
            'admin': {
                'name': 'admin',
                'description': 'System Administrator with full access',
                'permissions': {
                    'users': ['create', 'read', 'update', 'delete'],
                    'roles': ['create', 'read', 'update', 'delete'],
                    'sessions': ['create', 'read', 'update', 'delete'],
                    'documents': ['create', 'read', 'update', 'delete'],
                    'system': ['admin', 'configure']
                }
            },
            'teacher': {
                'name': 'teacher',
                'description': 'Teacher with session and document management access',
                'permissions': {
                    'users': ['read'],
                    'sessions': ['create', 'read', 'update', 'delete'],
                    'documents': ['create', 'read', 'update', 'delete'],
                    'students': ['read']
                }
            },
            'student': {
                'name': 'student',
                'description': 'Student with limited read-only access',
                'permissions': {
                    'sessions': ['read'],
                    'documents': ['read']
                }
            }
        }
    
    def create_default_roles(self) -> bool:
        """
        Create default roles if they don't exist
        
        Returns:
            True if successful, False otherwise
        """
        try:
            default_roles = self.get_default_roles()
            
            for role_name, role_data in default_roles.items():
                if not self.get_role_by_name(role_name):
                    self.create_role(
                        name=role_data['name'],
                        description=role_data['description'],
                        permissions=role_data['permissions']
                    )
                    logger.info(f"Created default role: {role_name}")
                else:
                    logger.info(f"Default role already exists: {role_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create default roles: {e}")
            return False