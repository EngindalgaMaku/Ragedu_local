"""
FastAPI Dependencies for RAG Education Assistant Auth Service
Provides reusable dependency functions for authentication and authorization
"""

import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# HTTP Bearer security scheme
security = HTTPBearer()


def get_auth_manager():
    """
    Dependency to get AuthManager instance
    This will be overridden by the FastAPI app
    """
    # This is a placeholder - the actual implementation will be injected by the app
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="AuthManager not configured"
    )


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager=Depends(get_auth_manager)
) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from JWT token
    
    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials
        auth_manager: AuthManager instance
        
    Returns:
        User data dictionary
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Check if user is already in request state (from middleware)
        user = getattr(request.state, 'user', None)
        if user:
            return user
        
        # Extract token
        token = credentials.credentials
        
        # Validate token and get user
        user = auth_manager.get_current_user(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Store in request state for future use
        request.state.user = user
        request.state.token = token
        
        logger.debug(f"Authenticated user: {user['username']}")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Dependency to get current active user
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User data dictionary
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.get('is_active', False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )
    
    return current_user


def require_permission(resource: str, action: str):
    """
    Dependency factory to require specific permissions
    
    Args:
        resource: Resource name (e.g., 'users', 'documents')
        action: Action name (e.g., 'create', 'read', 'update', 'delete')
        
    Returns:
        Dependency function
    """
    async def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_active_user),
        auth_manager=Depends(get_auth_manager)
    ) -> Dict[str, Any]:
        """
        Check if current user has required permission
        
        Args:
            current_user: Current authenticated user
            auth_manager: AuthManager instance
            
        Returns:
            User data if authorized
            
        Raises:
            HTTPException: If user lacks required permission
        """
        if not auth_manager.validate_permission(current_user, resource, action):
            logger.warning(f"User {current_user['username']} lacks permission: {resource}:{action}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for {resource}:{action}"
            )
        
        return current_user
    
    return permission_checker


def require_role(role_name: str):
    """
    Dependency factory to require specific role
    
    Args:
        role_name: Required role name
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        """
        Check if current user has required role
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User data if authorized
            
        Raises:
            HTTPException: If user doesn't have required role
        """
        user_role = current_user.get('role_name', '')
        if user_role != role_name:
            logger.warning(f"User {current_user['username']} lacks role: {role_name} (has: {user_role})")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {role_name} role"
            )
        
        return current_user
    
    return role_checker


def require_admin():
    """
    Dependency to require admin role
    
    Returns:
        Dependency function
    """
    return require_role("admin")


def require_teacher_or_admin():
    """
    Dependency to require teacher or admin role
    
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        """
        Check if current user has teacher or admin role
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User data if authorized
            
        Raises:
            HTTPException: If user doesn't have required role
        """
        user_role = current_user.get('role_name', '')
        if user_role not in ['teacher', 'admin']:
            logger.warning(f"User {current_user['username']} lacks teacher/admin role: (has: {user_role})")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Requires teacher or admin role"
            )
        
        return current_user
    
    return role_checker


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    auth_manager=Depends(get_auth_manager)
) -> Optional[Dict[str, Any]]:
    """
    Dependency to get current user if authenticated, None otherwise
    Useful for endpoints that work with or without authentication
    
    Args:
        request: FastAPI request object
        credentials: Optional HTTP Bearer credentials
        auth_manager: AuthManager instance
        
    Returns:
        User data dictionary if authenticated, None otherwise
    """
    try:
        if not credentials:
            return None
        
        # Check if user is already in request state
        user = getattr(request.state, 'user', None)
        if user:
            return user
        
        # Extract token
        token = credentials.credentials
        
        # Validate token and get user
        user = auth_manager.get_current_user(token)
        if user:
            # Store in request state
            request.state.user = user
            request.state.token = token
            logger.debug(f"Optionally authenticated user: {user['username']}")
        
        return user
        
    except Exception as e:
        logger.debug(f"Optional authentication failed: {e}")
        return None


def validate_session_ownership():
    """
    Dependency to validate that user owns the session they're trying to access
    Useful for session management endpoints
    
    Returns:
        Dependency function
    """
    async def session_ownership_checker(
        request: Request,
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        """
        Check if current user owns the session being accessed
        
        Args:
            request: FastAPI request object
            current_user: Current authenticated user
            
        Returns:
            User data if authorized
            
        Raises:
            HTTPException: If user doesn't own the session
        """
        # Get session_id from path parameters
        session_id = request.path_params.get('session_id')
        if not session_id:
            # If no session_id in path, allow access
            return current_user
        
        # For now, we'll implement basic ownership check
        # In a real implementation, you'd check if the session belongs to the user
        # This is a placeholder for session ownership validation
        
        # Admins can access any session
        if current_user.get('role_name') == 'admin':
            return current_user
        
        # TODO: Implement actual session ownership check
        # For now, allow access if user is authenticated
        return current_user
    
    return session_ownership_checker


class PermissionChecker:
    """
    Class-based permission checker for more complex permission logic
    """
    
    def __init__(self, resource: str, action: str, allow_self: bool = False):
        """
        Initialize permission checker
        
        Args:
            resource: Resource name
            action: Action name
            allow_self: Whether user can perform action on their own data
        """
        self.resource = resource
        self.action = action
        self.allow_self = allow_self
    
    async def __call__(
        self,
        request: Request,
        current_user: Dict[str, Any] = Depends(get_current_active_user),
        auth_manager=Depends(get_auth_manager)
    ) -> Dict[str, Any]:
        """
        Check permissions with additional logic
        
        Args:
            request: FastAPI request object
            current_user: Current authenticated user
            auth_manager: AuthManager instance
            
        Returns:
            User data if authorized
            
        Raises:
            HTTPException: If user lacks required permission
        """
        # Check basic permission
        if auth_manager.validate_permission(current_user, self.resource, self.action):
            return current_user
        
        # Check self-access if allowed
        if self.allow_self:
            user_id_param = request.path_params.get('user_id')
            if user_id_param and str(current_user['id']) == str(user_id_param):
                logger.debug(f"Allowing self-access for user {current_user['username']}")
                return current_user
        
        # Permission denied
        logger.warning(f"User {current_user['username']} lacks permission: {self.resource}:{self.action}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions for {self.resource}:{self.action}"
        )


# Common permission dependencies
require_users_read = require_permission("users", "read")
require_users_create = require_permission("users", "create")
require_users_update = require_permission("users", "update")
require_users_delete = require_permission("users", "delete")

require_roles_read = require_permission("roles", "read")
require_roles_create = require_permission("roles", "create")
require_roles_update = require_permission("roles", "update")
require_roles_delete = require_permission("roles", "delete")

require_sessions_read = require_permission("sessions", "read")
require_sessions_create = require_permission("sessions", "create")
require_sessions_update = require_permission("sessions", "update")
require_sessions_delete = require_permission("sessions", "delete")

# Permission checkers with self-access
users_read_or_self = PermissionChecker("users", "read", allow_self=True)
users_update_or_self = PermissionChecker("users", "update", allow_self=True)