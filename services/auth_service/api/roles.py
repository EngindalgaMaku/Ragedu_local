"""
Role Management API Endpoints for RAG Education Assistant Auth Service
Handles role CRUD operations and permission management
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

# Import schemas and dependencies
from auth.schemas import (
    RoleCreate, RoleUpdate, RoleResponse, RoleListResponse, BaseResponse,
    UserResponse, UserListResponse
)
from auth.dependencies import (
    get_auth_manager, get_current_active_user, require_admin, require_permission
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/roles", tags=["Role Management"])


@router.get("", response_model=RoleListResponse, summary="List Roles")
async def list_roles(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(100, ge=1, le=1000, description="Items per page"),
    current_user: Dict[str, Any] = Depends(require_permission("roles", "read")),
    auth_manager=Depends(get_auth_manager)
) -> RoleListResponse:
    """
    List all roles with pagination
    
    Args:
        page: Page number
        size: Items per page
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Paginated list of roles
    """
    try:
        # Calculate offset
        offset = (page - 1) * size
        
        # Get roles
        roles = auth_manager.role_model.list_roles(offset=offset, limit=size)
        total = auth_manager.role_model.get_role_count()
        
        # Convert to response format
        role_responses = []
        for role in roles:
            role_responses.append(RoleResponse(
                id=role['id'],
                name=role['name'],
                description=role['description'],
                permissions=role['permissions'],
                created_at=datetime.fromisoformat(role['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(role['updated_at'].replace('Z', '+00:00'))
            ))
        
        return RoleListResponse(
            roles=role_responses,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Role listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role listing service error"
        )


@router.get("/{role_id}", response_model=RoleResponse, summary="Get Role")
async def get_role(
    role_id: int,
    current_user: Dict[str, Any] = Depends(require_permission("roles", "read")),
    auth_manager=Depends(get_auth_manager)
) -> RoleResponse:
    """
    Get role by ID
    
    Args:
        role_id: Role ID
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Role information
        
    Raises:
        HTTPException: If role not found
    """
    try:
        role = auth_manager.role_model.get_role_by_id(role_id)
        
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        return RoleResponse(
            id=role['id'],
            name=role['name'],
            description=role['description'],
            permissions=role['permissions'],
            created_at=datetime.fromisoformat(role['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(role['updated_at'].replace('Z', '+00:00'))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Role retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role retrieval service error"
        )


@router.post("", response_model=RoleResponse, summary="Create Role", dependencies=[Depends(require_admin)])
async def create_role(
    role_data: RoleCreate,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> RoleResponse:
    """
    Create a new role (admin only)
    
    Args:
        role_data: Role creation data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Created role information
        
    Raises:
        HTTPException: If role creation fails
    """
    try:
        # Create role
        role_id = auth_manager.role_model.create_role(
            name=role_data.name,
            description=role_data.description,
            permissions=role_data.permissions
        )
        
        if not role_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create role - name may already exist"
            )
        
        # Get created role
        role = auth_manager.role_model.get_role_by_id(role_id)
        if not role:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Role created but could not be retrieved"
            )
        
        logger.info(f"Role {role_data.name} created by {current_user['username']}")
        
        return RoleResponse(
            id=role['id'],
            name=role['name'],
            description=role['description'],
            permissions=role['permissions'],
            created_at=datetime.fromisoformat(role['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(role['updated_at'].replace('Z', '+00:00'))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Role creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role creation service error"
        )


@router.put("/{role_id}", response_model=RoleResponse, summary="Update Role", dependencies=[Depends(require_admin)])
async def update_role(
    role_id: int,
    role_data: RoleUpdate,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> RoleResponse:
    """
    Update role information (admin only)
    
    Args:
        role_id: Role ID
        role_data: Role update data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Updated role information
        
    Raises:
        HTTPException: If role not found or update fails
    """
    try:
        # Check if role exists
        role = auth_manager.role_model.get_role_by_id(role_id)
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        # Prepare update data
        update_data = {}
        if role_data.name is not None:
            update_data['name'] = role_data.name
        if role_data.description is not None:
            update_data['description'] = role_data.description
        if role_data.permissions is not None:
            update_data['permissions'] = role_data.permissions
        
        # Update role
        success = auth_manager.role_model.update_role(role_id, **update_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update role"
            )
        
        # Get updated role
        updated_role = auth_manager.role_model.get_role_by_id(role_id)
        
        logger.info(f"Role {role_id} updated by {current_user['username']}")
        
        return RoleResponse(
            id=updated_role['id'],
            name=updated_role['name'],
            description=updated_role['description'],
            permissions=updated_role['permissions'],
            created_at=datetime.fromisoformat(updated_role['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(updated_role['updated_at'].replace('Z', '+00:00'))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Role update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role update service error"
        )


@router.delete("/{role_id}", response_model=BaseResponse, summary="Delete Role", dependencies=[Depends(require_admin)])
async def delete_role(
    role_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Delete role (admin only)
    Note: Cannot delete role if users are assigned to it
    
    Args:
        role_id: Role ID
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If role not found, has assigned users, or deletion fails
    """
    try:
        # Check if role exists
        role = auth_manager.role_model.get_role_by_id(role_id)
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        # Check if role has assigned users
        users_with_role = auth_manager.role_model.get_users_by_role(role_id)
        if users_with_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete role: {len(users_with_role)} users are assigned to this role"
            )
        
        # Delete role
        success = auth_manager.role_model.delete_role(role_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to delete role"
            )
        
        logger.info(f"Role {role_id} ({role['name']}) deleted by {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message=f"Role {role['name']} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Role deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role deletion service error"
        )


@router.get("/{role_id}/users", response_model=UserListResponse, summary="Get Users by Role")
async def get_users_by_role(
    role_id: int,
    current_user: Dict[str, Any] = Depends(require_permission("roles", "read")),
    auth_manager=Depends(get_auth_manager)
) -> UserListResponse:
    """
    Get all users assigned to a specific role
    
    Args:
        role_id: Role ID
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        List of users assigned to the role
        
    Raises:
        HTTPException: If role not found
    """
    try:
        # Check if role exists
        role = auth_manager.role_model.get_role_by_id(role_id)
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        # Get users with this role
        users = auth_manager.role_model.get_users_by_role(role_id)
        
        # Convert to response format
        user_responses = []
        for user in users:
            user_responses.append(UserResponse(
                id=user['id'],
                username=user['username'],
                email=user['email'],
                first_name=user['first_name'],
                last_name=user['last_name'],
                is_active=user['is_active'],
                role_id=role_id,
                role_name=role['name'],
                created_at=datetime.fromisoformat(user['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(user['updated_at'].replace('Z', '+00:00')) if user.get('updated_at') else None,
                last_login=datetime.fromisoformat(user['last_login'].replace('Z', '+00:00')) if user.get('last_login') else None
            ))
        
        return UserListResponse(
            users=user_responses,
            total=len(user_responses)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Users by role retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Users by role service error"
        )


@router.post("/{role_id}/permissions", response_model=BaseResponse, summary="Add Permission to Role", dependencies=[Depends(require_admin)])
async def add_permission_to_role(
    role_id: int,
    resource: str = Query(..., min_length=1, max_length=50, description="Resource name"),
    action: str = Query(..., min_length=1, max_length=50, description="Action name"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Add permission to role (admin only)
    
    Args:
        role_id: Role ID
        resource: Resource name (e.g., 'users', 'documents')
        action: Action name (e.g., 'create', 'read', 'update', 'delete')
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If role not found or operation fails
    """
    try:
        success = auth_manager.role_model.add_permission(role_id, resource, action)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        logger.info(f"Permission {resource}:{action} added to role {role_id} by {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message=f"Permission {resource}:{action} added to role"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add permission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Add permission service error"
        )


@router.delete("/{role_id}/permissions", response_model=BaseResponse, summary="Remove Permission from Role", dependencies=[Depends(require_admin)])
async def remove_permission_from_role(
    role_id: int,
    resource: str = Query(..., min_length=1, max_length=50, description="Resource name"),
    action: str = Query(..., min_length=1, max_length=50, description="Action name"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Remove permission from role (admin only)
    
    Args:
        role_id: Role ID
        resource: Resource name
        action: Action name
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If role not found or operation fails
    """
    try:
        success = auth_manager.role_model.remove_permission(role_id, resource, action)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        logger.info(f"Permission {resource}:{action} removed from role {role_id} by {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message=f"Permission {resource}:{action} removed from role"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove permission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Remove permission service error"
        )


@router.get("/{role_id}/permissions/{resource}", response_model=Dict[str, Any], summary="Check Role Permission")
async def check_role_permission(
    role_id: int,
    resource: str,
    action: str = Query(..., min_length=1, max_length=50, description="Action name"),
    current_user: Dict[str, Any] = Depends(require_permission("roles", "read")),
    auth_manager=Depends(get_auth_manager)
) -> Dict[str, Any]:
    """
    Check if role has specific permission
    
    Args:
        role_id: Role ID
        resource: Resource name
        action: Action name
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Permission check result
        
    Raises:
        HTTPException: If role not found
    """
    try:
        # Check if role exists
        role = auth_manager.role_model.get_role_by_id(role_id)
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        has_permission = auth_manager.role_model.has_permission(role_id, resource, action)
        
        return {
            "role_id": role_id,
            "role_name": role['name'],
            "resource": resource,
            "action": action,
            "has_permission": has_permission
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Permission check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Permission check service error"
        )


@router.post("/create-defaults", response_model=BaseResponse, summary="Create Default Roles", dependencies=[Depends(require_admin)])
async def create_default_roles(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Create default roles (admin only)
    Creates admin, teacher, and student roles if they don't exist
    
    Args:
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
    """
    try:
        success = auth_manager.role_model.create_default_roles()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create default roles"
            )
        
        logger.info(f"Default roles created/verified by {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message="Default roles created/verified successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create default roles failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Create default roles service error"
        )


@router.get("/defaults/definitions", response_model=Dict[str, Any], summary="Get Default Role Definitions")
async def get_default_role_definitions(
    current_user: Dict[str, Any] = Depends(require_permission("roles", "read")),
    auth_manager=Depends(get_auth_manager)
) -> Dict[str, Any]:
    """
    Get default role definitions
    
    Args:
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Default role definitions
    """
    try:
        default_roles = auth_manager.role_model.get_default_roles()
        return {
            "default_roles": default_roles,
            "total_roles": len(default_roles)
        }
        
    except Exception as e:
        logger.error(f"Get default roles failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Get default roles service error"
        )