"""
User Management API Endpoints for RAG Education Assistant Auth Service
Handles user CRUD operations, user activation/deactivation, and user statistics
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

# Import schemas and dependencies
from auth.schemas import (
    UserCreate, UserUpdate, UserResponse, UserListResponse, ResetPasswordRequest,
    BaseResponse, BatchUserActivate, BatchUserDeactivate, BatchOperationResponse,
    UserStatsResponse, PaginationParams, UserFilters, SortParams
)
from auth.dependencies import (
    get_auth_manager, get_current_active_user, require_admin,
    require_permission, users_read_or_self, users_update_or_self
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["User Management"])


@router.post("", response_model=UserResponse, summary="Create User", dependencies=[Depends(require_admin)])
async def create_user(
    user_data: UserCreate,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> UserResponse:
    """
    Create a new user (admin only)
    
    Args:
        user_data: User creation data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Created user information
        
    Raises:
        HTTPException: If user creation fails
    """
    try:
        # Validate role exists
        role = auth_manager.role_model.get_role_by_id(user_data.role_id)
        if not role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role ID"
            )
        
        # Create user
        user_id = auth_manager.user_model.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role_id=user_data.role_id,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            is_active=user_data.is_active
        )
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create user - username or email may already exist"
            )
        
        # Get created user
        user = auth_manager.user_model.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User created but could not be retrieved"
            )
        
        logger.info(f"User {user_data.username} created by {current_user['username']}")
        
        return UserResponse(
            id=user['id'],
            username=user['username'],
            email=user['email'],
            first_name=user['first_name'],
            last_name=user['last_name'],
            is_active=user['is_active'],
            role_id=user['role_id'],
            role_name=user['role_name'],
            created_at=datetime.fromisoformat(user['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(user['updated_at'].replace('Z', '+00:00')),
            last_login=datetime.fromisoformat(user['last_login'].replace('Z', '+00:00')) if user.get('last_login') else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation service error"
        )


@router.get("", response_model=UserListResponse, summary="List Users")
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(100, ge=1, le=1000, description="Items per page"),
    role_name: Optional[str] = Query(None, description="Filter by role name"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    search: Optional[str] = Query(None, min_length=1, max_length=100, description="Search in username, email, or name"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order"),
    current_user: Dict[str, Any] = Depends(require_permission("users", "read")),
    auth_manager=Depends(get_auth_manager)
) -> UserListResponse:
    """
    List users with filtering and pagination
    
    Args:
        page: Page number
        size: Items per page
        role_name: Filter by role name
        is_active: Filter by active status
        search: Search term
        sort_by: Field to sort by
        sort_order: Sort order (asc/desc)
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Paginated list of users
    """
    try:
        # Calculate offset
        offset = (page - 1) * size
        
        # Get users with filters
        users = auth_manager.user_model.list_users(
            role_name=role_name,
            is_active=is_active,
            offset=offset,
            limit=size
        )
        
        # Apply search filter if provided
        if search:
            search_term = search.lower()
            users = [
                user for user in users
                if (search_term in user['username'].lower() or
                    search_term in user['email'].lower() or
                    search_term in user['first_name'].lower() or
                    search_term in user['last_name'].lower())
            ]
        
        # Apply sorting (basic implementation)
        if sort_by in ['created_at', 'updated_at', 'last_login']:
            reverse = sort_order == "desc"
            users.sort(
                key=lambda x: x.get(sort_by) or '',
                reverse=reverse
            )
        
        # Get total count
        total = auth_manager.user_model.get_user_count(
            role_name=role_name,
            is_active=is_active
        )
        
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
                role_id=user.get('role_id', 0),
                role_name=user.get('role_name', ''),
                created_at=datetime.fromisoformat(user['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(user['updated_at'].replace('Z', '+00:00')),
                last_login=datetime.fromisoformat(user['last_login'].replace('Z', '+00:00')) if user.get('last_login') else None
            ))
        
        return UserListResponse(
            users=user_responses,
            total=total,
            page=page,
            size=size
        )
        
    except Exception as e:
        logger.error(f"User listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User listing service error"
        )


@router.get("/me", response_model=UserResponse, summary="Get Current User Profile")
async def get_current_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> UserResponse:
    """
    Get current authenticated user's profile
    
    Args:
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Current user information
    """
    try:
        user = auth_manager.user_model.get_user_by_id(current_user['id'])
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=user['id'],
            username=user['username'],
            email=user['email'],
            first_name=user['first_name'],
            last_name=user['last_name'],
            is_active=user['is_active'],
            role_id=user['role_id'],
            role_name=user['role_name'],
            created_at=datetime.fromisoformat(user['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(user['updated_at'].replace('Z', '+00:00')),
            last_login=datetime.fromisoformat(user['last_login'].replace('Z', '+00:00')) if user.get('last_login') else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User profile retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User profile retrieval service error"
        )


@router.put("/me", response_model=UserResponse, summary="Update Current User Profile")
async def update_current_user_profile(
    user_data: UserUpdate,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> UserResponse:
    """
    Update current authenticated user's profile (cannot change role or username)
    
    Args:
        user_data: User update data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Updated user information
    """
    try:
        user_id = current_user['id']
        
        # Check if user exists
        user = auth_manager.user_model.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prepare update data (exclude role_id and is_active for self-update)
        update_data = {}
        if user_data.username is not None:
            update_data['username'] = user_data.username
        if user_data.email is not None:
            update_data['email'] = user_data.email
        if user_data.first_name is not None:
            update_data['first_name'] = user_data.first_name
        if user_data.last_name is not None:
            update_data['last_name'] = user_data.last_name
        # Note: is_active and role_id cannot be changed by user themselves
        
        # Update user
        try:
            success = auth_manager.user_model.update_user(user_id, **update_data)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to update user profile"
                )
        except ValueError as ve:
            # Handle username/email already taken errors
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(ve)
            )
        
        # Get updated user
        updated_user = auth_manager.user_model.get_user_by_id(user_id)
        
        logger.info(f"User {user_id} updated their profile")
        
        return UserResponse(
            id=updated_user['id'],
            username=updated_user['username'],
            email=updated_user['email'],
            first_name=updated_user['first_name'],
            last_name=updated_user['last_name'],
            is_active=updated_user['is_active'],
            role_id=updated_user['role_id'],
            role_name=updated_user['role_name'],
            created_at=datetime.fromisoformat(updated_user['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(updated_user['updated_at'].replace('Z', '+00:00')),
            last_login=datetime.fromisoformat(updated_user['last_login'].replace('Z', '+00:00')) if updated_user.get('last_login') else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User profile update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User profile update service error"
        )


@router.get("/by-username/{username}", response_model=UserResponse, summary="Get User by Username")
async def get_user_by_username(
    username: str,
    auth_manager=Depends(get_auth_manager)
) -> UserResponse:
    """
    Get user by username (internal service endpoint - no auth required)
    
    Args:
        username: Username to lookup
        auth_manager: AuthManager instance
        
    Returns:
        User information
        
    Raises:
        HTTPException: If user not found
    """
    try:
        user = auth_manager.user_model.get_user_by_username(username)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=user['id'],
            username=user['username'],
            email=user['email'],
            first_name=user['first_name'],
            last_name=user['last_name'],
            is_active=user['is_active'],
            role_id=user['role_id'],
            role_name=user['role_name'],
            created_at=datetime.fromisoformat(user['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(user['updated_at'].replace('Z', '+00:00')),
            last_login=datetime.fromisoformat(user['last_login'].replace('Z', '+00:00')) if user.get('last_login') else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User retrieval by username failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User retrieval service error"
        )


@router.get("/by-id/{user_id}", response_model=UserResponse, summary="Get User by ID")
async def get_user_by_id_internal(
    user_id: int,
    auth_manager=Depends(get_auth_manager)
) -> UserResponse:
    """
    Get user by ID (internal service endpoint - no auth required)
    Used by other microservices to lookup user information
    
    Args:
        user_id: User ID to lookup
        auth_manager: AuthManager instance
        
    Returns:
        User information
        
    Raises:
        HTTPException: If user not found
    """
    try:
        user = auth_manager.user_model.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=user['id'],
            username=user['username'],
            email=user['email'],
            first_name=user['first_name'],
            last_name=user['last_name'],
            is_active=user['is_active'],
            role_id=user['role_id'],
            role_name=user['role_name'],
            created_at=datetime.fromisoformat(user['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(user['updated_at'].replace('Z', '+00:00')),
            last_login=datetime.fromisoformat(user['last_login'].replace('Z', '+00:00')) if user.get('last_login') else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User retrieval by ID failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User retrieval service error"
        )


@router.get("/{user_id}", response_model=UserResponse, summary="Get User")
async def get_user(
    user_id: int,
    current_user: Dict[str, Any] = Depends(users_read_or_self),
    auth_manager=Depends(get_auth_manager)
) -> UserResponse:
    """
    Get user by ID
    
    Args:
        user_id: User ID
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        User information
        
    Raises:
        HTTPException: If user not found
    """
    try:
        user = auth_manager.user_model.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=user['id'],
            username=user['username'],
            email=user['email'],
            first_name=user['first_name'],
            last_name=user['last_name'],
            is_active=user['is_active'],
            role_id=user['role_id'],
            role_name=user['role_name'],
            created_at=datetime.fromisoformat(user['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(user['updated_at'].replace('Z', '+00:00')),
            last_login=datetime.fromisoformat(user['last_login'].replace('Z', '+00:00')) if user.get('last_login') else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User retrieval service error"
        )


@router.put("/{user_id}", response_model=UserResponse, summary="Update User")
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: Dict[str, Any] = Depends(users_update_or_self),
    auth_manager=Depends(get_auth_manager)
) -> UserResponse:
    """
    Update user information
    
    Args:
        user_id: User ID
        user_data: User update data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Updated user information
        
    Raises:
        HTTPException: If user not found or update fails
    """
    try:
        # Check if user exists
        user = auth_manager.user_model.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate role if provided
        if user_data.role_id:
            role = auth_manager.role_model.get_role_by_id(user_data.role_id)
            if not role:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid role ID"
                )
        
        # Prepare update data
        update_data = {}
        if user_data.email is not None:
            update_data['email'] = user_data.email
        if user_data.first_name is not None:
            update_data['first_name'] = user_data.first_name
        if user_data.last_name is not None:
            update_data['last_name'] = user_data.last_name
        if user_data.is_active is not None:
            update_data['is_active'] = user_data.is_active
        if user_data.role_id is not None:
            update_data['role_id'] = user_data.role_id
        
        # Update user
        success = auth_manager.user_model.update_user(user_id, **update_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update user"
            )
        
        # Get updated user
        updated_user = auth_manager.user_model.get_user_by_id(user_id)
        
        logger.info(f"User {user_id} updated by {current_user['username']}")
        
        return UserResponse(
            id=updated_user['id'],
            username=updated_user['username'],
            email=updated_user['email'],
            first_name=updated_user['first_name'],
            last_name=updated_user['last_name'],
            is_active=updated_user['is_active'],
            role_id=updated_user['role_id'],
            role_name=updated_user['role_name'],
            created_at=datetime.fromisoformat(updated_user['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(updated_user['updated_at'].replace('Z', '+00:00')),
            last_login=datetime.fromisoformat(updated_user['last_login'].replace('Z', '+00:00')) if updated_user.get('last_login') else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User update service error"
        )


@router.delete("/{user_id}", response_model=BaseResponse, summary="Delete User", dependencies=[Depends(require_admin)])
async def delete_user(
    user_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Delete user (admin only)
    
    Args:
        user_id: User ID
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If user not found or deletion fails
    """
    try:
        # Prevent self-deletion
        if user_id == current_user['id']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Check if user exists
        user = auth_manager.user_model.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Delete user
        success = auth_manager.user_model.delete_user(user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to delete user"
            )
        
        logger.info(f"User {user_id} ({user['username']}) deleted by {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message=f"User {user['username']} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User deletion service error"
        )


@router.post("/{user_id}/activate", response_model=BaseResponse, summary="Activate User", dependencies=[Depends(require_admin)])
async def activate_user(
    user_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Activate user account (admin only)
    
    Args:
        user_id: User ID
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If user not found or activation fails
    """
    try:
        success = auth_manager.user_model.activate_user(user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User {user_id} activated by {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message="User activated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User activation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User activation service error"
        )


@router.post("/{user_id}/deactivate", response_model=BaseResponse, summary="Deactivate User", dependencies=[Depends(require_admin)])
async def deactivate_user(
    user_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Deactivate user account (admin only)
    
    Args:
        user_id: User ID
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If user not found or deactivation fails
    """
    try:
        # Prevent self-deactivation
        if user_id == current_user['id']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )
        
        success = auth_manager.user_model.deactivate_user(user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User {user_id} deactivated by {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message="User deactivated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User deactivation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User deactivation service error"
        )


@router.post("/{user_id}/reset-password", response_model=BaseResponse, summary="Reset User Password", dependencies=[Depends(require_admin)])
async def reset_user_password(
    user_id: int,
    reset_data: ResetPasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Reset user password (admin only)
    
    Args:
        user_id: User ID
        reset_data: Password reset data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If user not found or password reset fails
    """
    try:
        success = auth_manager.user_model.reset_password(user_id, reset_data.new_password)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Password reset for user {user_id} by {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message="Password reset successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset service error"
        )


@router.get("/stats/overview", response_model=UserStatsResponse, summary="Get User Statistics", dependencies=[Depends(require_admin)])
async def get_user_statistics(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> UserStatsResponse:
    """
    Get user statistics (admin only)
    
    Args:
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        User statistics
    """
    try:
        # Get total user count
        total_users = auth_manager.user_model.get_user_count()
        
        # Get active/inactive counts
        active_users = auth_manager.user_model.get_user_count(is_active=True)
        inactive_users = total_users - active_users
        
        # Get users by role
        roles = auth_manager.role_model.list_roles()
        users_by_role = {}
        for role in roles:
            count = auth_manager.user_model.get_user_count(role_name=role['name'])
            users_by_role[role['name']] = count
        
        return UserStatsResponse(
            total_users=total_users,
            active_users=active_users,
            inactive_users=inactive_users,
            users_by_role=users_by_role
        )
        
    except Exception as e:
        logger.error(f"User statistics retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User statistics service error"
        )


@router.post("/batch/activate", response_model=BatchOperationResponse, summary="Batch Activate Users", dependencies=[Depends(require_admin)])
async def batch_activate_users(
    batch_data: BatchUserActivate,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BatchOperationResponse:
    """
    Activate multiple users (admin only)
    
    Args:
        batch_data: Batch activation data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Batch operation result
    """
    success_count = 0
    failed_count = 0
    failed_items = []
    
    for user_id in batch_data.user_ids:
        try:
            success = auth_manager.user_model.activate_user(user_id)
            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_items.append({"user_id": user_id, "error": "User not found"})
        except Exception as e:
            failed_count += 1
            failed_items.append({"user_id": user_id, "error": str(e)})
    
    logger.info(f"Batch activation: {success_count} success, {failed_count} failed by {current_user['username']}")
    
    return BatchOperationResponse(
        success_count=success_count,
        failed_count=failed_count,
        failed_items=failed_items
    )


@router.post("/batch/deactivate", response_model=BatchOperationResponse, summary="Batch Deactivate Users", dependencies=[Depends(require_admin)])
async def batch_deactivate_users(
    batch_data: BatchUserDeactivate,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BatchOperationResponse:
    """
    Deactivate multiple users (admin only)
    
    Args:
        batch_data: Batch deactivation data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Batch operation result
    """
    success_count = 0
    failed_count = 0
    failed_items = []
    
    for user_id in batch_data.user_ids:
        try:
            # Prevent self-deactivation
            if user_id == current_user['id']:
                failed_count += 1
                failed_items.append({"user_id": user_id, "error": "Cannot deactivate your own account"})
                continue
            
            success = auth_manager.user_model.deactivate_user(user_id)
            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_items.append({"user_id": user_id, "error": "User not found"})
        except Exception as e:
            failed_count += 1
            failed_items.append({"user_id": user_id, "error": str(e)})
    
    logger.info(f"Batch deactivation: {success_count} success, {failed_count} failed by {current_user['username']}")
    
    return BatchOperationResponse(
        success_count=success_count,
        failed_count=failed_count,
        failed_items=failed_items
    )