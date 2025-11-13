"""
Authentication API Endpoints for RAG Education Assistant Auth Service
Handles login, logout, token refresh, and authentication operations
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer

# Import schemas and dependencies
from auth.schemas import (
    LoginRequest, TokenResponse, RefreshTokenRequest, RefreshTokenResponse,
    LogoutRequest, BaseResponse, ChangePasswordRequest, PermissionCheck,
    PermissionResponse, UserResponse, HealthResponse
)
from auth.dependencies import (
    get_auth_manager, get_current_user, get_current_active_user, get_optional_user
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


@router.post("/login", response_model=TokenResponse, summary="User Login")
async def login(
    request: Request,
    login_data: LoginRequest,
    auth_manager=Depends(get_auth_manager)
) -> TokenResponse:
    """
    Authenticate user and return JWT tokens
    
    Args:
        request: FastAPI request object
        login_data: Login credentials
        auth_manager: AuthManager instance
        
    Returns:
        JWT tokens and user information
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Get client IP for rate limiting
        client_ip = request.client.host if request.client else None
        
        # Authenticate user
        user = auth_manager.authenticate_user(
            username=login_data.username,
            password=login_data.password,
            ip_address=client_ip
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Create token data
        token_data = {
            "sub": user['username'],
            "user_id": user['id'],
            "role": user['role_name'],
            "permissions": user['permissions']
        }
        
        # Generate tokens
        access_token = auth_manager.create_access_token(token_data)
        refresh_token = auth_manager.create_refresh_token(token_data)
        
        # Create session
        user_agent = request.headers.get("user-agent")
        session_id = auth_manager.create_session(
            user_id=user['id'],
            access_token=access_token,
            refresh_token=refresh_token,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        if not session_id:
            logger.error("Failed to create session")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )
        
        # Prepare response
        user_response = UserResponse(
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
        
        logger.info(f"User {user['username']} logged in successfully")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=auth_manager.access_token_expire_minutes * 60,
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


@router.post("/refresh", response_model=RefreshTokenResponse, summary="Refresh Access Token")
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    auth_manager=Depends(get_auth_manager)
) -> RefreshTokenResponse:
    """
    Refresh access token using refresh token
    
    Args:
        refresh_data: Refresh token data
        auth_manager: AuthManager instance
        
    Returns:
        New access token and optionally new refresh token
        
    Raises:
        HTTPException: If refresh fails
    """
    try:
        result = auth_manager.refresh_access_token(refresh_data.refresh_token)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        logger.info("Access token refreshed successfully")
        
        return RefreshTokenResponse(
            access_token=result["access_token"],
            token_type="bearer",
            expires_in=auth_manager.access_token_expire_minutes * 60,
            refresh_token=result.get("refresh_token")  # Only if token rotation is enabled
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh service error"
        )


@router.post("/logout", response_model=BaseResponse, summary="User Logout")
async def logout(
    logout_data: LogoutRequest = None,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Logout user by invalidating tokens/sessions
    
    Args:
        logout_data: Logout request data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If logout fails
    """
    try:
        if not logout_data:
            logout_data = LogoutRequest()
        
        success = auth_manager.logout(
            refresh_token=logout_data.refresh_token,
            user_id=current_user['id'],
            all_sessions=logout_data.all_sessions
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Logout failed"
            )
        
        session_type = "all sessions" if logout_data.all_sessions else "current session"
        logger.info(f"User {current_user['username']} logged out from {session_type}")
        
        return BaseResponse(
            success=True,
            message=f"Successfully logged out from {session_type}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout service error"
        )


@router.get("/me", response_model=UserResponse, summary="Get Current User")
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> UserResponse:
    """
    Get current authenticated user information
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
    """
    return UserResponse(
        id=current_user['id'],
        username=current_user['username'],
        email=current_user['email'],
        first_name=current_user['first_name'],
        last_name=current_user['last_name'],
        is_active=current_user['is_active'],
        role_id=current_user['role_id'],
        role_name=current_user['role_name'],
        created_at=datetime.fromisoformat(current_user['created_at'].replace('Z', '+00:00')),
        updated_at=datetime.fromisoformat(current_user['updated_at'].replace('Z', '+00:00')),
        last_login=datetime.fromisoformat(current_user['last_login'].replace('Z', '+00:00')) if current_user.get('last_login') else None
    )


@router.put("/change-password", response_model=BaseResponse, summary="Change Password")
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Change user password
    
    Args:
        password_data: Password change data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If password change fails
    """
    try:
        success = auth_manager.user_model.change_password(
            user_id=current_user['id'],
            old_password=password_data.current_password,
            new_password=password_data.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        logger.info(f"Password changed for user {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message="Password changed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change service error"
        )


@router.post("/check-permission", response_model=PermissionResponse, summary="Check Permission")
async def check_permission(
    permission_data: PermissionCheck,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> PermissionResponse:
    """
    Check if current user has specific permission
    
    Args:
        permission_data: Permission check data
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Permission check result
    """
    has_permission = auth_manager.validate_permission(
        user=current_user,
        resource=permission_data.resource,
        action=permission_data.action
    )
    
    return PermissionResponse(
        has_permission=has_permission,
        resource=permission_data.resource,
        action=permission_data.action,
        user_id=current_user['id']
    )


@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check(
    auth_manager=Depends(get_auth_manager)
) -> HealthResponse:
    """
    Health check endpoint
    
    Args:
        auth_manager: AuthManager instance
        
    Returns:
        Health status
    """
    try:
        # Test database connection
        auth_manager.session_model.get_active_sessions_count()
        database_status = "connected"
    except Exception:
        database_status = "disconnected"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        database_status=database_status
    )


@router.delete("/sessions/{session_id}", response_model=BaseResponse, summary="Delete Session")
async def delete_session(
    session_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Delete a specific session
    
    Args:
        session_id: Session ID to delete
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If session deletion fails or user lacks permission
    """
    try:
        # Get session to check ownership
        session = auth_manager.session_model.get_session_by_id(session_id)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Check if user owns the session or is admin
        if session['user_id'] != current_user['id'] and current_user['role_name'] != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete session that doesn't belong to you"
            )
        
        # Delete session
        success = auth_manager.session_model.delete_session_by_id(session_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to delete session"
            )
        
        logger.info(f"Session {session_id} deleted by user {current_user['username']}")
        
        return BaseResponse(
            success=True,
            message="Session deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session deletion service error"
        )


@router.post("/validate-token", response_model=BaseResponse, summary="Validate Token")
async def validate_token(
    user: Dict[str, Any] = Depends(get_optional_user)
) -> BaseResponse:
    """
    Validate access token
    
    Args:
        user: Optional current user (from token)
        
    Returns:
        Token validation result
    """
    if user:
        return BaseResponse(
            success=True,
            message="Token is valid"
        )
    else:
        return BaseResponse(
            success=False,
            message="Token is invalid or expired"
        )


# Additional utility endpoints
@router.post("/cleanup-sessions", response_model=BaseResponse, summary="Cleanup Expired Sessions")
async def cleanup_expired_sessions(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    auth_manager=Depends(get_auth_manager)
) -> BaseResponse:
    """
    Cleanup expired sessions (admin only)
    
    Args:
        current_user: Current authenticated user
        auth_manager: AuthManager instance
        
    Returns:
        Cleanup result
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user['role_name'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        count = auth_manager.cleanup_expired_sessions()
        
        logger.info(f"Cleaned up {count} expired sessions")
        
        return BaseResponse(
            success=True,
            message=f"Cleaned up {count} expired sessions"
        )
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session cleanup service error"
        )