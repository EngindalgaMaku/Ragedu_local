"""
Pydantic Schemas for RAG Education Assistant Auth Service
Defines request and response models for API endpoints
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, EmailStr, validator, Field
import re


# Common response schemas
class BaseResponse(BaseModel):
    """Base response schema"""
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response schema"""
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel):
    """Paginated response schema"""
    items: List[Dict[str, Any]]
    total: int
    page: int = 1
    size: int = 100
    has_next: bool = False
    has_prev: bool = False


# Authentication schemas
class LoginRequest(BaseModel):
    """Login request schema"""
    username: str = Field(..., min_length=3, max_length=50, description="Username or email")
    password: str = Field(..., min_length=6, max_length=128, description="Password")
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format"""
        if '@' in v:
            # If it contains @, validate as email
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
                raise ValueError('Invalid email format')
        else:
            # Validate as username
            if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', v):
                raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v.lower()


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes in seconds
    user: 'UserResponse'


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema"""
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """Refresh token response schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 1800
    refresh_token: Optional[str] = None  # Only if token rotation is enabled


class LogoutRequest(BaseModel):
    """Logout request schema"""
    refresh_token: Optional[str] = None
    all_sessions: bool = False


# User schemas
class UserBase(BaseModel):
    """Base user schema"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    is_active: bool = True
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format"""
        if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', v):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v.lower()


class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 characters)")
    role_id: int = Field(..., gt=0, description="Role ID")
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        # Check for at least one digit
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v


class UserUpdate(BaseModel):
    """User update schema"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    is_active: Optional[bool] = None
    role_id: Optional[int] = Field(None, gt=0)
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format if provided"""
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', v):
                raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
            return v.lower()
        return v


class UserResponse(BaseModel):
    """User response schema"""
    id: int
    username: str
    email: str
    first_name: str
    last_name: str
    is_active: bool
    role_id: int
    role_name: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """User list response schema"""
    users: List[UserResponse]
    total: int
    page: int = 1
    size: int = 100


class ChangePasswordRequest(BaseModel):
    """Change password request schema"""
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v


class ResetPasswordRequest(BaseModel):
    """Reset password request schema (admin only)"""
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


# Role schemas
class RoleBase(BaseModel):
    """Base role schema"""
    name: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=1, max_length=255)
    permissions: Dict[str, List[str]] = Field(..., description="Permissions dictionary")


class RoleCreate(RoleBase):
    """Role creation schema"""
    pass


class RoleUpdate(BaseModel):
    """Role update schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    description: Optional[str] = Field(None, min_length=1, max_length=255)
    permissions: Optional[Dict[str, List[str]]] = None


class RoleResponse(BaseModel):
    """Role response schema"""
    id: int
    name: str
    description: str
    permissions: Dict[str, List[str]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RoleListResponse(BaseModel):
    """Role list response schema"""
    roles: List[RoleResponse]
    total: int


# Session schemas
class SessionResponse(BaseModel):
    """Session response schema"""
    id: int
    user_id: int
    expires_at: datetime
    created_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    class Config:
        from_attributes = True


class SessionDetailResponse(SessionResponse):
    """Detailed session response schema"""
    username: str
    email: str
    first_name: str
    last_name: str
    role_name: str


class SessionListResponse(BaseModel):
    """Session list response schema"""
    sessions: List[SessionDetailResponse]
    total: int
    page: int = 1
    size: int = 100


# Permission schemas
class PermissionCheck(BaseModel):
    """Permission check request schema"""
    resource: str = Field(..., min_length=1, max_length=50)
    action: str = Field(..., min_length=1, max_length=50)


class PermissionResponse(BaseModel):
    """Permission check response schema"""
    has_permission: bool
    resource: str
    action: str
    user_id: int


# Health check schema
class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = "healthy"
    timestamp: datetime
    version: str = "1.0.0"
    database_status: str = "connected"


# Statistics schemas
class UserStatsResponse(BaseModel):
    """User statistics response schema"""
    total_users: int
    active_users: int
    inactive_users: int
    users_by_role: Dict[str, int]


class SessionStatsResponse(BaseModel):
    """Session statistics response schema"""
    total_sessions: int
    active_sessions: int
    sessions_by_role: Dict[str, int]


class SystemStatsResponse(BaseModel):
    """System statistics response schema"""
    users: UserStatsResponse
    sessions: SessionStatsResponse
    uptime: str


# API validation schemas
class PaginationParams(BaseModel):
    """Pagination parameters schema"""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(100, ge=1, le=1000, description="Items per page")


class UserFilters(BaseModel):
    """User filtering parameters schema"""
    role_name: Optional[str] = None
    is_active: Optional[bool] = None
    search: Optional[str] = Field(None, min_length=1, max_length=100, description="Search in username, email, or name")


class SortParams(BaseModel):
    """Sorting parameters schema"""
    sort_by: str = Field("created_at", description="Field to sort by")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")


# Batch operation schemas
class BatchUserActivate(BaseModel):
    """Batch user activation schema"""
    user_ids: List[int] = Field(..., min_items=1, max_items=100)


class BatchUserDeactivate(BaseModel):
    """Batch user deactivation schema"""
    user_ids: List[int] = Field(..., min_items=1, max_items=100)


class BatchOperationResponse(BaseModel):
    """Batch operation response schema"""
    success_count: int
    failed_count: int
    failed_items: List[Dict[str, Any]] = []


# Update forward references
TokenResponse.model_rebuild()