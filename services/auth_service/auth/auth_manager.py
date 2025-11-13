"""
JWT Authentication Manager for RAG Education Assistant Auth Service
Handles JWT token creation, validation, and refresh functionality
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
import bcrypt
import secrets

# Set up logging
logger = logging.getLogger(__name__)

# Import database components
from database.database import DatabaseManager
from database.models.user import User
from database.models.role import Role
from database.models.session import UserSession


class AuthManager:
    """
    JWT Authentication Manager
    Manages token creation, validation, refresh, and user authentication
    """
    
    def __init__(self, 
                 secret_key: str = None,
                 algorithm: str = "HS256",
                 access_token_expire_minutes: int = 30,
                 refresh_token_expire_days: int = 7,
                 db_path: str = "data/rag_assistant.db"):
        """
        Initialize authentication manager
        
        Args:
            secret_key: JWT secret key (if None, generates a new one)
            algorithm: JWT algorithm
            access_token_expire_minutes: Access token expiration time in minutes
            refresh_token_expire_days: Refresh token expiration time in days
            db_path: Database file path
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        # Initialize database components
        self.db_manager = DatabaseManager(db_path)
        self.user_model = User(self.db_manager)
        self.role_model = Role(self.db_manager)
        self.session_model = UserSession(self.db_manager)
        
        # Password context for additional password operations
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Rate limiting storage (in production, use Redis)
        self._failed_attempts = {}
        self._max_attempts = 5
        self._lockout_duration = timedelta(minutes=15)
        
        logger.info("AuthManager initialized")
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(32)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token
        
        Args:
            data: Token payload data
            expires_delta: Custom expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Access token created for user: {data.get('sub')}")
        
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT refresh token
        
        Args:
            data: Token payload data
            expires_delta: Custom expiration time
            
        Returns:
            JWT refresh token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Refresh token created for user: {data.get('sub')}")
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
            token_type: Expected token type ("access" or "refresh")
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                logger.warning(f"Invalid token type: expected {token_type}, got {payload.get('type')}")
                return None
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
                logger.debug("Token expired")
                return None
            
            return payload
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str, ip_address: str = None) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username/email and password
        
        Args:
            username: Username or email
            password: Plain text password
            ip_address: Client IP address for rate limiting
            
        Returns:
            User data if authentication successful, None otherwise
        """
        # Check rate limiting
        if self._is_rate_limited(ip_address or username):
            logger.warning(f"Rate limit exceeded for {username}")
            return None
        
        try:
            # Attempt authentication using existing user model
            user = self.user_model.authenticate_user(username, password)
            
            if user:
                # Reset failed attempts on successful login
                self._reset_failed_attempts(ip_address or username)
                logger.info(f"User {username} authenticated successfully")
                return user
            else:
                # Record failed attempt
                self._record_failed_attempt(ip_address or username)
                logger.warning(f"Authentication failed for {username}")
                return None
                
        except Exception as e:
            logger.error(f"Authentication error for {username}: {e}")
            self._record_failed_attempt(ip_address or username)
            return None
    
    def create_session(self, user_id: int, access_token: str, refresh_token: str, 
                      ip_address: str = None, user_agent: str = None) -> Optional[int]:
        """
        Create a new user session with tokens
        
        Args:
            user_id: User ID
            access_token: Access token
            refresh_token: Refresh token  
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            # Store refresh token in session (access token is stateless)
            session_id = self.session_model.create_session(
                user_id=user_id,
                token=refresh_token,
                expires_in_hours=self.refresh_token_expire_days * 24,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            if session_id:
                logger.info(f"Session created for user {user_id} with ID {session_id}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """
        Create new access token using refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Dict with new access token and optionally new refresh token
        """
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token, "refresh")
            if not payload:
                return None
            
            # Validate session exists and is active
            session = self.session_model.validate_session(refresh_token)
            if not session:
                logger.warning("Refresh token session not found or expired")
                return None
            
            # Get current user data
            user = self.user_model.get_user_by_id(session['user_id'])
            if not user or not user['is_active']:
                logger.warning(f"User {session['user_id']} not found or inactive")
                return None
            
            # Create new access token
            token_data = {
                "sub": user['username'],
                "user_id": user['id'],
                "role": user['role_name'],
                "permissions": user['permissions']
            }
            
            new_access_token = self.create_access_token(token_data)
            
            # Optionally create new refresh token (token rotation)
            new_refresh_token = None
            if self._should_rotate_refresh_token(refresh_token):
                new_refresh_token = self.create_refresh_token(token_data)
                # Update session with new refresh token
                self.session_model.delete_session_by_token(refresh_token)
                self.session_model.create_session(
                    user_id=user['id'],
                    token=new_refresh_token,
                    expires_in_hours=self.refresh_token_expire_days * 24,
                    ip_address=session.get('ip_address'),
                    user_agent=session.get('user_agent')
                )
            
            result = {"access_token": new_access_token}
            if new_refresh_token:
                result["refresh_token"] = new_refresh_token
            
            logger.info(f"Access token refreshed for user {user['username']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return None
    
    def logout(self, refresh_token: str = None, user_id: int = None, all_sessions: bool = False) -> bool:
        """
        Logout user by invalidating sessions
        
        Args:
            refresh_token: Specific refresh token to invalidate
            user_id: User ID (for logout all sessions)
            all_sessions: Whether to logout from all sessions
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if all_sessions and user_id:
                # Logout from all sessions
                count = self.session_model.delete_user_sessions(user_id)
                logger.info(f"Logged out user {user_id} from {count} sessions")
                return count > 0
            elif refresh_token:
                # Logout from specific session
                success = self.session_model.delete_session_by_token(refresh_token)
                if success:
                    logger.info("Session logged out successfully")
                return success
            else:
                logger.warning("No refresh token or user_id provided for logout")
                return False
                
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    def get_current_user(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Get current user from access token
        
        Args:
            access_token: JWT access token
            
        Returns:
            User data if token is valid, None otherwise
        """
        try:
            payload = self.verify_token(access_token, "access")
            if not payload:
                return None
            
            # Get user from database
            user_id = payload.get("user_id")
            if not user_id:
                return None
            
            user = self.user_model.get_user_by_id(user_id)
            if not user or not user['is_active']:
                return None
            
            return user
            
        except Exception as e:
            logger.error(f"Failed to get current user: {e}")
            return None
    
    def validate_permission(self, user: Dict[str, Any], resource: str, action: str) -> bool:
        """
        Validate if user has specific permission
        
        Args:
            user: User data dictionary
            resource: Resource name
            action: Action name
            
        Returns:
            True if user has permission, False otherwise
        """
        try:
            permissions = user.get('permissions', {})
            if isinstance(permissions, str):
                permissions = json.loads(permissions)
            
            resource_permissions = permissions.get(resource, [])
            return action in resource_permissions
            
        except Exception as e:
            logger.error(f"Permission validation failed: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            count = self.session_model.cleanup_expired_sessions()
            logger.info(f"Cleaned up {count} expired sessions")
            return count
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def _is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited"""
        if identifier not in self._failed_attempts:
            return False
        
        attempts, last_attempt = self._failed_attempts[identifier]
        
        # Check if lockout period has expired
        if datetime.now() - last_attempt > self._lockout_duration:
            del self._failed_attempts[identifier]
            return False
        
        return attempts >= self._max_attempts
    
    def _record_failed_attempt(self, identifier: str):
        """Record a failed login attempt"""
        now = datetime.now()
        
        if identifier in self._failed_attempts:
            attempts, _ = self._failed_attempts[identifier]
            self._failed_attempts[identifier] = (attempts + 1, now)
        else:
            self._failed_attempts[identifier] = (1, now)
    
    def _reset_failed_attempts(self, identifier: str):
        """Reset failed login attempts"""
        if identifier in self._failed_attempts:
            del self._failed_attempts[identifier]
    
    def _should_rotate_refresh_token(self, refresh_token: str) -> bool:
        """
        Determine if refresh token should be rotated
        Currently rotates if token is more than half way to expiration
        """
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            
            issued_at = payload.get("iat")
            expires_at = payload.get("exp")
            
            if not issued_at or not expires_at:
                return True  # Rotate if we can't determine age
            
            now = datetime.now(timezone.utc).timestamp()
            token_age = now - issued_at
            token_lifetime = expires_at - issued_at
            
            # Rotate if token is more than 50% through its lifetime
            return token_age > (token_lifetime * 0.5)
            
        except Exception:
            return True  # Rotate on any error