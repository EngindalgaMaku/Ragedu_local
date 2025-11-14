"""
Authentication Middleware for RAG Education Assistant Auth Service
Handles request authentication and authorization
"""

import logging
from typing import Callable, List, Optional, Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import json

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle authentication for protected routes
    """
    
    def __init__(
        self,
        app,
        auth_manager,
        excluded_paths: List[str] = None,
        require_auth: bool = True
    ):
        """
        Initialize authentication middleware
        
        Args:
            app: FastAPI application instance
            auth_manager: AuthManager instance
            excluded_paths: List of paths to exclude from authentication
            require_auth: Whether authentication is required by default
        """
        super().__init__(app)
        self.auth_manager = auth_manager
        self.excluded_paths = excluded_paths or [
            "/docs", "/redoc", "/openapi.json", "/favicon.ico",
            "/auth/login", "/auth/refresh", "/health"
        ]
        self.require_auth = require_auth
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through authentication middleware
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object
        """
        # Skip authentication for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Extract token from Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            if self.require_auth:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication required", "error": "missing_token"}
                )
            else:
                return await call_next(request)
        
        # Extract token
        token = authorization.split(" ")[1]
        
        try:
            # Validate token and get user
            user = self.auth_manager.get_current_user(token)
            if not user:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or expired token", "error": "invalid_token"}
                )
            
            # Add user to request state
            request.state.user = user
            request.state.token = token
            
            # Log successful authentication
            logger.debug(f"Authenticated user: {user['username']}")
            
            # Continue to next middleware/endpoint
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            
            return response
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Authentication service error"}
            )


class RequirePermission:
    """
    Dependency class to require specific permissions for endpoints
    """
    
    def __init__(self, resource: str, action: str, auth_manager=None):
        """
        Initialize permission requirement
        
        Args:
            resource: Resource name (e.g., 'users', 'documents')
            action: Action name (e.g., 'create', 'read', 'update', 'delete')
            auth_manager: AuthManager instance (will be injected if None)
        """
        self.resource = resource
        self.action = action
        self.auth_manager = auth_manager
    
    def __call__(self, request: Request) -> Dict[str, Any]:
        """
        Check if current user has required permission
        
        Args:
            request: FastAPI request object
            
        Returns:
            User data if authorized
            
        Raises:
            HTTPException: If user lacks required permission
        """
        # Get user from request state (set by middleware)
        user = getattr(request.state, 'user', None)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Use auth_manager from middleware if not provided
        if not self.auth_manager:
            # This should be set by the application
            self.auth_manager = getattr(request.app.state, 'auth_manager', None)
        
        if not self.auth_manager:
            logger.error("AuthManager not available for permission check")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authorization service error"
            )
        
        # Check permission
        if not self.auth_manager.validate_permission(user, self.resource, self.action):
            logger.warning(f"User {user['username']} lacks permission: {self.resource}:{self.action}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for {self.resource}:{self.action}"
            )
        
        return user


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for authentication endpoints
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_requests: int = 10,
        rate_limited_paths: List[str] = None
    ):
        """
        Initialize rate limiting middleware
        
        Args:
            app: FastAPI application instance
            requests_per_minute: Maximum requests per minute per IP
            burst_requests: Maximum burst requests
            rate_limited_paths: Paths to apply rate limiting to
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_requests = burst_requests
        self.rate_limited_paths = rate_limited_paths or ["/auth/login", "/auth/register"]
        
        # In-memory storage (use Redis in production)
        self._request_counts = {}
        self._burst_counts = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through rate limiting middleware
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object
        """
        # Skip rate limiting for non-rate-limited paths
        if request.url.path not in self.rate_limited_paths:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limits
        if self._is_rate_limited(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "error": "rate_limit_exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self._record_request(client_ip)
        
        return await call_next(request)
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is rate limited"""
        import time
        current_time = time.time()
        
        # Clean old entries (simple cleanup)
        self._cleanup_old_entries(current_time)
        
        # Check burst limit
        burst_key = f"{client_ip}:burst"
        if burst_key in self._burst_counts:
            if self._burst_counts[burst_key] >= self.burst_requests:
                return True
        
        # Check per-minute limit
        minute_key = f"{client_ip}:{int(current_time // 60)}"
        if minute_key in self._request_counts:
            if self._request_counts[minute_key] >= self.requests_per_minute:
                return True
        
        return False
    
    def _record_request(self, client_ip: str):
        """Record a request for rate limiting"""
        import time
        current_time = time.time()
        
        # Record for burst limiting
        burst_key = f"{client_ip}:burst"
        self._burst_counts[burst_key] = self._burst_counts.get(burst_key, 0) + 1
        
        # Record for per-minute limiting
        minute_key = f"{client_ip}:{int(current_time // 60)}"
        self._request_counts[minute_key] = self._request_counts.get(minute_key, 0) + 1
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old rate limiting entries"""
        current_minute = int(current_time // 60)
        
        # Clean old minute entries
        old_keys = [k for k in self._request_counts.keys() 
                   if int(k.split(':')[1]) < current_minute - 2]
        for key in old_keys:
            del self._request_counts[key]
        
        # Clean burst entries every 5 seconds (daha sık cleanup)
        if int(current_time) % 5 == 0:
            self._burst_counts.clear()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses
    """
    
    def __init__(self, app):
        """
        Initialize security headers middleware
        
        Args:
            app: FastAPI application instance
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object with security headers
        """
        response = await call_next(request)
        
        # Add security headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        })
        
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware for the auth service
    """
    
    def __init__(
        self,
        app,
        allowed_origins: List[str] = None,
        allowed_methods: List[str] = None,
        allowed_headers: List[str] = None,
        allow_credentials: bool = True
    ):
        """
        Initialize CORS middleware
        
        Args:
            app: FastAPI application instance
            allowed_origins: List of allowed origins
            allowed_methods: List of allowed HTTP methods
            allowed_headers: List of allowed headers
            allow_credentials: Whether to allow credentials
        """
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or ["*"]
        self.allow_credentials = allow_credentials
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle CORS for requests
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object with CORS headers
        """
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = self._get_allowed_origin(origin)
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            return response
        
        # Process regular requests
        response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = self._get_allowed_origin(origin)
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
    
    def _get_allowed_origin(self, origin: Optional[str]) -> str:
        """Get allowed origin for the request"""
        # Never return wildcard when credentials are allowed
        if self.allow_credentials and origin and origin in self.allowed_origins:
            return origin
        
        # For non-credentialed requests, use wildcard if allowed
        if not self.allow_credentials and "*" in self.allowed_origins:
            return "*"
        
        if origin and origin in self.allowed_origins:
            return origin
        
        # Return first allowed origin instead of wildcard for credentials
        return self.allowed_origins[0] if self.allowed_origins else "null"