"""
Authentication module for RAG Education Assistant Auth Service
"""

from .auth_manager import AuthManager
from .middleware import AuthenticationMiddleware, RequirePermission
from .dependencies import get_current_user, get_current_active_user, require_permission
from .schemas import *

__all__ = [
    'AuthManager',
    'AuthenticationMiddleware',
    'RequirePermission',
    'get_current_user',
    'get_current_active_user',
    'require_permission'
]