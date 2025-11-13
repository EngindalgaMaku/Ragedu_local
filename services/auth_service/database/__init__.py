"""
Database package for RAG Education Assistant
Handles user authorization system with SQLite backend
"""

from .database import DatabaseManager
from .models.user import User
from .models.role import Role
from .models.session import UserSession

__all__ = ['DatabaseManager', 'User', 'Role', 'UserSession']