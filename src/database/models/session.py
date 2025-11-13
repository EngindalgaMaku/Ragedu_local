"""
Session model for RAG Education Assistant
Handles user session management and JWT tokens
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import secrets
import logging

logger = logging.getLogger(__name__)


class UserSession:
    """
    User session model for managing authentication sessions
    """
    
    def __init__(self, db_manager):
        """
        Initialize UserSession model
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    def create_session(self, user_id: int, token: str, expires_in_hours: int = 24,
                      ip_address: str = None, user_agent: str = None) -> Optional[int]:
        """
        Create a new user session
        
        Args:
            user_id: User ID
            token: Session token (JWT or similar)
            expires_in_hours: Session expiration in hours
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            Session ID if successful, None if failed
        """
        try:
            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            # Hash the token for secure storage
            token_hash = self.db.hash_token(token)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO user_sessions (user_id, token_hash, expires_at, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, token_hash, expires_at, ip_address, user_agent))
                
                conn.commit()
                session_id = cursor.lastrowid
                
                logger.info(f"Session created for user {user_id} with ID {session_id}")
                return session_id
                
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            return None
    
    def get_session_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get session by token
        
        Args:
            token: Session token
            
        Returns:
            Session data dictionary or None
        """
        try:
            token_hash = self.db.hash_token(token)
            return self.db.get_session_by_token(token_hash)
            
        except Exception as e:
            logger.error(f"Failed to get session by token: {e}")
            return None
    
    def get_session_by_id(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data dictionary or None
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT s.*, u.username, u.email, u.role_id, r.name as role_name, r.permissions
                    FROM user_sessions s
                    JOIN users u ON s.user_id = u.id
                    JOIN roles r ON u.role_id = r.id
                    WHERE s.id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Failed to get session by ID {session_id}: {e}")
            return None
    
    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate session token and return user data
        
        Args:
            token: Session token
            
        Returns:
            User data if session is valid, None otherwise
        """
        try:
            session = self.get_session_by_token(token)
            
            if not session:
                logger.debug("Session not found")
                return None
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session['expires_at'].replace('Z', '+00:00'))
            if datetime.utcnow() > expires_at:
                logger.debug(f"Session {session['id']} expired")
                # Clean up expired session
                self.delete_session_by_token(token)
                return None
            
            # Check if user is still active
            if not session.get('is_active', True):
                logger.debug(f"User {session['user_id']} is inactive")
                return None
            
            logger.debug(f"Session validated for user {session['username']}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to validate session: {e}")
            return None
    
    def refresh_session(self, token: str, expires_in_hours: int = 24) -> bool:
        """
        Refresh session expiration time
        
        Args:
            token: Session token
            expires_in_hours: New expiration in hours
            
        Returns:
            True if successful, False otherwise
        """
        try:
            token_hash = self.db.hash_token(token)
            new_expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            with self.db.get_connection() as conn:
                result = conn.execute("""
                    UPDATE user_sessions 
                    SET expires_at = ?
                    WHERE token_hash = ?
                """, (new_expires_at, token_hash))
                
                conn.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"Session refreshed")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to refresh session: {e}")
            return False
    
    def delete_session_by_token(self, token: str) -> bool:
        """
        Delete session by token
        
        Args:
            token: Session token
            
        Returns:
            True if successful, False otherwise
        """
        try:
            token_hash = self.db.hash_token(token)
            return self.db.delete_session(token_hash)
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def delete_session_by_id(self, session_id: int) -> bool:
        """
        Delete session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db.get_connection() as conn:
                result = conn.execute("""
                    DELETE FROM user_sessions WHERE id = ?
                """, (session_id,))
                
                conn.commit()
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"Session {session_id} deleted")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def delete_user_sessions(self, user_id: int) -> int:
        """
        Delete all sessions for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Number of sessions deleted
        """
        try:
            with self.db.get_connection() as conn:
                result = conn.execute("""
                    DELETE FROM user_sessions WHERE user_id = ?
                """, (user_id,))
                
                conn.commit()
                deleted_count = result.rowcount
                
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} sessions for user {user_id}")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete sessions for user {user_id}: {e}")
            return 0
    
    def get_user_sessions(self, user_id: int, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user
        
        Args:
            user_id: User ID
            active_only: Only return non-expired sessions
            
        Returns:
            List of session dictionaries
        """
        try:
            query = """
                SELECT id, user_id, expires_at, created_at, ip_address, user_agent
                FROM user_sessions
                WHERE user_id = ?
            """
            params = [user_id]
            
            if active_only:
                query += " AND expires_at > CURRENT_TIMESTAMP"
            
            query += " ORDER BY created_at DESC"
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(query, params)
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append(dict(row))
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to get sessions for user {user_id}: {e}")
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up all expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            with self.db.get_connection() as conn:
                result = conn.execute("""
                    DELETE FROM user_sessions 
                    WHERE expires_at < CURRENT_TIMESTAMP
                """)
                conn.commit()
                
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired sessions")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_active_sessions_count(self, user_id: int = None) -> int:
        """
        Get count of active sessions
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            Number of active sessions
        """
        try:
            query = """
                SELECT COUNT(*) as count
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.expires_at > CURRENT_TIMESTAMP AND u.is_active = 1
            """
            params = []
            
            if user_id:
                query += " AND s.user_id = ?"
                params.append(user_id)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(query, params)
                row = cursor.fetchone()
                return row['count'] if row else 0
                
        except Exception as e:
            logger.error(f"Failed to get active sessions count: {e}")
            return 0
    
    def list_active_sessions(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all active sessions with user information
        
        Args:
            offset: Pagination offset
            limit: Maximum number of results
            
        Returns:
            List of session dictionaries with user data
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT s.id, s.user_id, s.expires_at, s.created_at, s.ip_address,
                           u.username, u.email, u.first_name, u.last_name, r.name as role_name
                    FROM user_sessions s
                    JOIN users u ON s.user_id = u.id
                    JOIN roles r ON u.role_id = r.id
                    WHERE s.expires_at > CURRENT_TIMESTAMP AND u.is_active = 1
                    ORDER BY s.created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append(dict(row))
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to list active sessions: {e}")
            return []
    
    def generate_session_token(self) -> str:
        """
        Generate a secure session token
        
        Returns:
            Random token string
        """
        return secrets.token_urlsafe(32)
    
    def update_session_activity(self, token: str, ip_address: str = None, user_agent: str = None) -> bool:
        """
        Update session activity information
        
        Args:
            token: Session token
            ip_address: New IP address
            user_agent: New user agent
            
        Returns:
            True if successful, False otherwise
        """
        try:
            token_hash = self.db.hash_token(token)
            
            updates = []
            params = []
            
            if ip_address is not None:
                updates.append("ip_address = ?")
                params.append(ip_address)
            
            if user_agent is not None:
                updates.append("user_agent = ?")
                params.append(user_agent)
            
            if not updates:
                return True  # No updates needed
            
            params.append(token_hash)  # For WHERE clause
            
            query = f"""
                UPDATE user_sessions 
                SET {', '.join(updates)}
                WHERE token_hash = ?
            """
            
            with self.db.get_connection() as conn:
                result = conn.execute(query, params)
                conn.commit()
                
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
            return False
    
    def is_session_valid(self, token: str) -> bool:
        """
        Simple check if session token is valid (not expired)
        
        Args:
            token: Session token
            
        Returns:
            True if valid, False otherwise
        """
        session = self.validate_session(token)
        return session is not None