# RAG Education Assistant - User Authorization Database System

## Overview

This directory contains the complete user authorization system for the RAG Education Assistant, implemented using SQLite for compatibility and Python models for data management.

## Directory Structure

```
src/database/
├── __init__.py                 # Package initialization
├── database.py                # Database connection and management utilities
├── create_admin.py            # Interactive admin user creation script
├── test_database.py           # Comprehensive test suite
├── README.md                  # This documentation file
├── models/
│   ├── __init__.py            # Models package initialization
│   ├── user.py                # User model and authentication
│   ├── role.py                # Role and permission management
│   └── session.py             # Session management and JWT handling
└── migrations/
    ├── 001_create_tables.sql  # Initial table creation
    └── 002_insert_default_data.sql  # Default roles and users
```

## Database Schema

### Tables

#### `roles`

- **id**: Integer primary key
- **name**: Unique role name (admin, teacher, student)
- **description**: Role description
- **permissions**: JSON string containing resource permissions
- **created_at/updated_at**: Timestamps

#### `users`

- **id**: Integer primary key
- **username**: Unique username
- **email**: Unique email address
- **password_hash**: bcrypt hashed password
- **role_id**: Foreign key to roles table
- **first_name/last_name**: User names
- **is_active**: Boolean active status
- **created_at/updated_at/last_login**: Timestamps

#### `user_sessions`

- **id**: Integer primary key
- **user_id**: Foreign key to users table (CASCADE DELETE)
- **token_hash**: SHA-256 hashed session token
- **expires_at**: Session expiration timestamp
- **created_at**: Session creation timestamp
- **ip_address/user_agent**: Client information

### Indexes

Performance-optimized indexes on:

- `users(email, username, role_id, is_active)`
- `roles(name)`
- `user_sessions(token_hash, user_id, expires_at)`

## Default Roles and Permissions

### Admin Role

- **Full system access**: All CRUD operations on all resources
- **System administration**: Configure system settings
- **User management**: Create, modify, delete users and roles

### Teacher Role

- **Session management**: Create, read, update, delete learning sessions
- **Document management**: Upload, manage, and organize documents
- **Student monitoring**: View student progress and activities
- **Limited user access**: Read-only access to user information

### Student Role

- **Session participation**: Access to assigned learning sessions
- **Document access**: Read-only access to shared documents
- **Limited scope**: No administrative or management capabilities

## Usage Examples

### Initialize Database

```python
from src.database.database import DatabaseManager
from src.database.models.user import User
from src.database.models.role import Role
from src.database.models.session import UserSession

# Initialize database manager
db_manager = DatabaseManager("data/rag_assistant.db")

# Initialize models
user_model = User(db_manager)
role_model = Role(db_manager)
session_model = UserSession(db_manager)
```

### User Authentication

```python
# Authenticate user
user_data = user_model.authenticate_user("username", "password")
if user_data:
    print(f"Welcome {user_data['first_name']}!")
    print(f"Role: {user_data['role_name']}")
else:
    print("Authentication failed")
```

### Session Management

```python
# Create session
token = session_model.generate_session_token()
session_id = session_model.create_session(
    user_id=user_data['id'],
    token=token,
    expires_in_hours=24,
    ip_address="127.0.0.1"
)

# Validate session
session = session_model.validate_session(token)
if session:
    print(f"Session valid for {session['username']}")
```

### Permission Checking

```python
# Check user permission
has_permission = user_model.has_permission(
    user_id=user_data['id'],
    resource='documents',
    action='create'
)

if has_permission:
    # Allow document creation
    pass
```

## Security Features

### Password Security

- **bcrypt hashing**: Industry-standard password hashing with salt
- **Minimum complexity**: Enforced in application layer
- **Secure password reset**: Admin-controlled password reset functionality

### Session Security

- **Token hashing**: Session tokens stored as SHA-256 hashes
- **Expiration management**: Automatic cleanup of expired sessions
- **IP tracking**: Monitor session usage patterns
- **Secure token generation**: Cryptographically secure random tokens

### Database Security

- **Foreign key constraints**: Data integrity enforcement
- **SQL injection protection**: Parameterized queries throughout
- **Connection management**: Proper connection handling and cleanup

## Admin Management Scripts

### Create Admin User

```bash
python src/database/create_admin.py
```

Interactive script for:

- Creating new admin users
- Resetting admin passwords
- Listing existing admin accounts

### Run Database Tests

```bash
python src/database/test_database.py
```

Comprehensive test suite covering:

- Database connectivity
- Table and index creation
- Model functionality
- Authentication and authorization
- Session management
- Performance validation

## Migration System

### Running Migrations

```python
from src.database.database import DatabaseManager

db_manager = DatabaseManager()
db_manager.execute_migration('src/database/migrations/001_create_tables.sql')
db_manager.execute_migration('src/database/migrations/002_insert_default_data.sql')
```

### Migration Files

- **001_create_tables.sql**: Creates all database tables and indexes
- **002_insert_default_data.sql**: Inserts default roles and sample users

## Default Login Credentials

⚠️ **Security Notice**: Change default passwords immediately after installation!

### Default Admin Account

- **Username**: `admin`
- **Password**: `admin123`
- **Email**: `admin@rag-assistant.local`

### Demo Accounts

- **Teacher**: `teacher_demo` / `admin123`
- **Student**: `student_demo` / `admin123`

## Configuration

### Environment Variables

- `DB_PATH`: Override default database path (default: `data/rag_assistant.db`)

### Database Settings

- **Connection timeout**: 30 seconds
- **Foreign key enforcement**: Enabled
- **Default session expiration**: 24 hours
- **Automatic cleanup**: Expired sessions removed on access

## Best Practices

### Development

1. **Always use models**: Don't write raw SQL queries
2. **Handle exceptions**: Proper error handling throughout
3. **Test thoroughly**: Run test suite before deployment
4. **Use transactions**: For multi-step operations

### Production

1. **Secure database file**: Proper file permissions
2. **Regular backups**: Automated database backups
3. **Monitor sessions**: Regular cleanup of expired sessions
4. **Log security events**: Authentication failures and admin actions

### Security

1. **Change default passwords**: Immediately after installation
2. **Use HTTPS**: For all authentication requests
3. **Implement rate limiting**: Prevent brute force attacks
4. **Regular updates**: Keep dependencies updated

## Troubleshooting

### Common Issues

#### Database Locked Error

```python
# Solution: Ensure proper connection cleanup
with db_manager.get_connection() as conn:
    # Your database operations here
    pass  # Connection automatically closed
```

#### Permission Denied

- Check file permissions on database directory
- Ensure write access for application user

#### Authentication Failures

- Verify user is active (`is_active = True`)
- Check password complexity requirements
- Confirm role assignments

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('src.database').setLevel(logging.DEBUG)
```

## API Integration

The database system is designed to integrate with FastAPI or similar web frameworks:

```python
from fastapi import HTTPException, Depends
from src.database import get_db_manager, User, UserSession

async def get_current_user(token: str):
    db_manager = get_db_manager()
    session_model = UserSession(db_manager)

    session = session_model.validate_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    return session
```

## Performance Considerations

### Optimizations Implemented

- **Indexed queries**: All frequently used columns are indexed
- **Connection pooling**: Efficient connection management
- **Query optimization**: Minimal database calls per operation
- **Batch operations**: Efficient bulk operations where needed

### Monitoring

- **Query performance**: Monitor slow queries
- **Session cleanup**: Regular expired session removal
- **Database size**: Monitor growth and implement archiving if needed

## License and Support

This database system is part of the RAG Education Assistant project. For support and contributions, please refer to the main project documentation.

---

**Last Updated**: November 2025
**Database Version**: 1.0
**Python Compatibility**: 3.8+
**Dependencies**: sqlite3 (built-in), bcrypt, secrets
