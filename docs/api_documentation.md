# RAG Education Assistant - API Documentation

## Overview

This document describes the API endpoints for the RAG Education Assistant system, including authentication, user management, and document processing endpoints.

## Base URLs

- **API Gateway**: `http://localhost:8000` (Development) / `https://your-domain.com/api` (Production)
- **Auth Service**: `http://localhost:8006` (Development) / `https://your-domain.com/auth` (Production)
- **Frontend**: `http://localhost:3000` (Development) / `https://your-domain.com` (Production)

## Authentication

### Overview

The system uses JWT (JSON Web Tokens) for authentication with the following token types:

- **Access Token**: Short-lived (30 minutes), used for API requests
- **Refresh Token**: Long-lived (7 days), used to obtain new access tokens

### Authentication Header

Include the access token in the Authorization header for protected endpoints:

```
Authorization: Bearer <access_token>
```

## Auth Service API (Port 8006)

### Authentication Endpoints

#### POST `/auth/login`

Authenticate user and receive JWT tokens.

**Request Body:**

```json
{
  "username": "string",
  "password": "string"
}
```

**Response (200):**

```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "user_id": "string",
    "username": "string",
    "email": "string",
    "role": "admin|teacher|student",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

**Error Responses:**

- `401`: Invalid credentials
- `423`: Account locked
- `500`: Internal server error

#### POST `/auth/refresh`

Refresh access token using refresh token.

**Request Body:**

```json
{
  "refresh_token": "string"
}
```

**Response (200):**

```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### POST `/auth/logout`

🔒 **Protected** - Logout and invalidate tokens.

**Request Body:**

```json
{
  "refresh_token": "string"
}
```

**Response (200):**

```json
{
  "message": "Logged out successfully"
}
```

#### POST `/auth/logout-all`

🔒 **Protected** - Logout from all devices.

**Response (200):**

```json
{
  "message": "Logged out from all devices",
  "sessions_terminated": 3
}
```

#### GET `/auth/me`

🔒 **Protected** - Get current user information.

**Response (200):**

```json
{
  "user_id": "string",
  "username": "string",
  "email": "string",
  "role": "admin|teacher|student",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00Z",
  "permissions": ["create_session", "read_documents", "etc"]
}
```

### User Management Endpoints

#### GET `/users`

🔒 **Protected** (Admin only) - List all users.

**Query Parameters:**

- `role`: Filter by role (optional)
- `is_active`: Filter by active status (optional)
- `limit`: Maximum number of results (default: 50)
- `offset`: Number of results to skip (default: 0)

**Response (200):**

```json
{
  "users": [
    {
      "user_id": "string",
      "username": "string",
      "email": "string",
      "role": "string",
      "is_active": true,
      "created_at": "2024-01-01T00:00:00Z",
      "last_login": "2024-01-01T12:00:00Z"
    }
  ],
  "total": 10,
  "limit": 50,
  "offset": 0
}
```

#### POST `/users`

🔒 **Protected** (Admin only) - Create new user.

**Request Body:**

```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "role": "admin|teacher|student",
  "is_active": true
}
```

**Response (201):**

```json
{
  "user_id": "string",
  "username": "string",
  "email": "string",
  "role": "string",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### GET `/users/{user_id}`

🔒 **Protected** (Admin or own user) - Get user details.

**Response (200):**

```json
{
  "user_id": "string",
  "username": "string",
  "email": "string",
  "role": "string",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00Z",
  "last_login": "2024-01-01T12:00:00Z",
  "session_count": 5
}
```

#### PUT `/users/{user_id}`

🔒 **Protected** (Admin or own user) - Update user.

**Request Body:**

```json
{
  "email": "string",
  "role": "admin|teacher|student",
  "is_active": true
}
```

#### DELETE `/users/{user_id}`

🔒 **Protected** (Admin only) - Delete user.

**Response (200):**

```json
{
  "message": "User deleted successfully",
  "user_id": "string"
}
```

### Role Management Endpoints

#### GET `/roles`

🔒 **Protected** - List all roles.

**Response (200):**

```json
{
  "roles": [
    {
      "role_id": "string",
      "name": "admin",
      "description": "Administrator with full access",
      "permissions": ["*"]
    }
  ]
}
```

#### POST `/roles`

🔒 **Protected** (Admin only) - Create custom role.

**Request Body:**

```json
{
  "name": "string",
  "description": "string",
  "permissions": ["permission1", "permission2"]
}
```

### Health Check Endpoints

#### GET `/health`

Service health check.

**Response (200):**

```json
{
  "status": "healthy",
  "service": "RAG Education Assistant - Auth Service",
  "version": "1.0.0",
  "database": "connected",
  "active_sessions": 10,
  "environment": "development|production"
}
```

#### GET `/info`

Service information and statistics.

**Response (200):**

```json
{
  "service": "RAG Education Assistant - Auth Service",
  "version": "1.0.0",
  "features": ["JWT Authentication", "Role-Based Access Control"],
  "statistics": {
    "total_users": 25,
    "active_users": 20,
    "total_roles": 3,
    "active_sessions": 10
  }
}
```

## API Gateway Endpoints (Port 8000)

### Session Management

#### GET `/sessions`

🔒 **Protected** - List user sessions.

**Query Parameters:**

- `created_by`: Filter by creator (optional)
- `category`: Filter by category (optional)
- `status`: Filter by status (optional)
- `limit`: Maximum results (default: 50)

**Response (200):**

```json
{
  "sessions": [
    {
      "session_id": "string",
      "name": "string",
      "description": "string",
      "category": "lesson|homework|exam|research",
      "status": "active|inactive",
      "created_by": "string",
      "created_at": "2024-01-01T00:00:00Z",
      "document_count": 5,
      "total_chunks": 150,
      "query_count": 25
    }
  ]
}
```

#### POST `/sessions`

🔒 **Protected** - Create new session.

**Request Body:**

```json
{
  "name": "string",
  "description": "string",
  "category": "lesson|homework|exam|research",
  "grade_level": "string",
  "subject_area": "string",
  "learning_objectives": ["objective1", "objective2"],
  "tags": ["tag1", "tag2"],
  "is_public": false
}
```

#### GET `/sessions/{session_id}`

🔒 **Protected** - Get session details.

#### DELETE `/sessions/{session_id}`

🔒 **Protected** - Delete session.

**Query Parameters:**

- `create_backup`: Create backup before deletion (default: true)

### Document Processing

#### POST `/documents/convert-document-to-markdown`

🔒 **Protected** - Convert document to markdown.

**Request:** Multipart form with file upload

**Response (200):**

```json
{
  "success": true,
  "message": "Document successfully converted",
  "markdown_filename": "document.md",
  "metadata": {
    "pages": 10,
    "word_count": 2500
  }
}
```

#### POST `/documents/process-and-store`

🔒 **Protected** - Process documents and store vectors.

**Request Body (Form Data):**

- `session_id`: Target session ID
- `markdown_files`: JSON array of markdown filenames
- `chunk_strategy`: "semantic" or "fixed" (default: "semantic")
- `chunk_size`: Chunk size in characters (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 100)

#### GET `/documents/list-markdown`

🔒 **Protected** - List available markdown files.

**Response (200):**

```json
{
  "markdown_files": ["file1.md", "file2.md"],
  "count": 2
}
```

### Query Processing

#### POST `/rag/query`

🔒 **Protected** - Submit RAG query.

**Request Body:**

```json
{
  "session_id": "string",
  "query": "string",
  "top_k": 5,
  "use_rerank": true,
  "min_score": 0.1,
  "max_context_chars": 8000,
  "model": "llama-3.1-8b-instant"
}
```

**Response (200):**

```json
{
  "answer": "string",
  "sources": [
    {
      "content": "string",
      "source_file": "string",
      "score": 0.95,
      "metadata": {}
    }
  ]
}
```

### Health Check Endpoints

#### GET `/health`

API Gateway health check.

**Response (200):**

```json
{
  "status": "ok",
  "service": "api-gateway",
  "version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET `/health/services`

Check health of all microservices.

**Response (200):**

```json
{
  "gateway": "healthy",
  "overall_status": "healthy",
  "services": {
    "auth_service": {
      "status": "healthy",
      "url": "http://localhost:8006",
      "response_time": 0.025
    },
    "pdf_processor": {
      "status": "healthy",
      "url": "https://pdf-processor-url",
      "response_time": 0.15
    }
  }
}
```

#### GET `/health/comprehensive`

Comprehensive system health check including authentication flow.

**Response (200):**

```json
{
  "gateway": {
    "status": "healthy",
    "checks": [
      {
        "name": "session_manager",
        "status": "healthy",
        "details": "Session manager operational"
      }
    ]
  },
  "authentication": {
    "status": "healthy",
    "checks": [
      {
        "name": "auth_service_health",
        "status": "healthy",
        "details": "Auth service operational: 5 active sessions"
      }
    ]
  },
  "overall_status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Error Responses

### Standard Error Format

```json
{
  "error": "error_type",
  "detail": "Human-readable error message",
  "status_code": 400
}
```

### Common Error Codes

- **400**: Bad Request - Invalid request format or parameters
- **401**: Unauthorized - Invalid or missing authentication
- **403**: Forbidden - Insufficient permissions
- **404**: Not Found - Resource does not exist
- **409**: Conflict - Resource already exists
- **422**: Validation Error - Request validation failed
- **429**: Too Many Requests - Rate limit exceeded
- **500**: Internal Server Error - Server-side error
- **503**: Service Unavailable - External service unavailable

## Rate Limiting

Authentication endpoints are rate limited:

- **Default**: 60 requests per minute
- **Burst**: 10 additional requests allowed
- **Headers**: Rate limit information in response headers

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

## Demo Accounts

### Development Environment

The following demo accounts are available for testing:

| Username | Password | Role    | Description                     |
| -------- | -------- | ------- | ------------------------------- |
| admin    | admin    | admin   | Full system access              |
| teacher  | teacher  | teacher | Session and document management |
| student  | student  | student | Read-only access                |

⚠️ **Security Warning**: Change or disable demo accounts in production!

## SDK and Client Libraries

### JavaScript/TypeScript

Frontend integration example:

```typescript
// AuthContext integration
const { login, logout, user, isAuthenticated } = useAuth();

// Login
await login("username", "password");

// API calls with automatic token handling
const response = await fetch("/api/sessions", {
  headers: {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  },
});
```

### Python

Backend integration example:

```python
import httpx

# Login and get tokens
async with httpx.AsyncClient() as client:
    response = await client.post('http://localhost:8006/auth/login', json={
        'username': 'admin',
        'password': 'admin'
    })
    tokens = response.json()

    # Use access token for authenticated requests
    headers = {'Authorization': f"Bearer {tokens['access_token']}"}
    sessions = await client.get('http://localhost:8000/sessions', headers=headers)
```

## Postman Collection

A Postman collection with all endpoints is available at:
`docs/RAG_Education_Assistant.postman_collection.json`

This includes:

- Pre-configured authentication
- Environment variables for different deployments
- Example requests and responses
- Automated token refresh

## WebSocket Connections (Future)

Real-time features will be added in future versions:

- Live query processing status
- Real-time collaboration on sessions
- System health monitoring dashboard
- User activity notifications

## API Versioning

Current API version: **v1**

Future versions will be backward compatible with version-specific endpoints:

- `/v1/auth/login` (current)
- `/v2/auth/login` (future)

## OpenAPI Documentation

Interactive API documentation is available at:

- **Auth Service**: http://localhost:8006/docs
- **API Gateway**: http://localhost:8000/docs

These provide:

- Interactive endpoint testing
- Automatic request/response examples
- Schema validation
- Authentication integration
