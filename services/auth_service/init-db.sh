#!/bin/bash
set -e

echo "🔧 Initializing RAG Education Assistant Database..."

# Wait for database directory to be available
if [ ! -d "/app/data" ]; then
    echo "📁 Creating data directory..."
    mkdir -p /app/data
fi

# Set database path
DB_PATH="/app/data/rag_assistant.db"
echo "📍 Database path: $DB_PATH"

# Check if database already exists and has tables
# Check if database already exists and has tables. If so, exit.
if [ -f "$DB_PATH" ] && [ "$(sqlite3 "$DB_PATH" "SELECT COUNT(name) FROM sqlite_master WHERE type='table' AND name IN ('users', 'roles', 'user_sessions');" 2>/dev/null)" = "3" ]; then
    echo "✅ Database already initialized. Skipping."
    exit 0
fi

echo "🛠️  Running database migrations..."

# Copy migration files to container (if not already there)
MIGRATION_DIR="/app/migrations"
mkdir -p "$MIGRATION_DIR"

# Create migration files in container
cat > "$MIGRATION_DIR/001_create_tables.sql" << 'EOF'
-- Migration 001: Create tables for RAG Education Assistant User Authorization System
-- Created: 2025-11-01
-- Description: Initial database schema creation with users, roles, and sessions tables

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Create roles table
CREATE TABLE IF NOT EXISTS roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    permissions TEXT NOT NULL,  -- JSON string containing permissions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role_id INTEGER NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES roles(id)
);

-- Create user_sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT,
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role_id ON users(role_id);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

CREATE INDEX IF NOT EXISTS idx_roles_name ON roles(name);

CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at);

-- Create trigger to update updated_at timestamp on users table
CREATE TRIGGER IF NOT EXISTS update_users_updated_at
    AFTER UPDATE ON users
    FOR EACH ROW
    WHEN NEW.updated_at <= OLD.updated_at
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger to update updated_at timestamp on roles table
CREATE TRIGGER IF NOT EXISTS update_roles_updated_at
    AFTER UPDATE ON roles
    FOR EACH ROW
    WHEN NEW.updated_at <= OLD.updated_at
BEGIN
    UPDATE roles SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Verify tables were created successfully
SELECT 'Tables created successfully' as status;
EOF

cat > "$MIGRATION_DIR/002_insert_default_data.sql" << 'EOF'
-- Migration 002: Insert default data for RAG Education Assistant
-- Created: 2025-11-01
-- Description: Insert default roles and admin user

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Insert default roles with their permissions
INSERT OR IGNORE INTO roles (name, description, permissions) VALUES 
(
    'admin',
    'System Administrator with full access to all resources',
    '{"users": ["create", "read", "update", "delete"], "roles": ["create", "read", "update", "delete"], "sessions": ["create", "read", "update", "delete"], "documents": ["create", "read", "update", "delete"], "system": ["admin", "configure"]}'
),
(
    'teacher',
    'Teacher with session and document management access',
    '{"users": ["read"], "sessions": ["create", "read", "update", "delete"], "documents": ["create", "read", "update", "delete"], "students": ["read"]}'
),
(
    'student',
    'Student with limited read-only access to sessions and documents',
    '{"sessions": ["read"], "documents": ["read"]}'
);

-- Insert default admin user
-- Note: Default password is 'admin123' - should be changed on first login
-- Password hash generated with bcrypt for 'admin123'
INSERT OR IGNORE INTO users (
    username, 
    email, 
    password_hash, 
    role_id, 
    first_name, 
    last_name, 
    is_active
) VALUES (
    'admin',
    'admin@rag-assistant.local',
    '$2b$12$LQv3c1yqBwEHXb.NeGE0E.3r7.9d5qH5rJYC7KnQbJjMTwQAKN9Ke',  -- bcrypt hash for 'admin123'
    (SELECT id FROM roles WHERE name = 'admin'),
    'System',
    'Administrator',
    TRUE
);

-- Create sample teacher user (optional)
INSERT OR IGNORE INTO users (
    username,
    email,
    password_hash,
    role_id,
    first_name,
    last_name,
    is_active
) VALUES (
    'teacher_demo',
    'teacher@rag-assistant.local',
    '$2b$12$LQv3c1yqBwEHXb.NeGE0E.3r7.9d5qH5rJYC7KnQbJjMTwQAKN9Ke',  -- bcrypt hash for 'admin123'
    (SELECT id FROM roles WHERE name = 'teacher'),
    'Demo',
    'Teacher',
    TRUE
);

-- Create sample student user (optional)
INSERT OR IGNORE INTO users (
    username,
    email,
    password_hash,
    role_id,
    first_name,
    last_name,
    is_active
) VALUES (
    'student_demo',
    'student@rag-assistant.local',
    '$2b$12$LQv3c1yqBwEHXb.NeGE0E.3r7.9d5qH5rJYC7KnQbJjMTwQAKN9Ke',  -- bcrypt hash for 'admin123'
    (SELECT id FROM roles WHERE name = 'student'),
    'Demo',
    'Student',
    TRUE
);

-- Verify data insertion
SELECT 
    'Default data inserted successfully' as status,
    (SELECT COUNT(*) FROM roles) as roles_count,
    (SELECT COUNT(*) FROM users) as users_count;

-- Display created users for reference
SELECT 
    u.username,
    u.email,
    r.name as role,
    u.is_active
FROM users u
JOIN roles r ON u.role_id = r.id
ORDER BY r.name, u.username;
EOF

# Run migrations
echo "📝 Running migration 001: Create tables..."
sqlite3 "$DB_PATH" < "$MIGRATION_DIR/001_create_tables.sql"

echo "📊 Running migration 002: Insert default data..."
sqlite3 "$DB_PATH" < "$MIGRATION_DIR/002_insert_default_data.sql"

# Set proper permissions
chown appuser:appuser "$DB_PATH" 2>/dev/null || true
chmod 644 "$DB_PATH" 2>/dev/null || true

echo "✅ Database initialization completed successfully!"

# Show database info
echo "📈 Database Summary:"
sqlite3 "$DB_PATH" "SELECT 'Users: ' || COUNT(*) FROM users; SELECT 'Roles: ' || COUNT(*) FROM roles; SELECT 'Sessions: ' || COUNT(*) FROM user_sessions;"

echo ""
echo "🔑 Default Login Credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo "  (Please change the password after first login)"
echo ""