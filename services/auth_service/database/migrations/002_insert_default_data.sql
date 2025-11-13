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