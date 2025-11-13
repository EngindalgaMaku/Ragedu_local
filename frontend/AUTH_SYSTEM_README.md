# 🔐 RAG Education Assistant - Frontend Authentication System

A comprehensive, production-ready authentication system for the RAG Education Assistant frontend, featuring JWT token management, role-based access control, and seamless integration with the backend authentication microservice.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Components](#-components)
- [Hooks](#-hooks)
- [Types](#-types)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Security](#-security)
- [Testing](#-testing)
- [Deployment](#-deployment)

## ✨ Features

### Core Authentication

- **JWT Token Management** - Automatic token refresh and storage
- **Role-Based Access Control** - Admin, Teacher, Student roles with permissions
- **Session Management** - Persistent sessions with timeout handling
- **Multi-tab Synchronization** - Auth state syncs across browser tabs

### Security Features

- **Secure Token Storage** - Safe localStorage management with encryption
- **Auto-logout on Expiry** - Automatic logout when tokens expire
- **CSRF Protection** - Request signing and validation
- **XSS Protection** - Safe token handling and storage

### User Experience

- **Loading States** - Smooth loading indicators during auth operations
- **Error Handling** - User-friendly error messages and recovery
- **Remember Me** - Optional persistent sessions
- **Demo Credentials** - Built-in demo accounts for testing

### Developer Experience

- **TypeScript Support** - Full type safety throughout
- **Custom Hooks** - Specialized hooks for different auth operations
- **Component Protection** - Easy-to-use protection components
- **Hot Reload Compatible** - Works seamlessly with Next.js dev server

## 🏗️ Architecture

```
frontend/
├── types/auth.ts                    # TypeScript type definitions
├── contexts/AuthContext.tsx         # Main authentication context
├── hooks/useAuth.ts                # Authentication hooks
├── lib/
│   ├── token-manager.ts            # Token management utility
│   ├── api-client.ts               # Axios client with auth
│   └── auth-api.ts                 # Auth service API functions
├── components/auth/
│   ├── ProtectedRoute.tsx          # Route protection
│   ├── AuthGuard.tsx               # Component protection
│   ├── RoleGuard.tsx               # Role-based protection
│   ├── LoginForm.tsx               # Enhanced login form
│   └── LogoutButton.tsx            # Enhanced logout button
└── utils/permissions.ts            # Permission utilities
```

## 🚀 Quick Start

### 1. Environment Setup

Copy the environment template:

```bash
cp frontend/.env.example frontend/.env.local
```

Configure your environment variables:

```env
NEXT_PUBLIC_AUTH_SERVICE_URL=http://localhost:8002
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2. Authentication Provider Setup

The AuthProvider is already configured in your app layout:

```tsx
// frontend/app/layout.tsx
import { AuthProvider } from "@/contexts/AuthContext";

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
```

### 3. Basic Usage

```tsx
// In any component
import { useAuth } from "@/hooks/useAuth";

function MyComponent() {
  const { user, isAuthenticated, login, logout } = useAuth();

  if (!isAuthenticated) {
    return <div>Please log in</div>;
  }

  return (
    <div>
      <h1>Welcome, {user?.first_name}!</h1>
      <button onClick={() => logout()}>Logout</button>
    </div>
  );
}
```

## 🧩 Components

### ProtectedRoute

Protects entire routes based on authentication and roles:

```tsx
import ProtectedRoute from '@/components/auth/ProtectedRoute';

// Protect a route for teachers and admins only
<ProtectedRoute requiredRole="teacher">
  <TeacherDashboard />
</ProtectedRoute>

// Protect with specific permissions
<ProtectedRoute requiredPermissions={[{resource: 'users', action: 'create'}]}>
  <AdminPanel />
</ProtectedRoute>
```

### AuthGuard

Protects individual components:

```tsx
import AuthGuard from '@/components/auth/AuthGuard';

// Show component only to specific roles
<AuthGuard roles={['admin', 'teacher']} fallback={<div>Access denied</div>}>
  <SensitiveComponent />
</AuthGuard>

// Show component with permissions
<AuthGuard permissions={[{resource: 'documents', action: 'delete'}]}>
  <DeleteButton />
</AuthGuard>
```

### RoleGuard

Simple role-based protection:

```tsx
import { RoleGuard } from "@/components/auth/RoleGuard";

<RoleGuard allowedRoles={["admin"]}>
  <AdminOnlyFeature />
</RoleGuard>;
```

### LoginForm

Enhanced login form with validation:

```tsx
import LoginForm from "@/components/auth/LoginForm";

<LoginForm
  onLogin={(response) => console.log("Login success:", response)}
  onError={(error) => console.log("Login error:", error)}
  showRememberMe={true}
  className="w-full max-w-md"
/>;
```

### LogoutButton

Customizable logout button:

```tsx
import LogoutButton from '@/components/auth/LogoutButton';

// Basic logout button
<LogoutButton />

// Custom logout button
<LogoutButton
  onLogout={() => console.log('Logged out')}
  className="custom-style"
  logoutAll={true} // For admins - logout all sessions
/>
```

## 🎣 Hooks

### useAuth()

Main authentication hook:

```tsx
const {
  user, // Current user object
  isAuthenticated, // Boolean auth status
  isLoading, // Loading state
  permissions, // User permissions
  login, // Login function
  logout, // Logout function
  refreshToken, // Manual token refresh
  checkPermission, // Check specific permission
  hasRole, // Check user role
  updateLastActivity, // Update activity timestamp
} = useAuth();
```

### useLogin()

Specialized login hook:

```tsx
const { login, isLoading, error } = useLogin();

const handleLogin = async (credentials) => {
  try {
    await login(credentials);
    // Handle success
  } catch (error) {
    // Handle error
  }
};
```

### useLogout()

Specialized logout hook:

```tsx
const { logout, logoutAll, isLoading } = useLogout();

// Logout current session
await logout();

// Logout all sessions (admin only)
await logoutAll();
```

### usePermissions()

Permission checking hook:

```tsx
const { checkPermission, hasPermission, getUserPermissions } = usePermissions();

const canEdit = checkPermission("documents", "update");
const userPerms = getUserPermissions();
```

### useRoles()

Role management hook:

```tsx
const { userRole, hasRole, isAdmin, isTeacher, isStudent } = useRoles();

if (isAdmin) {
  // Admin-specific logic
}
```

## 📝 Types

### Core Types

```typescript
interface User {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  is_active: boolean;
  role_id: number;
  role_name: string;
  created_at: string;
  updated_at: string;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  permissions: Record<string, string[]>;
  tokens: TokenInfo;
  lastActivity: number;
}

interface TokenInfo {
  access_token: string;
  refresh_token: string;
  expires_at: number;
  user: User;
}
```

## ⚙️ Configuration

### Environment Variables

```env
# Required
NEXT_PUBLIC_AUTH_SERVICE_URL=http://localhost:8002
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional
NEXT_PUBLIC_SESSION_TIMEOUT=3600000      # 1 hour in ms
NEXT_PUBLIC_TOKEN_REFRESH_THRESHOLD=300000 # 5 minutes in ms
NEXT_PUBLIC_MAX_LOGIN_ATTEMPTS=5
NEXT_PUBLIC_SHOW_DEBUG_INFO=false
```

### API Client Configuration

```typescript
// Automatically configured with:
const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_AUTH_SERVICE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});
```

## 🔧 Usage Examples

### Role-Based Dashboard

```tsx
function Dashboard() {
  const { user, hasRole } = useAuth();

  if (hasRole("student")) {
    return <StudentInterface />;
  }

  if (hasRole("teacher")) {
    return <TeacherInterface />;
  }

  if (hasRole("admin")) {
    return <AdminInterface />;
  }

  return <DefaultInterface />;
}
```

### Permission-Based Features

```tsx
function DocumentManager() {
  const { checkPermission } = usePermissions();

  return (
    <div>
      <DocumentList />

      {checkPermission("documents", "create") && <CreateDocumentButton />}

      {checkPermission("documents", "delete") && <DeleteDocumentButton />}
    </div>
  );
}
```

### Protected API Calls

```tsx
// API calls automatically include auth tokens
import { useApi } from "@/hooks/useAuth";

function MyComponent() {
  const api = useApi();

  const fetchData = async () => {
    try {
      // Token is automatically attached
      const response = await api.get("/protected-endpoint");
      return response.data;
    } catch (error) {
      // Automatic token refresh and retry on 401
      console.error("API Error:", error);
    }
  };
}
```

## 🛡️ Security

### Token Security

- Tokens are stored in localStorage with encryption
- Automatic cleanup on logout
- Secure token refresh mechanism
- Protection against XSS attacks

### Session Security

- Automatic logout on token expiry
- Activity-based session timeout
- Cross-tab synchronization
- Session fixation protection

### Permission Security

- Server-side permission validation
- Client-side permission caching
- Role hierarchy enforcement
- Permission-based UI rendering

## 🧪 Testing

Comprehensive testing strategy included:

```bash
# Run the test plan
See: frontend/tests/auth-integration.test.md

# Manual testing checklist
- Login/logout flows
- Role-based access
- Permission checks
- Token refresh
- Error handling
- UI responsiveness
```

### Demo Accounts

```typescript
// Built-in demo accounts for testing
const demoAccounts = {
  admin: { username: "admin", password: "admin123" },
  teacher: { username: "teacher", password: "teacher123" },
  student: { username: "student", password: "student123" },
};
```

## 🚢 Deployment

### Pre-deployment Checklist

- [ ] Update environment variables for production
- [ ] Test all auth flows in staging environment
- [ ] Verify token expiry and refresh logic
- [ ] Test role-based access controls
- [ ] Check security headers and HTTPS
- [ ] Validate error handling and recovery

### Production Configuration

```env
# Production environment
NEXT_PUBLIC_AUTH_SERVICE_URL=https://your-auth-service.com
NEXT_PUBLIC_API_URL=https://your-api.com
NEXT_PUBLIC_SHOW_DEBUG_INFO=false
NEXT_PUBLIC_ENABLE_CONSOLE_LOGS=false
```

## 📚 Additional Resources

- **Authentication Service API**: See `services/auth_service/README.md`
- **Integration Tests**: See `frontend/tests/auth-integration.test.md`
- **Type Definitions**: See `frontend/types/auth.ts`
- **Permission Matrix**: See `frontend/utils/permissions.ts`

## 🤝 Contributing

When extending the authentication system:

1. **Follow TypeScript patterns** - All new code should be fully typed
2. **Update tests** - Add tests for new functionality
3. **Document changes** - Update this README for any new features
4. **Security review** - Have security-sensitive changes reviewed
5. **Test integration** - Verify changes work with auth service

## 📞 Support

For issues with the authentication system:

1. Check the integration test plan for debugging steps
2. Review the auth service logs for backend issues
3. Verify environment configuration
4. Check browser developer tools for client-side issues

---

**Built with ❤️ for the RAG Education Assistant**

This authentication system provides enterprise-grade security and user experience for educational technology applications.
