/**
 * TypeScript Type Definitions for RAG Education Assistant Authentication
 * Frontend types that match the backend auth service schemas
 */

// ===== USER TYPES =====

export interface User {
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
  last_login?: string | null;
}

export interface UserCreate {
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  password: string;
  role_id: number;
  is_active?: boolean;
}

export interface UserUpdate {
  email?: string;
  first_name?: string;
  last_name?: string;
  is_active?: boolean;
  role_id?: number;
}

// ===== AUTHENTICATION TYPES =====

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface RefreshTokenRequest {
  refresh_token: string;
}

export interface RefreshTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token?: string;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

export interface LogoutRequest {
  refresh_token?: string;
  all_sessions?: boolean;
}

// ===== TOKEN TYPES =====

export interface TokenPayload {
  sub: string; // username
  user_id: number;
  role: string;
  permissions: Record<string, string[]>;
  exp: number;
  iat: number;
}

export interface TokenInfo {
  access_token: string;
  refresh_token: string;
  expires_at: number;
  user: User;
}

// ===== ROLE & PERMISSION TYPES =====

export interface Role {
  id: number;
  name: string;
  description: string;
  permissions: Record<string, string[]>;
  created_at: string;
  updated_at: string;
}

export interface Permission {
  resource: string;
  action: string;
}

export interface PermissionCheck {
  resource: string;
  action: string;
}

export interface PermissionResponse {
  has_permission: boolean;
  resource: string;
  action: string;
  user_id: number;
}

// ===== SESSION TYPES =====

export interface Session {
  id: number;
  user_id: number;
  expires_at: string;
  created_at: string;
  ip_address?: string;
  user_agent?: string;
}

export interface SessionDetail extends Session {
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  role_name: string;
}

// ===== AUTH STATE TYPES =====

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  permissions: Record<string, string[]>;
  tokens: {
    access_token: string | null;
    refresh_token: string | null;
    expires_at: number | null;
  };
  lastActivity: number;
}

export interface AuthContextType extends AuthState {
  login: (credentials: LoginCredentials) => Promise<LoginResponse>;
  logout: (logoutData?: LogoutRequest) => Promise<void>;
  refreshToken: () => Promise<RefreshTokenResponse | null>;
  updateUser: (userData: UserUpdate) => Promise<User>;
  changePassword: (passwordData: ChangePasswordRequest) => Promise<void>;
  checkPermission: (resource: string, action: string) => boolean;
  hasRole: (roleName: string) => boolean;
  isTokenExpired: () => boolean;
  updateLastActivity: () => void;
}

// ===== API RESPONSE TYPES =====

export interface BaseResponse {
  success: boolean;
  message?: string;
}

export interface ErrorResponse extends BaseResponse {
  success: false;
  error: string;
  details?: Record<string, any>;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ApiError {
  message: string;
  status: number;
  details?: Record<string, any>;
}

// ===== COMPONENT TYPES =====

export interface LoginFormProps {
  onLogin?: (response: LoginResponse) => void;
  onError?: (error: ApiError) => void;
  redirectTo?: string;
  showRememberMe?: boolean;
  className?: string;
}

export interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredPermissions?: Permission[];
  requiredRole?: string;
  fallback?: React.ReactNode;
  redirectTo?: string;
}

export interface AuthGuardProps {
  children: React.ReactNode;
  permissions?: Permission[];
  roles?: string[];
  fallback?: React.ReactNode;
  showError?: boolean;
}

export interface RoleGuardProps {
  children: React.ReactNode;
  allowedRoles: string[];
  fallback?: React.ReactNode;
  strict?: boolean; // If true, requires exact role match
}

export interface LogoutButtonProps {
  onLogout?: () => void;
  logoutAll?: boolean;
  className?: string;
  children?: React.ReactNode;
}

// ===== HOOK TYPES =====

export interface UseAuthOptions {
  redirectTo?: string;
  requireAuth?: boolean;
}

export interface UseApiOptions {
  baseURL?: string;
  timeout?: number;
  retries?: number;
  onTokenRefresh?: (tokens: TokenInfo) => void;
  onAuthError?: (error: ApiError) => void;
}

export interface ApiClientConfig {
  baseURL: string;
  timeout: number;
  retries: number;
  retryDelay: number;
}

// ===== STORAGE TYPES =====

export interface StorageKeys {
  ACCESS_TOKEN: "access_token";
  REFRESH_TOKEN: "refresh_token";
  USER_DATA: "user_data";
  EXPIRES_AT: "expires_at";
  LAST_ACTIVITY: "last_activity";
  PERMISSIONS: "permissions";
}

export interface StoredAuthData {
  access_token: string;
  refresh_token: string;
  user_data: User;
  expires_at: number;
  permissions: Record<string, string[]>;
}

// ===== VALIDATION TYPES =====

export interface ValidationRule {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  custom?: (value: string) => string | null;
}

export interface ValidationRules {
  username: ValidationRule;
  email: ValidationRule;
  password: ValidationRule;
  firstName: ValidationRule;
  lastName: ValidationRule;
}

export interface ValidationErrors {
  [key: string]: string | null;
}

// ===== EVENT TYPES =====

export type AuthEventType =
  | "login"
  | "logout"
  | "token_refresh"
  | "permission_denied"
  | "session_expired"
  | "user_updated";

export interface AuthEvent {
  type: AuthEventType;
  payload?: any;
  timestamp: number;
}

export type AuthEventListener = (event: AuthEvent) => void;

// ===== UTILITY TYPES =====

export type RoleLevel = "admin" | "teacher" | "student";

export interface RoleHierarchy {
  admin: string[];
  teacher: string[];
  student: string[];
}

export interface PermissionMatrix {
  [resource: string]: {
    [action: string]: RoleLevel[];
  };
}

// ===== CONSTANTS =====

export const AUTH_ROLES = {
  ADMIN: "admin",
  TEACHER: "teacher",
  STUDENT: "student",
} as const;

export const AUTH_PERMISSIONS = {
  USERS: {
    CREATE: "create",
    READ: "read",
    UPDATE: "update",
    DELETE: "delete",
  },
  ROLES: {
    CREATE: "create",
    READ: "read",
    UPDATE: "update",
    DELETE: "delete",
  },
  SESSIONS: {
    CREATE: "create",
    READ: "read",
    UPDATE: "update",
    DELETE: "delete",
  },
  DOCUMENTS: {
    CREATE: "create",
    READ: "read",
    UPDATE: "update",
    DELETE: "delete",
  },
  SYSTEM: {
    ADMIN: "admin",
    CONFIGURE: "configure",
  },
} as const;

export const AUTH_STORAGE_KEYS: StorageKeys = {
  ACCESS_TOKEN: "access_token",
  REFRESH_TOKEN: "refresh_token",
  USER_DATA: "user_data",
  EXPIRES_AT: "expires_at",
  LAST_ACTIVITY: "last_activity",
  PERMISSIONS: "permissions",
};

// ===== TYPE GUARDS =====

export function isUser(obj: any): obj is User {
  return (
    obj &&
    typeof obj.id === "number" &&
    typeof obj.username === "string" &&
    typeof obj.email === "string" &&
    typeof obj.role_name === "string"
  );
}

export function isLoginResponse(obj: any): obj is LoginResponse {
  return (
    obj &&
    typeof obj.access_token === "string" &&
    typeof obj.refresh_token === "string" &&
    isUser(obj.user)
  );
}

export function isApiError(obj: any): obj is ApiError {
  return (
    obj && typeof obj.message === "string" && typeof obj.status === "number"
  );
}

export function isTokenExpired(expiresAt: number): boolean {
  return Date.now() >= expiresAt * 1000;
}

export function hasPermission(
  userPermissions: Record<string, string[]>,
  resource: string,
  action: string
): boolean {
  const resourcePermissions = userPermissions[resource];
  return resourcePermissions && resourcePermissions.includes(action);
}

export function hasRole(userRole: string, allowedRoles: string[]): boolean {
  return allowedRoles.includes(userRole);
}
