/**
 * Custom Authentication Hooks for RAG Education Assistant
 * Provides specialized hooks for different authentication use cases
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  LoginCredentials,
  LoginResponse,
  ApiError,
  User,
  ChangePasswordRequest,
  Permission,
  UseAuthOptions,
  AUTH_ROLES,
} from "../types/auth";
import { useAuth as useAuthContext } from "../contexts/AuthContext";
import { apiClient } from "../lib/api-client";
import { tokenManager } from "../lib/token-manager";

// ===== MAIN AUTH HOOK =====

/**
 * Main authentication hook with enhanced functionality
 */
export function useAuth(options: UseAuthOptions = {}) {
  const auth = useAuthContext();
  const router = useRouter();

  const { redirectTo = "/login", requireAuth = false } = options;

  // Redirect if auth is required but user is not authenticated
  useEffect(() => {
    if (requireAuth && !auth.isLoading && !auth.isAuthenticated) {
      router.push(redirectTo);
    }
  }, [requireAuth, auth.isLoading, auth.isAuthenticated, redirectTo, router]);

  return auth;
}

// ===== LOGIN HOOK =====

/**
 * Hook for handling login operations with loading states and error handling
 */
export function useLogin() {
  const auth = useAuthContext();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const login = useCallback(
    async (
      credentials: LoginCredentials,
      redirectTo: string = "/"
    ): Promise<boolean> => {
      setIsLoading(true);
      setError(null);

      try {
        const loginResponse = await auth.login(credentials);
        // Role-based redirection
        if (loginResponse.user) {
          const userRole = loginResponse.user.role_name?.toLowerCase();
          if (userRole === AUTH_ROLES.ADMIN.toLowerCase()) {
            router.push("/admin");
          } else if (userRole === AUTH_ROLES.STUDENT.toLowerCase()) {
            // Öğrenci için student dashboard'a yönlendir
            router.push("/student");
          } else {
            // Öğretmen ve diğer roller için redirectTo kullan
            router.push(redirectTo);
          }
        } else {
          router.push(redirectTo);
        }
        return true;
      } catch (err: any) {
        setError(err);
        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [auth, router]
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    login,
    isLoading,
    error,
    clearError,
    isAuthenticated: auth.isAuthenticated,
  };
}

// ===== LOGOUT HOOK =====

/**
 * Hook for handling logout operations
 */
export function useLogout() {
  const auth = useAuthContext();
  const [isLoading, setIsLoading] = useState(false);

  const logout = useCallback(
    async (allSessions: boolean = false): Promise<void> => {
      setIsLoading(true);

      try {
        await auth.logout({ all_sessions: allSessions });
      } finally {
        setIsLoading(false);
      }
    },
    [auth]
  );

  const logoutAll = useCallback(async (): Promise<void> => {
    await logout(true);
  }, [logout]);

  return {
    logout,
    logoutAll,
    isLoading,
  };
}

// ===== REQUIRE AUTH HOOK =====

/**
 * Hook that ensures authentication and redirects if not authenticated
 */
export function useRequireAuth(redirectTo: string = "/login") {
  const auth = useAuthContext();
  const router = useRouter();

  useEffect(() => {
    if (!auth.isLoading && !auth.isAuthenticated) {
      router.push(redirectTo);
    }
  }, [auth.isAuthenticated, auth.isLoading, redirectTo, router]);

  return {
    ...auth,
    isReady: !auth.isLoading && auth.isAuthenticated,
  };
}

// ===== PERMISSION HOOKS =====

/**
 * Hook for checking user permissions
 */
export function usePermissions(requiredPermissions?: Permission[]) {
  const auth = useAuthContext();

  const hasAllPermissions = useCallback((): boolean => {
    if (!requiredPermissions || requiredPermissions.length === 0) return true;

    return requiredPermissions.every(({ resource, action }) =>
      auth.checkPermission(resource, action)
    );
  }, [auth, requiredPermissions]);

  const hasAnyPermission = useCallback((): boolean => {
    if (!requiredPermissions || requiredPermissions.length === 0) return true;

    return requiredPermissions.some(({ resource, action }) =>
      auth.checkPermission(resource, action)
    );
  }, [auth, requiredPermissions]);

  const checkPermission = useCallback(
    (resource: string, action: string): boolean => {
      return auth.checkPermission(resource, action);
    },
    [auth]
  );

  return {
    hasAllPermissions: hasAllPermissions(),
    hasAnyPermission: hasAnyPermission(),
    checkPermission,
    permissions: auth.permissions,
    isAuthenticated: auth.isAuthenticated,
  };
}

/**
 * Hook for role-based access control
 */
export function useRoles(allowedRoles?: string[]) {
  const auth = useAuthContext();

  const hasRole = useCallback(
    (role: string): boolean => {
      return auth.hasRole(role);
    },
    [auth]
  );

  const hasAnyRole = useCallback(
    (roles: string[]): boolean => {
      return roles.some((role) => auth.hasRole(role));
    },
    [auth]
  );

  const hasAllowedRole = useCallback((): boolean => {
    if (!allowedRoles || allowedRoles.length === 0) return true;
    return hasAnyRole(allowedRoles);
  }, [allowedRoles, hasAnyRole]);

  const isAdmin = useCallback((): boolean => {
    return hasRole(AUTH_ROLES.ADMIN);
  }, [hasRole]);

  const isTeacher = useCallback((): boolean => {
    return hasRole(AUTH_ROLES.TEACHER);
  }, [hasRole]);

  const isStudent = useCallback((): boolean => {
    return hasRole(AUTH_ROLES.STUDENT);
  }, [hasRole]);

  return {
    hasRole,
    hasAnyRole,
    hasAllowedRole: hasAllowedRole(),
    isAdmin: isAdmin(),
    isTeacher: isTeacher(),
    isStudent: isStudent(),
    userRole: auth.user?.role_name || null,
    isAuthenticated: auth.isAuthenticated,
  };
}

// ===== USER MANAGEMENT HOOKS =====

/**
 * Hook for managing user profile updates
 */
export function useUserProfile() {
  const auth = useAuthContext();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const updateProfile = useCallback(
    async (userData: Partial<User>): Promise<boolean> => {
      if (!auth.user?.id) return false;

      setIsLoading(true);
      setError(null);

      try {
        await auth.updateUser(userData);
        return true;
      } catch (err: any) {
        setError(err);
        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [auth]
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    user: auth.user,
    updateProfile,
    isLoading,
    error,
    clearError,
    isAuthenticated: auth.isAuthenticated,
  };
}

/**
 * Hook for password management
 */
export function usePasswordChange() {
  const auth = useAuthContext();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [success, setSuccess] = useState(false);

  const changePassword = useCallback(
    async (passwordData: ChangePasswordRequest): Promise<boolean> => {
      setIsLoading(true);
      setError(null);
      setSuccess(false);

      try {
        await auth.changePassword(passwordData);
        setSuccess(true);
        return true;
      } catch (err: any) {
        setError(err);
        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [auth]
  );

  const clearMessages = useCallback(() => {
    setError(null);
    setSuccess(false);
  }, []);

  return {
    changePassword,
    isLoading,
    error,
    success,
    clearMessages,
  };
}

// ===== TOKEN MANAGEMENT HOOKS =====

/**
 * Hook for token management and monitoring
 */
export function useTokens() {
  const auth = useAuthContext();
  const [tokenExpiry, setTokenExpiry] = useState<string | null>(null);
  const [isExpiring, setIsExpiring] = useState(false);

  // Update token expiry information
  useEffect(() => {
    const updateTokenInfo = () => {
      const expiry = tokenManager.getFormattedTimeUntilExpiry();
      const timeUntilExpiry = tokenManager.getTimeUntilExpiry();

      setTokenExpiry(expiry);
      setIsExpiring(
        timeUntilExpiry !== null && timeUntilExpiry < 5 * 60 * 1000
      ); // 5 minutes
    };

    if (auth.isAuthenticated) {
      updateTokenInfo();
      const interval = setInterval(updateTokenInfo, 30000); // Update every 30 seconds

      return () => clearInterval(interval);
    } else {
      setTokenExpiry(null);
      setIsExpiring(false);
    }
  }, [auth.isAuthenticated]);

  const refreshToken = useCallback(async (): Promise<boolean> => {
    try {
      const result = await auth.refreshToken();
      return result !== null;
    } catch {
      return false;
    }
  }, [auth]);

  return {
    tokenExpiry,
    isExpiring,
    isTokenExpired: auth.isTokenExpired(),
    refreshToken,
    isAuthenticated: auth.isAuthenticated,
  };
}

// ===== SESSION MANAGEMENT HOOKS =====

/**
 * Hook for session management and activity tracking
 */
export function useSession() {
  const auth = useAuthContext();
  const [sessionInfo, setSessionInfo] = useState({
    lastActivity: Date.now(),
    isActive: true,
  });

  // Update session information
  useEffect(() => {
    const updateSessionInfo = () => {
      const lastActivity = tokenManager.getLastActivity();
      const isTimedOut = tokenManager.isSessionTimedOut();

      setSessionInfo({
        lastActivity: lastActivity || Date.now(),
        isActive: !isTimedOut,
      });
    };

    if (auth.isAuthenticated) {
      updateSessionInfo();
      const interval = setInterval(updateSessionInfo, 60000); // Update every minute

      return () => clearInterval(interval);
    }
  }, [auth.isAuthenticated]);

  const updateActivity = useCallback(() => {
    auth.updateLastActivity();
    setSessionInfo((prev) => ({
      ...prev,
      lastActivity: Date.now(),
      isActive: true,
    }));
  }, [auth]);

  return {
    ...sessionInfo,
    updateActivity,
    isAuthenticated: auth.isAuthenticated,
  };
}

// ===== API INTEGRATION HOOKS =====

/**
 * Hook for making authenticated API requests
 */
export function useApi() {
  const auth = useAuthContext();

  const request = useCallback(
    async <T = any>(
      method: "GET" | "POST" | "PUT" | "DELETE" | "PATCH",
      url: string,
      data?: any,
      config?: any
    ): Promise<T> => {
      if (!auth.isAuthenticated) {
        throw new Error("User not authenticated");
      }

      switch (method) {
        case "GET":
          return apiClient.get<T>(url, config);
        case "POST":
          return apiClient.post<T>(url, data, config);
        case "PUT":
          return apiClient.put<T>(url, data, config);
        case "DELETE":
          return apiClient.delete<T>(url, config);
        case "PATCH":
          return apiClient.patch<T>(url, data, config);
        default:
          throw new Error(`Unsupported method: ${method}`);
      }
    },
    [auth.isAuthenticated]
  );

  const get = useCallback(
    <T = any>(url: string, config?: any): Promise<T> => {
      return request<T>("GET", url, undefined, config);
    },
    [request]
  );

  const post = useCallback(
    <T = any>(url: string, data?: any, config?: any): Promise<T> => {
      return request<T>("POST", url, data, config);
    },
    [request]
  );

  const put = useCallback(
    <T = any>(url: string, data?: any, config?: any): Promise<T> => {
      return request<T>("PUT", url, data, config);
    },
    [request]
  );

  const del = useCallback(
    <T = any>(url: string, config?: any): Promise<T> => {
      return request<T>("DELETE", url, undefined, config);
    },
    [request]
  );

  const patch = useCallback(
    <T = any>(url: string, data?: any, config?: any): Promise<T> => {
      return request<T>("PATCH", url, data, config);
    },
    [request]
  );

  return {
    request,
    get,
    post,
    put,
    delete: del,
    patch,
    isAuthenticated: auth.isAuthenticated,
  };
}

// ===== UTILITY HOOKS =====

/**
 * Hook for authentication status monitoring
 */
export function useAuthStatus() {
  const auth = useAuthContext();
  const [status, setStatus] = useState({
    isInitialized: false,
    hasValidSession: false,
    needsLogin: false,
  });

  useEffect(() => {
    setStatus({
      isInitialized: !auth.isLoading,
      hasValidSession: auth.isAuthenticated && !auth.isTokenExpired(),
      needsLogin: !auth.isLoading && !auth.isAuthenticated,
    });
  }, [auth.isLoading, auth.isAuthenticated, auth.isTokenExpired]);

  return {
    ...status,
    user: auth.user,
    isLoading: auth.isLoading,
  };
}

/**
 * Hook for development/debugging authentication info
 */
export function useAuthDebug() {
  const auth = useAuthContext();

  if (process.env.NODE_ENV !== "development") {
    return null;
  }

  return {
    user: auth.user,
    permissions: auth.permissions,
    tokens: {
      hasAccess: !!tokenManager.getAccessToken(),
      hasRefresh: !!tokenManager.getRefreshToken(),
      expiry: tokenManager.getFormattedTimeUntilExpiry(),
      isExpired: tokenManager.isTokenExpired(),
    },
    session: {
      lastActivity: tokenManager.getLastActivity(),
      isTimedOut: tokenManager.isSessionTimedOut(),
    },
    api: apiClient.getAuthStatus(),
  };
}

// ===== DEFAULT EXPORTS =====

export default useAuth;
