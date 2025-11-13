/**
 * Authentication Context for RAG Education Assistant
 * Provides global authentication state management and authentication operations
 */

"use client";

import React, {
  createContext,
  useContext,
  useReducer,
  useEffect,
  ReactNode,
  useCallback,
} from "react";
import { useRouter } from "next/navigation";
import {
  AuthState,
  AuthContextType,
  LoginCredentials,
  LoginResponse,
  RefreshTokenResponse,
  ChangePasswordRequest,
  LogoutRequest,
  User,
  UserUpdate,
  ApiError,
  TokenInfo,
  AUTH_ROLES,
  hasPermission,
  hasRole,
} from "../types/auth";
import { tokenManager } from "../lib/token-manager";
import { apiClient } from "../lib/api-client";

// ===== AUTH STATE MANAGEMENT =====

type AuthAction =
  | { type: "LOGIN_START" }
  | {
      type: "LOGIN_SUCCESS";
      payload: { user: User; permissions: Record<string, string[]> };
    }
  | { type: "LOGIN_FAILURE"; payload: string }
  | { type: "LOGOUT" }
  | { type: "TOKEN_REFRESH"; payload: { user: User } }
  | { type: "USER_UPDATE"; payload: User }
  | { type: "SET_LOADING"; payload: boolean }
  | {
      type: "RESTORE_SESSION";
      payload: { user: User; permissions: Record<string, string[]> };
    }
  | { type: "CLEAR_ERROR" }
  | { type: "UPDATE_ACTIVITY" };

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: true,
  permissions: {},
  tokens: {
    access_token: null,
    refresh_token: null,
    expires_at: null,
  },
  lastActivity: Date.now(),
};

function authReducer(state: AuthState, action: AuthAction): AuthState {
  switch (action.type) {
    case "LOGIN_START":
      return {
        ...state,
        isLoading: true,
      };

    case "LOGIN_SUCCESS":
      return {
        ...state,
        user: action.payload.user,
        isAuthenticated: true,
        isLoading: false,
        permissions: action.payload.permissions,
        lastActivity: Date.now(),
      };

    case "LOGIN_FAILURE":
      return {
        ...state,
        user: null,
        isAuthenticated: false,
        isLoading: false,
        permissions: {},
        tokens: {
          access_token: null,
          refresh_token: null,
          expires_at: null,
        },
      };

    case "LOGOUT":
      return {
        ...initialState,
        isLoading: false,
      };

    case "TOKEN_REFRESH":
      return {
        ...state,
        user: action.payload.user,
        lastActivity: Date.now(),
      };

    case "USER_UPDATE":
      return {
        ...state,
        user: action.payload,
        lastActivity: Date.now(),
      };

    case "SET_LOADING":
      return {
        ...state,
        isLoading: action.payload,
      };

    case "RESTORE_SESSION":
      return {
        ...state,
        user: action.payload.user,
        isAuthenticated: true,
        isLoading: false,
        permissions: action.payload.permissions,
        lastActivity: Date.now(),
      };

    case "UPDATE_ACTIVITY":
      return {
        ...state,
        lastActivity: Date.now(),
      };

    default:
      return state;
  }
}

// ===== CONTEXT CREATION =====

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// ===== AUTH PROVIDER COMPONENT =====

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [state, dispatch] = useReducer(authReducer, initialState);
  const router = useRouter();

  // ===== SESSION RESTORATION =====

  /**
   * Restore user session from stored tokens
   */
  const restoreSession = useCallback(async () => {
    try {
      if (typeof window === "undefined") {
        dispatch({ type: "SET_LOADING", payload: false });
        return;
      }

      const tokenInfo = tokenManager.getTokenInfo();

      if (!tokenInfo || tokenManager.isTokenExpired()) {
        // Clear expired tokens
        tokenManager.clearTokens();
        dispatch({ type: "SET_LOADING", payload: false });
        return;
      }

      // Try to get current user info
      const userData = await apiClient.getCurrentUser();
      const permissions = tokenManager.getPermissions();

      dispatch({
        type: "RESTORE_SESSION",
        payload: {
          user: userData,
          permissions,
        },
      });
    } catch (error) {
      console.error("Session restoration failed:", error);
      tokenManager.clearTokens();
      dispatch({ type: "LOGOUT" });
    }
  }, []);

  // ===== AUTHENTICATION METHODS =====

  /**
   * Login user with credentials
   */
  const login = useCallback(
    async (credentials: LoginCredentials): Promise<LoginResponse> => {
      dispatch({ type: "LOGIN_START" });

      try {
        console.log(
          "AuthContext: Starting login with credentials:",
          credentials
        );
        const loginResponse = await apiClient.login(credentials);
        console.log("AuthContext: Login response received:", loginResponse);

        // Extract permissions from token payload if available
        const permissions = tokenManager.getPermissions();
        console.log("AuthContext: Extracted permissions:", permissions);

        dispatch({
          type: "LOGIN_SUCCESS",
          payload: {
            user: loginResponse.user,
            permissions,
          },
        });

        console.log("AuthContext: Login success dispatched");
        return loginResponse;
      } catch (error: any) {
        console.error("AuthContext: Login failed:", error);
        dispatch({ type: "LOGIN_FAILURE", payload: error.message });
        throw error;
      }
    },
    []
  );

  /**
   * Logout user
   */
  const logout = useCallback(
    async (logoutData?: LogoutRequest): Promise<void> => {
      try {
        await apiClient.logout(logoutData?.all_sessions);
      } catch (error) {
        console.warn(
          "Logout API call failed, but clearing local state:",
          error
        );
      } finally {
        dispatch({ type: "LOGOUT" });
        router.push("/login");
      }
    },
    [router]
  );

  /**
   * Refresh access token
   */
  const refreshToken =
    useCallback(async (): Promise<RefreshTokenResponse | null> => {
      try {
        const refreshResponse = await apiClient.refreshAccessToken();

        if (refreshResponse && state.user) {
          dispatch({
            type: "TOKEN_REFRESH",
            payload: { user: state.user },
          });
        }

        return refreshResponse;
      } catch (error) {
        console.error("Token refresh failed:", error);
        dispatch({ type: "LOGOUT" });
        router.push("/login");
        return null;
      }
    }, [state.user, router]);

  /**
   * Update user information
   */
  const updateUser = useCallback(
    async (userData: UserUpdate): Promise<User> => {
      try {
        const updatedUser = await apiClient.put<User>(
          `/users/${state.user?.id}`,
          userData
        );

        dispatch({
          type: "USER_UPDATE",
          payload: updatedUser,
        });

        return updatedUser;
      } catch (error: any) {
        throw error;
      }
    },
    [state.user?.id]
  );

  /**
   * Change user password
   */
  const changePassword = useCallback(
    async (passwordData: ChangePasswordRequest): Promise<void> => {
      try {
        await apiClient.changePassword(
          passwordData.current_password,
          passwordData.new_password
        );
      } catch (error: any) {
        throw error;
      }
    },
    []
  );

  // ===== PERMISSION METHODS =====

  /**
   * Check if user has specific permission
   */
  const checkPermission = useCallback(
    (resource: string, action: string): boolean => {
      return hasPermission(state.permissions, resource, action);
    },
    [state.permissions]
  );

  /**
   * Check if user has specific role
   */
  const hasUserRole = useCallback(
    (roleName: string): boolean => {
      if (!state.user?.role_name) return false;
      return hasRole(state.user.role_name, [roleName]);
    },
    [state.user?.role_name]
  );

  /**
   * Check if token is expired
   */
  const isTokenExpired = useCallback((): boolean => {
    return tokenManager.isTokenExpired();
  }, []);

  /**
   * Update last activity timestamp
   */
  const updateLastActivity = useCallback((): void => {
    tokenManager.updateLastActivity();
    dispatch({ type: "UPDATE_ACTIVITY" });
  }, []);

  // ===== EFFECTS =====

  /**
   * Initialize authentication state on mount
   */
  useEffect(() => {
    restoreSession();
  }, [restoreSession]);

  /**
   * Setup API client callbacks
   */
  useEffect(() => {
    // Handle authentication errors
    apiClient.onAuthenticationError((error: ApiError) => {
      console.error("Authentication error:", error);
      dispatch({ type: "LOGOUT" });
      router.push("/login");
    });

    // Handle token refresh
    apiClient.onTokenRefreshed((tokens: TokenInfo) => {
      if (state.user) {
        dispatch({
          type: "TOKEN_REFRESH",
          payload: { user: tokens.user },
        });
      }
    });
  }, [router, state.user]);

  /**
   * Setup token manager callbacks
   */
  useEffect(() => {
    // Handle token expiry
    tokenManager.onTokenExpiry(() => {
      dispatch({ type: "LOGOUT" });
      router.push("/login");
    });

    // Handle token refresh needed
    tokenManager.onTokenRefresh((tokenInfo: TokenInfo) => {
      refreshToken();
    });

    // Cleanup on unmount
    return () => {
      tokenManager.destroy();
    };
  }, [refreshToken, router]);

  /**
   * Activity tracking
   */
  useEffect(() => {
    if (!state.isAuthenticated) return;

    const activityEvents = [
      "mousedown",
      "mousemove",
      "keypress",
      "scroll",
      "touchstart",
      "click",
    ];

    const handleActivity = () => {
      updateLastActivity();
    };

    activityEvents.forEach((event) => {
      document.addEventListener(event, handleActivity, { passive: true });
    });

    return () => {
      activityEvents.forEach((event) => {
        document.removeEventListener(event, handleActivity);
      });
    };
  }, [state.isAuthenticated, updateLastActivity]);

  // ===== CONTEXT VALUE =====

  const contextValue: AuthContextType = {
    ...state,
    login,
    logout,
    refreshToken,
    updateUser,
    changePassword,
    checkPermission,
    hasRole: hasUserRole,
    isTokenExpired,
    updateLastActivity,
  };

  return (
    <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>
  );
}

// ===== CUSTOM HOOK =====

/**
 * Custom hook to use authentication context
 */
export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);

  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }

  return context;
}

// ===== UTILITY HOOKS =====

/**
 * Hook for components that require authentication
 */
export function useRequireAuth(redirectTo: string = "/login") {
  const auth = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!auth.isLoading && !auth.isAuthenticated) {
      router.push(redirectTo);
    }
  }, [auth.isAuthenticated, auth.isLoading, redirectTo, router]);

  return auth;
}

/**
 * Hook for permission-based rendering
 */
export function usePermissions(
  requiredPermissions?: Array<{ resource: string; action: string }>
) {
  const auth = useAuth();

  const hasAllPermissions = useCallback(() => {
    if (!requiredPermissions || requiredPermissions.length === 0) return true;

    return requiredPermissions.every(({ resource, action }) =>
      auth.checkPermission(resource, action)
    );
  }, [auth, requiredPermissions]);

  const hasAnyPermission = useCallback(() => {
    if (!requiredPermissions || requiredPermissions.length === 0) return true;

    return requiredPermissions.some(({ resource, action }) =>
      auth.checkPermission(resource, action)
    );
  }, [auth, requiredPermissions]);

  return {
    hasAllPermissions: hasAllPermissions(),
    hasAnyPermission: hasAnyPermission(),
    permissions: auth.permissions,
    checkPermission: auth.checkPermission,
  };
}

/**
 * Hook for role-based operations
 */
export function useRoles(allowedRoles?: string[]) {
  const auth = useAuth();

  const hasAllowedRole = useCallback(() => {
    if (!allowedRoles || allowedRoles.length === 0) return true;
    if (!auth.user?.role_name) return false;

    return allowedRoles.includes(auth.user.role_name);
  }, [auth.user?.role_name, allowedRoles]);

  const isAdmin = useCallback(() => {
    return auth.hasRole(AUTH_ROLES.ADMIN);
  }, [auth]);

  const isTeacher = useCallback(() => {
    return auth.hasRole(AUTH_ROLES.TEACHER);
  }, [auth]);

  const isStudent = useCallback(() => {
    return auth.hasRole(AUTH_ROLES.STUDENT);
  }, [auth]);

  return {
    hasAllowedRole: hasAllowedRole(),
    isAdmin: isAdmin(),
    isTeacher: isTeacher(),
    isStudent: isStudent(),
    userRole: auth.user?.role_name || null,
  };
}

// ===== DEFAULT EXPORT =====

export default AuthProvider;
