/**
 * Token Manager for RAG Education Assistant
 * Handles secure token storage, validation, and management
 */

import {
  TokenInfo,
  StoredAuthData,
  User,
  AUTH_STORAGE_KEYS,
  isTokenExpired,
} from "../types/auth";

// ===== CONSTANTS =====

const TOKEN_REFRESH_BUFFER = 5 * 60 * 1000; // 5 minutes buffer before expiry
const ACTIVITY_UPDATE_INTERVAL = 30 * 1000; // Update activity every 30 seconds
const SESSION_TIMEOUT = 8 * 60 * 60 * 1000; // 8 hours of inactivity

// ===== TOKEN STORAGE CLASS =====

class TokenManager {
  private static instance: TokenManager;
  private activityTimer: NodeJS.Timeout | null = null;
  private refreshTimer: NodeJS.Timeout | null = null;
  private onTokenExpired?: () => void;
  private onTokenRefreshed?: (tokenInfo: TokenInfo) => void;

  private constructor() {
    this.initializeActivityTracking();
    this.initializeTokenRefreshTimer();
  }

  /**
   * Singleton pattern - get token manager instance
   */
  public static getInstance(): TokenManager {
    if (!TokenManager.instance) {
      TokenManager.instance = new TokenManager();
    }
    return TokenManager.instance;
  }

  // ===== TOKEN STORAGE METHODS =====

  /**
   * Store authentication tokens and user data
   */
  public storeTokens(tokenInfo: TokenInfo): boolean {
    try {
      if (typeof window === "undefined") return false;

      const authData: StoredAuthData = {
        access_token: tokenInfo.access_token,
        refresh_token: tokenInfo.refresh_token,
        user_data: tokenInfo.user,
        expires_at: tokenInfo.expires_at,
        permissions: {}, // Will be updated from user context
      };

      // Store tokens securely
      localStorage.setItem(
        AUTH_STORAGE_KEYS.ACCESS_TOKEN,
        tokenInfo.access_token
      );
      localStorage.setItem(
        AUTH_STORAGE_KEYS.REFRESH_TOKEN,
        tokenInfo.refresh_token
      );
      localStorage.setItem(
        AUTH_STORAGE_KEYS.USER_DATA,
        JSON.stringify(tokenInfo.user)
      );
      localStorage.setItem(
        AUTH_STORAGE_KEYS.EXPIRES_AT,
        tokenInfo.expires_at.toString()
      );
      localStorage.setItem(
        AUTH_STORAGE_KEYS.LAST_ACTIVITY,
        Date.now().toString()
      );

      this.scheduleTokenRefresh(tokenInfo.expires_at);
      this.updateLastActivity();

      return true;
    } catch (error) {
      console.error("Failed to store tokens:", error);
      return false;
    }
  }

  /**
   * Retrieve stored access token
   */
  public getAccessToken(): string | null {
    try {
      if (typeof window === "undefined") return null;

      const token = localStorage.getItem(AUTH_STORAGE_KEYS.ACCESS_TOKEN);
      const expiresAt = this.getTokenExpiry();

      // Check if token is expired
      if (token && expiresAt && this.isTokenExpired()) {
        this.handleTokenExpiry();
        return null;
      }

      return token;
    } catch (error) {
      console.error("Failed to retrieve access token:", error);
      return null;
    }
  }

  /**
   * Retrieve stored refresh token
   */
  public getRefreshToken(): string | null {
    try {
      if (typeof window === "undefined") return null;
      return localStorage.getItem(AUTH_STORAGE_KEYS.REFRESH_TOKEN);
    } catch (error) {
      console.error("Failed to retrieve refresh token:", error);
      return null;
    }
  }

  /**
   * Get stored user data
   */
  public getUserData(): User | null {
    try {
      if (typeof window === "undefined") return null;

      const userData = localStorage.getItem(AUTH_STORAGE_KEYS.USER_DATA);
      return userData ? JSON.parse(userData) : null;
    } catch (error) {
      console.error("Failed to retrieve user data:", error);
      return null;
    }
  }

  /**
   * Get token expiry timestamp
   */
  public getTokenExpiry(): number | null {
    try {
      if (typeof window === "undefined") return null;

      const expiresAt = localStorage.getItem(AUTH_STORAGE_KEYS.EXPIRES_AT);
      return expiresAt ? parseInt(expiresAt, 10) : null;
    } catch (error) {
      console.error("Failed to retrieve token expiry:", error);
      return null;
    }
  }

  /**
   * Store user permissions
   */
  public storePermissions(permissions: Record<string, string[]>): void {
    try {
      if (typeof window === "undefined") return;
      localStorage.setItem(
        AUTH_STORAGE_KEYS.PERMISSIONS,
        JSON.stringify(permissions)
      );
    } catch (error) {
      console.error("Failed to store permissions:", error);
    }
  }

  /**
   * Get stored permissions
   */
  public getPermissions(): Record<string, string[]> {
    try {
      if (typeof window === "undefined") return {};

      const permissions = localStorage.getItem(AUTH_STORAGE_KEYS.PERMISSIONS);
      return permissions ? JSON.parse(permissions) : {};
    } catch (error) {
      console.error("Failed to retrieve permissions:", error);
      return {};
    }
  }

  // ===== TOKEN VALIDATION METHODS =====

  /**
   * Check if access token is expired
   */
  public isTokenExpired(): boolean {
    const expiresAt = this.getTokenExpiry();
    if (!expiresAt) return true;

    return isTokenExpired(expiresAt);
  }

  /**
   * Check if token needs refresh (within buffer time)
   */
  public needsRefresh(): boolean {
    const expiresAt = this.getTokenExpiry();
    if (!expiresAt) return true;

    const now = Date.now();
    const expiryTime = expiresAt * 1000;

    return expiryTime - now <= TOKEN_REFRESH_BUFFER;
  }

  /**
   * Check if user has valid authentication
   */
  public isAuthenticated(): boolean {
    const accessToken = this.getAccessToken();
    const userData = this.getUserData();

    return !!(accessToken && userData && !this.isTokenExpired());
  }

  /**
   * Validate token format (basic JWT structure check)
   */
  public isValidTokenFormat(token: string): boolean {
    try {
      const parts = token.split(".");
      return parts.length === 3;
    } catch {
      return false;
    }
  }

  /**
   * Decode JWT token payload (client-side only for non-sensitive data)
   */
  public decodeTokenPayload(token: string): any | null {
    try {
      const parts = token.split(".");
      if (parts.length !== 3) return null;

      const payload = parts[1];
      const decoded = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
      return JSON.parse(decoded);
    } catch (error) {
      console.error("Failed to decode token payload:", error);
      return null;
    }
  }

  // ===== TOKEN CLEANUP METHODS =====

  /**
   * Clear all stored authentication data
   */
  public clearTokens(): void {
    try {
      if (typeof window === "undefined") return;

      // Clear all auth-related localStorage items
      Object.values(AUTH_STORAGE_KEYS).forEach((key) => {
        localStorage.removeItem(key);
      });

      // Clear timers
      this.clearTimers();
    } catch (error) {
      console.error("Failed to clear tokens:", error);
    }
  }

  /**
   * Clear only expired tokens
   */
  public clearExpiredTokens(): void {
    if (this.isTokenExpired()) {
      this.clearTokens();
    }
  }

  // ===== ACTIVITY TRACKING METHODS =====

  /**
   * Update last activity timestamp
   */
  public updateLastActivity(): void {
    try {
      if (typeof window === "undefined") return;
      localStorage.setItem(
        AUTH_STORAGE_KEYS.LAST_ACTIVITY,
        Date.now().toString()
      );
    } catch (error) {
      console.error("Failed to update last activity:", error);
    }
  }

  /**
   * Get last activity timestamp
   */
  public getLastActivity(): number | null {
    try {
      if (typeof window === "undefined") return null;

      const lastActivity = localStorage.getItem(
        AUTH_STORAGE_KEYS.LAST_ACTIVITY
      );
      return lastActivity ? parseInt(lastActivity, 10) : null;
    } catch (error) {
      console.error("Failed to retrieve last activity:", error);
      return null;
    }
  }

  /**
   * Check if session has timed out due to inactivity
   */
  public isSessionTimedOut(): boolean {
    const lastActivity = this.getLastActivity();
    if (!lastActivity) return true;

    const timeSinceActivity = Date.now() - lastActivity;
    return timeSinceActivity > SESSION_TIMEOUT;
  }

  // ===== TIMER MANAGEMENT METHODS =====

  /**
   * Initialize activity tracking
   */
  private initializeActivityTracking(): void {
    if (typeof window === "undefined") return;

    // Track user activity
    const events = [
      "mousedown",
      "mousemove",
      "keypress",
      "scroll",
      "touchstart",
      "click",
    ];

    const updateActivity = () => {
      this.updateLastActivity();
    };

    events.forEach((event) => {
      document.addEventListener(event, updateActivity, { passive: true });
    });

    // Periodic activity updates
    this.activityTimer = setInterval(() => {
      if (this.isAuthenticated()) {
        this.updateLastActivity();

        // Check for session timeout
        if (this.isSessionTimedOut()) {
          this.handleSessionTimeout();
        }
      }
    }, ACTIVITY_UPDATE_INTERVAL);
  }

  /**
   * Initialize token refresh timer
   */
  private initializeTokenRefreshTimer(): void {
    const checkAndScheduleRefresh = () => {
      const expiresAt = this.getTokenExpiry();
      if (expiresAt) {
        this.scheduleTokenRefresh(expiresAt);
      }
    };

    // Check immediately and then periodically
    checkAndScheduleRefresh();
    setInterval(checkAndScheduleRefresh, 60000); // Check every minute
  }

  /**
   * Schedule automatic token refresh
   */
  private scheduleTokenRefresh(expiresAt: number): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }

    const now = Date.now();
    const expiryTime = expiresAt * 1000;
    const refreshTime = expiryTime - TOKEN_REFRESH_BUFFER;
    const delay = Math.max(0, refreshTime - now);

    this.refreshTimer = setTimeout(() => {
      if (this.needsRefresh() && this.onTokenRefreshed) {
        const tokenInfo = this.getTokenInfo();
        if (tokenInfo) {
          this.onTokenRefreshed(tokenInfo);
        }
      }
    }, delay);
  }

  /**
   * Clear all timers
   */
  private clearTimers(): void {
    if (this.activityTimer) {
      clearInterval(this.activityTimer);
      this.activityTimer = null;
    }

    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
  }

  // ===== EVENT HANDLERS =====

  /**
   * Handle token expiry
   */
  private handleTokenExpiry(): void {
    this.clearTokens();

    if (this.onTokenExpired) {
      this.onTokenExpired();
    }
  }

  /**
   * Handle session timeout
   */
  private handleSessionTimeout(): void {
    this.clearTokens();

    if (this.onTokenExpired) {
      this.onTokenExpired();
    }
  }

  // ===== CALLBACK REGISTRATION =====

  /**
   * Register callback for token expiry
   */
  public onTokenExpiry(callback: () => void): void {
    this.onTokenExpired = callback;
  }

  /**
   * Register callback for token refresh needed
   */
  public onTokenRefresh(callback: (tokenInfo: TokenInfo) => void): void {
    this.onTokenRefreshed = callback;
  }

  // ===== UTILITY METHODS =====

  /**
   * Get complete token information
   */
  public getTokenInfo(): TokenInfo | null {
    const accessToken = this.getAccessToken();
    const refreshToken = this.getRefreshToken();
    const userData = this.getUserData();
    const expiresAt = this.getTokenExpiry();

    if (!accessToken || !refreshToken || !userData || !expiresAt) {
      return null;
    }

    return {
      access_token: accessToken,
      refresh_token: refreshToken,
      expires_at: expiresAt,
      user: userData,
    };
  }

  /**
   * Get authentication headers
   */
  public getAuthHeaders(): Record<string, string> {
    const accessToken = this.getAccessToken();

    if (!accessToken) {
      return {};
    }

    return {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    };
  }

  /**
   * Check if tokens exist in storage
   */
  public hasStoredTokens(): boolean {
    if (typeof window === "undefined") return false;

    const accessToken = localStorage.getItem(AUTH_STORAGE_KEYS.ACCESS_TOKEN);
    const refreshToken = localStorage.getItem(AUTH_STORAGE_KEYS.REFRESH_TOKEN);

    return !!(accessToken && refreshToken);
  }

  /**
   * Get time until token expiry in milliseconds
   */
  public getTimeUntilExpiry(): number | null {
    const expiresAt = this.getTokenExpiry();
    if (!expiresAt) return null;

    const now = Date.now();
    const expiryTime = expiresAt * 1000;

    return Math.max(0, expiryTime - now);
  }

  /**
   * Format time until expiry as human-readable string
   */
  public getFormattedTimeUntilExpiry(): string | null {
    const timeUntilExpiry = this.getTimeUntilExpiry();
    if (!timeUntilExpiry) return null;

    const minutes = Math.floor(timeUntilExpiry / (60 * 1000));
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else {
      return `${minutes}m`;
    }
  }

  /**
   * Cleanup on app unmount
   */
  public destroy(): void {
    this.clearTimers();
  }
}

// ===== SINGLETON EXPORT =====

export const tokenManager = TokenManager.getInstance();

// ===== UTILITY FUNCTIONS =====

/**
 * Create token info object
 */
export function createTokenInfo(
  accessToken: string,
  refreshToken: string,
  expiresIn: number,
  user: User
): TokenInfo {
  const expiresAt = Math.floor(Date.now() / 1000) + expiresIn;

  return {
    access_token: accessToken,
    refresh_token: refreshToken,
    expires_at: expiresAt,
    user,
  };
}

/**
 * Validate token structure
 */
export function validateTokenStructure(token: string): boolean {
  return tokenManager.isValidTokenFormat(token);
}

/**
 * Get token payload without validation
 */
export function getTokenPayload(token: string): any | null {
  return tokenManager.decodeTokenPayload(token);
}

/**
 * Check if user is authenticated
 */
export function isUserAuthenticated(): boolean {
  return tokenManager.isAuthenticated();
}

/**
 * Clear all authentication data
 */
export function clearAuthData(): void {
  tokenManager.clearTokens();
}

export default tokenManager;
