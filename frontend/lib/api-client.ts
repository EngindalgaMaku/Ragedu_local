/**
 * API Client for RAG Education Assistant
 * Configured axios client with authentication, token refresh, and error handling
 */

import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
} from "axios";

// Extend axios request config to include metadata
interface ExtendedAxiosRequestConfig extends InternalAxiosRequestConfig {
  metadata?: {
    startTime: number;
  };
  _retry?: boolean;
  retryCount?: number;
}
import {
  ApiError,
  LoginCredentials,
  LoginResponse,
  RefreshTokenRequest,
  RefreshTokenResponse,
  TokenInfo,
  User,
  ApiClientConfig,
} from "../types/auth";
import { tokenManager } from "./token-manager";

// ===== CONSTANTS =====

const DEFAULT_CONFIG: ApiClientConfig = {
  // Always use /api for browser requests (Next.js rewrites will proxy to api-gateway)
  // For server-side, this will be handled by next.config.js rewrites
  baseURL: "/api",
  timeout: 30000, // 30 seconds
  retries: 3,
  retryDelay: 1000,
};

const RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504];
const NO_AUTH_ENDPOINTS = ["/auth/login", "/auth/refresh", "/health"];

// ===== API CLIENT CLASS =====

class ApiClient {
  private axiosInstance: AxiosInstance;
  private refreshPromise: Promise<RefreshTokenResponse | null> | null = null;
  private config: ApiClientConfig;
  private onAuthError?: (error: ApiError) => void;
  private onTokenRefresh?: (tokens: TokenInfo) => void;

  constructor(config: Partial<ApiClientConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.axiosInstance = this.createAxiosInstance();
    this.setupInterceptors();
  }

  // ===== AXIOS INSTANCE SETUP =====

  /**
   * Create axios instance with base configuration
   */
  private createAxiosInstance(): AxiosInstance {
    return axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  /**
   * Setup request and response interceptors
   */
  private setupInterceptors(): void {
    this.setupRequestInterceptor();
    this.setupResponseInterceptor();
  }

  /**
   * Setup request interceptor for adding auth headers
   */
  private setupRequestInterceptor(): void {
    this.axiosInstance.interceptors.request.use(
      (config: ExtendedAxiosRequestConfig) => {
        // Skip auth for certain endpoints
        const isNoAuthEndpoint = NO_AUTH_ENDPOINTS.some((endpoint) =>
          config.url?.includes(endpoint)
        );

        if (!isNoAuthEndpoint) {
          const accessToken = tokenManager.getAccessToken();

          if (accessToken) {
            config.headers.Authorization = `Bearer ${accessToken}`;
          }
        }

        // Add request metadata
        config.metadata = {
          startTime: Date.now(),
        };

        return config;
      },
      (error: AxiosError) => {
        return Promise.reject(this.createApiError(error));
      }
    );
  }

  /**
   * Setup response interceptor for handling errors and token refresh
   */
  private setupResponseInterceptor(): void {
    this.axiosInstance.interceptors.response.use(
      (response: AxiosResponse) => {
        // Log successful responses in development
        if (process.env.NODE_ENV === "development") {
          const config = response.config as ExtendedAxiosRequestConfig;
          const duration = Date.now() - (config.metadata?.startTime || 0);
          console.log(
            `✅ ${response.config.method?.toUpperCase()} ${
              response.config.url
            } - ${response.status} (${duration}ms)`
          );
        }

        return response;
      },
      async (error: AxiosError) => {
        const originalRequest = error.config as ExtendedAxiosRequestConfig;

        // Skip token refresh for login endpoint - login failures should not trigger refresh
        const isLoginEndpoint = originalRequest.url?.includes("/auth/login");
        const isNoAuthEndpoint = NO_AUTH_ENDPOINTS.some((endpoint) =>
          originalRequest.url?.includes(endpoint)
        );

        // Handle 401 Unauthorized - Token expired or invalid
        // BUT: Skip token refresh for login/auth endpoints (they should return 401 directly)
        if (
          error.response?.status === 401 &&
          !originalRequest._retry &&
          !isNoAuthEndpoint &&
          !isLoginEndpoint
        ) {
          originalRequest._retry = true;

          try {
            // Attempt to refresh token
            const refreshResult = await this.refreshAccessToken();

            if (refreshResult) {
              // Retry original request with new token
              originalRequest.headers.Authorization = `Bearer ${refreshResult.access_token}`;
              return this.axiosInstance(originalRequest);
            }
          } catch (refreshError) {
            // Refresh failed, trigger auth error
            this.handleAuthError();
            return Promise.reject(this.createApiError(error));
          }
        }

        // Handle network errors and retryable status codes
        if (this.shouldRetry(error, originalRequest)) {
          return this.retryRequest(originalRequest);
        }

        // Log error in development
        if (process.env.NODE_ENV === "development") {
          const duration =
            Date.now() - (originalRequest.metadata?.startTime || 0);
          console.error(
            `❌ ${originalRequest.method?.toUpperCase()} ${
              originalRequest.url
            } - ${error.response?.status || "Network Error"} (${duration}ms)`
          );
        }

        return Promise.reject(this.createApiError(error));
      }
    );
  }

  // ===== AUTHENTICATION METHODS =====

  /**
   * Login user and store tokens
   */
  public async login(credentials: LoginCredentials): Promise<LoginResponse> {
    try {
      console.log(
        "API Client: Login attempt with baseURL:",
        this.config.baseURL
      );
      console.log(
        "API Client: Full URL will be:",
        `${this.config.baseURL}/auth/login`
      );
      console.log("API Client: Credentials:", credentials);

      const response = await this.axiosInstance.post<LoginResponse>(
        "/auth/login",
        credentials
      );

      console.log(
        "API Client: Login response received:",
        response.status,
        response.data
      );

      const loginData = response.data;

      // Store tokens
      const tokenInfo: TokenInfo = {
        access_token: loginData.access_token,
        refresh_token: loginData.refresh_token,
        expires_at: Math.floor(Date.now() / 1000) + loginData.expires_in,
        user: loginData.user,
      };

      tokenManager.storeTokens(tokenInfo);

      // Store permissions if available
      if (loginData.user && "permissions" in loginData.user) {
        const permissions = (loginData.user as any).permissions;
        tokenManager.storePermissions(permissions);
      }

      console.log("API Client: Login successful, tokens stored");
      return loginData;
    } catch (error) {
      console.error("API Client: Login failed:", error);
      throw this.createApiError(error as AxiosError);
    }
  }

  /**
   * Refresh access token
   */
  public async refreshAccessToken(): Promise<RefreshTokenResponse | null> {
    // Prevent multiple simultaneous refresh requests
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    const refreshToken = tokenManager.getRefreshToken();
    if (!refreshToken) {
      return null;
    }

    this.refreshPromise = this.performTokenRefresh(refreshToken);

    try {
      const result = await this.refreshPromise;
      return result;
    } finally {
      this.refreshPromise = null;
    }
  }

  /**
   * Perform token refresh request
   */
  private async performTokenRefresh(
    refreshToken: string
  ): Promise<RefreshTokenResponse | null> {
    try {
      const response = await this.axiosInstance.post<RefreshTokenResponse>(
        "/auth/refresh",
        { refresh_token: refreshToken } as RefreshTokenRequest
      );

      const refreshData = response.data;

      // Update stored tokens
      const userData = tokenManager.getUserData();
      if (userData) {
        const tokenInfo: TokenInfo = {
          access_token: refreshData.access_token,
          refresh_token: refreshData.refresh_token || refreshToken,
          expires_at: Math.floor(Date.now() / 1000) + refreshData.expires_in,
          user: userData,
        };

        tokenManager.storeTokens(tokenInfo);

        // Notify callback
        if (this.onTokenRefresh) {
          this.onTokenRefresh(tokenInfo);
        }
      }

      return refreshData;
    } catch (error) {
      // Clear invalid tokens
      tokenManager.clearTokens();
      throw error;
    }
  }

  /**
   * Logout user
   */
  public async logout(allSessions: boolean = false): Promise<void> {
    const refreshToken = tokenManager.getRefreshToken();

    try {
      await this.axiosInstance.post("/auth/logout", {
        refresh_token: refreshToken,
        all_sessions: allSessions,
      });
    } catch (error) {
      // Even if logout request fails, clear local tokens
      console.warn("Logout request failed, but clearing local tokens", error);
    } finally {
      tokenManager.clearTokens();
    }
  }

  /**
   * Get current user info
   */
  public async getCurrentUser(): Promise<User> {
    const response = await this.axiosInstance.get<User>("/auth/me");
    return response.data;
  }

  /**
   * Change user password
   */
  public async changePassword(
    currentPassword: string,
    newPassword: string
  ): Promise<void> {
    await this.axiosInstance.put("/auth/change-password", {
      current_password: currentPassword,
      new_password: newPassword,
    });
  }

  /**
   * Check user permission
   */
  public async checkPermission(
    resource: string,
    action: string
  ): Promise<boolean> {
    const response = await this.axiosInstance.post("/auth/check-permission", {
      resource,
      action,
    });
    return response.data.has_permission;
  }

  // ===== HTTP METHODS =====

  /**
   * GET request
   */
  public async get<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await this.axiosInstance.get<T>(url, config);
    return response.data;
  }

  /**
   * POST request
   */
  public async post<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await this.axiosInstance.post<T>(url, data, config);
    return response.data;
  }

  /**
   * PUT request
   */
  public async put<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await this.axiosInstance.put<T>(url, data, config);
    return response.data;
  }

  /**
   * DELETE request
   */
  public async delete<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await this.axiosInstance.delete<T>(url, config);
    return response.data;
  }

  /**
   * PATCH request
   */
  public async patch<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const response = await this.axiosInstance.patch<T>(url, data, config);
    return response.data;
  }

  // ===== ERROR HANDLING =====

  /**
   * Create standardized API error
   */
  private createApiError(error: AxiosError): ApiError {
    if (error.response) {
      // Server responded with error status
      const responseData = error.response.data as any;

      // FastAPI uses 'detail' field, but also check 'error' and 'message'
      const errorMessage =
        responseData?.detail ||
        responseData?.error ||
        responseData?.message ||
        "Server error";

      return {
        message: errorMessage,
        status: error.response.status,
        details: responseData?.details || {},
      };
    } else if (error.request) {
      // Network error
      return {
        message: "Network error - please check your connection",
        status: 0,
        details: { networkError: true },
      };
    } else {
      // Request setup error
      return {
        message: error.message || "Request configuration error",
        status: 0,
        details: { configError: true },
      };
    }
  }

  /**
   * Handle authentication errors
   */
  private handleAuthError(): void {
    tokenManager.clearTokens();

    if (this.onAuthError) {
      this.onAuthError({
        message: "Authentication failed",
        status: 401,
        details: { authError: true },
      });
    }
  }

  // ===== RETRY LOGIC =====

  /**
   * Check if request should be retried
   */
  private shouldRetry(
    error: AxiosError,
    config: ExtendedAxiosRequestConfig
  ): boolean {
    const retryCount = config.retryCount || 0;

    if (retryCount >= this.config.retries) {
      return false;
    }

    // Retry on network errors
    if (!error.response) {
      return true;
    }

    // Retry on specific status codes
    return RETRY_STATUS_CODES.includes(error.response.status);
  }

  /**
   * Retry failed request
   */
  private async retryRequest(
    config: ExtendedAxiosRequestConfig
  ): Promise<AxiosResponse> {
    config.retryCount = (config.retryCount || 0) + 1;

    // Calculate delay with exponential backoff
    const delay = this.config.retryDelay * Math.pow(2, config.retryCount - 1);

    await new Promise((resolve) => setTimeout(resolve, delay));

    return this.axiosInstance(config);
  }

  // ===== CALLBACK REGISTRATION =====

  /**
   * Register callback for authentication errors
   */
  public onAuthenticationError(callback: (error: ApiError) => void): void {
    this.onAuthError = callback;
  }

  /**
   * Register callback for token refresh
   */
  public onTokenRefreshed(callback: (tokens: TokenInfo) => void): void {
    this.onTokenRefresh = callback;
  }

  // ===== UTILITY METHODS =====

  /**
   * Get raw axios instance for advanced usage
   */
  public getAxiosInstance(): AxiosInstance {
    return this.axiosInstance;
  }

  /**
   * Update base URL
   */
  public updateBaseURL(baseURL: string): void {
    this.config.baseURL = baseURL;
    this.axiosInstance.defaults.baseURL = baseURL;
  }

  /**
   * Check if user is authenticated
   */
  public isAuthenticated(): boolean {
    return tokenManager.isAuthenticated();
  }

  /**
   * Get authentication status
   */
  public getAuthStatus(): {
    isAuthenticated: boolean;
    user: User | null;
    tokenExpiry: string | null;
  } {
    return {
      isAuthenticated: this.isAuthenticated(),
      user: tokenManager.getUserData(),
      tokenExpiry: tokenManager.getFormattedTimeUntilExpiry(),
    };
  }
}

// ===== SINGLETON EXPORT =====

export const apiClient = new ApiClient();

// ===== CONFIGURATION FUNCTION =====

/**
 * Configure API client with custom settings
 */
export function configureApiClient(config: Partial<ApiClientConfig>): void {
  // Create new instance with updated config
  const newClient = new ApiClient(config);

  // Replace singleton methods
  Object.setPrototypeOf(apiClient, newClient);
  Object.assign(apiClient, newClient);
}

// ===== UTILITY FUNCTIONS =====

/**
 * Create API client instance with custom config
 */
export function createApiClient(
  config: Partial<ApiClientConfig> = {}
): ApiClient {
  return new ApiClient(config);
}

/**
 * Handle API errors consistently
 */
export function handleApiError(error: any): ApiError {
  if (error.response) {
    const responseData = error.response.data;
    // FastAPI uses 'detail' field, but also check 'error' and 'message'
    const errorMessage =
      responseData?.detail ||
      responseData?.error ||
      responseData?.message ||
      "Server error";

    return {
      message: errorMessage,
      status: error.response.status,
      details: responseData?.details || {},
    };
  } else if (error.request) {
    return {
      message: "Network error - please check your connection",
      status: 0,
      details: { networkError: true },
    };
  } else {
    return {
      message: error.message || "Unknown error",
      status: 0,
      details: { unknownError: true },
    };
  }
}

/**
 * Check if error is authentication related
 */
export function isAuthError(error: ApiError): boolean {
  return error.status === 401 || error.details?.authError === true;
}

/**
 * Check if error is network related
 */
export function isNetworkError(error: ApiError): boolean {
  return error.status === 0 && error.details?.networkError === true;
}

export default apiClient;
