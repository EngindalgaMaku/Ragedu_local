/**
 * useErrorHandler Hook - Centralized error handling for session detail page
 * Provides consistent error handling, logging, and user feedback mechanisms
 */

import { useCallback, useEffect, useRef } from "react";
import { useSessionStore } from "../stores/sessionStore";
import { useChunksStore } from "../stores/chunksStore";
import { useRagSettingsStore } from "../stores/ragSettingsStore";

// Error types and categories
export type ErrorCategory =
  | "session"
  | "chunks"
  | "rag"
  | "api"
  | "validation"
  | "network"
  | "authentication"
  | "permission"
  | "unknown";

export type ErrorSeverity = "low" | "medium" | "high" | "critical";

export interface ErrorContext {
  category: ErrorCategory;
  severity: ErrorSeverity;
  code?: string;
  details?: Record<string, any>;
  timestamp: string;
  sessionId?: string;
  userId?: string;
  action?: string;
  component?: string;
}

export interface ProcessedError {
  id: string;
  message: string;
  originalError: Error | string | unknown;
  context: ErrorContext;
  userMessage: string;
  technicalMessage: string;
  recoverable: boolean;
  retryable: boolean;
  reportable: boolean;
}

export interface ErrorHandlerOptions {
  enableLogging?: boolean;
  enableReporting?: boolean;
  enableToasts?: boolean;
  logLevel?: "debug" | "info" | "warn" | "error";
  maxRetries?: number;
  retryDelay?: number;
}

// Default options
const DEFAULT_OPTIONS: ErrorHandlerOptions = {
  enableLogging: true,
  enableReporting: false, // Would be enabled in production
  enableToasts: true,
  logLevel: "error",
  maxRetries: 3,
  retryDelay: 1000,
};

// Error classification rules
const ERROR_CLASSIFICATIONS = {
  // API Errors
  400: {
    category: "validation" as ErrorCategory,
    severity: "medium" as ErrorSeverity,
    recoverable: true,
    retryable: false,
  },
  401: {
    category: "authentication" as ErrorCategory,
    severity: "high" as ErrorSeverity,
    recoverable: false,
    retryable: false,
  },
  403: {
    category: "permission" as ErrorCategory,
    severity: "high" as ErrorSeverity,
    recoverable: false,
    retryable: false,
  },
  404: {
    category: "api" as ErrorCategory,
    severity: "medium" as ErrorSeverity,
    recoverable: true,
    retryable: false,
  },
  429: {
    category: "api" as ErrorCategory,
    severity: "medium" as ErrorSeverity,
    recoverable: true,
    retryable: true,
  },
  500: {
    category: "api" as ErrorCategory,
    severity: "high" as ErrorSeverity,
    recoverable: true,
    retryable: true,
  },
  502: {
    category: "network" as ErrorCategory,
    severity: "high" as ErrorSeverity,
    recoverable: true,
    retryable: true,
  },
  503: {
    category: "network" as ErrorCategory,
    severity: "high" as ErrorSeverity,
    recoverable: true,
    retryable: true,
  },
  504: {
    category: "network" as ErrorCategory,
    severity: "medium" as ErrorSeverity,
    recoverable: true,
    retryable: true,
  },
};

// User-friendly error messages
const ERROR_MESSAGES = {
  session: {
    default: "Session işlemi sırasında bir hata oluştu",
    load_failed: "Oturum verileri yüklenirken hata oluştu",
    save_failed: "Oturum ayarları kaydedilemedi",
    delete_failed: "Oturum silinemedi",
  },
  chunks: {
    default: "Belge parçaları işlenirken hata oluştu",
    load_failed: "Belge parçaları yüklenemedi",
    process_failed: "Dosyalar işlenemedi",
    improve_failed: "Belge parçası geliştirilemedi",
  },
  rag: {
    default: "RAG sistemi hatası",
    config_failed: "RAG ayarları kaydedilemedi",
    query_failed: "Sorgu işlenemedi",
    model_unavailable: "Seçilen model kullanılamıyor",
  },
  api: {
    default: "API hatası oluştu",
    network: "Ağ bağlantısı hatası",
    timeout: "İstek zaman aşımına uğradı",
    server: "Sunucu hatası",
  },
  validation: {
    default: "Geçersiz veri",
    required_field: "Zorunlu alan eksik",
    invalid_format: "Geçersiz format",
  },
  authentication: {
    default: "Kimlik doğrulama hatası",
    expired: "Oturum süresi doldu",
    invalid: "Geçersiz kimlik bilgileri",
  },
  permission: {
    default: "Yetki hatası",
    access_denied: "Erişim reddedildi",
    insufficient_permissions: "Yetersiz yetki",
  },
  network: {
    default: "Ağ bağlantısı hatası",
    timeout: "Bağlantı zaman aşımı",
    offline: "İnternet bağlantısı yok",
  },
  unknown: {
    default: "Beklenmeyen bir hata oluştu",
  },
};

export const useErrorHandler = (options: ErrorHandlerOptions = {}) => {
  const config = { ...DEFAULT_OPTIONS, ...options };
  const retryCounters = useRef<Map<string, number>>(new Map());

  // Store actions for clearing errors
  const sessionActions = useSessionStore();
  const chunksActions = useChunksStore();
  const ragActions = useRagSettingsStore();

  // Error processing function
  const processError = useCallback(
    (
      error: Error | string | unknown,
      context: Partial<ErrorContext> = {}
    ): ProcessedError => {
      const id = `error-${Date.now()}-${Math.random()
        .toString(36)
        .substr(2, 9)}`;
      const timestamp = new Date().toISOString();

      let message = "";
      let code = "";
      let statusCode: number | undefined;

      // Extract error information
      if (error instanceof Error) {
        message = error.message;
        code = (error as any).code || "";
        statusCode = (error as any).status || (error as any).statusCode;
      } else if (typeof error === "string") {
        message = error;
      } else if (typeof error === "object" && error !== null) {
        const errorObj = error as any;
        message = errorObj.message || errorObj.error || "Unknown error";
        code = errorObj.code || "";
        statusCode = errorObj.status || errorObj.statusCode;
      } else {
        message = "Unknown error occurred";
      }

      // Determine error category and severity
      let category = context.category || "unknown";
      let severity = context.severity || "medium";
      let recoverable = true;
      let retryable = false;

      // Classify based on status code
      if (
        statusCode &&
        ERROR_CLASSIFICATIONS[statusCode as keyof typeof ERROR_CLASSIFICATIONS]
      ) {
        const classification =
          ERROR_CLASSIFICATIONS[
            statusCode as keyof typeof ERROR_CLASSIFICATIONS
          ];
        category = classification.category;
        severity = classification.severity;
        recoverable = classification.recoverable;
        retryable = classification.retryable || false;
      }

      // Determine if error should be reported
      const reportable = severity === "high" || severity === "critical";

      // Generate user-friendly message
      const categoryMessages =
        ERROR_MESSAGES[category] || ERROR_MESSAGES.unknown;
      const userMessage =
        context.action &&
        categoryMessages[context.action as keyof typeof categoryMessages]
          ? categoryMessages[context.action as keyof typeof categoryMessages]
          : categoryMessages.default;

      return {
        id,
        message,
        originalError: error,
        context: {
          category,
          severity,
          code,
          timestamp,
          ...context,
        },
        userMessage,
        technicalMessage: message,
        recoverable,
        retryable,
        reportable,
      };
    },
    []
  );

  // Logging function
  const logError = useCallback(
    (processedError: ProcessedError) => {
      if (!config.enableLogging) return;

      const { context, message, userMessage } = processedError;
      const logData = {
        id: processedError.id,
        category: context.category,
        severity: context.severity,
        message,
        userMessage,
        context: context.details,
        timestamp: context.timestamp,
        sessionId: context.sessionId,
        action: context.action,
        component: context.component,
      };

      switch (context.severity) {
        case "critical":
        case "high":
          console.error("[ErrorHandler]", logData);
          break;
        case "medium":
          console.warn("[ErrorHandler]", logData);
          break;
        case "low":
          console.info("[ErrorHandler]", logData);
          break;
      }
    },
    [config.enableLogging]
  );

  // Error reporting function (placeholder for production implementation)
  const reportError = useCallback(
    (processedError: ProcessedError) => {
      if (!config.enableReporting || !processedError.reportable) return;

      // In production, this would send error to monitoring service
      // e.g., Sentry, LogRocket, etc.
      console.log("Would report error:", processedError);
    },
    [config.enableReporting]
  );

  // Show user notification (placeholder for toast implementation)
  const showNotification = useCallback(
    (processedError: ProcessedError) => {
      if (!config.enableToasts) return;

      // In production, this would use a toast library
      // For now, just log the user message
      console.log(`[User Notification] ${processedError.userMessage}`);
    },
    [config.enableToasts]
  );

  // Retry mechanism
  const retry = useCallback(
    async (
      operation: () => Promise<any>,
      errorKey: string,
      maxRetries: number = config.maxRetries || 3,
      delay: number = config.retryDelay || 1000
    ) => {
      const currentRetries = retryCounters.current.get(errorKey) || 0;

      if (currentRetries >= maxRetries) {
        retryCounters.current.delete(errorKey);
        throw new Error(`Max retries (${maxRetries}) exceeded for ${errorKey}`);
      }

      try {
        const result = await operation();
        retryCounters.current.delete(errorKey); // Reset on success
        return result;
      } catch (error) {
        retryCounters.current.set(errorKey, currentRetries + 1);

        if (currentRetries < maxRetries - 1) {
          await new Promise((resolve) => setTimeout(resolve, delay));
          return retry(operation, errorKey, maxRetries, delay);
        } else {
          throw error;
        }
      }
    },
    [config.maxRetries, config.retryDelay]
  );

  // Main error handling function
  const handleError = useCallback(
    async (
      error: Error | string | unknown,
      context: Partial<ErrorContext> = {},
      options: {
        clearExisting?: boolean;
        showNotification?: boolean;
        attempt?: number;
      } = {}
    ) => {
      const processedError = processError(error, context);

      // Log the error
      logError(processedError);

      // Report if necessary
      if (processedError.reportable) {
        reportError(processedError);
      }

      // Show user notification
      if (options.showNotification !== false) {
        showNotification(processedError);
      }

      // Update store error states based on category
      if (options.clearExisting) {
        clearAllErrors();
      }

      switch (processedError.context.category) {
        case "session":
          sessionActions.setError(processedError.userMessage);
          break;
        case "chunks":
          chunksActions.setError(processedError.userMessage);
          break;
        case "rag":
          ragActions.setError(processedError.userMessage);
          break;
        default:
          // For general errors, set session error as fallback
          sessionActions.setError(processedError.userMessage);
          break;
      }

      return processedError;
    },
    [
      processError,
      logError,
      reportError,
      showNotification,
      sessionActions,
      chunksActions,
      ragActions,
    ]
  );

  // Error recovery functions
  const clearError = useCallback(
    (category: ErrorCategory) => {
      switch (category) {
        case "session":
          sessionActions.clearMessages();
          break;
        case "chunks":
          chunksActions.clearMessages();
          break;
        case "rag":
          ragActions.clearMessages();
          break;
        default:
          clearAllErrors();
          break;
      }
    },
    [sessionActions, chunksActions, ragActions]
  );

  const clearAllErrors = useCallback(() => {
    sessionActions.clearMessages();
    chunksActions.clearMessages();
    ragActions.clearMessages();
  }, [sessionActions, chunksActions, ragActions]);

  // Wrapped operation functions with automatic error handling
  const withErrorHandling = useCallback(
    <T extends any[], R>(
      operation: (...args: T) => Promise<R>,
      context: Partial<ErrorContext> = {}
    ) => {
      return async (...args: T): Promise<R> => {
        try {
          return await operation(...args);
        } catch (error) {
          await handleError(error, context);
          throw error; // Re-throw for caller to handle if needed
        }
      };
    },
    [handleError]
  );

  const withErrorBoundary = useCallback(
    <T extends any[], R>(
      operation: (...args: T) => Promise<R>,
      fallback: R,
      context: Partial<ErrorContext> = {}
    ) => {
      return async (...args: T): Promise<R> => {
        try {
          return await operation(...args);
        } catch (error) {
          await handleError(error, context);
          return fallback;
        }
      };
    },
    [handleError]
  );

  // Auto-clear errors after timeout
  useEffect(() => {
    const timeouts: NodeJS.Timeout[] = [];

    // Auto-clear session errors after 10 seconds
    if (sessionActions.error) {
      const timeout = setTimeout(() => {
        sessionActions.clearMessages();
      }, 10000);
      timeouts.push(timeout);
    }

    // Auto-clear chunks errors after 10 seconds
    if (chunksActions.error) {
      const timeout = setTimeout(() => {
        chunksActions.clearMessages();
      }, 10000);
      timeouts.push(timeout);
    }

    // Auto-clear RAG errors after 10 seconds
    if (ragActions.error) {
      const timeout = setTimeout(() => {
        ragActions.clearMessages();
      }, 10000);
      timeouts.push(timeout);
    }

    return () => {
      timeouts.forEach((timeout) => clearTimeout(timeout));
    };
  }, [sessionActions, chunksActions, ragActions]);

  return {
    // Main error handling
    handleError,
    processError,

    // Error recovery
    clearError,
    clearAllErrors,

    // Wrapped operations
    withErrorHandling,
    withErrorBoundary,
    retry,

    // Utility functions
    logError,
    reportError,
    showNotification,

    // Configuration
    config,
  };
};

// Convenience hooks for specific error categories
export const useSessionErrorHandler = () => {
  const errorHandler = useErrorHandler();

  const handleSessionError = useCallback(
    (error: any, action?: string) => {
      return errorHandler.handleError(error, {
        category: "session",
        action,
        component: "SessionDetailPage",
      });
    },
    [errorHandler]
  );

  return {
    ...errorHandler,
    handleSessionError,
  };
};

export const useChunksErrorHandler = () => {
  const errorHandler = useErrorHandler();

  const handleChunksError = useCallback(
    (error: any, action?: string) => {
      return errorHandler.handleError(error, {
        category: "chunks",
        action,
        component: "ChunksManagement",
      });
    },
    [errorHandler]
  );

  return {
    ...errorHandler,
    handleChunksError,
  };
};

export const useRagErrorHandler = () => {
  const errorHandler = useErrorHandler();

  const handleRagError = useCallback(
    (error: any, action?: string) => {
      return errorHandler.handleError(error, {
        category: "rag",
        action,
        component: "RagSettings",
      });
    },
    [errorHandler]
  );

  return {
    ...errorHandler,
    handleRagError,
  };
};

export default useErrorHandler;
