"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  RefreshCw,
  Bug,
  Home,
  ChevronDown,
  ChevronUp,
  Copy,
  ExternalLink,
} from "lucide-react";
import { useErrorHandler } from "../../hooks/useErrorHandler";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  showDetails?: boolean;
  enableRetry?: boolean;
  className?: string;
  variant?: "default" | "card" | "minimal" | "full";
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
  retryCount: number;
}

// Error Boundary Class Component
class ErrorBoundaryClass extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: props.showDetails || false,
      retryCount: 0,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo,
    });

    // Call custom error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log error
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleRetry = () => {
    this.setState((prevState) => ({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: prevState.retryCount + 1,
    }));
  };

  toggleDetails = () => {
    this.setState((prevState) => ({
      showDetails: !prevState.showDetails,
    }));
  };

  copyErrorDetails = async () => {
    const { error, errorInfo } = this.state;
    const errorText = `
Hata: ${error?.message || "Bilinmeyen hata"}
Stack: ${error?.stack || "Stack bilgisi yok"}
Component Stack: ${errorInfo?.componentStack || "Component stack bilgisi yok"}
Retry Count: ${this.state.retryCount}
Timestamp: ${new Date().toISOString()}
    `.trim();

    try {
      await navigator.clipboard.writeText(errorText);
      // In a real implementation, you'd show a toast here
      console.log("Error details copied to clipboard");
    } catch (err) {
      console.error("Failed to copy error details:", err);
    }
  };

  render() {
    if (this.state.hasError) {
      const {
        fallback,
        variant = "default",
        className,
        enableRetry = true,
      } = this.props;
      const { error, errorInfo, showDetails, retryCount } = this.state;

      // Custom fallback
      if (fallback) {
        return fallback;
      }

      return (
        <ErrorFallbackUI
          error={error}
          errorInfo={errorInfo}
          showDetails={showDetails}
          retryCount={retryCount}
          onRetry={enableRetry ? this.handleRetry : undefined}
          onToggleDetails={this.toggleDetails}
          onCopyDetails={this.copyErrorDetails}
          variant={variant}
          className={className}
        />
      );
    }

    return this.props.children;
  }
}

// Error Fallback UI Component
interface ErrorFallbackUIProps {
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
  retryCount: number;
  onRetry?: () => void;
  onToggleDetails: () => void;
  onCopyDetails: () => void;
  variant?: "default" | "card" | "minimal" | "full";
  className?: string;
}

function ErrorFallbackUI({
  error,
  errorInfo,
  showDetails,
  retryCount,
  onRetry,
  onToggleDetails,
  onCopyDetails,
  variant = "default",
  className,
}: ErrorFallbackUIProps) {
  const errorMessage = error?.message || "Beklenmeyen bir hata oluştu";
  const isNetworkError =
    errorMessage.includes("fetch") || errorMessage.includes("network");
  const isChunkError = error?.stack?.includes("ChunkLoadError");

  if (variant === "minimal") {
    return (
      <Alert variant="destructive" className={className}>
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Hata</AlertTitle>
        <AlertDescription className="flex items-center justify-between">
          <span>{errorMessage}</span>
          {onRetry && (
            <Button variant="outline" size="sm" onClick={onRetry}>
              <RefreshCw className="w-3 h-3 mr-1" />
              Yeniden Dene
            </Button>
          )}
        </AlertDescription>
      </Alert>
    );
  }

  if (variant === "card") {
    return (
      <Card className={cn("border-destructive", className)}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="w-5 h-5" />
            Bir Hata Oluştu
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">{errorMessage}</p>
          <div className="flex gap-2">
            {onRetry && (
              <Button onClick={onRetry} size="sm">
                <RefreshCw className="w-4 h-4 mr-2" />
                Yeniden Dene
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={onToggleDetails}>
              <Bug className="w-4 h-4 mr-2" />
              {showDetails ? "Detayları Gizle" : "Detayları Göster"}
            </Button>
          </div>

          {showDetails && (
            <div className="mt-4 p-4 bg-muted rounded-md">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium">Teknik Detaylar</h4>
                <Button variant="ghost" size="sm" onClick={onCopyDetails}>
                  <Copy className="w-3 h-3 mr-1" />
                  Kopyala
                </Button>
              </div>
              <pre className="text-xs overflow-auto max-h-32">
                {error?.stack || error?.message}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  if (variant === "full") {
    return (
      <div
        className={cn(
          "min-h-screen flex items-center justify-center bg-background",
          className
        )}
      >
        <div className="max-w-md w-full space-y-6 p-6">
          <div className="text-center">
            <AlertTriangle className="mx-auto h-12 w-12 text-destructive mb-4" />
            <h1 className="text-2xl font-bold text-foreground mb-2">
              Bir Şeyler Ters Gitti
            </h1>
            <p className="text-muted-foreground mb-6">
              {isChunkError
                ? "Sayfa yüklenirken bir hata oluştu. Lütfen sayfayı yenileyin."
                : isNetworkError
                ? "Ağ bağlantısında bir sorun var. İnternet bağlantınızı kontrol edin."
                : errorMessage}
            </p>
          </div>

          <div className="space-y-3">
            {onRetry && (
              <Button onClick={onRetry} className="w-full">
                <RefreshCw className="w-4 h-4 mr-2" />
                Yeniden Dene
                {retryCount > 0 && (
                  <span className="ml-2 text-xs opacity-70">
                    ({retryCount}. deneme)
                  </span>
                )}
              </Button>
            )}

            <Button
              variant="outline"
              onClick={() => (window.location.href = "/")}
              className="w-full"
            >
              <Home className="w-4 h-4 mr-2" />
              Ana Sayfaya Dön
            </Button>

            <Button
              variant="ghost"
              onClick={onToggleDetails}
              className="w-full"
            >
              {showDetails ? (
                <>
                  <ChevronUp className="w-4 h-4 mr-2" />
                  Detayları Gizle
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4 mr-2" />
                  Teknik Detayları Göster
                </>
              )}
            </Button>
          </div>

          {showDetails && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="text-base flex items-center justify-between">
                  Hata Detayları
                  <Button variant="ghost" size="sm" onClick={onCopyDetails}>
                    <Copy className="w-3 h-3" />
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <h4 className="text-sm font-medium mb-1">Hata Mesajı:</h4>
                  <code className="text-xs bg-muted p-2 rounded block">
                    {error?.message}
                  </code>
                </div>

                {error?.stack && (
                  <div>
                    <h4 className="text-sm font-medium mb-1">Stack Trace:</h4>
                    <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-32">
                      {error.stack}
                    </pre>
                  </div>
                )}

                <div className="text-xs text-muted-foreground">
                  <p>Zaman: {new Date().toLocaleString("tr-TR")}</p>
                  <p>Deneme: {retryCount + 1}</p>
                  <p>User Agent: {navigator.userAgent}</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    );
  }

  // Default variant
  return (
    <div className={cn("p-6 space-y-4", className)}>
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Bir Hata Oluştu</AlertTitle>
        <AlertDescription>
          {errorMessage}
          {retryCount > 0 && (
            <span className="block text-xs mt-1 opacity-70">
              {retryCount}. yeniden deneme
            </span>
          )}
        </AlertDescription>
      </Alert>

      <div className="flex gap-2">
        {onRetry && (
          <Button onClick={onRetry} size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Yeniden Dene
          </Button>
        )}
        <Button variant="outline" size="sm" onClick={onToggleDetails}>
          {showDetails ? "Detayları Gizle" : "Detayları Göster"}
        </Button>
      </div>

      {showDetails && (
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium">Teknik Detaylar</h4>
              <Button variant="ghost" size="sm" onClick={onCopyDetails}>
                <Copy className="w-3 h-3 mr-1" />
                Kopyala
              </Button>
            </div>
            <pre className="text-xs bg-muted p-3 rounded overflow-auto max-h-40">
              {error?.stack || error?.message}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Hook-based Error Boundary wrapper
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryConfig?: Omit<ErrorBoundaryProps, "children">
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundaryClass {...errorBoundaryConfig}>
      <Component {...props} />
    </ErrorBoundaryClass>
  );

  WrappedComponent.displayName = `withErrorBoundary(${
    Component.displayName || Component.name
  })`;

  return WrappedComponent;
}

// Hook for handling errors in functional components
export function useErrorBoundaryHandler() {
  const errorHandler = useErrorHandler();

  const captureError = React.useCallback(
    (error: Error, context?: string) => {
      errorHandler.handleError(error, {
        category: "unknown",
        severity: "high",
        component: context,
      });
    },
    [errorHandler]
  );

  return { captureError };
}

// Specialized error boundaries for different contexts
export function SessionErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundaryClass
      variant="card"
      onError={(error: any, errorInfo: any) => {
        console.error("Session Error:", error, errorInfo);
      }}
    >
      {children}
    </ErrorBoundaryClass>
  );
}

export function TabErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundaryClass variant="default" enableRetry={true}>
      {children}
    </ErrorBoundaryClass>
  );
}

export function APIErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundaryClass
      variant="minimal"
      onError={(error: any, errorInfo: any) => {
        // Log API errors to monitoring service
        console.error("API Error:", error, errorInfo);
      }}
    >
      {children}
    </ErrorBoundaryClass>
  );
}

// Main export
export default ErrorBoundaryClass;
export { ErrorBoundaryClass as ErrorBoundary };
