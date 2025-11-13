/**
 * ProtectedRoute Component for RAG Education Assistant
 * Protects routes based on authentication status, permissions, and roles
 */

"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ProtectedRouteProps, Permission, AUTH_ROLES } from "../../types/auth";
import { useAuth, usePermissions, useRoles } from "../../hooks/useAuth";

// ===== LOADING COMPONENT =====

const LoadingSpinner: React.FC = () => (
  <div className="flex items-center justify-center min-h-screen bg-background">
    <div className="flex flex-col items-center space-y-4">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      <p className="text-muted-foreground">Loading...</p>
    </div>
  </div>
);

// ===== ACCESS DENIED COMPONENT =====

interface AccessDeniedProps {
  title?: string;
  message?: string;
  showBackButton?: boolean;
}

const AccessDenied: React.FC<AccessDeniedProps> = ({
  title = "Access Denied",
  message = "You don't have permission to access this page.",
  showBackButton = true,
}) => {
  const router = useRouter();

  return (
    <div className="flex items-center justify-center min-h-screen bg-background px-4">
      <div className="max-w-md w-full bg-card rounded-2xl shadow-2xl p-8 text-center">
        <div className="w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center mx-auto mb-6">
          <svg
            className="w-8 h-8 text-destructive"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.664-.833-2.464 0L4.35 16.5c-.77.833.192 2.5 1.732 2.5z"
            />
          </svg>
        </div>

        <h1 className="text-2xl font-bold text-foreground mb-4">{title}</h1>

        <p className="text-muted-foreground mb-8">{message}</p>

        {showBackButton && (
          <div className="space-y-3">
            <button
              onClick={() => router.back()}
              className="btn btn-outline w-full"
            >
              Go Back
            </button>
            <button
              onClick={() => router.push("/")}
              className="btn btn-primary w-full"
            >
              Go Home
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// ===== MAIN PROTECTED ROUTE COMPONENT =====

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredPermissions = [],
  requiredRole,
  fallback,
  redirectTo = "/login",
}) => {
  const auth = useAuth();
  const router = useRouter();
  const [isChecking, setIsChecking] = useState(true);

  // Check permissions if required
  const { hasAllPermissions } = usePermissions(requiredPermissions);

  // Check role if required
  const { hasAllowedRole } = useRoles(requiredRole ? [requiredRole] : []);

  useEffect(() => {
    // Wait for auth to initialize
    if (auth.isLoading) return;

    setIsChecking(false);

    // Redirect to login if not authenticated
    if (!auth.isAuthenticated) {
      const currentPath = window.location.pathname + window.location.search;
      const loginUrl = `${redirectTo}?redirect=${encodeURIComponent(
        currentPath
      )}`;
      router.push(loginUrl);
      return;
    }
  }, [auth.isLoading, auth.isAuthenticated, redirectTo, router]);

  // Show loading while checking authentication
  if (auth.isLoading || isChecking) {
    return fallback || <LoadingSpinner />;
  }

  // Not authenticated - this case is handled by useEffect redirect
  if (!auth.isAuthenticated) {
    return fallback || <LoadingSpinner />;
  }

  // Check role requirement
  if (requiredRole && !hasAllowedRole) {
    const roleMessage = `You need ${requiredRole} role to access this page.`;
    return (
      fallback || (
        <AccessDenied title="Insufficient Permissions" message={roleMessage} />
      )
    );
  }

  // Check permission requirements
  if (requiredPermissions.length > 0 && !hasAllPermissions) {
    const permissionList = requiredPermissions
      .map((p) => `${p.resource}:${p.action}`)
      .join(", ");
    const permissionMessage = `You need the following permissions: ${permissionList}`;

    return (
      fallback || (
        <AccessDenied
          title="Insufficient Permissions"
          message={permissionMessage}
        />
      )
    );
  }

  // All checks passed, render children
  return <>{children}</>;
};

// ===== HELPER COMPONENTS =====

/**
 * AdminRoute - Shorthand for admin-only routes
 */
export const AdminRoute: React.FC<Omit<ProtectedRouteProps, "requiredRole">> = (
  props
) => <ProtectedRoute {...props} requiredRole={AUTH_ROLES.ADMIN} />;

/**
 * TeacherRoute - Shorthand for teacher-accessible routes
 */
export const TeacherRoute: React.FC<
  Omit<ProtectedRouteProps, "requiredRole">
> = (props) => <ProtectedRoute {...props} requiredRole={AUTH_ROLES.TEACHER} />;

/**
 * StudentRoute - Shorthand for student-accessible routes
 */
export const StudentRoute: React.FC<
  Omit<ProtectedRouteProps, "requiredRole">
> = (props) => <ProtectedRoute {...props} requiredRole={AUTH_ROLES.STUDENT} />;

/**
 * AuthenticatedRoute - Shorthand for routes that just require authentication
 */
export const AuthenticatedRoute: React.FC<
  Pick<ProtectedRouteProps, "children" | "fallback" | "redirectTo">
> = (props) => <ProtectedRoute {...props} />;

// ===== HIGHER ORDER COMPONENT =====

/**
 * HOC to wrap components with route protection
 */
export function withProtectedRoute<P extends object>(
  Component: React.ComponentType<P>,
  protection: Omit<ProtectedRouteProps, "children">
) {
  const WrappedComponent = (props: P) => (
    <ProtectedRoute {...protection}>
      <Component {...props} />
    </ProtectedRoute>
  );

  WrappedComponent.displayName = `withProtectedRoute(${
    Component.displayName || Component.name
  })`;

  return WrappedComponent;
}

/**
 * HOC for admin-only components
 */
export function withAdminRoute<P extends object>(
  Component: React.ComponentType<P>
) {
  return withProtectedRoute(Component, { requiredRole: AUTH_ROLES.ADMIN });
}

/**
 * HOC for teacher-accessible components
 */
export function withTeacherRoute<P extends object>(
  Component: React.ComponentType<P>
) {
  return withProtectedRoute(Component, { requiredRole: AUTH_ROLES.TEACHER });
}

/**
 * HOC for student-accessible components
 */
export function withStudentRoute<P extends object>(
  Component: React.ComponentType<P>
) {
  return withProtectedRoute(Component, { requiredRole: AUTH_ROLES.STUDENT });
}

// ===== UTILITY FUNCTIONS =====

/**
 * Create a protected route with specific permissions
 */
export function createPermissionRoute(permissions: Permission[]) {
  return (
    props: Pick<ProtectedRouteProps, "children" | "fallback" | "redirectTo">
  ) => <ProtectedRoute {...props} requiredPermissions={permissions} />;
}

/**
 * Create a route that requires multiple roles (any one of them)
 */
export function createMultiRoleRoute(roles: string[]) {
  return (
    props: Pick<ProtectedRouteProps, "children" | "fallback" | "redirectTo">
  ) => {
    const { hasAnyRole } = useRoles(roles);
    const [isChecking, setIsChecking] = useState(true);
    const auth = useAuth();
    const router = useRouter();

    useEffect(() => {
      if (auth.isLoading) return;
      setIsChecking(false);

      if (!auth.isAuthenticated) {
        router.push("/login");
      }
    }, [auth.isLoading, auth.isAuthenticated, router]);

    if (auth.isLoading || isChecking) {
      return props.fallback || <LoadingSpinner />;
    }

    if (!auth.isAuthenticated) {
      return props.fallback || <LoadingSpinner />;
    }

    if (!hasAnyRole) {
      const roleMessage = `You need one of the following roles: ${roles.join(
        ", "
      )}`;
      return (
        props.fallback || (
          <AccessDenied
            title="Insufficient Permissions"
            message={roleMessage}
          />
        )
      );
    }

    return <>{props.children}</>;
  };
}

export default ProtectedRoute;
