/**
 * AuthGuard Component for RAG Education Assistant
 * Provides component-level authentication and authorization protection
 */

"use client";

import React from "react";
import { AuthGuardProps, Permission } from "../../types/auth";
import { useAuth, usePermissions, useRoles } from "../../hooks/useAuth";

// ===== DEFAULT FALLBACK COMPONENTS =====

const DefaultLoadingFallback: React.FC = () => (
  <div className="flex items-center justify-center p-4">
    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
  </div>
);

const DefaultUnauthenticatedFallback: React.FC = () => (
  <div className="text-center p-6 bg-card rounded-lg border border-border">
    <div className="w-12 h-12 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
      <svg
        className="w-6 h-6 text-muted-foreground"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
        />
      </svg>
    </div>
    <p className="text-sm text-muted-foreground">
      Please log in to access this content.
    </p>
  </div>
);

const DefaultUnauthorizedFallback: React.FC = () => (
  <div className="text-center p-6 bg-card rounded-lg border border-border">
    <div className="w-12 h-12 bg-destructive/10 rounded-full flex items-center justify-center mx-auto mb-4">
      <svg
        className="w-6 h-6 text-destructive"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728L5.636 5.636m12.728 12.728L5.636 5.636"
        />
      </svg>
    </div>
    <p className="text-sm text-muted-foreground">
      You don't have permission to access this content.
    </p>
  </div>
);

// ===== MAIN AUTH GUARD COMPONENT =====

const AuthGuard: React.FC<AuthGuardProps> = ({
  children,
  permissions = [],
  roles = [],
  fallback,
  showError = true,
}) => {
  const auth = useAuth();
  const { hasAllPermissions } = usePermissions(permissions);
  const { hasAnyRole } = useRoles(roles);

  // Show loading state
  if (auth.isLoading) {
    return fallback || <DefaultLoadingFallback />;
  }

  // Check authentication
  if (!auth.isAuthenticated) {
    if (!showError) return null;
    return fallback || <DefaultUnauthenticatedFallback />;
  }

  // Check role requirements
  if (roles.length > 0 && !hasAnyRole) {
    if (!showError) return null;
    return fallback || <DefaultUnauthorizedFallback />;
  }

  // Check permission requirements
  if (permissions.length > 0 && !hasAllPermissions) {
    if (!showError) return null;
    return fallback || <DefaultUnauthorizedFallback />;
  }

  // All checks passed, render children
  return <>{children}</>;
};

// ===== SPECIALIZED AUTH GUARD VARIANTS =====

/**
 * Simple authentication guard - only checks if user is logged in
 */
export const AuthRequired: React.FC<
  Pick<AuthGuardProps, "children" | "fallback" | "showError">
> = (props) => <AuthGuard {...props} />;

/**
 * Permission-based guard
 */
interface PermissionGuardProps
  extends Pick<AuthGuardProps, "children" | "fallback" | "showError"> {
  resource: string;
  action: string;
}

export const PermissionGuard: React.FC<PermissionGuardProps> = ({
  resource,
  action,
  ...props
}) => <AuthGuard {...props} permissions={[{ resource, action }]} />;

/**
 * Multiple permissions guard (requires ALL permissions)
 */
interface MultiPermissionGuardProps
  extends Pick<AuthGuardProps, "children" | "fallback" | "showError"> {
  permissions: Permission[];
  requireAll?: boolean; // Default: true (requires all permissions)
}

export const MultiPermissionGuard: React.FC<MultiPermissionGuardProps> = ({
  permissions,
  requireAll = true,
  children,
  fallback,
  showError,
}) => {
  const auth = useAuth();

  if (auth.isLoading) {
    return fallback || <DefaultLoadingFallback />;
  }

  if (!auth.isAuthenticated) {
    if (!showError) return null;
    return fallback || <DefaultUnauthenticatedFallback />;
  }

  const hasPermission = requireAll
    ? permissions.every(({ resource, action }) =>
        auth.checkPermission(resource, action)
      )
    : permissions.some(({ resource, action }) =>
        auth.checkPermission(resource, action)
      );

  if (!hasPermission) {
    if (!showError) return null;
    return fallback || <DefaultUnauthorizedFallback />;
  }

  return <>{children}</>;
};

/**
 * Admin-only guard
 */
export const AdminGuard: React.FC<
  Pick<AuthGuardProps, "children" | "fallback" | "showError">
> = (props) => <AuthGuard {...props} roles={["admin"]} />;

/**
 * Teacher or Admin guard
 */
export const TeacherGuard: React.FC<
  Pick<AuthGuardProps, "children" | "fallback" | "showError">
> = (props) => <AuthGuard {...props} roles={["admin", "teacher"]} />;

/**
 * Any authenticated user guard (student, teacher, or admin)
 */
export const StudentGuard: React.FC<
  Pick<AuthGuardProps, "children" | "fallback" | "showError">
> = (props) => <AuthGuard {...props} roles={["admin", "teacher", "student"]} />;

// ===== CONDITIONAL RENDERING HELPERS =====

/**
 * Show content only if user has specific permission
 */
interface ShowIfPermissionProps {
  resource: string;
  action: string;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const ShowIfPermission: React.FC<ShowIfPermissionProps> = ({
  resource,
  action,
  children,
  fallback = null,
}) => {
  const auth = useAuth();

  if (!auth.isAuthenticated || !auth.checkPermission(resource, action)) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

/**
 * Show content only if user has specific role
 */
interface ShowIfRoleProps {
  role: string;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const ShowIfRole: React.FC<ShowIfRoleProps> = ({
  role,
  children,
  fallback = null,
}) => {
  const auth = useAuth();

  if (!auth.isAuthenticated || !auth.hasRole(role)) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

/**
 * Show content only if user has any of the specified roles
 */
interface ShowIfAnyRoleProps {
  roles: string[];
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const ShowIfAnyRole: React.FC<ShowIfAnyRoleProps> = ({
  roles,
  children,
  fallback = null,
}) => {
  const { hasAnyRole } = useRoles(roles);

  if (!hasAnyRole) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

/**
 * Show content only if user is authenticated
 */
interface ShowIfAuthenticatedProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const ShowIfAuthenticated: React.FC<ShowIfAuthenticatedProps> = ({
  children,
  fallback = null,
}) => {
  const auth = useAuth();

  if (!auth.isAuthenticated) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

/**
 * Show content only if user is NOT authenticated
 */
export const ShowIfNotAuthenticated: React.FC<ShowIfAuthenticatedProps> = ({
  children,
  fallback = null,
}) => {
  const auth = useAuth();

  if (auth.isAuthenticated) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

// ===== HIGHER ORDER COMPONENTS =====

/**
 * HOC to wrap components with auth guard
 */
export function withAuthGuard<P extends object>(
  Component: React.ComponentType<P>,
  guardProps: Omit<AuthGuardProps, "children">
) {
  const WrappedComponent = (props: P) => (
    <AuthGuard {...guardProps}>
      <Component {...props} />
    </AuthGuard>
  );

  WrappedComponent.displayName = `withAuthGuard(${
    Component.displayName || Component.name
  })`;

  return WrappedComponent;
}

/**
 * HOC for admin-only components
 */
export function withAdminGuard<P extends object>(
  Component: React.ComponentType<P>
) {
  return withAuthGuard(Component, { roles: ["admin"] });
}

/**
 * HOC for teacher-accessible components
 */
export function withTeacherGuard<P extends object>(
  Component: React.ComponentType<P>
) {
  return withAuthGuard(Component, { roles: ["admin", "teacher"] });
}

/**
 * HOC for permission-based components
 */
export function withPermissionGuard<P extends object>(
  Component: React.ComponentType<P>,
  permissions: Permission[]
) {
  return withAuthGuard(Component, { permissions });
}

// ===== UTILITY FUNCTIONS =====

/**
 * Create a custom auth guard with specific requirements
 */
export function createAuthGuard(guardProps: Omit<AuthGuardProps, "children">) {
  return (props: Pick<AuthGuardProps, "children" | "fallback">) => (
    <AuthGuard {...guardProps} {...props} />
  );
}

/**
 * Create a permission-specific guard
 */
export function createPermissionGuard(resource: string, action: string) {
  return createAuthGuard({ permissions: [{ resource, action }] });
}

/**
 * Create a role-specific guard
 */
export function createRoleGuard(roles: string[]) {
  return createAuthGuard({ roles });
}

export default AuthGuard;
