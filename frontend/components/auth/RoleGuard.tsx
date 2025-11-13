/**
 * RoleGuard Component for RAG Education Assistant
 * Provides role-based access control for components
 */

"use client";

import React from "react";
import { RoleGuardProps, AUTH_ROLES } from "../../types/auth";
import { useRoles } from "../../hooks/useAuth";

// ===== DEFAULT FALLBACK COMPONENTS =====

const DefaultLoadingFallback: React.FC = () => (
  <div className="flex items-center justify-center p-4">
    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
  </div>
);

const DefaultUnauthorizedFallback: React.FC<{ allowedRoles: string[] }> = ({
  allowedRoles,
}) => (
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
          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    </div>
    <p className="text-sm text-muted-foreground mb-2">
      Access restricted to: {allowedRoles.join(", ")}
    </p>
    <p className="text-xs text-muted-foreground">
      You don't have the required role to access this content.
    </p>
  </div>
);

// ===== MAIN ROLE GUARD COMPONENT =====

const RoleGuard: React.FC<RoleGuardProps> = ({
  children,
  allowedRoles,
  fallback,
  strict = false,
}) => {
  const { hasAnyRole, userRole, isAuthenticated } = useRoles(allowedRoles);

  // Show loading state while auth is initializing
  // Note: useRoles internally uses useAuth which handles loading state
  if (!isAuthenticated) {
    return fallback || <DefaultLoadingFallback />;
  }

  // In strict mode, require exact role match
  if (strict) {
    const hasExactRole = userRole && allowedRoles.includes(userRole);
    if (!hasExactRole) {
      return (
        fallback || <DefaultUnauthorizedFallback allowedRoles={allowedRoles} />
      );
    }
  } else {
    // In non-strict mode, use role hierarchy
    if (!hasAnyRole) {
      return (
        fallback || <DefaultUnauthorizedFallback allowedRoles={allowedRoles} />
      );
    }
  }

  // All checks passed, render children
  return <>{children}</>;
};

// ===== SPECIALIZED ROLE GUARD VARIANTS =====

/**
 * Admin-only role guard
 */
interface AdminOnlyProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const AdminOnly: React.FC<AdminOnlyProps> = ({ children, fallback }) => (
  <RoleGuard allowedRoles={[AUTH_ROLES.ADMIN]} fallback={fallback} strict>
    {children}
  </RoleGuard>
);

/**
 * Teacher or higher role guard (Teacher or Admin)
 */
export const TeacherOrHigher: React.FC<AdminOnlyProps> = ({
  children,
  fallback,
}) => (
  <RoleGuard
    allowedRoles={[AUTH_ROLES.ADMIN, AUTH_ROLES.TEACHER]}
    fallback={fallback}
  >
    {children}
  </RoleGuard>
);

/**
 * Teacher-only role guard (strict)
 */
export const TeacherOnly: React.FC<AdminOnlyProps> = ({
  children,
  fallback,
}) => (
  <RoleGuard allowedRoles={[AUTH_ROLES.TEACHER]} fallback={fallback} strict>
    {children}
  </RoleGuard>
);

/**
 * Student or higher role guard (any authenticated user)
 */
export const StudentOrHigher: React.FC<AdminOnlyProps> = ({
  children,
  fallback,
}) => (
  <RoleGuard
    allowedRoles={[AUTH_ROLES.ADMIN, AUTH_ROLES.TEACHER, AUTH_ROLES.STUDENT]}
    fallback={fallback}
  >
    {children}
  </RoleGuard>
);

/**
 * Student-only role guard (strict)
 */
export const StudentOnly: React.FC<AdminOnlyProps> = ({
  children,
  fallback,
}) => (
  <RoleGuard allowedRoles={[AUTH_ROLES.STUDENT]} fallback={fallback} strict>
    {children}
  </RoleGuard>
);

// ===== ROLE HIERARCHY COMPONENTS =====

/**
 * Role hierarchy guard - checks if user's role is at or above minimum level
 */
interface RoleHierarchyGuardProps {
  children: React.ReactNode;
  minimumRole: "student" | "teacher" | "admin";
  fallback?: React.ReactNode;
}

export const RoleHierarchyGuard: React.FC<RoleHierarchyGuardProps> = ({
  children,
  minimumRole,
  fallback,
}) => {
  const roleHierarchy = {
    student: [AUTH_ROLES.STUDENT, AUTH_ROLES.TEACHER, AUTH_ROLES.ADMIN],
    teacher: [AUTH_ROLES.TEACHER, AUTH_ROLES.ADMIN],
    admin: [AUTH_ROLES.ADMIN],
  };

  const allowedRoles = roleHierarchy[minimumRole] || [];

  return (
    <RoleGuard allowedRoles={allowedRoles} fallback={fallback}>
      {children}
    </RoleGuard>
  );
};

// ===== CONDITIONAL RENDERING HELPERS =====

/**
 * Render content only for specific role
 */
interface ShowForRoleProps {
  role: string;
  children: React.ReactNode;
  fallback?: React.ReactNode;
  strict?: boolean;
}

export const ShowForRole: React.FC<ShowForRoleProps> = ({
  role,
  children,
  fallback = null,
  strict = false,
}) => (
  <RoleGuard allowedRoles={[role]} fallback={fallback} strict={strict}>
    {children}
  </RoleGuard>
);

/**
 * Render content for multiple roles
 */
interface ShowForRolesProps {
  roles: string[];
  children: React.ReactNode;
  fallback?: React.ReactNode;
  strict?: boolean;
}

export const ShowForRoles: React.FC<ShowForRolesProps> = ({
  roles,
  children,
  fallback = null,
  strict = false,
}) => (
  <RoleGuard allowedRoles={roles} fallback={fallback} strict={strict}>
    {children}
  </RoleGuard>
);

/**
 * Render different content based on user role
 */
interface RoleBasedRenderProps {
  adminContent?: React.ReactNode;
  teacherContent?: React.ReactNode;
  studentContent?: React.ReactNode;
  defaultContent?: React.ReactNode;
}

export const RoleBasedRender: React.FC<RoleBasedRenderProps> = ({
  adminContent,
  teacherContent,
  studentContent,
  defaultContent = null,
}) => {
  const { userRole } = useRoles();

  switch (userRole) {
    case AUTH_ROLES.ADMIN:
      return <>{adminContent || defaultContent}</>;
    case AUTH_ROLES.TEACHER:
      return <>{teacherContent || defaultContent}</>;
    case AUTH_ROLES.STUDENT:
      return <>{studentContent || defaultContent}</>;
    default:
      return <>{defaultContent}</>;
  }
};

// ===== ROLE-SPECIFIC MENU/NAVIGATION HELPERS =====

/**
 * Admin navigation items
 */
export const AdminNav: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <AdminOnly>{children}</AdminOnly>;

/**
 * Teacher navigation items
 */
export const TeacherNav: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <TeacherOrHigher>{children}</TeacherOrHigher>;

/**
 * Student navigation items
 */
export const StudentNav: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <StudentOrHigher>{children}</StudentOrHigher>;

// ===== HIGHER ORDER COMPONENTS =====

/**
 * HOC to wrap components with role guard
 */
export function withRoleGuard<P extends object>(
  Component: React.ComponentType<P>,
  allowedRoles: string[],
  strict: boolean = false
) {
  const WrappedComponent = (props: P) => (
    <RoleGuard allowedRoles={allowedRoles} strict={strict}>
      <Component {...props} />
    </RoleGuard>
  );

  WrappedComponent.displayName = `withRoleGuard(${
    Component.displayName || Component.name
  })`;

  return WrappedComponent;
}

/**
 * HOC for admin-only components
 */
export function withAdminOnly<P extends object>(
  Component: React.ComponentType<P>
) {
  return withRoleGuard(Component, [AUTH_ROLES.ADMIN], true);
}

/**
 * HOC for teacher-accessible components
 */
export function withTeacherAccess<P extends object>(
  Component: React.ComponentType<P>
) {
  return withRoleGuard(Component, [AUTH_ROLES.ADMIN, AUTH_ROLES.TEACHER]);
}

/**
 * HOC for student-accessible components
 */
export function withStudentAccess<P extends object>(
  Component: React.ComponentType<P>
) {
  return withRoleGuard(Component, [
    AUTH_ROLES.ADMIN,
    AUTH_ROLES.TEACHER,
    AUTH_ROLES.STUDENT,
  ]);
}

// ===== ROLE VALIDATION HOOKS =====

/**
 * Hook for role validation with detailed information
 */
export function useRoleValidation(
  requiredRoles: string[],
  strict: boolean = false
) {
  const { userRole, hasAnyRole, isAuthenticated } = useRoles(requiredRoles);

  const isAuthorized = strict
    ? userRole && requiredRoles.includes(userRole)
    : hasAnyRole;

  const getRoleLevel = (role: string): number => {
    switch (role) {
      case AUTH_ROLES.ADMIN:
        return 3;
      case AUTH_ROLES.TEACHER:
        return 2;
      case AUTH_ROLES.STUDENT:
        return 1;
      default:
        return 0;
    }
  };

  const userRoleLevel = getRoleLevel(userRole || "");
  const requiredRoleLevel = Math.min(...requiredRoles.map(getRoleLevel));

  return {
    isAuthenticated,
    userRole,
    userRoleLevel,
    requiredRoles,
    requiredRoleLevel,
    isAuthorized: isAuthenticated && isAuthorized,
    hasHigherRole: userRoleLevel > requiredRoleLevel,
    hasExactRole: userRole && requiredRoles.includes(userRole),
    missingRoles: requiredRoles.filter((role) => role !== userRole),
  };
}

// ===== UTILITY FUNCTIONS =====

/**
 * Create a custom role guard with specific roles
 */
export function createRoleGuard(
  allowedRoles: string[],
  strict: boolean = false
) {
  return (props: { children: React.ReactNode; fallback?: React.ReactNode }) => (
    <RoleGuard {...props} allowedRoles={allowedRoles} strict={strict} />
  );
}

/**
 * Create a minimum role requirement guard
 */
export function createMinimumRoleGuard(
  minimumRole: "student" | "teacher" | "admin"
) {
  return (props: { children: React.ReactNode; fallback?: React.ReactNode }) => (
    <RoleHierarchyGuard {...props} minimumRole={minimumRole} />
  );
}

/**
 * Check if a role is valid
 */
export function isValidRole(role: string): boolean {
  return Object.values(AUTH_ROLES).includes(role as any);
}

/**
 * Get role hierarchy level
 */
export function getRoleHierarchyLevel(role: string): number {
  switch (role) {
    case AUTH_ROLES.ADMIN:
      return 3;
    case AUTH_ROLES.TEACHER:
      return 2;
    case AUTH_ROLES.STUDENT:
      return 1;
    default:
      return 0;
  }
}

/**
 * Compare role hierarchy levels
 */
export function compareRoles(role1: string, role2: string): number {
  return getRoleHierarchyLevel(role1) - getRoleHierarchyLevel(role2);
}

/**
 * Check if role1 is higher than role2 in hierarchy
 */
export function isHigherRole(role1: string, role2: string): boolean {
  return compareRoles(role1, role2) > 0;
}

export default RoleGuard;
