/**
 * Permission Utilities for RAG Education Assistant
 * Provides helper functions for permission and role management
 */

import {
  Permission,
  PermissionMatrix,
  RoleHierarchy,
  RoleLevel,
  User,
  AUTH_ROLES,
  AUTH_PERMISSIONS,
} from "../types/auth";

// ===== ROLE HIERARCHY CONFIGURATION =====

/**
 * Role hierarchy configuration - higher level roles include lower level permissions
 */
export const ROLE_HIERARCHY: RoleHierarchy = {
  admin: ["admin", "teacher", "student"],
  teacher: ["teacher", "student"],
  student: ["student"],
};

/**
 * Role priority levels for comparison
 */
export const ROLE_LEVELS: Record<string, number> = {
  [AUTH_ROLES.ADMIN]: 3,
  [AUTH_ROLES.TEACHER]: 2,
  [AUTH_ROLES.STUDENT]: 1,
};

// ===== PERMISSION MATRIX =====

/**
 * Default permission matrix defining what each role can do
 */
export const PERMISSION_MATRIX: PermissionMatrix = {
  users: {
    create: ["admin"],
    read: ["admin", "teacher"],
    update: ["admin"],
    delete: ["admin"],
  },
  roles: {
    create: ["admin"],
    read: ["admin", "teacher"],
    update: ["admin"],
    delete: ["admin"],
  },
  sessions: {
    create: ["admin", "teacher", "student"],
    read: ["admin", "teacher", "student"],
    update: ["admin", "teacher", "student"],
    delete: ["admin", "teacher", "student"],
  },
  documents: {
    create: ["admin", "teacher"],
    read: ["admin", "teacher", "student"],
    update: ["admin", "teacher"],
    delete: ["admin", "teacher"],
  },
  system: {
    admin: ["admin"],
    configure: ["admin"],
  },
  analytics: {
    read: ["admin", "teacher"],
    export: ["admin"],
  },
  feedback: {
    create: ["admin", "teacher", "student"],
    read: ["admin", "teacher"],
    update: ["admin", "teacher"],
    delete: ["admin"],
  },
};

// ===== PERMISSION CHECKING FUNCTIONS =====

/**
 * Check if user has specific permission
 */
export function hasPermission(
  userPermissions: Record<string, string[]>,
  resource: string,
  action: string
): boolean {
  const resourcePermissions = userPermissions[resource];
  return resourcePermissions ? resourcePermissions.includes(action) : false;
}

/**
 * Check if user has any of the specified permissions
 */
export function hasAnyPermission(
  userPermissions: Record<string, string[]>,
  permissions: Permission[]
): boolean {
  return permissions.some(({ resource, action }) =>
    hasPermission(userPermissions, resource, action)
  );
}

/**
 * Check if user has all specified permissions
 */
export function hasAllPermissions(
  userPermissions: Record<string, string[]>,
  permissions: Permission[]
): boolean {
  return permissions.every(({ resource, action }) =>
    hasPermission(userPermissions, resource, action)
  );
}

/**
 * Get all permissions for a specific resource
 */
export function getResourcePermissions(
  userPermissions: Record<string, string[]>,
  resource: string
): string[] {
  return userPermissions[resource] || [];
}

/**
 * Check if user has permission for a resource (any action)
 */
export function hasResourceAccess(
  userPermissions: Record<string, string[]>,
  resource: string
): boolean {
  const resourcePermissions = userPermissions[resource];
  return resourcePermissions ? resourcePermissions.length > 0 : false;
}

// ===== ROLE CHECKING FUNCTIONS =====

/**
 * Check if user has specific role
 */
export function hasRole(userRole: string, requiredRole: string): boolean {
  return userRole === requiredRole;
}

/**
 * Check if user has any of the specified roles
 */
export function hasAnyRole(userRole: string, allowedRoles: string[]): boolean {
  return allowedRoles.includes(userRole);
}

/**
 * Check if user role is at or above minimum required level
 */
export function hasMinimumRole(userRole: string, minimumRole: string): boolean {
  const userLevel = ROLE_LEVELS[userRole] || 0;
  const requiredLevel = ROLE_LEVELS[minimumRole] || 0;
  return userLevel >= requiredLevel;
}

/**
 * Check if user role has higher priority than comparison role
 */
export function isHigherRole(
  userRole: string,
  comparisonRole: string
): boolean {
  const userLevel = ROLE_LEVELS[userRole] || 0;
  const comparisonLevel = ROLE_LEVELS[comparisonRole] || 0;
  return userLevel > comparisonLevel;
}

/**
 * Get role hierarchy for a specific role
 */
export function getRoleHierarchy(role: string): string[] {
  switch (role) {
    case AUTH_ROLES.ADMIN:
      return ROLE_HIERARCHY.admin;
    case AUTH_ROLES.TEACHER:
      return ROLE_HIERARCHY.teacher;
    case AUTH_ROLES.STUDENT:
      return ROLE_HIERARCHY.student;
    default:
      return [];
  }
}

// ===== PERMISSION MATRIX FUNCTIONS =====

/**
 * Check permission using the permission matrix
 */
export function checkPermissionMatrix(
  userRole: string,
  resource: string,
  action: string
): boolean {
  const resourceMatrix = PERMISSION_MATRIX[resource];
  if (!resourceMatrix) return false;

  const allowedRoles = resourceMatrix[action];
  if (!allowedRoles) return false;

  return allowedRoles.includes(userRole as RoleLevel);
}

/**
 * Get all permissions for a role from the permission matrix
 */
export function getRolePermissions(role: string): Record<string, string[]> {
  const permissions: Record<string, string[]> = {};

  Object.entries(PERMISSION_MATRIX).forEach(([resource, actions]) => {
    const allowedActions: string[] = [];

    Object.entries(actions).forEach(([action, allowedRoles]) => {
      if (allowedRoles.includes(role as RoleLevel)) {
        allowedActions.push(action);
      }
    });

    if (allowedActions.length > 0) {
      permissions[resource] = allowedActions;
    }
  });

  return permissions;
}

/**
 * Get permissions with role hierarchy applied
 */
export function getRolePermissionsWithHierarchy(
  role: string
): Record<string, string[]> {
  const permissions: Record<string, string[]> = {};
  const roleHierarchy = getRoleHierarchy(role);

  Object.entries(PERMISSION_MATRIX).forEach(([resource, actions]) => {
    const allowedActions: string[] = [];

    Object.entries(actions).forEach(([action, allowedRoles]) => {
      if (
        roleHierarchy.some((hierarchyRole) =>
          allowedRoles.includes(hierarchyRole as RoleLevel)
        )
      ) {
        allowedActions.push(action);
      }
    });

    if (allowedActions.length > 0) {
      permissions[resource] = allowedActions;
    }
  });

  return permissions;
}

// ===== USER PERMISSION UTILITIES =====

/**
 * Extract permissions from user object
 */
export function extractUserPermissions(user: User): Record<string, string[]> {
  // Get permissions from role
  const rolePermissions = getRolePermissionsWithHierarchy(user.role_name);

  // You can extend this to include user-specific permissions if needed
  return rolePermissions;
}

/**
 * Check if user can perform action on resource
 */
export function canUserPerformAction(
  user: User | null,
  resource: string,
  action: string
): boolean {
  if (!user) return false;

  // Check using permission matrix with role hierarchy
  return checkPermissionMatrix(user.role_name, resource, action);
}

/**
 * Get all actions user can perform on a resource
 */
export function getUserResourceActions(
  user: User | null,
  resource: string
): string[] {
  if (!user) return [];

  const userPermissions = extractUserPermissions(user);
  return getResourcePermissions(userPermissions, resource);
}

/**
 * Check if user can access any admin features
 */
export function canAccessAdmin(user: User | null): boolean {
  if (!user) return false;
  return user.role_name === AUTH_ROLES.ADMIN;
}

/**
 * Check if user can manage other users
 */
export function canManageUsers(user: User | null): boolean {
  return (
    canUserPerformAction(user, "users", "create") ||
    canUserPerformAction(user, "users", "update") ||
    canUserPerformAction(user, "users", "delete")
  );
}

/**
 * Check if user can view analytics
 */
export function canViewAnalytics(user: User | null): boolean {
  return canUserPerformAction(user, "analytics", "read");
}

/**
 * Check if user can manage documents
 */
export function canManageDocuments(user: User | null): boolean {
  return (
    canUserPerformAction(user, "documents", "create") ||
    canUserPerformAction(user, "documents", "update") ||
    canUserPerformAction(user, "documents", "delete")
  );
}

// ===== PERMISSION VALIDATION =====

/**
 * Validate permission object structure
 */
export function validatePermission(permission: any): permission is Permission {
  return (
    permission &&
    typeof permission === "object" &&
    typeof permission.resource === "string" &&
    typeof permission.action === "string"
  );
}

/**
 * Validate permissions array
 */
export function validatePermissions(permissions: any[]): Permission[] {
  return permissions.filter(validatePermission);
}

/**
 * Validate role name
 */
export function validateRole(role: string): boolean {
  return Object.values(AUTH_ROLES).includes(role as any);
}

/**
 * Sanitize role name
 */
export function sanitizeRole(role: string): string {
  const lowerRole = role.toLowerCase();
  switch (lowerRole) {
    case "admin":
    case "administrator":
      return AUTH_ROLES.ADMIN;
    case "teacher":
    case "instructor":
    case "educator":
      return AUTH_ROLES.TEACHER;
    case "student":
    case "learner":
      return AUTH_ROLES.STUDENT;
    default:
      return AUTH_ROLES.STUDENT; // Default to student role
  }
}

// ===== PERMISSION DEBUGGING =====

/**
 * Get detailed permission information for debugging
 */
export function getPermissionDebugInfo(user: User | null): any {
  if (!user) return { user: null, permissions: {}, role: null };

  const userPermissions = extractUserPermissions(user);
  const roleHierarchy = getRoleHierarchy(user.role_name);

  return {
    user: {
      id: user.id,
      username: user.username,
      role: user.role_name,
    },
    permissions: userPermissions,
    roleHierarchy,
    roleLevel: ROLE_LEVELS[user.role_name] || 0,
    canAccessAdmin: canAccessAdmin(user),
    canManageUsers: canManageUsers(user),
    canManageDocuments: canManageDocuments(user),
    canViewAnalytics: canViewAnalytics(user),
  };
}

/**
 * Log permission check for debugging
 */
export function debugPermissionCheck(
  user: User | null,
  resource: string,
  action: string,
  result: boolean
): void {
  if (process.env.NODE_ENV === "development") {
    console.group(`Permission Check: ${resource}.${action}`);
    console.log("User:", user?.username || "Anonymous");
    console.log("Role:", user?.role_name || "None");
    console.log("Resource:", resource);
    console.log("Action:", action);
    console.log("Result:", result ? "✅ ALLOWED" : "❌ DENIED");

    if (user) {
      const permissions = extractUserPermissions(user);
      console.log("User Permissions:", permissions);
      console.log("Resource Permissions:", permissions[resource] || "None");
    }

    console.groupEnd();
  }
}

// ===== EXPORT CONSTANTS =====

export { AUTH_ROLES, AUTH_PERMISSIONS };

// ===== DEFAULT EXPORT =====

export default {
  hasPermission,
  hasAnyPermission,
  hasAllPermissions,
  hasRole,
  hasAnyRole,
  hasMinimumRole,
  isHigherRole,
  canUserPerformAction,
  extractUserPermissions,
  getRolePermissions,
  getRolePermissionsWithHierarchy,
  validatePermission,
  validateRole,
  sanitizeRole,
  getPermissionDebugInfo,
  debugPermissionCheck,
  ROLE_HIERARCHY,
  ROLE_LEVELS,
  PERMISSION_MATRIX,
};
