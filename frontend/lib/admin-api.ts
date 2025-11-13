/**
 * Admin API Service
 * Centralized API functions for admin panel backend communication
 * Portlar merkezi config'den alınır
 */

import { apiClient } from "./api-client";
import { URLS } from "@/config/ports";

// ===== CONFIGURATION =====

const MAIN_GATEWAY_URL = URLS.API_GATEWAY;
const AUTH_SERVICE_URL = URLS.AUTH_SERVICE;

// ===== TYPE DEFINITIONS =====

export interface AdminStats {
  totalUsers: number;
  activeUsers: number;
  activeSessions: number;
  totalSessions: number;
  totalRoles: number;
}

// Import SessionMeta from api.ts for learning sessions
export interface LearningSession {
  session_id: string;
  name: string;
  description: string;
  category: string;
  status: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  last_accessed: string;
  grade_level: string;
  subject_area: string;
  learning_objectives: string[];
  tags: string[];
  document_count: number;
  total_chunks: number;
  query_count: number;
  student_entry_count: number;
  user_rating: number;
  is_public: boolean;
  backup_count: number;
  rag_settings?: {
    model?: string;
    chain_type?: "stuff" | "refine" | "map_reduce";
    top_k?: number;
    use_rerank?: boolean;
    min_score?: number;
    max_context_chars?: number;
    use_direct_llm?: boolean;
    embedding_model?: string;
  } | null;
}

export interface ActivityLog {
  id: number;
  type:
    | "user_created"
    | "user_login"
    | "user_logout"
    | "role_assigned"
    | "session_expired";
  message: string;
  user: string;
  user_id?: number;
  timestamp: string;
  metadata?: any;
}

export interface SystemHealth {
  status: "healthy" | "warning" | "critical";
  uptime: string;
  lastBackup: string;
  diskUsage: string;
  memoryUsage: string;
  services: {
    auth_service: boolean;
    main_gateway: boolean;
    database: boolean;
  };
}

export interface AdminUser {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  is_active: boolean;
  role_name: string;
  created_at: string;
  last_login: string | null;
  permissions?: string[];
}

export interface AdminSession {
  id: number;
  user_id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  role_name: string;
  ip_address: string;
  user_agent: string;
  created_at: string;
  expires_at: string;
  last_activity: string;
  is_active: boolean;
}

export interface AdminRole {
  id: number;
  name: string;
  description: string;
  permissions: Record<string, string[]>;
  user_count?: number;
  created_at: string;
  updated_at: string;
}

export interface CreateUserRequest {
  username: string;
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  role_name: string;
  is_active?: boolean;
}

export interface UpdateUserRequest {
  id: number;
  username?: string;
  email?: string;
  first_name?: string;
  last_name?: string;
  role_name?: string;
  is_active?: boolean;
}

export interface CreateRoleRequest {
  name: string;
  description: string;
  permissions: Record<string, string[]>;
}

export interface UpdateRoleRequest {
  id: number;
  name?: string;
  description?: string;
  permissions?: Record<string, string[]>;
}

// ===== ADMIN STATISTICS =====

export async function getAdminStats(): Promise<AdminStats> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${AUTH_SERVICE_URL}/admin/stats`, {
    method: "GET",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Failed to fetch admin stats: ${response.status}`
    );
  }

  return await response.json();
}

// ===== ACTIVITY LOGS =====

export async function getActivityLogs(
  limit: number = 10
): Promise<ActivityLog[]> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(
    `${AUTH_SERVICE_URL}/admin/activity-logs?limit=${limit}`,
    {
      method: "GET",
      credentials: "include",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Failed to fetch activity logs: ${response.status}`
    );
  }

  return await response.json();
}

// ===== SYSTEM HEALTH =====

export async function getSystemHealth(): Promise<SystemHealth> {
  try {
    // Check both services
    const [authCheck, gatewayCheck] = await Promise.allSettled([
      fetch(`${AUTH_SERVICE_URL}/health`, { method: "GET" }),
      fetch(`${MAIN_GATEWAY_URL}/health`, { method: "GET" }),
    ]);

    const authHealthy = authCheck.status === "fulfilled" && authCheck.value.ok;
    const gatewayHealthy =
      gatewayCheck.status === "fulfilled" && gatewayCheck.value.ok;

    // Get detailed health info from auth service if available
    let healthData: SystemHealth = {
      status: authHealthy && gatewayHealthy ? "healthy" : "warning",
      uptime: "99.9%",
      lastBackup: "2 saat önce",
      diskUsage: "45%",
      memoryUsage: "67%",
      services: {
        auth_service: authHealthy,
        main_gateway: gatewayHealthy,
        database: true, // Assume healthy if services are responding
      },
    };

    if (authHealthy) {
      try {
        const detailedHealth = await fetch(
          `${AUTH_SERVICE_URL}/admin/system-health`,
          {
            headers: {
              Authorization: `Bearer ${getAccessToken()}`,
              "Content-Type": "application/json",
            },
          }
        );

        if (detailedHealth.ok) {
          const data = await detailedHealth.json();
          healthData = { ...healthData, ...data };
        }
      } catch (error) {
        console.warn("Could not fetch detailed health data:", error);
      }
    }

    return healthData;
  } catch (error) {
    console.error("Error checking system health:", error);
    return {
      status: "critical",
      uptime: "Unknown",
      lastBackup: "Unknown",
      diskUsage: "Unknown",
      memoryUsage: "Unknown",
      services: {
        auth_service: false,
        main_gateway: false,
        database: false,
      },
    };
  }
}

// ===== USER MANAGEMENT =====

export async function getAdminUsers(): Promise<AdminUser[]> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${AUTH_SERVICE_URL}/admin/users`, {
    method: "GET",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Kullanıcılar yüklenemedi: ${response.status}`
    );
  }

  return await response.json();
}

export async function createAdminUser(
  userData: CreateUserRequest
): Promise<AdminUser> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${AUTH_SERVICE_URL}/admin/users`, {
    method: "POST",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(userData),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Kullanıcı oluşturulamadı: ${response.status}`
    );
  }

  return await response.json();
}

export async function updateAdminUser(
  userData: UpdateUserRequest
): Promise<AdminUser> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(
    `${AUTH_SERVICE_URL}/admin/users/${userData.id}`,
    {
      method: "PUT",
      credentials: "include",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Kullanıcı güncellenemedi: ${response.status}`
    );
  }

  return await response.json();
}

export async function changePassword(
  userId: number,
  newPassword: string
): Promise<void> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(
    `${AUTH_SERVICE_URL}/admin/users/${userId}/password`,
    {
      method: "PATCH",
      credentials: "include",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ new_password: newPassword }),
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Şifre değiştirilemedi: ${response.status}`
    );
  }
}

export async function deleteAdminUser(userId: number): Promise<void> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${AUTH_SERVICE_URL}/admin/users/${userId}`, {
    method: "DELETE",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Kullanıcı silinemedi: ${response.status}`
    );
  }
}

export async function bulkUpdateUsers(
  action: "activate" | "deactivate",
  userIds: number[]
): Promise<void> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${AUTH_SERVICE_URL}/admin/users/bulk`, {
    method: "PATCH",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      action,
      user_ids: userIds,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Toplu güncelleme başarısız: ${response.status}`
    );
  }
}

// ===== SESSION MANAGEMENT =====

export async function getAdminSessions(): Promise<AdminSession[]> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${AUTH_SERVICE_URL}/admin/sessions`, {
    method: "GET",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Oturumlar yüklenemedi: ${response.status}`
    );
  }

  return await response.json();
}

export async function terminateSession(sessionId: number): Promise<void> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(
    `${AUTH_SERVICE_URL}/admin/sessions/${sessionId}`,
    {
      method: "DELETE",
      credentials: "include",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Oturum sonlandırılamadı: ${response.status}`
    );
  }
}

export async function terminateUserSessions(userId: number): Promise<void> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(
    `${AUTH_SERVICE_URL}/admin/users/${userId}/sessions`,
    {
      method: "DELETE",
      credentials: "include",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Kullanıcı oturumları sonlandırılamadı: ${response.status}`
    );
  }
}

// ===== ROLE MANAGEMENT =====

export async function getAdminRoles(): Promise<AdminRole[]> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${AUTH_SERVICE_URL}/admin/roles`, {
    method: "GET",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Roller yüklenemedi: ${response.status}`
    );
  }

  return await response.json();
}

export async function createAdminRole(
  roleData: CreateRoleRequest
): Promise<AdminRole> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${AUTH_SERVICE_URL}/admin/roles`, {
    method: "POST",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(roleData),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Rol oluşturulamadı: ${response.status}`
    );
  }

  return await response.json();
}

export async function updateAdminRole(
  roleData: UpdateRoleRequest
): Promise<AdminRole> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(
    `${AUTH_SERVICE_URL}/admin/roles/${roleData.id}`,
    {
      method: "PUT",
      credentials: "include",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(roleData),
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Rol güncellenemedi: ${response.status}`
    );
  }

  return await response.json();
}

export async function deleteAdminRole(roleId: number): Promise<void> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${AUTH_SERVICE_URL}/admin/roles/${roleId}`, {
    method: "DELETE",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Rol silinemedi: ${response.status}`
    );
  }
}

// ===== UTILITY FUNCTIONS =====

function getAccessToken(): string | null {
  if (typeof window !== "undefined") {
    return localStorage.getItem("access_token");
  }
  return null;
}

// ===== ERROR HANDLING =====

export class AdminApiError extends Error {
  constructor(message: string, public status?: number, public details?: any) {
    super(message);
    this.name = "AdminApiError";
  }
}

// ===== LEARNING SESSIONS MANAGEMENT =====

export async function getAdminLearningSessions(): Promise<LearningSession[]> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${MAIN_GATEWAY_URL}/sessions`, {
    method: "GET",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Ders oturumları yüklenemedi: ${response.status}`
    );
  }

  return await response.json();
}

export async function deleteAdminLearningSession(
  sessionId: string
): Promise<void> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(`${MAIN_GATEWAY_URL}/sessions/${sessionId}`, {
    method: "DELETE",
    credentials: "include",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Ders oturumu silinemedi: ${response.status}`
    );
  }
}

export async function updateAdminLearningSessionStatus(
  sessionId: string,
  status: "active" | "inactive"
): Promise<LearningSession> {
  const token = getAccessToken();
  if (!token) {
    throw new Error("No access token found");
  }

  const response = await fetch(
    `${MAIN_GATEWAY_URL}/sessions/${sessionId}/status`,
    {
      method: "PATCH",
      credentials: "include",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ status }),
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        errorData.message ||
        `Ders oturumu durumu güncellenemedi: ${response.status}`
    );
  }

  const result = await response.json();
  return result.updated_session || result;
}

// ===== API CLIENT WRAPPER =====

export const adminApiClient = {
  // Statistics
  getStats: getAdminStats,

  // Activity Logs
  getActivityLogs,

  // System Health
  getSystemHealth,

  // User Management
  getUsers: getAdminUsers,
  createUser: createAdminUser,
  updateUser: updateAdminUser,
  changePassword,
  deleteUser: deleteAdminUser,
  bulkUpdateUsers,

  // Session Management (User Login Sessions)
  getSessions: getAdminSessions,
  terminateSession,
  terminateUserSessions,

  // Learning Sessions Management
  getLearningSessions: getAdminLearningSessions,
  deleteLearningSession: deleteAdminLearningSession,
  updateLearningSessionStatus: updateAdminLearningSessionStatus,

  // Role Management
  getRoles: getAdminRoles,
  createRole: createAdminRole,
  updateRole: updateAdminRole,
  deleteRole: deleteAdminRole,
};

export default adminApiClient;
