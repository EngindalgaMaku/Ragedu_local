"use client";

import React, { useState, useEffect } from "react";
import ModernAdminLayout from "../components/ModernAdminLayout";
import RoleTable from "./components/RoleTable";
import RoleModal from "./components/RoleModal";
import {
  adminApiClient,
  AdminRole,
  CreateRoleRequest,
  UpdateRoleRequest,
} from "@/lib/admin-api";

export default function RolesPage() {
  const [roles, setRoles] = useState<AdminRole[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [selectedRole, setSelectedRole] = useState<AdminRole | null>(null);
  const [modalMode, setModalMode] = useState<"create" | "edit">("create");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchRoles();
  }, []);

  const fetchRoles = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await adminApiClient.getRoles();
      setRoles(data);
    } catch (error) {
      console.error("Failed to fetch roles:", error);
      setError("Failed to load roles");
      setRoles([]);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateRole = () => {
    setSelectedRole(null);
    setModalMode("create");
    setShowModal(true);
  };

  const handleEditRole = (role: AdminRole) => {
    setSelectedRole(role);
    setModalMode("edit");
    setShowModal(true);
  };

  const handleDeleteRole = async (roleId: number) => {
    const role = roles.find((r) => r.id === roleId);
    if (role && (role.user_count || 0) > 0) {
      alert(
        `Cannot delete role "${role.name}" because it is assigned to ${
          role.user_count || 0
        } user(s). Please reassign those users first.`
      );
      return;
    }

    if (confirm("Are you sure you want to delete this role?")) {
      try {
        await adminApiClient.deleteRole(roleId);
        setRoles(roles.filter((role) => role.id !== roleId));
        setError(null);
      } catch (error) {
        console.error("Failed to delete role:", error);
        setError("Failed to delete role");
      }
    }
  };

  const handleRoleSaved = async (
    roleData: CreateRoleRequest | UpdateRoleRequest
  ) => {
    try {
      if (modalMode === "create") {
        const newRole = await adminApiClient.createRole(
          roleData as CreateRoleRequest
        );
        setRoles([...roles, { ...newRole, user_count: 0 }]);
      } else {
        const updatedRole = await adminApiClient.updateRole(
          roleData as UpdateRoleRequest
        );
        setRoles(
          roles.map((role) =>
            role.id === updatedRole.id
              ? { ...updatedRole, user_count: role.user_count || 0 }
              : role
          )
        );
      }
      setShowModal(false);
      setSelectedRole(null);
      setError(null);
    } catch (error) {
      console.error("Failed to save role:", error);
      setError("Failed to save role");
    }
  };

  const getTotalPermissions = (permissions: Record<string, string[]>) => {
    return Object.values(permissions).flat().length;
  };

  const getTotalUsers = () => {
    return roles.reduce((total, role) => total + (role.user_count || 0), 0);
  };

  return (
    <ModernAdminLayout
      title="Rol Yönetimi"
      description="Kullanıcı rollerini ve yetkilerini yönetin"
    >
      <div className="space-y-3">
        {/* Action Bar */}
        <div className="flex justify-between items-center bg-white dark:bg-gray-800 p-3 rounded-lg shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Roller ({roles.length})
          </h2>
          <button
            onClick={handleCreateRole}
            className="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-md transition-colors"
          >
            <svg
              className="w-4 h-4 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6v6m0 0v6m0-6h6m-6 0H6"
              />
            </svg>
            Yeni Rol
          </button>
        </div>

        {/* Roles Table */}
        <RoleTable
          roles={roles}
          loading={loading}
          onEdit={handleEditRole}
          onDelete={handleDeleteRole}
        />

        {/* Role Modal */}
        {showModal && (
          <RoleModal
            role={selectedRole}
            mode={modalMode}
            onSave={handleRoleSaved}
            onClose={() => {
              setShowModal(false);
              setSelectedRole(null);
            }}
          />
        )}
      </div>
    </ModernAdminLayout>
  );
}
