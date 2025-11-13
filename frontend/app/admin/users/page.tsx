"use client";

import React, { useState, useEffect } from "react";
import ModernAdminLayout from "../components/ModernAdminLayout";
import UserTable from "./components/UserTable";
import UserModal from "./components/UserModal";
import UserStats from "./components/UserStats";
import {
  adminApiClient,
  AdminUser,
  CreateUserRequest,
  UpdateUserRequest,
} from "@/lib/admin-api";

export default function UsersPage() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState<AdminUser | null>(null);
  const [modalMode, setModalMode] = useState<"create" | "edit">("create");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await adminApiClient.getUsers();
      setUsers(data);
    } catch (error) {
      console.error("Failed to fetch users:", error);
      setError("Failed to load users");
      setUsers([]);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateUser = () => {
    setSelectedUser(null);
    setModalMode("create");
    setShowModal(true);
  };

  const handleEditUser = (user: AdminUser) => {
    setSelectedUser(user);
    setModalMode("edit");
    setShowModal(true);
  };

  const handleDeleteUser = async (userId: number) => {
    // Find the user to check if it's the admin
    const userToDelete = users.find((user) => user.id === userId);

    // Prevent deletion of admin user
    if (userToDelete?.username === "admin") {
      setError("Sistem administratörü kullanıcısı silinemez!");
      return;
    }

    if (confirm("Bu kullanıcıyı silmek istediğinizden emin misiniz?")) {
      try {
        await adminApiClient.deleteUser(userId);
        setUsers(users.filter((user) => user.id !== userId));
        setError(null);
      } catch (error) {
        console.error("Failed to delete user:", error);
        setError("Kullanıcı silinemedi");
      }
    }
  };

  const handleUserSaved = async (userData: any) => {
    try {
      if (modalMode === "create") {
        const newUser = await adminApiClient.createUser(
          userData as CreateUserRequest
        );
        setUsers([...users, newUser]);
      } else {
        // For edit mode, separate password handling
        const { password, ...updateData } = userData;

        // Update user info first
        const updatedUser = await adminApiClient.updateUser(
          updateData as UpdateUserRequest
        );

        // If password was provided, update it separately
        if (password && password.trim()) {
          await adminApiClient.changePassword(userData.id, password);
        }

        setUsers(
          users.map((user) => (user.id === updatedUser.id ? updatedUser : user))
        );
      }
      setShowModal(false);
      setSelectedUser(null);
      setError(null);
    } catch (error) {
      console.error("Failed to save user:", error);
      setError(
        "Kullanıcı kaydedilemedi: " +
          (error instanceof Error ? error.message : "Bilinmeyen hata")
      );
    }
  };

  const handleBulkAction = async (action: string, selectedIds: number[]) => {
    try {
      // Double-check to ensure admin user is not included in bulk operations
      const filteredIds = selectedIds.filter((id) => {
        const user = users.find((u) => u.id === id);
        return user?.username !== "admin";
      });

      if (action === "activate" || action === "deactivate") {
        if (filteredIds.length === 0) {
          setError("Seçili kullanıcı bulunamadı");
          return;
        }

        await adminApiClient.bulkUpdateUsers(action, filteredIds);
        // Refresh users list
        await fetchUsers();
        setError(null);
      }
    } catch (error) {
      console.error("Failed to perform bulk action:", error);
      setError("Toplu işlem başarısız oldu");
    }
  };

  return (
    <ModernAdminLayout
      title="Kullanıcı Yönetimi"
      description="Sistem kullanıcılarını ve yetkilerini yönetin"
    >
      <div className="space-y-3">
        {/* Error Message */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg
                  className="h-5 w-5 text-red-400"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-red-800 dark:text-red-200">
                  {error}
                </p>
              </div>
              <div className="ml-auto pl-3">
                <button
                  onClick={() => setError(null)}
                  className="inline-flex rounded-md bg-red-50 dark:bg-red-900/20 p-1.5 text-red-500 hover:bg-red-100 dark:hover:bg-red-900/40"
                >
                  <svg
                    className="h-5 w-5"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                      clipRule="evenodd"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Action Bar */}
        <div className="flex justify-between items-center bg-white dark:bg-gray-800 p-3 rounded-lg shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Kullanıcılar ({users.length})
          </h2>
          <button
            onClick={handleCreateUser}
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
            Yeni Kullanıcı
          </button>
        </div>

        {/* Users Table */}
        <UserTable
          users={users}
          loading={loading}
          onEdit={handleEditUser}
          onDelete={handleDeleteUser}
          onBulkAction={handleBulkAction}
        />

        {/* User Modal */}
        {showModal && (
          <UserModal
            user={selectedUser}
            mode={modalMode}
            onSave={handleUserSaved}
            onClose={() => {
              setShowModal(false);
              setSelectedUser(null);
            }}
          />
        )}
      </div>
    </ModernAdminLayout>
  );
}
