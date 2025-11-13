"use client";

import React, { useState, useMemo } from "react";
import { AdminUser } from "@/lib/admin-api";

interface UserTableProps {
  users: AdminUser[];
  loading: boolean;
  onEdit: (user: AdminUser) => void;
  onDelete: (userId: number) => void;
  onBulkAction: (action: string, selectedIds: number[]) => void;
}

type SortKey = keyof AdminUser;
type SortOrder = "asc" | "desc";

export default function UserTable({
  users,
  loading,
  onEdit,
  onDelete,
  onBulkAction,
}: UserTableProps) {
  const [selectedUsers, setSelectedUsers] = useState<Set<number>>(new Set());
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [searchTerm, setSearchTerm] = useState("");
  const [roleFilter, setRoleFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);

  // Get unique roles for filter dropdown
  const uniqueRoles = useMemo(() => {
    return Array.from(new Set(users.map((user) => user.role_name))).sort();
  }, [users]);

  // Filter and sort users
  const filteredAndSortedUsers = useMemo(() => {
    let filtered = users.filter((user) => {
      const matchesSearch =
        user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
        user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
        `${user.first_name} ${user.last_name}`
          .toLowerCase()
          .includes(searchTerm.toLowerCase());

      const matchesRole = !roleFilter || user.role_name === roleFilter;
      const matchesStatus =
        !statusFilter ||
        (statusFilter === "active" && user.is_active) ||
        (statusFilter === "inactive" && !user.is_active);

      return matchesSearch && matchesRole && matchesStatus;
    });

    // Sort
    filtered.sort((a, b) => {
      const aValue = a[sortKey];
      const bValue = b[sortKey];

      // Handle null values
      if (aValue === null || aValue === undefined) {
        if (bValue === null || bValue === undefined) return 0;
        return sortOrder === "asc" ? 1 : -1;
      }
      if (bValue === null || bValue === undefined) {
        return sortOrder === "asc" ? -1 : 1;
      }

      if (aValue < bValue) return sortOrder === "asc" ? -1 : 1;
      if (aValue > bValue) return sortOrder === "asc" ? 1 : -1;
      return 0;
    });

    return filtered;
  }, [users, searchTerm, roleFilter, statusFilter, sortKey, sortOrder]);

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedUsers.length / itemsPerPage);
  const paginatedUsers = filteredAndSortedUsers.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortOrder("asc");
    }
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      // Exclude admin user from bulk selection
      const selectableUsers = paginatedUsers.filter(
        (user) => user.username !== "admin"
      );
      setSelectedUsers(new Set(selectableUsers.map((user) => user.id)));
    } else {
      setSelectedUsers(new Set());
    }
  };

  const handleSelectUser = (userId: number, checked: boolean) => {
    const user = users.find((u) => u.id === userId);
    // Prevent selecting admin user for bulk operations
    if (user?.username === "admin" && checked) {
      return;
    }

    const newSelected = new Set(selectedUsers);
    if (checked) {
      newSelected.add(userId);
    } else {
      newSelected.delete(userId);
    }
    setSelectedUsers(newSelected);
  };

  const handleBulkAction = (action: string) => {
    if (selectedUsers.size > 0) {
      onBulkAction(action, Array.from(selectedUsers));
      setSelectedUsers(new Set());
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "Hiçbir zaman";
    return new Date(dateString).toLocaleDateString("tr-TR", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getSortIcon = (key: SortKey) => {
    if (sortKey !== key) {
      return (
        <svg
          className="w-4 h-4 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"
          />
        </svg>
      );
    }

    return sortOrder === "asc" ? (
      <svg
        className="w-4 h-4 text-indigo-600"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M7 16V4m0 0L3 8m4-4l4 4"
        />
      </svg>
    ) : (
      <svg
        className="w-4 h-4 text-indigo-600"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M17 8l4 4m0 0l-4 4m4-4H3"
        />
      </svg>
    );
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded mb-4"></div>
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div
                key={i}
                className="h-12 bg-gray-200 dark:bg-gray-700 rounded"
              ></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden w-full">
      {/* Filters and Search */}
      <div className="p-3 border-b border-gray-200 dark:border-gray-700 overflow-x-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div>
            <input
              type="text"
              placeholder="Kullanıcı ara..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            />
          </div>

          <div>
            <select
              value={roleFilter}
              onChange={(e) => setRoleFilter(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="">Tüm Roller</option>
              {uniqueRoles.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
          </div>

          <div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="">Tüm Durum</option>
              <option value="active">Aktif</option>
              <option value="inactive">İnaktif</option>
            </select>
          </div>

          {/* Bulk Actions */}
          {selectedUsers.size > 0 && (
            <div className="flex space-x-2">
              <button
                onClick={() => handleBulkAction("activate")}
                className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-md transition-colors"
              >
                Aktifleştir ({selectedUsers.size})
              </button>
              <button
                onClick={() => handleBulkAction("deactivate")}
                className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-md transition-colors"
              >
                Pasifleştir ({selectedUsers.size})
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto w-full">
        <table className="w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              <th className="px-3 py-2 text-left">
                <input
                  type="checkbox"
                  checked={
                    selectedUsers.size === paginatedUsers.length &&
                    paginatedUsers.length > 0
                  }
                  onChange={(e) => handleSelectAll(e.target.checked)}
                  className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                />
              </th>
              <th
                className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("username")}
              >
                <div className="flex items-center space-x-1">
                  <span>Kullanıcı Adı</span>
                  {getSortIcon("username")}
                </div>
              </th>
              <th
                className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("email")}
              >
                <div className="flex items-center space-x-1">
                  <span>Email</span>
                  {getSortIcon("email")}
                </div>
              </th>
              <th
                className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("role_name")}
              >
                <div className="flex items-center space-x-1">
                  <span>Rol</span>
                  {getSortIcon("role_name")}
                </div>
              </th>
              <th
                className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("is_active")}
              >
                <div className="flex items-center space-x-1">
                  <span>Durum</span>
                  {getSortIcon("is_active")}
                </div>
              </th>
              <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                İşlemler
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {paginatedUsers.map((user) => (
              <tr
                key={user.id}
                className="hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <td className="px-3 py-2">
                  {/* Disable checkbox for admin user */}
                  {user.username === "admin" ? (
                    <input
                      type="checkbox"
                      disabled
                      className="rounded border-gray-300 text-gray-400 cursor-not-allowed"
                      title="Sistem administratörü toplu işlemlere dahil edilemez"
                    />
                  ) : (
                    <input
                      type="checkbox"
                      checked={selectedUsers.has(user.id)}
                      onChange={(e) =>
                        handleSelectUser(user.id, e.target.checked)
                      }
                      className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                    />
                  )}
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="flex-shrink-0 h-8 w-8">
                      <div className="h-8 w-8 rounded-full bg-indigo-100 dark:bg-indigo-900 flex items-center justify-center">
                        <span className="text-sm font-medium text-indigo-600 dark:text-indigo-400">
                          {user.first_name.charAt(0)}
                          {user.last_name.charAt(0)}
                        </span>
                      </div>
                    </div>
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {user.username}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {user.first_name} {user.last_name}
                      </div>
                      <div className="text-xs text-gray-400 dark:text-gray-500 flex items-center gap-1 mt-0.5">
                        <svg
                          className="w-3 h-3"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z"
                            clipRule="evenodd"
                          />
                        </svg>
                        {formatDate(user.last_login)}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <div className="text-sm text-gray-900 dark:text-white">
                    {user.email}
                  </div>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <span
                    className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      user.role_name === "admin"
                        ? "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
                        : user.role_name === "teacher"
                        ? "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                        : "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                    }`}
                  >
                    {user.role_name}
                  </span>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <span
                    className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      user.is_active
                        ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                        : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                    }`}
                  >
                    {user.is_active ? "Aktif" : "İnaktif"}
                  </span>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <div className="flex justify-center items-center gap-2">
                    <button
                      onClick={() => onEdit(user)}
                      className="p-1.5 text-indigo-600 hover:text-indigo-900 dark:text-indigo-400 dark:hover:text-indigo-300 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 rounded transition-colors"
                      title="Düzenle"
                    >
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                        />
                      </svg>
                    </button>
                    {/* Check if this is the original admin user - protect from deletion */}
                    {user.username === "admin" ? (
                      <button
                        className="p-1.5 text-gray-400 cursor-not-allowed rounded"
                        title="Sistem administratörü silinemez"
                        disabled
                      >
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                          />
                        </svg>
                      </button>
                    ) : (
                      <button
                        onClick={() => onDelete(user.id)}
                        className="p-1.5 text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                        title="Sil"
                      >
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                          />
                        </svg>
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="bg-white dark:bg-gray-800 px-3 py-2 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                {(currentPage - 1) * itemsPerPage + 1} -{" "}
                {Math.min(
                  currentPage * itemsPerPage,
                  filteredAndSortedUsers.length
                )}{" "}
                / {filteredAndSortedUsers.length} sonuç gösteriliyor
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Önceki
              </button>
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Sayfa {currentPage} / {totalPages}
              </span>
              <button
                onClick={() =>
                  setCurrentPage(Math.min(totalPages, currentPage + 1))
                }
                disabled={currentPage === totalPages}
                className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Sonraki
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
