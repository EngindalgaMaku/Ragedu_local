"use client";

import React, { useState, useMemo } from "react";
import { LearningSession, AdminUser } from "@/lib/admin-api";

interface SessionsTableProps {
  sessions: LearningSession[];
  users: AdminUser[];
  loading: boolean;
  onView: (session: LearningSession) => void;
  onDelete: (sessionId: string) => void;
  onToggleStatus: (sessionId: string, currentStatus: string) => void;
  onBulkAction: (action: string, selectedIds: string[]) => void;
}

type SortKey = keyof LearningSession;
type SortOrder = "asc" | "desc";

export default function SessionsTable({
  sessions,
  users,
  loading,
  onView,
  onDelete,
  onToggleStatus,
  onBulkAction,
}: SessionsTableProps) {
  // Create user map for quick lookup
  const userMap = useMemo(() => {
    const map = new Map<number | string, AdminUser>();
    users.forEach((user) => {
      map.set(user.id.toString(), user);
      map.set(user.id, user);
    });
    return map;
  }, [users]);

  // Get teacher name from created_by (which might be user_id or username)
  const getTeacherName = (createdBy: string): string => {
    // Try to find by user_id first
    const userById = userMap.get(createdBy);
    if (userById) {
      return `${userById.first_name} ${userById.last_name}`;
    }
    // Try to find by username
    const userByUsername = users.find((u) => u.username === createdBy);
    if (userByUsername) {
      return `${userByUsername.first_name} ${userByUsername.last_name}`;
    }
    // If not found, return as is (might be username already)
    return createdBy;
  };
  const [selectedSessions, setSelectedSessions] = useState<Set<string>>(
    new Set()
  );
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [searchTerm, setSearchTerm] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [teacherFilter, setTeacherFilter] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const [backingUp, setBackingUp] = useState<string | null>(null);
  const [restoreModalOpen, setRestoreModalOpen] = useState(false);
  const [restoreFile, setRestoreFile] = useState<File | null>(null);
  const [restoring, setRestoring] = useState(false);

  // Get unique values for filter dropdowns
  const uniqueCategories = useMemo(() => {
    return Array.from(
      new Set(sessions.map((session) => session.category).filter(Boolean))
    ).sort();
  }, [sessions]);

  const uniqueTeachers = useMemo(() => {
    return Array.from(
      new Set(
        sessions
          .map((session) => getTeacherName(session.created_by))
          .filter(Boolean)
      )
    ).sort();
  }, [sessions, userMap]);

  // Filter and sort sessions
  const filteredAndSortedSessions = useMemo(() => {
    let filtered = sessions.filter((session) => {
      const matchesSearch =
        session.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        session.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        session.created_by.toLowerCase().includes(searchTerm.toLowerCase()) ||
        session.subject_area.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesCategory =
        !categoryFilter || session.category === categoryFilter;
      const matchesStatus = !statusFilter || session.status === statusFilter;
      const matchesTeacher =
        !teacherFilter || getTeacherName(session.created_by) === teacherFilter;

      return (
        matchesSearch && matchesCategory && matchesStatus && matchesTeacher
      );
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

      if (typeof aValue === "string" && typeof bValue === "string") {
        return sortOrder === "asc"
          ? aValue.localeCompare(bValue, "tr", { sensitivity: "base" })
          : bValue.localeCompare(aValue, "tr", { sensitivity: "base" });
      }

      if (aValue < bValue) return sortOrder === "asc" ? -1 : 1;
      if (aValue > bValue) return sortOrder === "asc" ? 1 : -1;
      return 0;
    });

    return filtered;
  }, [
    sessions,
    searchTerm,
    categoryFilter,
    statusFilter,
    teacherFilter,
    sortKey,
    sortOrder,
  ]);

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedSessions.length / itemsPerPage);
  const paginatedSessions = filteredAndSortedSessions.slice(
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
      setSelectedSessions(
        new Set(paginatedSessions.map((session) => session.session_id))
      );
    } else {
      setSelectedSessions(new Set());
    }
  };

  const handleSelectSession = (sessionId: string, checked: boolean) => {
    const newSelected = new Set(selectedSessions);
    if (checked) {
      newSelected.add(sessionId);
    } else {
      newSelected.delete(sessionId);
    }
    setSelectedSessions(newSelected);
  };

  const handleBulkAction = (action: string) => {
    if (selectedSessions.size > 0) {
      onBulkAction(action, Array.from(selectedSessions));
      setSelectedSessions(new Set());
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "Bilinmiyor";
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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          <div>
            <input
              type="text"
              placeholder="Ders oturumu ara..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            />
          </div>

          <div>
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="">Tüm Kategoriler</option>
              {uniqueCategories.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </div>

          <div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="">Tüm Durum</option>
              <option value="active">Aktif</option>
              <option value="inactive">İnaktif</option>
            </select>
          </div>

          <div>
            <select
              value={teacherFilter}
              onChange={(e) => setTeacherFilter(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="">Tüm Öğretmenler</option>
              {uniqueTeachers.map((teacher) => (
                <option key={teacher} value={teacher}>
                  {teacher}
                </option>
              ))}
            </select>
          </div>

          {/* Bulk Actions */}
          <div className="flex space-x-2">
            {selectedSessions.size > 0 && (
              <>
                <button
                  onClick={() => handleBulkAction("activate")}
                  className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-md transition-colors"
                >
                  Aktifleştir ({selectedSessions.size})
                </button>
                <button
                  onClick={() => handleBulkAction("deactivate")}
                  className="px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-white text-sm rounded-md transition-colors"
                >
                  Pasifleştir ({selectedSessions.size})
                </button>
                <button
                  onClick={() => handleBulkAction("delete")}
                  className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-md transition-colors"
                >
                  Sil ({selectedSessions.size})
                </button>
              </>
            )}
            <button
              onClick={() => setRestoreModalOpen(true)}
              className="px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-md transition-colors"
              title="Yedekten Geri Yükle"
            >
              📥 Yedekten Geri Yükle
            </button>
          </div>
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
                    selectedSessions.size === paginatedSessions.length &&
                    paginatedSessions.length > 0
                  }
                  onChange={(e) => handleSelectAll(e.target.checked)}
                  className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                />
              </th>
              <th
                className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("name")}
              >
                <div className="flex items-center space-x-1">
                  <span>Oturum Adı</span>
                  {getSortIcon("name")}
                </div>
              </th>
              <th
                className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("created_by")}
              >
                <div className="flex items-center space-x-1">
                  <span>Öğretmen</span>
                  {getSortIcon("created_by")}
                </div>
              </th>
              <th
                className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("category")}
              >
                <div className="flex items-center space-x-1">
                  <span>Kategori</span>
                  {getSortIcon("category")}
                </div>
              </th>
              <th
                className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("status")}
              >
                <div className="flex items-center space-x-1">
                  <span>Durum</span>
                  {getSortIcon("status")}
                </div>
              </th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                İstatistikler
              </th>
              <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                İşlemler
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {paginatedSessions.map((session) => (
              <tr
                key={session.session_id}
                className="hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <td className="px-3 py-2">
                  <input
                    type="checkbox"
                    checked={selectedSessions.has(session.session_id)}
                    onChange={(e) =>
                      handleSelectSession(session.session_id, e.target.checked)
                    }
                    className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="flex-shrink-0 h-8 w-8">
                      <div className="h-8 w-8 rounded-full bg-indigo-100 dark:bg-indigo-900 flex items-center justify-center">
                        <svg
                          className="w-4 h-4 text-indigo-600 dark:text-indigo-400"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                          />
                        </svg>
                      </div>
                    </div>
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {session.name}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400 truncate max-w-xs">
                        {session.description}
                      </div>
                      <div className="text-xs text-gray-400 dark:text-gray-500 flex items-center gap-1 mt-0.5">
                        <span>
                          {session.grade_level} • {session.subject_area}
                        </span>
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <div className="text-sm text-gray-900 dark:text-white font-medium">
                    {getTeacherName(session.created_by)}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {formatDate(session.created_at)}
                  </div>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                    {session.category}
                  </span>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <span
                    className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      session.status === "active"
                        ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                        : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                    }`}
                  >
                    {session.status === "active" ? "Aktif" : "İnaktif"}
                  </span>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                    <div className="flex items-center gap-1">
                      <svg
                        className="w-3 h-3"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <span>{session.document_count} Belge</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <svg
                        className="w-3 h-3"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3z" />
                      </svg>
                      <span>{session.student_entry_count} Öğrenci</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <svg
                        className="w-3 h-3"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span>{session.query_count} Soru</span>
                    </div>
                  </div>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <div className="flex justify-center items-center gap-2">
                    <button
                      onClick={() => onView(session)}
                      className="p-1.5 text-indigo-600 hover:text-indigo-900 dark:text-indigo-400 dark:hover:text-indigo-300 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 rounded transition-colors"
                      title="Detayları Görüntüle"
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
                          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                        />
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                        />
                      </svg>
                    </button>
                    <button
                      onClick={() =>
                        onToggleStatus(session.session_id, session.status)
                      }
                      className={`p-1.5 ${
                        session.status === "active"
                          ? "text-yellow-600 hover:text-yellow-900 dark:text-yellow-400 dark:hover:text-yellow-300 hover:bg-yellow-50 dark:hover:bg-yellow-900/20"
                          : "text-green-600 hover:text-green-900 dark:text-green-400 dark:hover:text-green-300 hover:bg-green-50 dark:hover:bg-green-900/20"
                      } rounded transition-colors`}
                      title={
                        session.status === "active"
                          ? "Pasifleştir"
                          : "Aktifleştir"
                      }
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
                          d={
                            session.status === "active"
                              ? "M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"
                              : "M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h8m2 4H7a2 2 0 01-2-2V7a2 2 0 012-2h10a2 2 0 012 2v11a2 2 0 01-2 2z"
                          }
                        />
                      </svg>
                    </button>
                    <button
                      onClick={async () => {
                        setBackingUp(session.session_id);
                        try {
                          const response = await fetch(
                            `/api/sessions/${session.session_id}/backup/download`
                          );
                          if (response.ok) {
                            const blob = await response.blob();
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `session_backup_${session.session_id}_${Date.now()}.json`;
                            document.body.appendChild(a);
                            a.click();
                            window.URL.revokeObjectURL(url);
                            document.body.removeChild(a);
                            alert("Yedek başarıyla indirildi!");
                          } else {
                            alert("Yedek indirme başarısız!");
                          }
                        } catch (error) {
                          console.error("Backup error:", error);
                          alert("Yedek indirme hatası!");
                        } finally {
                          setBackingUp(null);
                        }
                      }}
                      disabled={backingUp === session.session_id}
                      className="p-1.5 text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-colors disabled:opacity-50"
                      title="Yedek Al"
                    >
                      {backingUp === session.session_id ? (
                        <svg
                          className="w-4 h-4 animate-spin"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                      ) : (
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
                            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                          />
                        </svg>
                      )}
                    </button>
                    <button
                      onClick={() => onDelete(session.session_id)}
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
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* No Data State */}
      {filteredAndSortedSessions.length === 0 && !loading && (
        <div className="text-center py-12">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
            />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
            Ders oturumu bulunamadı
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {searchTerm || categoryFilter || statusFilter || teacherFilter
              ? "Arama kriterlerinize uygun ders oturumu bulunamadı."
              : "Henüz hiç ders oturumu oluşturulmamış."}
          </p>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="bg-white dark:bg-gray-800 px-3 py-2 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                {(currentPage - 1) * itemsPerPage + 1} -{" "}
                {Math.min(
                  currentPage * itemsPerPage,
                  filteredAndSortedSessions.length
                )}{" "}
                / {filteredAndSortedSessions.length} sonuç gösteriliyor
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

      {/* Restore Modal */}
      {restoreModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Yedekten Geri Yükle
              </h3>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Yedek Dosyası Seç (JSON)
                </label>
                <input
                  type="file"
                  accept=".json"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setRestoreFile(e.target.files[0]);
                    }
                  }}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Yeni Oturum ID (Boş bırakırsanız orijinal ID kullanılır)
                </label>
                <input
                  type="text"
                  id="newSessionId"
                  placeholder="Orijinal session_id kullanılacak"
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => {
                    setRestoreModalOpen(false);
                    setRestoreFile(null);
                  }}
                  className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                >
                  İptal
                </button>
                <button
                  onClick={async () => {
                    if (!restoreFile) {
                      alert("Lütfen bir yedek dosyası seçin!");
                      return;
                    }
                    setRestoring(true);
                    try {
                      const text = await restoreFile.text();
                      const backupData = JSON.parse(text);
                      const newSessionId = (
                        document.getElementById("newSessionId") as HTMLInputElement
                      )?.value.trim() || null;
                      
                      const response = await fetch("/api/sessions/restore", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          backup_data: backupData,
                          new_session_id: newSessionId,
                          restore_chunks: true,
                          restore_topics: true,
                          restore_kb: true,
                          restore_qa: true,
                        }),
                      });
                      
                      if (response.ok) {
                        const result = await response.json();
                        alert(
                          `Geri yükleme başarılı!\n${result.message}\nSayfayı yenilemeniz gerekebilir.`
                        );
                        setRestoreModalOpen(false);
                        setRestoreFile(null);
                        window.location.reload();
                      } else {
                        const error = await response.json();
                        alert(`Geri yükleme başarısız: ${error.detail || "Bilinmeyen hata"}`);
                      }
                    } catch (error) {
                      console.error("Restore error:", error);
                      alert("Geri yükleme hatası! Dosya formatını kontrol edin.");
                    } finally {
                      setRestoring(false);
                    }
                  }}
                  disabled={!restoreFile || restoring}
                  className="px-4 py-2 text-sm font-medium text-white bg-purple-600 rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {restoring ? "Yükleniyor..." : "Geri Yükle"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
