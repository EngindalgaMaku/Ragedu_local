"use client";

import React, { useState, useMemo } from "react";
import Link from "next/link";

interface Session {
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
}

interface SessionTableProps {
  sessions: Session[];
  loading: boolean;
  selectedSessions: Set<number>;
  onSelectionChange: (selected: Set<number>) => void;
  onTerminateSession: (sessionId: number) => void;
  onTerminateUserSessions: (userId: number) => void;
}

type SortKey =
  | "username"
  | "role_name"
  | "created_at"
  | "expires_at"
  | "last_activity";
type SortOrder = "asc" | "desc";

export default function SessionTable({
  sessions,
  loading,
  selectedSessions,
  onSelectionChange,
  onTerminateSession,
  onTerminateUserSessions,
}: SessionTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<
    "all" | "active" | "expired"
  >("all");
  const [roleFilter, setRoleFilter] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);

  const getBrowserName = (userAgent: string) => {
    if (userAgent.includes("Chrome")) return "Chrome";
    if (userAgent.includes("Firefox")) return "Firefox";
    if (userAgent.includes("Safari")) return "Safari";
    if (userAgent.includes("Edge")) return "Edge";
    return "Unknown";
  };

  const getOSName = (userAgent: string) => {
    if (userAgent.includes("Windows NT 10.0")) return "Windows 10";
    if (userAgent.includes("Windows NT")) return "Windows";
    if (userAgent.includes("Mac OS X")) return "macOS";
    if (userAgent.includes("iPhone")) return "iOS";
    if (userAgent.includes("Android")) return "Android";
    if (userAgent.includes("Linux")) return "Linux";
    return "Unknown";
  };

  const isSessionActive = (session: Session) => {
    return new Date(session.expires_at) > new Date();
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getRelativeTime = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  // Get unique roles for filter dropdown
  const uniqueRoles = useMemo(() => {
    return Array.from(
      new Set(sessions.map((session) => session.role_name))
    ).sort();
  }, [sessions]);

  // Filter and sort sessions
  const filteredAndSortedSessions = useMemo(() => {
    let filtered = sessions.filter((session) => {
      const matchesSearch =
        session.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
        session.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
        `${session.first_name} ${session.last_name}`
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        session.ip_address.includes(searchTerm);

      const matchesRole = !roleFilter || session.role_name === roleFilter;

      let matchesStatus = true;
      if (statusFilter === "active") {
        matchesStatus = isSessionActive(session);
      } else if (statusFilter === "expired") {
        matchesStatus = !isSessionActive(session);
      }

      return matchesSearch && matchesRole && matchesStatus;
    });

    // Sort
    filtered.sort((a, b) => {
      let aValue: any = a[sortKey];
      let bValue: any = b[sortKey];

      if (sortKey.includes("_at")) {
        aValue = new Date(aValue).getTime();
        bValue = new Date(bValue).getTime();
      }

      if (aValue < bValue) return sortOrder === "asc" ? -1 : 1;
      if (aValue > bValue) return sortOrder === "asc" ? 1 : -1;
      return 0;
    });

    return filtered;
  }, [sessions, searchTerm, roleFilter, statusFilter, sortKey, sortOrder]);

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
      onSelectionChange(
        new Set(paginatedSessions.map((session) => session.id))
      );
    } else {
      onSelectionChange(new Set());
    }
  };

  const handleSelectSession = (sessionId: number, checked: boolean) => {
    const newSelected = new Set(selectedSessions);
    if (checked) {
      newSelected.add(sessionId);
    } else {
      newSelected.delete(sessionId);
    }
    onSelectionChange(newSelected);
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
                className="h-16 bg-gray-200 dark:bg-gray-700 rounded"
              ></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden">
      {/* Filters and Search */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <input
              type="text"
              placeholder="Search sessions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            />
          </div>

          <div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="all">All Sessions</option>
              <option value="active">Active Only</option>
              <option value="expired">Expired Only</option>
            </select>
          </div>

          <div>
            <select
              value={roleFilter}
              onChange={(e) => setRoleFilter(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="">All Roles</option>
              {uniqueRoles.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
          </div>

          <div className="flex justify-end">
            <span className="text-sm text-gray-500 dark:text-gray-400 py-2">
              {filteredAndSortedSessions.length} sessions
            </span>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              <th className="px-6 py-3 text-left">
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
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("username")}
              >
                <div className="flex items-center space-x-1">
                  <span>User</span>
                  {getSortIcon("username")}
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("role_name")}
              >
                <div className="flex items-center space-x-1">
                  <span>Role</span>
                  {getSortIcon("role_name")}
                </div>
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Device & Location
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("created_at")}
              >
                <div className="flex items-center space-x-1">
                  <span>Started</span>
                  {getSortIcon("created_at")}
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => handleSort("last_activity")}
              >
                <div className="flex items-center space-x-1">
                  <span>Last Activity</span>
                  {getSortIcon("last_activity")}
                </div>
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {paginatedSessions.map((session) => {
              const isActive = isSessionActive(session);

              return (
                <tr
                  key={session.id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700"
                >
                  <td className="px-6 py-4">
                    <input
                      type="checkbox"
                      checked={selectedSessions.has(session.id)}
                      onChange={(e) =>
                        handleSelectSession(session.id, e.target.checked)
                      }
                      className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                    />
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-8 w-8">
                        <div className="h-8 w-8 rounded-full bg-indigo-100 dark:bg-indigo-900 flex items-center justify-center">
                          <span className="text-sm font-medium text-indigo-600 dark:text-indigo-400">
                            {session.first_name.charAt(0)}
                            {session.last_name.charAt(0)}
                          </span>
                        </div>
                      </div>
                      <div className="ml-4">
                        <Link
                          href={`/admin/users/${session.user_id}`}
                          className="text-sm font-medium text-indigo-600 hover:text-indigo-900 dark:text-indigo-400 dark:hover:text-indigo-300"
                        >
                          {session.first_name} {session.last_name}
                        </Link>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                          @{session.username}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        session.role_name === "admin"
                          ? "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
                          : session.role_name === "teacher"
                          ? "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                          : "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                      }`}
                    >
                      {session.role_name}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-900 dark:text-white">
                      {getBrowserName(session.user_agent)} on{" "}
                      {getOSName(session.user_agent)}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {session.ip_address}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    <div>{formatDate(session.created_at)}</div>
                    <div className="text-xs">
                      {getRelativeTime(session.created_at)}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    <div>{formatDate(session.last_activity)}</div>
                    <div className="text-xs">
                      {getRelativeTime(session.last_activity)}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div
                        className={`w-2 h-2 rounded-full mr-2 ${
                          isActive ? "bg-green-500" : "bg-red-500"
                        }`}
                      ></div>
                      <span
                        className={`text-sm font-medium ${
                          isActive
                            ? "text-green-600 dark:text-green-400"
                            : "text-red-600 dark:text-red-400"
                        }`}
                      >
                        {isActive ? "Active" : "Expired"}
                      </span>
                    </div>
                    {!isActive && (
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        Expired {getRelativeTime(session.expires_at)}
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex justify-end space-x-2">
                      <button
                        onClick={() => onTerminateSession(session.id)}
                        className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                        disabled={!isActive}
                      >
                        {isActive ? "Terminate" : "Remove"}
                      </button>
                      <button
                        onClick={() => onTerminateUserSessions(session.user_id)}
                        className="text-orange-600 hover:text-orange-900 dark:text-orange-400 dark:hover:text-orange-300"
                      >
                        Terminate All
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="bg-white dark:bg-gray-800 px-4 py-3 border-t border-gray-200 dark:border-gray-700 sm:px-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                Showing {(currentPage - 1) * itemsPerPage + 1} to{" "}
                {Math.min(
                  currentPage * itemsPerPage,
                  filteredAndSortedSessions.length
                )}{" "}
                of {filteredAndSortedSessions.length} results
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Previous
              </button>
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() =>
                  setCurrentPage(Math.min(totalPages, currentPage + 1))
                }
                disabled={currentPage === totalPages}
                className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
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
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
            />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
            No sessions found
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {searchTerm || roleFilter || statusFilter !== "all"
              ? "Try adjusting your search criteria."
              : "No active sessions at the moment."}
          </p>
        </div>
      )}
    </div>
  );
}
