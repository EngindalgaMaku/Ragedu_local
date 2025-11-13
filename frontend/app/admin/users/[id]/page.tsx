"use client";

import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import ModernAdminLayout from "../../components/ModernAdminLayout";
import Link from "next/link";
import { AdminUser, AdminSession } from "@/lib/admin-api";
import adminApi from "@/lib/admin-api";

export default function UserDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [user, setUser] = useState<AdminUser | null>(null);
  const [sessions, setSessions] = useState<AdminSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<
    "profile" | "sessions" | "activity"
  >("profile");

  const userId = params?.id as string;

  useEffect(() => {
    const fetchUserDetails = async () => {
      if (!userId) return;

      setLoading(true);
      setError(null);
      try {
        // Fetch all users and find the specific one
        const allUsers = await adminApi.getUsers();
        const userData = allUsers.find((user) => user.id === parseInt(userId));

        if (!userData) {
          throw new Error("User not found");
        }

        setUser(userData);

        // Fetch user sessions
        const userSessions = await adminApi.getSessions();
        // Filter sessions for this specific user
        const filteredSessions = userSessions.filter(
          (session) => session.user_id === parseInt(userId)
        );
        setSessions(filteredSessions);
      } catch (error) {
        console.error("Failed to fetch user details:", error);
        setError("Failed to load user details. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchUserDetails();
  }, [userId]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const handleTerminateSession = async (sessionId: number) => {
    if (confirm("Are you sure you want to terminate this session?")) {
      try {
        await adminApi.terminateSession(sessionId);
        setSessions(sessions.filter((session) => session.id !== sessionId));
      } catch (error) {
        console.error("Failed to terminate session:", error);
        alert("Failed to terminate session. Please try again.");
      }
    }
  };

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
    if (userAgent.includes("Linux")) return "Linux";
    return "Unknown";
  };

  if (loading) {
    return (
      <ModernAdminLayout
        title="User Details"
        description="Loading user information..."
      >
        <div className="animate-pulse">
          <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-48 mb-4"></div>
            <div className="space-y-3">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
            </div>
          </div>
        </div>
      </ModernAdminLayout>
    );
  }

  if (error || (!loading && !user)) {
    return (
      <ModernAdminLayout
        title="User Not Found"
        description="The requested user could not be found"
      >
        <div className="text-center">
          <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-100 dark:bg-red-900">
            <svg
              className="h-6 w-6 text-red-600 dark:text-red-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 8.5c-.77.833-.192 2.5 1.732 2.5z"
              />
            </svg>
          </div>
          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
            {error ? "Error Loading User" : "User not found"}
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {error || `The user with ID ${userId} could not be found.`}
          </p>
          <div className="mt-6">
            <Link
              href="/admin/users"
              className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Back to Users
            </Link>
          </div>
        </div>
      </ModernAdminLayout>
    );
  }

  return (
    <ModernAdminLayout
      title={user ? `${user.first_name} ${user.last_name}` : "User Details"}
      description={
        user ? `User details for ${user.username}` : "Loading user details"
      }
    >
      {!user ? (
        <div className="animate-pulse">
          <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-48 mb-4"></div>
            <div className="space-y-3">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Header */}
          <div className="bg-white dark:bg-gray-800 shadow rounded-lg">
            <div className="px-4 py-5 sm:px-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="h-12 w-12 rounded-full bg-indigo-100 dark:bg-indigo-900 flex items-center justify-center">
                      <span className="text-lg font-medium text-indigo-600 dark:text-indigo-400">
                        {user.first_name.charAt(0)}
                        {user.last_name.charAt(0)}
                      </span>
                    </div>
                  </div>
                  <div className="ml-4">
                    <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                      {user.first_name} {user.last_name}
                    </h1>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      @{user.username} • {user.email}
                    </p>
                    <div className="flex items-center mt-1 space-x-4">
                      <span
                        className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          user.is_active
                            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                            : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                        }`}
                      >
                        {user.is_active ? "Active" : "Inactive"}
                      </span>
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
                    </div>
                  </div>
                </div>
                <div className="flex space-x-3">
                  <Link
                    href={`/admin/users?action=edit&id=${user.id}`}
                    className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Edit User
                  </Link>
                  <Link
                    href="/admin/users"
                    className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Back to Users
                  </Link>
                </div>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="bg-white dark:bg-gray-800 shadow rounded-lg">
            <div className="border-b border-gray-200 dark:border-gray-700">
              <nav className="-mb-px flex space-x-8 px-6">
                {[
                  { key: "profile", label: "Profile" },
                  { key: "sessions", label: "Sessions" },
                  { key: "activity", label: "Activity" },
                ].map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setActiveTab(tab.key as any)}
                    className={`py-4 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab.key
                        ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                        : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>

            <div className="p-6">
              {activeTab === "profile" && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                        Basic Information
                      </h3>
                      <dl className="space-y-3">
                        <div>
                          <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            Username
                          </dt>
                          <dd className="text-sm text-gray-900 dark:text-white">
                            {user.username}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            Email
                          </dt>
                          <dd className="text-sm text-gray-900 dark:text-white">
                            {user.email}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            Full Name
                          </dt>
                          <dd className="text-sm text-gray-900 dark:text-white">
                            {user.first_name} {user.last_name}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            Role
                          </dt>
                          <dd className="text-sm text-gray-900 dark:text-white capitalize">
                            {user.role_name}
                          </dd>
                        </div>
                      </dl>
                    </div>

                    <div>
                      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                        Account Details
                      </h3>
                      <dl className="space-y-3">
                        <div>
                          <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            Status
                          </dt>
                          <dd className="text-sm text-gray-900 dark:text-white">
                            {user.is_active ? "Active" : "Inactive"}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            Created
                          </dt>
                          <dd className="text-sm text-gray-900 dark:text-white">
                            {formatDate(user.created_at)}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            Last Updated
                          </dt>
                          <dd className="text-sm text-gray-900 dark:text-white">
                            {formatDate(user.created_at)}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            Last Login
                          </dt>
                          <dd className="text-sm text-gray-900 dark:text-white">
                            {user.last_login
                              ? formatDate(user.last_login)
                              : "Never"}
                          </dd>
                        </div>
                      </dl>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === "sessions" && (
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                    User Sessions ({sessions.length})
                  </h3>
                  <div className="space-y-4">
                    {sessions.length === 0 ? (
                      <p className="text-gray-500 dark:text-gray-400">
                        No active sessions found.
                      </p>
                    ) : (
                      sessions.map((session) => (
                        <div
                          key={session.id}
                          className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-4">
                                <div className="flex-shrink-0">
                                  <div className="w-10 h-10 bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center">
                                    <svg
                                      className="w-5 h-5 text-gray-600 dark:text-gray-400"
                                      fill="none"
                                      stroke="currentColor"
                                      viewBox="0 0 24 24"
                                    >
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                                      />
                                    </svg>
                                  </div>
                                </div>
                                <div>
                                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                                    {getBrowserName(session.user_agent)} on{" "}
                                    {getOSName(session.user_agent)}
                                  </div>
                                  <div className="text-sm text-gray-500 dark:text-gray-400">
                                    {session.ip_address} • Started{" "}
                                    {formatDate(session.created_at)}
                                  </div>
                                  <div className="text-xs text-gray-500 dark:text-gray-400">
                                    Expires: {formatDate(session.expires_at)}
                                  </div>
                                </div>
                              </div>
                            </div>
                            <button
                              onClick={() => handleTerminateSession(session.id)}
                              className="ml-4 text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300 text-sm font-medium"
                            >
                              Terminate
                            </button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}

              {activeTab === "activity" && (
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                    Recent Activity
                  </h3>
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
                        d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                      No activity data
                    </h3>
                    <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                      Activity logging is not yet implemented.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </ModernAdminLayout>
  );
}
