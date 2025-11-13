"use client";

import React, { useState, useEffect } from "react";
import Widget from "./Widget";
import { adminApiClient, ActivityLog } from "@/lib/admin-api";

interface Activity {
  id: number;
  type:
    | "user_created"
    | "user_login"
    | "user_logout"
    | "role_assigned"
    | "session_expired";
  message: string;
  user: string;
  timestamp: string;
  icon: React.ReactNode;
  color: string;
}

const getActivityIcon = (type: ActivityLog["type"]) => {
  switch (type) {
    case "user_created":
      return (
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
            d="M12 6v6m0 0v6m0-6h6m-6 0H6"
          />
        </svg>
      );
    case "user_login":
      return (
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
            d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"
          />
        </svg>
      );
    case "user_logout":
      return (
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
            d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
          />
        </svg>
      );
    case "role_assigned":
      return (
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
            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
          />
        </svg>
      );
    case "session_expired":
      return (
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
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      );
    default:
      return (
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
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      );
  }
};

const getActivityColor = (type: ActivityLog["type"]) => {
  switch (type) {
    case "user_created":
      return "bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-400";
    case "user_login":
      return "bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400";
    case "user_logout":
      return "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400";
    case "role_assigned":
      return "bg-purple-100 text-purple-600 dark:bg-purple-900 dark:text-purple-400";
    case "session_expired":
      return "bg-yellow-100 text-yellow-600 dark:bg-yellow-900 dark:text-yellow-400";
    default:
      return "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400";
  }
};

const formatRelativeTime = (timestamp: string) => {
  const now = new Date();
  const date = new Date(timestamp);
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return "Az önce";
  if (diffMins < 60) return `${diffMins} dakika önce`;
  if (diffHours < 24) return `${diffHours} saat önce`;
  return `${diffDays} gün önce`;
};

export default function ActivityFeedWidget() {
  const [activities, setActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchActivities = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await adminApiClient.getActivityLogs(10);
      const formattedActivities: Activity[] = data.map((log) => ({
        id: log.id,
        type: log.type,
        message: log.message,
        user: log.user,
        timestamp: formatRelativeTime(log.timestamp),
        icon: getActivityIcon(log.type),
        color: getActivityColor(log.type),
      }));
      setActivities(formattedActivities);
    } catch (error) {
      console.error("Etkinlik kayıtları alınamadı:", error);
      setError("Etkinlik kayıtları alınamadı");
      setActivities([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchActivities();
  }, []);

  const viewAllButton = (
    <button className="text-xs text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 px-2 py-1 rounded hover:bg-indigo-50 dark:hover:bg-indigo-900/20 transition-colors">
      Tümünü Gör
    </button>
  );

  // Don't render if no activities
  if (activities.length === 0 && !loading && !error) {
    return null;
  }

  return (
    <Widget
      title="Son Etkinlikler"
      loading={loading}
      error={error}
      onRefresh={fetchActivities}
      actions={viewAllButton}
    >
      {activities.length > 0 && (
        <div className="space-y-2">
          {activities.slice(0, 5).map((activity, index) => (
            <div 
              key={activity.id} 
              className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors border border-gray-200 dark:border-gray-700"
            >
              <div className={`flex-shrink-0 p-2 rounded-lg ${activity.color}`}>
                {activity.icon}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs mb-0.5">
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {activity.message}
                  </span>
                  <span className="text-indigo-600 dark:text-indigo-400 ml-1 font-medium">
                    {activity.user}
                  </span>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                  </svg>
                  {activity.timestamp}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </Widget>
  );
}
