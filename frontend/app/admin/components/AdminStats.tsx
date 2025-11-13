"use client";

import React, { useState, useEffect } from "react";
import { adminApiClient, AdminStats as AdminStatsType } from "@/lib/admin-api";

export default function AdminStats() {
  const [stats, setStats] = useState<AdminStatsType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = await adminApiClient.getStats();
        setStats(data);
      } catch (error) {
        console.error("Failed to fetch admin statistics:", error);
        setError("Failed to load statistics");
        // Fallback to default values on error
        setStats({
          totalUsers: 0,
          activeUsers: 0,
          activeSessions: 0,
          totalSessions: 0,
          totalRoles: 0,
        });
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow animate-pulse"
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-20 mb-2"></div>
                <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-16"></div>
              </div>
              <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
            </div>
            <div className="mt-4">
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-12"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg
              className="h-5 w-5 text-red-400"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800 dark:text-red-200">
              Failed to load statistics
            </h3>
            <div className="mt-2 text-sm text-red-700 dark:text-red-300">
              <p>{error || "Unknown error occurred"}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const statCards = [
    {
      title: "Toplam Kullanıcı",
      value: stats.totalUsers,
      change: "+12%",
      changeType: "increase" as const,
      icon: (
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z"
          />
        </svg>
      ),
      color: "bg-blue-500",
    },
    {
      title: "Aktif Kullanıcı",
      value: stats.activeUsers,
      change: "+5%",
      changeType: "increase" as const,
      icon: (
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
          />
        </svg>
      ),
      color: "bg-green-500",
    },
    {
      title: "Aktif Oturumlar",
      value: stats.activeSessions,
      change: "-2%",
      changeType: "decrease" as const,
      icon: (
        <svg
          className="w-6 h-6"
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
      ),
      color: "bg-yellow-500",
    },
    {
      title: "Toplam Oturumlar",
      value: stats.totalSessions,
      change: "+8%",
      changeType: "increase" as const,
      icon: (
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
          />
        </svg>
      ),
      color: "bg-purple-500",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {statCards.map((card, index) => (
        <div
          key={index}
          className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow hover:shadow-md transition-shadow"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                {card.title}
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {card.value.toLocaleString()}
              </p>
            </div>
            <div className={`${card.color} p-3 rounded-lg`}>
              <span className="text-white">{card.icon}</span>
            </div>
          </div>
          <div className="mt-4 flex items-center">
            <span
              className={`text-sm font-medium ${
                card.changeType === "increase"
                  ? "text-green-600 dark:text-green-400"
                  : "text-red-600 dark:text-red-400"
              }`}
            >
              {card.change}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">
              geçen aydan beri
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
