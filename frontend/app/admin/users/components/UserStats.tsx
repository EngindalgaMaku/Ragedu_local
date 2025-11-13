"use client";

import React from "react";
import { AdminUser } from "@/lib/admin-api";

interface UserStatsProps {
  users: AdminUser[];
  loading: boolean;
}

export default function UserStats({ users, loading }: UserStatsProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow animate-pulse"
          >
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded-md"></div>
              </div>
              <div className="ml-5 w-0 flex-1">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-20 mb-2"></div>
                <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-16"></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  const totalUsers = users.length;
  const activeUsers = users.filter((user) => user.is_active).length;
  const inactiveUsers = totalUsers - activeUsers;

  const roleStats = users.reduce((acc, user) => {
    acc[user.role_name] = (acc[user.role_name] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Calculate recently active users (logged in within last 7 days)
  const sevenDaysAgo = new Date();
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
  const recentlyActiveUsers = users.filter((user) => {
    if (!user.last_login) return false;
    const lastLogin = new Date(user.last_login);
    return lastLogin > sevenDaysAgo;
  }).length;

  const stats = [
    {
      name: "Total Users",
      value: totalUsers,
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
      color: "text-blue-600 dark:text-blue-400",
      bgColor: "bg-blue-100 dark:bg-blue-900",
    },
    {
      name: "Active Users",
      value: activeUsers,
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
            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      color: "text-green-600 dark:text-green-400",
      bgColor: "bg-green-100 dark:bg-green-900",
    },
    {
      name: "Inactive Users",
      value: inactiveUsers,
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
            d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728L5.636 5.636m12.728 12.728L18.364 5.636M5.636 18.364l12.728-12.728"
          />
        </svg>
      ),
      color: "text-red-600 dark:text-red-400",
      bgColor: "bg-red-100 dark:bg-red-900",
    },
    {
      name: "Recently Active",
      value: recentlyActiveUsers,
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
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      color: "text-purple-600 dark:text-purple-400",
      bgColor: "bg-purple-100 dark:bg-purple-900",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Main Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <div
            key={stat.name}
            className="bg-white dark:bg-gray-800 overflow-hidden shadow rounded-lg"
          >
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className={`${stat.bgColor} rounded-md p-3`}>
                    <div className={stat.color}>{stat.icon}</div>
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                      {stat.name}
                    </dt>
                    <dd className="text-lg font-medium text-gray-900 dark:text-white">
                      {stat.value}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Role Distribution */}
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white mb-4">
          Role Distribution
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(roleStats).map(([role, count]) => {
            const percentage =
              totalUsers > 0 ? ((count / totalUsers) * 100).toFixed(1) : "0";
            const roleColor =
              role === "admin"
                ? "bg-purple-500"
                : role === "teacher"
                ? "bg-blue-500"
                : "bg-green-500";

            return (
              <div
                key={role}
                className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
              >
                <div className="flex items-center">
                  <div
                    className={`w-3 h-3 ${roleColor} rounded-full mr-3`}
                  ></div>
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                      {role}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {percentage}% of users
                    </div>
                  </div>
                </div>
                <div className="text-lg font-semibold text-gray-900 dark:text-white">
                  {count}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* User Activity Chart (Mock) */}
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white mb-4">
          User Registration Trend (Last 7 Days)
        </h3>
        <div className="flex items-end justify-between h-32 space-x-2">
          {Array.from({ length: 7 }).map((_, index) => {
            const height = Math.random() * 80 + 20; // Mock data
            const day = new Date();
            day.setDate(day.getDate() - (6 - index));

            return (
              <div key={index} className="flex-1 flex flex-col items-center">
                <div
                  className="bg-indigo-500 rounded-t-md w-full transition-all duration-300 hover:bg-indigo-600"
                  style={{ height: `${height}%` }}
                ></div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                  {day.toLocaleDateString("en-US", { weekday: "short" })}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
