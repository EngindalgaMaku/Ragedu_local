"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { adminApiClient, SystemHealth } from "@/lib/admin-api";

interface QuickAction {
  title: string;
  description: string;
  href: string;
  icon: React.ReactNode;
  color: string;
}

const quickActions: QuickAction[] = [
  {
    title: "Kullanıcı Oluştur",
    description: "Sisteme yeni bir kullanıcı ekle",
    href: "/admin/users?action=create",
    color: "bg-blue-500 hover:bg-blue-600",
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
          d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"
        />
      </svg>
    ),
  },
  {
    title: "Rolleri Yönet",
    description: "Kullanıcı rollerini ve izinlerini yapılandır",
    href: "/admin/roles",
    color: "bg-purple-500 hover:bg-purple-600",
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
          d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
        />
      </svg>
    ),
  },
  {
    title: "Aktif Oturumlar",
    description: "Kullanıcı oturumlarını izle ve yönet",
    href: "/admin/sessions",
    color: "bg-green-500 hover:bg-green-600",
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
  },
  {
    title: "Sistem Ayarları",
    description: "Sistem genelindeki ayarları yapılandır",
    href: "/admin/settings",
    color: "bg-gray-500 hover:bg-gray-600",
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
          d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
        />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
        />
      </svg>
    ),
  },
];

const getStatusColor = (status: SystemHealth["status"]) => {
  switch (status) {
    case "healthy":
      return {
        dot: "bg-green-500",
        text: "text-green-600 dark:text-green-400",
        label: "Sağlıklı",
      };
    case "warning":
      return {
        dot: "bg-yellow-500",
        text: "text-yellow-600 dark:text-yellow-400",
        label: "Uyarı",
      };
    case "critical":
      return {
        dot: "bg-red-500",
        text: "text-red-600 dark:text-red-400",
        label: "Kritik",
      };
    default:
      return {
        dot: "bg-gray-500",
        text: "text-gray-600 dark:text-gray-400",
        label: "Bilinmiyor",
      };
  }
};

export default function QuickActions() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);

  useEffect(() => {
    const fetchSystemHealth = async () => {
      setHealthLoading(true);
      try {
        const data = await adminApiClient.getSystemHealth();
        setSystemHealth(data);
      } catch (error) {
        console.error("Failed to fetch system health:", error);
        // Fallback to default values on error
        setSystemHealth({
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
        });
      } finally {
        setHealthLoading(false);
      }
    };

    fetchSystemHealth();
  }, []);
  return (
    <div className="space-y-6">
      {/* Quick Actions */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Hızlı Eylemler
        </h3>
        <div className="grid grid-cols-1 gap-4">
          {quickActions.map((action, index) => (
            <Link
              key={index}
              href={action.href}
              className={`${action.color} text-white p-4 rounded-lg transition-colors group`}
            >
              <div className="flex items-start space-x-3">
                <span className="text-white group-hover:scale-110 transition-transform">
                  {action.icon}
                </span>
                <div>
                  <h4 className="font-medium text-white">{action.title}</h4>
                  <p className="text-sm text-white/80 mt-1">
                    {action.description}
                  </p>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* System Health */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Sistem Durumu
        </h3>

        {healthLoading ? (
          <div className="space-y-4 animate-pulse">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="flex items-center justify-between">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-20"></div>
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-16"></div>
              </div>
            ))}
          </div>
        ) : systemHealth ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Durum
              </span>
              <div className="flex items-center space-x-2">
                <div
                  className={`w-2 h-2 ${
                    getStatusColor(systemHealth.status).dot
                  } rounded-full`}
                ></div>
                <span
                  className={`text-sm font-medium ${
                    getStatusColor(systemHealth.status).text
                  }`}
                >
                  {getStatusColor(systemHealth.status).label}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Çalışma Süresi
              </span>
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                {systemHealth.uptime}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Son Yedekleme
              </span>
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                {systemHealth.lastBackup}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Disk Kullanımı
              </span>
              <div className="flex items-center space-x-2">
                <div className="w-16 h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                  <div className="w-7 h-2 bg-blue-500 rounded-full"></div>
                </div>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {systemHealth.diskUsage}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Bellek Kullanımı
              </span>
              <div className="flex items-center space-x-2">
                <div className="w-16 h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                  <div className="w-10 h-2 bg-yellow-500 rounded-full"></div>
                </div>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {systemHealth.memoryUsage}
                </span>
              </div>
            </div>

            {/* Service Status */}
            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Servisler
              </h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    Auth Service
                  </span>
                  <div
                    className={`w-2 h-2 rounded-full ${
                      systemHealth.services.auth_service
                        ? "bg-green-500"
                        : "bg-red-500"
                    }`}
                  ></div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    Main Gateway
                  </span>
                  <div
                    className={`w-2 h-2 rounded-full ${
                      systemHealth.services.main_gateway
                        ? "bg-green-500"
                        : "bg-red-500"
                    }`}
                  ></div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    Database
                  </span>
                  <div
                    className={`w-2 h-2 rounded-full ${
                      systemHealth.services.database
                        ? "bg-green-500"
                        : "bg-red-500"
                    }`}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-4">
            <svg
              className="mx-auto h-8 w-8 text-red-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="mt-2 text-sm text-red-600 dark:text-red-400">
              Sistem durumu alınamadı
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
