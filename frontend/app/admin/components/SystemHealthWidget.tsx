"use client";

import React, { useState, useEffect } from "react";
import Widget from "./Widget";
import { adminApiClient, SystemHealth } from "@/lib/admin-api";

const getStatusColor = (status: SystemHealth["status"]) => {
  switch (status) {
    case "healthy":
      return {
        dot: "bg-green-500",
        text: "text-green-600 dark:text-green-400",
        bg: "bg-green-50 dark:bg-green-900/20",
        label: "Sağlıklı",
      };
    case "warning":
      return {
        dot: "bg-yellow-500",
        text: "text-yellow-600 dark:text-yellow-400",
        bg: "bg-yellow-50 dark:bg-yellow-900/20",
        label: "Uyarı",
      };
    case "critical":
      return {
        dot: "bg-red-500",
        text: "text-red-600 dark:text-red-400",
        bg: "bg-red-50 dark:bg-red-900/20",
        label: "Kritik",
      };
    default:
      return {
        dot: "bg-gray-500",
        text: "text-gray-600 dark:text-gray-400",
        bg: "bg-gray-50 dark:bg-gray-900/20",
        label: "Bilinmiyor",
      };
  }
};

const getUsageColor = (usage: string) => {
  const numericValue = parseInt(usage.replace("%", ""));
  if (numericValue < 50) return "bg-green-500";
  if (numericValue < 80) return "bg-yellow-500";
  return "bg-red-500";
};

const getUsageWidth = (usage: string) => {
  const numericValue = parseInt(usage.replace("%", ""));
  return Math.min(Math.max(numericValue, 5), 100);
};

export default function SystemHealthWidget() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSystemHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await adminApiClient.getSystemHealth();
      setSystemHealth(data);
    } catch (error) {
      console.error("Sistem durumu alınamadı:", error);
      setError("Sistem durumu alınamadı");
      // Fallback to default values on error
      setSystemHealth({
        status: "critical",
        uptime: "Bilinmiyor",
        lastBackup: "Bilinmiyor",
        diskUsage: "0%",
        memoryUsage: "0%",
        services: {
          auth_service: false,
          main_gateway: false,
          database: false,
        },
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemHealth();
  }, []);

  return (
    <Widget
      title="Sistem Durumu"
      loading={loading}
      error={error}
      onRefresh={fetchSystemHealth}
    >
      {systemHealth && (
        <div className="space-y-3">
          {/* Overall Status - Compact */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 lg:gap-3">
            <div className="p-4 rounded-lg bg-slate-800/40 border border-slate-800 backdrop-blur">
              <div className="flex items-center gap-3">
                <div
                  className={`w-12 h-12 ${
                    getStatusColor(systemHealth.status).dot
                  } rounded-xl flex items-center justify-center shadow-md`}
                >
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                    Sistem Durumu
                  </h3>
                  <p
                    className={`text-base font-bold ${
                      getStatusColor(systemHealth.status).text
                    }`}
                  >
                    {getStatusColor(systemHealth.status).label}
                  </p>
                </div>
              </div>
            </div>
            <div className="p-4 rounded-lg bg-slate-800/40 border border-slate-800 backdrop-blur">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center shadow-md">
                  <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                    Çalışma Süresi
                  </h3>
                  <p className="text-base font-bold text-blue-700 dark:text-blue-300">
                    {systemHealth.uptime}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* System Metrics - Compact */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 lg:gap-3">
            {/* Disk Usage */}
            <div className="p-4 rounded-lg bg-slate-800/40 border border-slate-800 backdrop-blur">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M3 12v3c0 1.657 3.134 3 7 3s7-1.343 7-3v-3c0 1.657-3.134 3-7 3s-7-1.343-7-3z"></path>
                    <path d="M3 7v3c0 1.657 3.134 3 7 3s7-1.343 7-3V7c0 1.657-3.134 3-7 3S3 8.657 3 7z"></path>
                    <path d="M17 5c0 1.657-3.134 3-7 3S3 6.657 3 5s3.134-3 7-3 7 1.343 7 3z"></path>
                  </svg>
                  <span className="text-xs font-bold text-gray-800 dark:text-gray-200">
                    Disk
                  </span>
                </div>
                <span className="text-lg font-extrabold text-blue-700 dark:text-blue-300">
                  {systemHealth.diskUsage}
                </span>
              </div>
              <div className="w-full bg-blue-200 dark:bg-blue-900/40 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-2 rounded-full ${getUsageColor(
                    systemHealth.diskUsage
                  )} transition-all duration-500`}
                  style={{ width: `${getUsageWidth(systemHealth.diskUsage)}%` }}
                ></div>
              </div>
            </div>

            {/* Memory Usage */}
            <div className="p-4 rounded-lg bg-slate-800/40 border border-slate-800 backdrop-blur">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-purple-600 dark:text-purple-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M13 7H7v6h6V7z"></path>
                    <path fillRule="evenodd" d="M7 2a1 1 0 012 0v1h2V2a1 1 0 112 0v1h2a2 2 0 012 2v2h1a1 1 0 110 2h-1v2h1a1 1 0 110 2h-1v2a2 2 0 01-2 2h-2v1a1 1 0 11-2 0v-1H9v1a1 1 0 11-2 0v-1H5a2 2 0 01-2-2v-2H2a1 1 0 110-2h1V9H2a1 1 0 010-2h1V5a2 2 0 012-2h2V2zM5 5h10v10H5V5z" clipRule="evenodd"></path>
                  </svg>
                  <span className="text-xs font-bold text-gray-800 dark:text-gray-200">
                    Bellek
                  </span>
                </div>
                <span className="text-lg font-extrabold text-purple-700 dark:text-purple-300">
                  {systemHealth.memoryUsage}
                </span>
              </div>
              <div className="w-full bg-purple-200 dark:bg-purple-900/40 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-2 rounded-full ${getUsageColor(
                    systemHealth.memoryUsage
                  )} transition-all duration-500`}
                  style={{
                    width: `${getUsageWidth(systemHealth.memoryUsage)}%`,
                  }}
                ></div>
              </div>
            </div>
          </div>

          {/* Service Status - Compact */}
          <div>
            <h4 className="text-sm font-bold text-slate-100 mb-3">
              Servisler
            </h4>
            <div className="grid grid-cols-3 gap-3">
              {Object.entries(systemHealth.services).map(
                ([service, isActive]) => (
                  <div
                    key={service}
                    className={`p-3 rounded-lg transition-all duration-300 ${isActive ? "bg-green-600/15" : "bg-red-600/15"} border border-slate-800 backdrop-blur`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div
                        className={`w-2 h-2 rounded-full ${
                          isActive ? "bg-green-500" : "bg-red-500"
                        } animate-pulse`}
                      ></div>
                      <span
                        className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                          isActive
                            ? "bg-green-500 text-white"
                            : "bg-red-500 text-white"
                        }`}
                      >
                        {isActive ? "✓" : "✗"}
                      </span>
                    </div>
                    <span className="text-xs font-semibold text-slate-200 capitalize">
                      {service.replace("_", " ")}
                    </span>
                  </div>
                )
              )}
            </div>
          </div>
        </div>
      )}
    </Widget>
  );
}
