"use client";

import React, { useState, useEffect } from "react";
import Widget from "./Widget";
import { adminApiClient, AdminStats as AdminStatsType } from "@/lib/admin-api";

export default function StatsWidget() {
  const [stats, setStats] = useState<AdminStatsType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await adminApiClient.getStats();
      setStats(data);
    } catch (error) {
      console.error("Failed to fetch admin statistics:", error);
      setError("İstatistikler alınamadı");
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

  useEffect(() => {
    fetchStats();
  }, []);

  const statCards = stats
    ? [
        {
          title: "Toplam Kullanıcı",
          value: stats.totalUsers,
          change: "+12%",
          changeType: "increase" as const,
          icon: "👥",
          color: "from-blue-500 to-blue-600",
        },
        {
          title: "Aktif Kullanıcı",
          value: stats.activeUsers,
          change: "+5%",
          changeType: "increase" as const,
          icon: "👤",
          color: "from-green-500 to-green-600",
        },
        {
          title: "Aktif Oturumlar",
          value: stats.activeSessions,
          change: "-2%",
          changeType: "decrease" as const,
          icon: "🔗",
          color: "from-yellow-500 to-yellow-600",
        },
        {
          title: "Toplam Oturumlar",
          value: stats.totalSessions,
          change: "+8%",
          changeType: "increase" as const,
          icon: "📊",
          color: "from-purple-500 to-purple-600",
        },
      ]
    : [];

  return (
    <Widget
      title="Sistem İstatistikleri"
      loading={loading}
      error={error}
      onRefresh={fetchStats}
    >
      {stats && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 lg:gap-3">
          {statCards.map((card, index) => (
            <div key={index} className="relative group">
              {/* Compact Stat Card */}
              <div className="relative p-4 rounded-lg bg-slate-800/40 border border-slate-800 hover:border-indigo-600/60 backdrop-blur transition-all duration-300">
                {/* Background gradient overlay */}
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${card.color} opacity-0 group-hover:opacity-10 rounded-xl transition-opacity duration-300`}
                ></div>

                <div className="relative">
                  {/* Icon and Badge */}
                  <div className="flex items-center justify-between mb-2">
                    <div className={`flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br ${card.color} shadow-sm`}>
                      <span className="text-xl">{card.icon}</span>
                    </div>
                    <div
                      className={`px-2 py-0.5 rounded-full bg-gradient-to-r ${card.color} text-white text-xs font-bold`}
                    >
                      {card.change}
                    </div>
                  </div>

                  {/* Title */}
                  <h3 className="text-xs font-semibold text-slate-300 mb-1 uppercase tracking-wide">
                    {card.title}
                  </h3>

                  {/* Value */}
                  <p className="text-2xl font-extrabold text-white">
                    {card.value.toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Widget>
  );
}
