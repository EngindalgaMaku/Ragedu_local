"use client";

import React, { useEffect, useState } from "react";
import { adminApiClient, AdminStats as AdminStatsType, SystemHealth } from "@/lib/admin-api";
import ActivityFeedWidget from "./ActivityFeedWidget";
import QuickActionsWidget from "./QuickActionsWidget";

type ServiceMap = Record<"auth_service" | "main_gateway" | "database", boolean>;

function usageColor(value: number): string {
  if (value < 50) return "bg-green-500";
  if (value < 80) return "bg-yellow-500";
  return "bg-red-500";
}

function parsePercent(p: string): number {
  const n = parseInt(String(p).replace(/[^0-9]/g, ""));
  return Number.isFinite(n) ? Math.min(Math.max(n, 0), 100) : 0;
}

export default function DashboardV2() {
  const [stats, setStats] = useState<AdminStatsType | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      try {
        const [s, h] = await Promise.all([adminApiClient.getStats(), adminApiClient.getSystemHealth()]);
        if (!mounted) return;
        setStats(s);
        setHealth(h);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, []);

  const kpis =
    stats && [
      { label: "Toplam Kullanıcı", value: stats.totalUsers, color: "from-indigo-500 to-indigo-600", icon: "👥" },
      { label: "Aktif Kullanıcı", value: stats.activeUsers, color: "from-emerald-500 to-emerald-600", icon: "👤" },
      { label: "Aktif Oturumlar", value: stats.activeSessions, color: "from-amber-500 to-amber-600", icon: "🔗" },
      { label: "Toplam Oturumlar", value: stats.totalSessions, color: "from-fuchsia-500 to-fuchsia-600", icon: "📊" },
    ];

  return (
    <div className="space-y-4">
      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {kpis?.map((k, i) => (
          <div key={i} className="rounded-lg bg-slate-800/40 border border-slate-800 p-4 backdrop-blur">
            <div className="flex items-center justify-between">
              <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${k.color} flex items-center justify-center`}>
                <span className="text-lg">{k.icon}</span>
              </div>
              <div className="text-2xl font-extrabold text-white">{k.value}</div>
            </div>
            <div className="mt-2 text-xs font-semibold uppercase tracking-wide text-slate-300">{k.label}</div>
          </div>
        ))}
        {!kpis &&
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="rounded-lg bg-slate-800/30 border border-slate-800 p-4 animate-pulse h-20" />
          ))}
      </div>

      {/* Health + Resources */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <div className="xl:col-span-2 space-y-3">
          <div className="rounded-lg bg-slate-800/40 border border-slate-800 p-4 backdrop-blur">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${health?.status === "healthy" ? "bg-emerald-500" : health?.status === "warning" ? "bg-amber-500" : "bg-rose-500"}`}>
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <div className="text-sm font-semibold text-slate-200">Sistem Durumu</div>
                  <div className="text-base font-bold text-white">{health?.status ?? "—"}</div>
                </div>
              </div>
              <div className="text-sm text-slate-300">
                Çalışma Süresi: <span className="font-semibold text-indigo-300">{health?.uptime ?? "—"}</span>
              </div>
            </div>

            {/* Resources */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <div className="flex items-center justify-between text-xs text-slate-300 mb-2">
                  <span>Disk</span>
                  <span className="font-bold">{health?.diskUsage ?? "0%"}</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-700 overflow-hidden">
                  <div
                    className={`h-2 rounded-full ${usageColor(parsePercent(health?.diskUsage ?? "0%"))}`}
                    style={{ width: `${parsePercent(health?.diskUsage ?? "0%")} %`.replace(" ", "") }}
                  />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-xs text-slate-300 mb-2">
                  <span>Bellek</span>
                  <span className="font-bold">{health?.memoryUsage ?? "0%"}</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-700 overflow-hidden">
                  <div
                    className={`h-2 rounded-full ${usageColor(parsePercent(health?.memoryUsage ?? "0%"))}`}
                    style={{ width: `${parsePercent(health?.memoryUsage ?? "0%")} %`.replace(" ", "") }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Services */}
          <div className="rounded-lg bg-slate-800/40 border border-slate-800 p-4 backdrop-blur">
            <div className="text-sm font-bold text-slate-200 mb-3">Servisler</div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {(Object.entries((health?.services as ServiceMap) ?? { auth_service: false, main_gateway: false, database: false }) as [keyof ServiceMap, boolean][]).map(
                ([key, active]) => (
                  <div key={key} className={`rounded-lg px-4 py-3 border ${active ? "bg-emerald-500/10 border-emerald-600/50" : "bg-rose-500/10 border-rose-600/50"}`}>
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-semibold text-slate-200 capitalize">{String(key).replace("_", " ")}</div>
                      <span className={`w-2.5 h-2.5 rounded-full ${active ? "bg-emerald-500" : "bg-rose-500"}`} />
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        </div>

        {/* Right rail */}
        <div className="space-y-3">
          <QuickActionsWidget />
          <ActivityFeedWidget />
        </div>
      </div>
    </div>
  );
}



