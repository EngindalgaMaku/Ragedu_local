"use client";

import React, { useEffect, useState } from "react";
import {
  adminApiClient,
  AdminStats as AdminStatsType,
  SystemHealth,
} from "@/lib/admin-api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import {
  Users,
  UserCheck,
  Link,
  BarChart3,
  Activity,
  HardDrive,
  MemoryStick,
  Server,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import ModernActivityFeedWidget from "./ModernActivityFeedWidget";
import ModernQuickActionsWidget from "./ModernQuickActionsWidget";

type ServiceMap = Record<"auth_service" | "main_gateway" | "database", boolean>;

function parsePercent(p: string): number {
  const n = parseInt(String(p).replace(/[^0-9]/g, ""));
  return Number.isFinite(n) ? Math.min(Math.max(n, 0), 100) : 0;
}

function getHealthIcon(status: string) {
  switch (status) {
    case "healthy":
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    case "warning":
      return <AlertCircle className="w-5 h-5 text-yellow-500" />;
    default:
      return <XCircle className="w-5 h-5 text-red-500" />;
  }
}

function getHealthColor(status: string) {
  switch (status) {
    case "healthy":
      return "from-green-500 to-green-600";
    case "warning":
      return "from-yellow-500 to-yellow-600";
    default:
      return "from-red-500 to-red-600";
  }
}

export default function ModernDashboard() {
  const [stats, setStats] = useState<AdminStatsType | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    setLoading(true);
    try {
      const [s, h] = await Promise.all([
        adminApiClient.getStats(),
        adminApiClient.getSystemHealth(),
      ]);
      setStats(s);
      setHealth(h);
    } catch (error) {
      console.error("Failed to load dashboard data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const kpis = stats
    ? [
        {
          label: "Toplam Kullanıcı",
          value: stats.totalUsers,
          icon: Users,
          change: "+12%",
          changeType: "increase" as const,
          color: "from-blue-500 to-blue-600",
        },
        {
          label: "Aktif Kullanıcı",
          value: stats.activeUsers,
          icon: UserCheck,
          change: "+5%",
          changeType: "increase" as const,
          color: "from-green-500 to-green-600",
        },
        {
          label: "Aktif Oturumlar",
          value: stats.activeSessions,
          icon: Link,
          change: "-2%",
          changeType: "decrease" as const,
          color: "from-orange-500 to-orange-600",
        },
        {
          label: "Toplam Oturumlar",
          value: stats.totalSessions,
          icon: BarChart3,
          change: "+8%",
          changeType: "increase" as const,
          color: "from-purple-500 to-purple-600",
        },
      ]
    : [];

  if (loading) {
    return (
      <div className="space-y-6">
        {/* Loading skeleton */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-6">
                <div className="h-20 bg-muted rounded"></div>
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2 animate-pulse">
            <CardContent className="p-6">
              <div className="h-64 bg-muted rounded"></div>
            </CardContent>
          </Card>
          <Card className="animate-pulse">
            <CardContent className="p-6">
              <div className="h-64 bg-muted rounded"></div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Refresh button */}
      <div className="flex items-center justify-end">
        <Button
          onClick={loadData}
          variant="outline"
          size="sm"
          disabled={loading}
          className="min-h-[44px]"
        >
          <RefreshCw
            className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`}
          />
          <span className="hidden sm:inline">Yenile</span>
          <span className="sm:hidden">↻</span>
        </Button>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        {kpis.map((kpi, index) => (
          <Card key={index} className="hover:shadow-md transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {kpi.label}
              </CardTitle>
              <div className={`p-2 rounded-md bg-gradient-to-r ${kpi.color}`}>
                <kpi.icon className="h-4 w-4 text-white" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {kpi.value.toLocaleString()}
              </div>
              <div className="flex items-center text-xs text-muted-foreground">
                {kpi.changeType === "increase" ? (
                  <TrendingUp className="mr-1 h-4 w-4 text-green-500" />
                ) : (
                  <TrendingDown className="mr-1 h-4 w-4 text-red-500" />
                )}
                <span
                  className={`font-medium ${
                    kpi.changeType === "increase"
                      ? "text-green-600"
                      : "text-red-600"
                  }`}
                >
                  {kpi.change}
                </span>
                <span className="ml-1">geçen aydan</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid gap-4 sm:gap-6 grid-cols-1 lg:grid-cols-3">
        {/* System Health Section */}
        <div className="lg:col-span-2 space-y-4 sm:space-y-6">
          {/* System Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Sistem Durumu
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 sm:space-y-6">
              {/* Overall Health */}
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-0">
                <div className="flex items-center space-x-3">
                  {getHealthIcon(health?.status || "unknown")}
                  <div>
                    <div className="font-semibold capitalize">
                      {health?.status || "Bilinmiyor"}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Genel sistem durumu
                    </div>
                  </div>
                </div>
                <Badge variant="secondary" className="text-xs">
                  <span className="hidden sm:inline">Çalışma Süresi: </span>{health?.uptime || "—"}
                </Badge>
              </div>

              {/* Resource Usage */}
              <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <HardDrive className="h-4 w-4" />
                      <span>Disk Kullanımı</span>
                    </div>
                    <span className="font-medium">
                      {health?.diskUsage || "0%"}
                    </span>
                  </div>
                  <Progress
                    value={parsePercent(health?.diskUsage || "0%")}
                    className="h-2"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <MemoryStick className="h-4 w-4" />
                      <span>Bellek Kullanımı</span>
                    </div>
                    <span className="font-medium">
                      {health?.memoryUsage || "0%"}
                    </span>
                  </div>
                  <Progress
                    value={parsePercent(health?.memoryUsage || "0%")}
                    className="h-2"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Services Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                Servis Durumları
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
                {(
                  Object.entries(
                    (health?.services as ServiceMap) ?? {
                      auth_service: false,
                      main_gateway: false,
                      database: false,
                    }
                  ) as [keyof ServiceMap, boolean][]
                ).map(([key, active]) => (
                  <div
                    key={key}
                    className={`flex items-center justify-between p-3 rounded-lg border ${
                      active
                        ? "bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800"
                        : "bg-red-50 border-red-200 dark:bg-red-950 dark:border-red-800"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {active ? (
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-600" />
                      )}
                      <span className="text-sm font-medium capitalize">
                        {String(key).replace("_", " ")}
                      </span>
                    </div>
                    <Badge
                      variant={active ? "default" : "destructive"}
                      className="text-xs"
                    >
                      {active ? "Aktif" : "İnaktif"}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right sidebar */}
        <div className="space-y-4 sm:space-y-6">
          <ModernQuickActionsWidget />
          <ModernActivityFeedWidget />
        </div>
      </div>
    </div>
  );
}
