"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { adminApiClient, ActivityLog } from "@/lib/admin-api";
import {
  UserPlus,
  LogIn,
  LogOut,
  Shield,
  Clock,
  Info,
  RefreshCw,
  ExternalLink,
} from "lucide-react";

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
  icon: React.ComponentType<any>;
  color: string;
}

const getActivityIcon = (type: ActivityLog["type"]) => {
  switch (type) {
    case "user_created":
      return UserPlus;
    case "user_login":
      return LogIn;
    case "user_logout":
      return LogOut;
    case "role_assigned":
      return Shield;
    case "session_expired":
      return Clock;
    default:
      return Info;
  }
};

const getActivityColor = (type: ActivityLog["type"]) => {
  switch (type) {
    case "user_created":
      return "text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-950";
    case "user_login":
      return "text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-950";
    case "user_logout":
      return "text-gray-600 bg-gray-50 dark:text-gray-400 dark:bg-gray-950";
    case "role_assigned":
      return "text-purple-600 bg-purple-50 dark:text-purple-400 dark:bg-purple-950";
    case "session_expired":
      return "text-yellow-600 bg-yellow-50 dark:text-yellow-400 dark:bg-yellow-950";
    default:
      return "text-gray-600 bg-gray-50 dark:text-gray-400 dark:bg-gray-950";
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

export default function ModernActivityFeedWidget() {
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

  // Don't render if no activities
  if (activities.length === 0 && !loading && !error) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">
            Son Etkinlikler
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchActivities}
              disabled={loading}
            >
              <RefreshCw
                className={`h-4 w-4 ${loading ? "animate-spin" : ""}`}
              />
            </Button>
            <Button variant="ghost" size="sm">
              <ExternalLink className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="flex items-center gap-3 animate-pulse">
                <div className="h-10 w-10 bg-muted rounded-lg"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-muted rounded w-3/4"></div>
                  <div className="h-3 bg-muted rounded w-1/2"></div>
                </div>
              </div>
            ))}
          </div>
        ) : error ? (
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground">{error}</p>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchActivities}
              className="mt-2"
            >
              Tekrar Dene
            </Button>
          </div>
        ) : activities.length > 0 ? (
          <div className="space-y-3">
            {activities.slice(0, 5).map((activity) => {
              const IconComponent = activity.icon;
              return (
                <div
                  key={activity.id}
                  className="flex items-center gap-3 p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors"
                >
                  <div
                    className={`flex-shrink-0 p-2 rounded-lg ${activity.color}`}
                  >
                    <IconComponent className="h-4 w-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm mb-1">
                      <span className="font-medium">{activity.message}</span>
                      <Badge variant="secondary" className="ml-2 text-xs">
                        {activity.user}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {activity.timestamp}
                    </p>
                  </div>
                </div>
              );
            })}

            {activities.length > 5 && (
              <div className="text-center pt-2">
                <Button variant="ghost" size="sm" className="text-xs">
                  {activities.length - 5} daha fazla etkinlik
                </Button>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="w-12 h-12 bg-muted rounded-full flex items-center justify-center mx-auto mb-3">
              <Info className="h-6 w-6 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">Henüz etkinlik yok</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
