"use client";
import React, { useState, useEffect, useCallback } from "react";
import { getApiUrl } from "@/lib/api";

export interface Notification {
  id: string;
  title: string;
  message: string;
  type: "success" | "info" | "warning" | "error";
  timestamp: string;
  read: boolean;
  sessionId?: string;
  actionType?:
    | "processing_complete"
    | "chunking_complete"
    | "upload_complete"
    | "general";
}

interface NotificationSystemProps {
  className?: string;
}

// Local storage key for notifications
const NOTIFICATIONS_STORAGE_KEY = "dashboard_notifications";

// Notification storage utilities
const loadNotificationsFromStorage = (): Notification[] => {
  if (typeof window === "undefined") return [];
  try {
    const stored = localStorage.getItem(NOTIFICATIONS_STORAGE_KEY);
    if (!stored) return [];
    const notifications = JSON.parse(stored);
    // Clean old notifications (older than 7 days)
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    return notifications.filter(
      (n: Notification) => new Date(n.timestamp) > weekAgo
    );
  } catch (error) {
    console.error("Error loading notifications from storage:", error);
    return [];
  }
};

const saveNotificationsToStorage = (notifications: Notification[]) => {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(
      NOTIFICATIONS_STORAGE_KEY,
      JSON.stringify(notifications)
    );
  } catch (error) {
    console.error("Error saving notifications to storage:", error);
  }
};

export default function NotificationSystem({
  className = "",
}: NotificationSystemProps) {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isPolling, setIsPolling] = useState(false);

  // Load notifications from localStorage on mount
  useEffect(() => {
    const stored = loadNotificationsFromStorage();
    setNotifications(stored);
  }, []);

  // Save notifications to localStorage whenever notifications change
  useEffect(() => {
    if (notifications.length > 0) {
      saveNotificationsToStorage(notifications);
    }
  }, [notifications]);

  // Add new notification
  const addNotification = useCallback(
    (notification: Omit<Notification, "id" | "timestamp" | "read">) => {
      const newNotification: Notification = {
        ...notification,
        id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString(),
        read: false,
      };

      setNotifications((prev) => {
        const updated = [newNotification, ...prev].slice(0, 50); // Keep max 50 notifications
        saveNotificationsToStorage(updated);
        return updated;
      });

      // Auto-close dropdown if it was open
      setTimeout(() => {
        setIsOpen(false);
      }, 3000);
    },
    []
  );

  // Mark notification as read
  const markAsRead = useCallback((id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  }, []);

  // Mark all as read
  const markAllAsRead = useCallback(() => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  }, []);

  // Delete notification
  const deleteNotification = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  // Clear all notifications
  const clearAllNotifications = useCallback(() => {
    setNotifications([]);
    localStorage.removeItem(NOTIFICATIONS_STORAGE_KEY);
  }, []);

  // Poll for new notifications from backend - Fixed dependencies
  const pollNotifications = useCallback(async () => {
    if (isPolling) return;

    setIsPolling(true);
    try {
      const response = await fetch(`${getApiUrl()}/v1/notifications/pending`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
      });

      if (response.ok) {
        const data = await response.json();
        if (data.notifications && Array.isArray(data.notifications)) {
          // Add new notifications directly to state to avoid dependency issues
          const newNotifs = data.notifications.map((notif: any) => ({
            id: `notif_${Date.now()}_${Math.random()
              .toString(36)
              .substr(2, 9)}`,
            title: notif.title || "Yeni Bildirim",
            message: notif.message || "",
            type: notif.type || "info",
            sessionId: notif.session_id,
            actionType: notif.action_type || "general",
            timestamp: new Date().toISOString(),
            read: false,
          }));

          if (newNotifs.length > 0) {
            setNotifications((prev) => {
              const updated = [...newNotifs, ...prev].slice(0, 50);
              saveNotificationsToStorage(updated);
              return updated;
            });

            // Mark as delivered on backend
            const notificationIds = data.notifications.map((n: any) => n.id);
            await fetch(`${getApiUrl()}/v1/notifications/delivered`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ notification_ids: notificationIds }),
              credentials: "include",
            }).catch(console.error);
          }
        }
      }
    } catch (error) {
      console.error("Error polling notifications:", error);
    } finally {
      setIsPolling(false);
    }
  }, [isPolling]); // Removed addNotification dependency

  // Start polling when component mounts - Fixed to prevent re-runs
  useEffect(() => {
    let mounted = true;
    let interval: NodeJS.Timeout | null = null;

    // Initial poll
    const startPolling = async () => {
      if (mounted) {
        await pollNotifications();
        // Set up polling interval (every 2 minutes)
        interval = setInterval(() => {
          if (mounted) {
            pollNotifications();
          }
        }, 120000);
      }
    };

    startPolling();

    return () => {
      mounted = false;
      if (interval) {
        clearInterval(interval);
      }
    };
  }, []); // Empty dependency array - only run once on mount

  // Helper functions
  const unreadCount = notifications.filter((n) => !n.read).length;
  const recentNotifications = notifications.slice(0, 10); // Show last 10

  // Get notification icon based on type
  const getNotificationIcon = (type: Notification["type"]) => {
    switch (type) {
      case "success":
        return "✅";
      case "error":
        return "❌";
      case "warning":
        return "⚠️";
      case "info":
      default:
        return "ℹ️";
    }
  };

  // Get notification color classes
  const getNotificationColors = (type: Notification["type"]) => {
    switch (type) {
      case "success":
        return "border-green-200 bg-green-50";
      case "error":
        return "border-red-200 bg-red-50";
      case "warning":
        return "border-yellow-200 bg-yellow-50";
      case "info":
      default:
        return "border-blue-200 bg-blue-50";
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffMinutes = Math.floor(diffMs / (1000 * 60));

      if (diffMinutes < 1) return "Az önce";
      if (diffMinutes < 60) return `${diffMinutes} dakika önce`;

      const diffHours = Math.floor(diffMinutes / 60);
      if (diffHours < 24) return `${diffHours} saat önce`;

      const diffDays = Math.floor(diffHours / 24);
      if (diffDays < 7) return `${diffDays} gün önce`;

      return date.toLocaleDateString("tr-TR");
    } catch (error) {
      return "Bilinmiyor";
    }
  };

  return (
    <div className={`relative ${className}`}>
      {/* Notification Bell Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-3 text-gray-600 hover:text-gray-900 transition-colors rounded-full hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        title={`${unreadCount} okunmamış bildirim`}
      >
        {/* Bell Icon */}
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
          />
        </svg>

        {/* Badge for unread notifications */}
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-red-500 rounded-full animate-pulse">
            {unreadCount > 99 ? "99+" : unreadCount}
          </span>
        )}

        {/* Polling indicator */}
        {isPolling && (
          <span className="absolute -bottom-1 -right-1 w-3 h-3 bg-blue-500 rounded-full animate-spin">
            <span className="absolute inset-0 w-3 h-3 bg-blue-400 rounded-full animate-ping"></span>
          </span>
        )}
      </button>

      {/* Notification Dropdown */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          ></div>

          {/* Dropdown Content */}
          <div className="absolute right-0 mt-2 w-96 bg-white rounded-lg shadow-lg border border-gray-200 z-50 max-h-96 overflow-y-auto">
            {/* Header */}
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 rounded-t-lg">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-900">
                  Bildirimler
                </h3>
                <div className="flex items-center gap-2">
                  {unreadCount > 0 && (
                    <button
                      onClick={markAllAsRead}
                      className="text-sm text-blue-600 hover:text-blue-800 font-medium"
                    >
                      Tümünü Okundu İşaretle
                    </button>
                  )}
                  {notifications.length > 0 && (
                    <button
                      onClick={clearAllNotifications}
                      className="text-sm text-red-600 hover:text-red-800 font-medium ml-2"
                    >
                      Temizle
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Notifications List */}
            <div className="max-h-80 overflow-y-auto">
              {recentNotifications.length === 0 ? (
                <div className="px-4 py-8 text-center text-gray-500">
                  <svg
                    className="w-12 h-12 mx-auto mb-3 text-gray-300"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                    />
                  </svg>
                  <p>Henüz bildirim yok</p>
                </div>
              ) : (
                recentNotifications.map((notification) => (
                  <div
                    key={notification.id}
                    className={`px-4 py-3 border-b border-gray-100 hover:bg-gray-50 transition-colors ${
                      !notification.read
                        ? "bg-blue-25 border-l-4 border-l-blue-500"
                        : ""
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 mt-1">
                        <span className="text-lg">
                          {getNotificationIcon(notification.type)}
                        </span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1">
                            <p
                              className={`text-sm font-medium ${
                                !notification.read
                                  ? "text-gray-900"
                                  : "text-gray-700"
                              }`}
                            >
                              {notification.title}
                            </p>
                            <p
                              className={`text-sm mt-1 ${
                                !notification.read
                                  ? "text-gray-700"
                                  : "text-gray-500"
                              }`}
                            >
                              {notification.message}
                            </p>
                            <p className="text-xs text-gray-400 mt-1">
                              {formatTimestamp(notification.timestamp)}
                            </p>
                          </div>
                          <div className="flex items-center gap-1">
                            {!notification.read && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  markAsRead(notification.id);
                                }}
                                className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                              >
                                Okundu
                              </button>
                            )}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                deleteNotification(notification.id);
                              }}
                              className="text-gray-400 hover:text-red-500 ml-1"
                              title="Sil"
                            >
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
                                  d="M6 18L18 6M6 6l12 12"
                                />
                              </svg>
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Footer */}
            {notifications.length > 10 && (
              <div className="px-4 py-2 bg-gray-50 rounded-b-lg border-t border-gray-200">
                <p className="text-sm text-gray-500 text-center">
                  {notifications.length - 10} daha fazla bildirim var
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// Export utilities for external use
export const NotificationUtils = {
  addNotification: (
    notification: Omit<Notification, "id" | "timestamp" | "read">
  ) => {
    // This will be used by other components to add notifications manually
    const notifications = loadNotificationsFromStorage();
    const newNotification: Notification = {
      ...notification,
      id: `manual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      read: false,
    };
    const updated = [newNotification, ...notifications].slice(0, 50);
    saveNotificationsToStorage(updated);

    // Trigger custom event to update UI
    window.dispatchEvent(
      new CustomEvent("notification-added", {
        detail: { notification: newNotification },
      })
    );
  },

  // Add processing completion notification
  addProcessingCompleteNotification: (
    sessionId: string,
    message: string,
    success: boolean = true
  ) => {
    NotificationUtils.addNotification({
      title: success ? "İşlem Tamamlandı" : "İşlem Başarısız",
      message: message,
      type: success ? "success" : "error",
      sessionId: sessionId,
      actionType: "processing_complete",
    });
  },
};
