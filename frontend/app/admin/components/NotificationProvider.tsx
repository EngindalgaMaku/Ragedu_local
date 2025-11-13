"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
} from "react";

interface Notification {
  id: string;
  title: string;
  message: string;
  type: "success" | "error" | "warning" | "info";
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, "id">) => void;
  removeNotification: (id: string) => void;
  clearAllNotifications: () => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(
  undefined
);

interface NotificationProviderProps {
  children: React.ReactNode;
}

export function NotificationProvider({ children }: NotificationProviderProps) {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = useCallback(
    (notification: Omit<Notification, "id">) => {
      const id = `notification-${Date.now()}-${Math.random()
        .toString(36)
        .substr(2, 9)}`;
      const newNotification: Notification = {
        ...notification,
        id,
        duration: notification.duration ?? 5000,
      };

      setNotifications((prev) => [...prev, newNotification]);

      // Auto-remove notification after duration
      if (newNotification.duration && newNotification.duration > 0) {
        setTimeout(() => {
          removeNotification(id);
        }, newNotification.duration);
      }
    },
    []
  );

  const removeNotification = useCallback((id: string) => {
    setNotifications((prev) =>
      prev.filter((notification) => notification.id !== id)
    );
  }, []);

  const clearAllNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  return (
    <NotificationContext.Provider
      value={{
        notifications,
        addNotification,
        removeNotification,
        clearAllNotifications,
      }}
    >
      {children}
      <NotificationContainer />
    </NotificationContext.Provider>
  );
}

export function useNotifications() {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error(
      "useNotifications must be used within a NotificationProvider"
    );
  }
  return context;
}

function NotificationContainer() {
  const { notifications, removeNotification } = useNotifications();

  if (notifications.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 space-y-4 max-w-sm w-full">
      {notifications.map((notification) => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onRemove={() => removeNotification(notification.id)}
        />
      ))}
    </div>
  );
}

interface NotificationItemProps {
  notification: Notification;
  onRemove: () => void;
}

function NotificationItem({ notification, onRemove }: NotificationItemProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [isRemoving, setIsRemoving] = useState(false);

  useEffect(() => {
    // Trigger animation after mount
    setTimeout(() => setIsVisible(true), 10);
  }, []);

  const handleRemove = () => {
    setIsRemoving(true);
    setTimeout(() => {
      onRemove();
    }, 300);
  };

  const getIcon = () => {
    switch (notification.type) {
      case "success":
        return (
          <svg
            className="w-5 h-5 text-green-600"
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
        );
      case "error":
        return (
          <svg
            className="w-5 h-5 text-red-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 8.5c-.77.833-.192 2.5 1.732 2.5z"
            />
          </svg>
        );
      case "warning":
        return (
          <svg
            className="w-5 h-5 text-yellow-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 8.5c-.77.833-.192 2.5 1.732 2.5z"
            />
          </svg>
        );
      case "info":
        return (
          <svg
            className="w-5 h-5 text-blue-600"
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

  const getColorClasses = () => {
    switch (notification.type) {
      case "success":
        return "bg-green-50 border-green-200 dark:bg-green-900 dark:border-green-700";
      case "error":
        return "bg-red-50 border-red-200 dark:bg-red-900 dark:border-red-700";
      case "warning":
        return "bg-yellow-50 border-yellow-200 dark:bg-yellow-900 dark:border-yellow-700";
      case "info":
        return "bg-blue-50 border-blue-200 dark:bg-blue-900 dark:border-blue-700";
    }
  };

  return (
    <div
      className={`
        transform transition-all duration-300 ease-in-out
        ${
          isVisible && !isRemoving
            ? "translate-x-0 opacity-100 scale-100"
            : "translate-x-full opacity-0 scale-95"
        }
        ${getColorClasses()}
        border rounded-lg shadow-lg p-4 max-w-sm w-full
      `}
    >
      <div className="flex items-start">
        <div className="flex-shrink-0 mr-3 mt-0.5">{getIcon()}</div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium text-gray-900 dark:text-white">
            {notification.title}
          </h4>
          <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
            {notification.message}
          </p>
          {notification.action && (
            <div className="mt-3">
              <button
                onClick={notification.action.onClick}
                className="text-sm font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
              >
                {notification.action.label}
              </button>
            </div>
          )}
        </div>
        <button
          onClick={handleRemove}
          className="flex-shrink-0 ml-4 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300"
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
  );
}

// Utility hook for common notification actions
export function useNotificationHelpers() {
  const { addNotification } = useNotifications();

  return {
    notifySuccess: (
      title: string,
      message: string,
      action?: Notification["action"]
    ) => {
      addNotification({ type: "success", title, message, action });
    },
    notifyError: (
      title: string,
      message: string,
      action?: Notification["action"]
    ) => {
      addNotification({ type: "error", title, message, action });
    },
    notifyWarning: (
      title: string,
      message: string,
      action?: Notification["action"]
    ) => {
      addNotification({ type: "warning", title, message, action });
    },
    notifyInfo: (
      title: string,
      message: string,
      action?: Notification["action"]
    ) => {
      addNotification({ type: "info", title, message, action });
    },
  };
}
