import { NextRequest, NextResponse } from "next/server";

// Notification tipini tanımla
interface Notification {
  id: string;
  type: "success" | "error" | "warning" | "info";
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  sessionId?: string;
  userId?: string;
}

// Bildirimleri import et (aynı in-memory store)
// Gerçek projede shared storage kullanılmalı (Redis, DB)
let notifications: Notification[] = [];

// Global bildirimleri almak için - geçici çözüm
function getGlobalNotifications(): Notification[] {
  // Bu fonksiyon gerçek projede shared storage'dan veri almalı
  if (typeof global !== "undefined" && (global as any).__notifications) {
    return (global as any).__notifications;
  }
  return notifications;
}

function setGlobalNotifications(notifs: Notification[]) {
  if (typeof global !== "undefined") {
    (global as any).__notifications = notifs;
  }
  notifications = notifs;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { notificationIds } = body;

    if (!Array.isArray(notificationIds)) {
      return NextResponse.json(
        { error: "notificationIds must be an array" },
        { status: 400 }
      );
    }

    const currentNotifications = getGlobalNotifications();

    // Belirtilen ID'lerdeki bildirimleri okundu olarak işaretle
    let updatedCount = 0;
    currentNotifications.forEach((notification: Notification) => {
      if (notificationIds.includes(notification.id)) {
        notification.read = true;
        updatedCount++;
      }
    });

    setGlobalNotifications(currentNotifications);

    console.log(`📖 Marked ${updatedCount} notifications as read`);

    return NextResponse.json({
      success: true,
      message: `${updatedCount} notifications marked as read`,
      updatedCount,
    });
  } catch (error) {
    console.error("Error marking notifications as read:", error);
    return NextResponse.json(
      { error: "Failed to mark notifications as read" },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const { notificationIds } = body;

    if (!Array.isArray(notificationIds)) {
      return NextResponse.json(
        { error: "notificationIds must be an array" },
        { status: 400 }
      );
    }

    const currentNotifications = getGlobalNotifications();

    // Belirtilen ID'lerdeki bildirimleri sil
    const filteredNotifications = currentNotifications.filter(
      (notification: Notification) => !notificationIds.includes(notification.id)
    );

    const deletedCount =
      currentNotifications.length - filteredNotifications.length;
    setGlobalNotifications(filteredNotifications);

    console.log(`🗑️ Deleted ${deletedCount} notifications`);

    return NextResponse.json({
      success: true,
      message: `${deletedCount} notifications deleted`,
      deletedCount,
    });
  } catch (error) {
    console.error("Error deleting notifications:", error);
    return NextResponse.json(
      { error: "Failed to delete notifications" },
      { status: 500 }
    );
  }
}
