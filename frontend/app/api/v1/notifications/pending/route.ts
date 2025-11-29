import { NextRequest, NextResponse } from "next/server";

// Basit bir in-memory bildirim sistemi - gerçek projede veritabanı kullanılmalı
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

// Global shared storage - hem pending hem delivered route'ları kullanabilir
function getGlobalNotifications(): Notification[] {
  if (typeof global !== "undefined" && (global as any).__notifications) {
    return (global as any).__notifications;
  }
  return [];
}

function setGlobalNotifications(notifs: Notification[]) {
  if (typeof global !== "undefined") {
    (global as any).__notifications = notifs;
  }
}

function addGlobalNotification(notification: Notification) {
  const currentNotifications = getGlobalNotifications();
  currentNotifications.push(notification);

  // Eski bildirimleri temizle (son 100 bildirim)
  if (currentNotifications.length > 100) {
    const trimmedNotifications = currentNotifications.slice(-100);
    setGlobalNotifications(trimmedNotifications);
  } else {
    setGlobalNotifications(currentNotifications);
  }
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const userId = searchParams.get("user_id");
    const sessionId = searchParams.get("session_id");

    // Global storage'dan bildirimleri al
    const allNotifications = getGlobalNotifications();

    // Kullanıcıya özel bildirimleri filtrele (şimdilik hepsini dön)
    let filteredNotifications = allNotifications.filter((n) => !n.read);

    if (userId) {
      filteredNotifications = filteredNotifications.filter(
        (n) => !n.userId || n.userId === userId
      );
    }

    if (sessionId) {
      filteredNotifications = filteredNotifications.filter(
        (n) => !n.sessionId || n.sessionId === sessionId
      );
    }

    // En yeni bildirimleri önce göster
    filteredNotifications.sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );

    return NextResponse.json({
      notifications: filteredNotifications,
      count: filteredNotifications.length,
    });
  } catch (error) {
    console.error("Error fetching notifications:", error);
    return NextResponse.json(
      { error: "Failed to fetch notifications" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, title, message, sessionId, userId } = body;

    if (!title || !message) {
      return NextResponse.json(
        { error: "Title and message are required" },
        { status: 400 }
      );
    }

    const notification: Notification = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      type: type || "info",
      title,
      message,
      timestamp: new Date().toISOString(),
      read: false,
      sessionId,
      userId,
    };

    addGlobalNotification(notification);

    console.log(`✅ New notification created: ${title}`);

    return NextResponse.json({
      success: true,
      notification,
      message: "Notification created successfully",
    });
  } catch (error) {
    console.error("Error creating notification:", error);
    return NextResponse.json(
      { error: "Failed to create notification" },
      { status: 500 }
    );
  }
}

// Test bildirimleri eklemek için utility fonksiyon (export kaldırıldı - Next.js API route uyumluluğu için)
function addTestNotification() {
  const testNotification: Notification = {
    id: Date.now().toString(),
    type: "success",
    title: "Test Bildirimi",
    message: "Bu bir test bildirimidir.",
    timestamp: new Date().toISOString(),
    read: false,
  };

  addGlobalNotification(testNotification);
}

// Background processing tamamlandığında bildirim gönderen fonksiyon (export kaldırıldı - Next.js API route uyumluluğu için)
function addBackgroundProcessingNotification(
  sessionId: string,
  chunkCount: number,
  fileCount: number,
  isSuccess: boolean = true
) {
  const notification: Notification = {
    id: `bg_process_${sessionId}_${Date.now()}`,
    type: isSuccess ? "success" : "warning",
    title: isSuccess ? "İşlem Tamamlandı!" : "İşlem Kısmen Tamamlandı",
    message: `${fileCount} dosya işlendi, ${chunkCount} chunk oluşturuldu. Session: ${sessionId.slice(
      0,
      8
    )}...`,
    timestamp: new Date().toISOString(),
    read: false,
    sessionId,
  };

  addGlobalNotification(notification);
  console.log(
    `🔔 Background processing notification added for session ${sessionId}`
  );
}
