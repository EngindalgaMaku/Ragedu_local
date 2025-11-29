import { NextRequest, NextResponse } from "next/server";

// Test bildirimlerini oluşturmak için basit endpoint
export async function POST(request: NextRequest) {
  try {
    // Kendi pending endpoint'imizi çağırarak test bildirimi oluştur
    const testNotification = {
      type: "success" as const,
      title: "Test Bildirimi",
      message: `Test bildirimi oluşturuldu: ${new Date().toLocaleString(
        "tr-TR"
      )}`,
      sessionId: "test-session-123",
    };

    // Internal API çağrısı yap
    const baseUrl = request.nextUrl.origin;
    const createResponse = await fetch(
      `${baseUrl}/api/v1/notifications/pending`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(testNotification),
      }
    );

    if (!createResponse.ok) {
      throw new Error("Failed to create test notification");
    }

    const result = await createResponse.json();

    console.log("🧪 Test notification created successfully");

    return NextResponse.json({
      success: true,
      message: "Test bildirimi başarıyla oluşturuldu",
      notification: result.notification,
    });
  } catch (error) {
    console.error("Error creating test notification:", error);
    return NextResponse.json(
      { error: "Failed to create test notification" },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const count = parseInt(searchParams.get("count") || "1");

    const notifications = [];

    for (let i = 0; i < Math.min(count, 10); i++) {
      const types = ["success", "info", "warning", "error"] as const;
      const type = types[Math.floor(Math.random() * types.length)];

      const testNotification = {
        type,
        title: `Test Bildirimi ${i + 1}`,
        message: `Bu ${
          i + 1
        }. test bildirimidir. Oluşturulma zamanı: ${new Date().toLocaleString(
          "tr-TR"
        )}`,
        sessionId: "test-session-" + Math.random().toString(36).substr(2, 9),
      };

      // Internal API çağrısı yap
      const baseUrl = request.nextUrl.origin;
      const createResponse = await fetch(
        `${baseUrl}/api/v1/notifications/pending`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(testNotification),
        }
      );

      if (createResponse.ok) {
        const result = await createResponse.json();
        notifications.push(result.notification);
      }

      // Kısa bekleme
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    console.log(
      `🧪 ${notifications.length} test notifications created successfully`
    );

    return NextResponse.json({
      success: true,
      message: `${notifications.length} test bildirimi oluşturuldu`,
      notifications,
    });
  } catch (error) {
    console.error("Error creating test notifications:", error);
    return NextResponse.json(
      { error: "Failed to create test notifications" },
      { status: 500 }
    );
  }
}
