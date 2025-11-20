"use client";

import React from "react";
import { useAuth, useRequireAuth } from "@/contexts/AuthContext";
import { useRouter, usePathname } from "next/navigation";
import LogoutButton from "@/components/LogoutButton";
import Link from "next/link";
import { LayoutDashboard, MessageSquare } from "lucide-react";

export default function StudentLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const auth = useRequireAuth();
  const router = useRouter();
  const pathname = usePathname();

  // Redirect if not a student
  React.useEffect(() => {
    if (auth.user && auth.user.role_name !== "student") {
      // Redirect to appropriate page based on role
      if (auth.user.role_name === "admin") {
        router.push("/admin");
      } else if (auth.user.role_name === "teacher") {
        router.push("/");
      } else {
        router.push("/");
      }
    }
  }, [auth.user, router]);

  if (auth.isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (!auth.user || auth.user.role_name !== "student") {
    return null; // Will redirect
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Navigation Bar */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">📚</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">RAG Temelli Eğitim Asistanı</h1>
                <p className="text-xs text-gray-500">Yapay Zeka Destekli Öğretim Platformu</p>
              </div>
            </div>
            
          <div className="flex items-center gap-4">
            <LogoutButton />
          </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex gap-2 border-t border-gray-200 pt-3">
            <Link
              href="/student"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                pathname === "/student"
                  ? "bg-blue-100 text-blue-700"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              <LayoutDashboard className="w-4 h-4" />
              Dashboard
            </Link>
            <Link
              href="/student/chat"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                pathname === "/student/chat"
                  ? "bg-blue-100 text-blue-700"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              Soru & Cevap
            </Link>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main>{children}</main>
    </div>
  );
}


