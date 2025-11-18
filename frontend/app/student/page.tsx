"use client";

import React from "react";
import { useAuth, useRequireAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import StudentDashboard from "@/components/StudentDashboard";
import LogoutButton from "@/components/LogoutButton";

export default function StudentPage() {
  const auth = useRequireAuth();
  const router = useRouter();

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
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">📚</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">RAG Eğitim Asistanı</h1>
              <p className="text-xs text-gray-500">Öğrenci Paneli</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <a
              href="/profile"
              className="text-sm font-medium text-gray-700 hover:text-blue-600 transition-colors"
            >
              Profil
            </a>
            <LogoutButton />
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="py-6">
        <StudentDashboard userId={auth.user.id.toString()} />
      </main>
    </div>
  );
}

