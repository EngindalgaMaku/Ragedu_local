"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import {
  LayoutDashboard,
  BookOpen,
  FolderOpen,
  Bot,
  Menu,
  X,
  ChevronLeft,
  LogOut,
  User,
  Home,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
// Dropdown menu components will be created inline

type TabType = "dashboard" | "sessions" | "upload" | "query";

interface TeacherLayoutProps {
  children: React.ReactNode;
  activeTab?: TabType;
  onTabChange?: (tab: TabType) => void;
}

const navigationItems: Array<{
  id: TabType;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
  desc: string;
}> = [
  {
    id: "dashboard",
    name: "Kontrol Paneli",
    icon: LayoutDashboard,
    desc: "Genel Bakış",
  },
  {
    id: "sessions",
    name: "Ders Oturumları",
    icon: BookOpen,
    desc: "Sınıf Yönetimi",
  },
  {
    id: "upload",
    name: "Belge Merkezi",
    icon: FolderOpen,
    desc: "Materyal Yükleme",
  },
  {
    id: "query",
    name: "Eğitim Asistanı",
    icon: Bot,
    desc: "Soru-Cevap",
  },
];

function TeacherLayout({
  children,
  activeTab = "dashboard",
  onTabChange,
}: TeacherLayoutProps) {
  const { user, logout } = useAuth();
  const router = useRouter();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const handleLogout = async () => {
    try {
      await logout();
      router.push("/login");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const handleTabClick = (tabId: TabType) => {
    if (onTabChange) {
      onTabChange(tabId);
    }
    setSidebarOpen(false);
  };

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 overflow-hidden">
      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 ${
          sidebarCollapsed ? "w-16" : "w-64"
        } transform ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0 transition-all duration-300 ease-in-out bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col`}
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200 dark:border-gray-700">
          {!sidebarCollapsed && (
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">RAG</span>
              </div>
              <div>
                <h1 className="text-sm font-bold text-gray-900 dark:text-white">
                  Eğitim Asistanı
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Öğretmen Paneli
                </p>
              </div>
            </div>
          )}
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="hidden lg:flex h-8 w-8"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            >
              {sidebarCollapsed ? (
                <ChevronLeft className="h-4 w-4" />
              ) : (
                <ChevronLeft className="h-4 w-4 rotate-180" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden h-8 w-8"
              onClick={() => setSidebarOpen(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;

            return (
              <button
                key={item.id}
                onClick={() => handleTabClick(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 ${
                  isActive
                    ? "bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-md"
                    : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                } ${sidebarCollapsed ? "justify-center" : ""}`}
                title={sidebarCollapsed ? item.name : undefined}
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {!sidebarCollapsed && (
                  <div className="flex-1 text-left">
                    <div className="text-sm font-medium">{item.name}</div>
                    <div
                      className={`text-xs ${
                        isActive
                          ? "text-white/80"
                          : "text-gray-500 dark:text-gray-400"
                      }`}
                    >
                      {item.desc}
                    </div>
                  </div>
                )}
              </button>
            );
          })}
        </nav>

        {/* User Profile & Footer */}
        <div className="border-t border-gray-200 dark:border-gray-700 p-4 space-y-4">
          {/* User Profile */}
          {!sidebarCollapsed ? (
            <div className="relative">
              <Button
                variant="ghost"
                className="w-full justify-start gap-3 h-auto p-2 hover:bg-gray-100 dark:hover:bg-gray-700"
                onClick={() => setUserMenuOpen(!userMenuOpen)}
              >
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-white text-xs font-bold">
                    {user?.first_name?.charAt(0) ||
                      user?.username?.charAt(0) ||
                      "U"}
                  </span>
                </div>
                <div className="flex-1 text-left min-w-0">
                  <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {user?.first_name} {user?.last_name}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                    {user?.email}
                  </div>
                </div>
              </Button>
              {userMenuOpen && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setUserMenuOpen(false)}
                  />
                  <Card className="absolute bottom-full left-0 mb-2 w-56 z-50 shadow-xl">
                    <div className="p-1">
                      <Button
                        variant="ghost"
                        className="w-full justify-start gap-2"
                        onClick={() => {
                          router.push("/profile");
                          setUserMenuOpen(false);
                        }}
                      >
                        <User className="h-4 w-4" />
                        Profil
                      </Button>
                      <Button
                        variant="ghost"
                        className="w-full justify-start gap-2"
                        onClick={() => {
                          router.push("/");
                          setUserMenuOpen(false);
                        }}
                      >
                        <Home className="h-4 w-4" />
                        Ana Sayfa
                      </Button>
                      <div className="border-t my-1" />
                      <Button
                        variant="ghost"
                        className="w-full justify-start gap-2 text-red-600 hover:text-red-700 hover:bg-red-50"
                        onClick={handleLogout}
                      >
                        <LogOut className="h-4 w-4" />
                        Çıkış Yap
                      </Button>
                    </div>
                  </Card>
                </>
              )}
            </div>
          ) : (
            <div className="flex justify-center relative">
              <Button
                variant="ghost"
                size="icon"
                className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600"
                onClick={() => setUserMenuOpen(!userMenuOpen)}
              >
                <span className="text-white text-xs font-bold">
                  {user?.first_name?.charAt(0) ||
                    user?.username?.charAt(0) ||
                    "U"}
                </span>
              </Button>
              {userMenuOpen && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setUserMenuOpen(false)}
                  />
                  <Card className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 z-50 shadow-xl">
                    <div className="p-1">
                      <Button
                        variant="ghost"
                        className="w-full justify-start gap-2"
                        onClick={() => {
                          router.push("/profile");
                          setUserMenuOpen(false);
                        }}
                      >
                        <User className="h-4 w-4" />
                        Profil
                      </Button>
                      <Button
                        variant="ghost"
                        className="w-full justify-start gap-2"
                        onClick={() => {
                          router.push("/");
                          setUserMenuOpen(false);
                        }}
                      >
                        <Home className="h-4 w-4" />
                        Ana Sayfa
                      </Button>
                      <div className="border-t my-1" />
                      <Button
                        variant="ghost"
                        className="w-full justify-start gap-2 text-red-600 hover:text-red-700 hover:bg-red-50"
                        onClick={handleLogout}
                      >
                        <LogOut className="h-4 w-4" />
                        Çıkış Yap
                      </Button>
                    </div>
                  </Card>
                </>
              )}
            </div>
          )}

          {/* Footer - Copyright */}
          {!sidebarCollapsed && (
            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <p className="text-xs text-gray-500 dark:text-gray-400 text-center leading-relaxed">
                © 2025 Engin DALGA
                <br />
                MAKÜ Yüksek Lisans Ödevi
              </p>
            </div>
          )}
        </div>
      </aside>

      {/* Top Header - Fixed Position */}
      <header 
        className="fixed top-0 right-0 h-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between z-40"
        style={{
          left: sidebarCollapsed ? '4rem' : '16rem',
          paddingLeft: '1.5rem',
          paddingRight: '1.5rem'
        }}
      >
        <div className="flex items-center gap-4 flex-1">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-5 w-5" />
          </Button>
        </div>
        <div className="flex items-center gap-2">
          {/* User Menu Dropdown */}
          <div className="relative">
            <Button
              variant="ghost"
              className="flex items-center gap-2 h-auto p-2 hover:bg-gray-100 dark:hover:bg-gray-700"
              onClick={() => setUserMenuOpen(!userMenuOpen)}
            >
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white text-xs font-bold">
                  {user?.first_name?.charAt(0) ||
                    user?.username?.charAt(0) ||
                    "U"}
                </span>
              </div>
              <div className="hidden md:block text-left">
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                  {user?.first_name} {user?.last_name}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {user?.email}
                </div>
              </div>
            </Button>
            {userMenuOpen && (
              <>
                <div
                  className="fixed inset-0 z-40"
                  onClick={() => setUserMenuOpen(false)}
                />
                <Card className="absolute top-full right-0 mt-2 w-56 z-50 shadow-xl">
                  <div className="p-1">
                    <Button
                      variant="ghost"
                      className="w-full justify-start gap-2"
                      onClick={() => {
                        router.push("/profile");
                        setUserMenuOpen(false);
                      }}
                    >
                      <User className="h-4 w-4" />
                      Profil
                    </Button>
                    <Button
                      variant="ghost"
                      className="w-full justify-start gap-2"
                      onClick={() => {
                        router.push("/");
                        setUserMenuOpen(false);
                      }}
                    >
                      <Home className="h-4 w-4" />
                      Ana Sayfa
                    </Button>
                    <div className="border-t my-1" />
                    <Button
                      variant="ghost"
                      className="w-full justify-start gap-2 text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-400 dark:hover:text-red-300 dark:hover:bg-red-900/20"
                      onClick={handleLogout}
                    >
                      <LogOut className="h-4 w-4" />
                      Çıkış Yap
                    </Button>
                  </div>
                </Card>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div 
        className="flex-1 flex flex-col transition-all duration-300 min-w-0"
        style={{
          marginLeft: sidebarCollapsed ? '4rem' : '16rem',
          marginTop: '4rem'
        }}
      >
        {/* Page Content */}
        <main className="flex-1 overflow-y-auto px-4 lg:px-6 py-0 bg-gray-50 dark:bg-gray-900">
          {children}
        </main>
      </div>
    </div>
  );
}

export default TeacherLayout;
