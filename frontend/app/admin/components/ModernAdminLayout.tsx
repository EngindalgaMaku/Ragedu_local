"use client";

import React from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { NotificationProvider } from "./NotificationProvider";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Menu,
  X,
  Settings,
  Users,
  BarChart3,
  FileText,
  Shield,
  Home,
  LogOut,
  User,
  Bell,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

interface AdminLayoutProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
}

const sidebarItems = [
  { icon: BarChart3, label: "Dashboard", href: "/admin", active: true },
  { icon: Users, label: "Kullanıcılar", href: "/admin/users" },
  { icon: Shield, label: "Roller", href: "/admin/roles" },
  { icon: FileText, label: "Oturumlar", href: "/admin/sessions" },
  { icon: FileText, label: "RAG Testleri", href: "/admin/rag-tests" },
  { icon: FileText, label: "Markdown", href: "/admin/markdown" },
  { icon: Settings, label: "Ayarlar", href: "/admin/settings" },
];

export default function ModernAdminLayout({
  children,
  title,
  description,
}: AdminLayoutProps) {
  const { user, logout } = useAuth();
  const router = useRouter();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const handleLogout = async () => {
    try {
      await logout();
      router.push("/auth/login");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  // Redirect non-admin users
  useEffect(() => {
    if (user && user.role_name !== "admin") {
      router.push("/dashboard");
    }
  }, [user, router]);

  // Show loading while checking auth
  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  // Block non-admin users
  if (user.role_name !== "admin") {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <Card className="max-w-md">
          <CardContent className="p-8 text-center">
            <div className="w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <Shield className="w-8 h-8 text-destructive" />
            </div>
            <h2 className="text-2xl font-bold mb-2">Erişim Reddedildi</h2>
            <p className="text-muted-foreground mb-6">
              Bu sayfaya erişim yetkiniz yok. Sadece admin kullanıcıları bu
              paneli görebilir.
            </p>
            <Button onClick={() => router.push("/dashboard")}>
              Ana Sayfaya Dön
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <NotificationProvider>
      <div className="min-h-screen bg-slate-50">
        {/* Mobile sidebar overlay */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-40 lg:hidden bg-black/50 backdrop-blur-sm"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Sidebar */}
        <aside
          className={`fixed inset-y-0 left-0 z-50 ${
            sidebarCollapsed ? "w-16" : "w-64"
          } transform ${
            sidebarOpen ? "translate-x-0" : "-translate-x-full"
          } lg:translate-x-0 transition-all duration-300 ease-in-out border-r border-slate-700 bg-slate-900`}
        >
          <div className="flex flex-col h-full overflow-hidden">
            {/* Logo/Header */}
            <div
              className={`p-6 border-b border-slate-700 ${
                sidebarCollapsed ? "px-4" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                {!sidebarCollapsed && (
                  <h1 className="text-xl font-bold text-white">RAG System</h1>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                  className="hidden lg:flex text-slate-400 hover:text-white hover:bg-slate-800"
                >
                  {sidebarCollapsed ? (
                    <ChevronRight className="h-4 w-4" />
                  ) : (
                    <ChevronLeft className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-4 py-4 space-y-2">
              {sidebarItems.map((item, index) => (
                <Button
                  key={index}
                  variant="ghost"
                  className={`w-full justify-start ${
                    sidebarCollapsed ? "px-2" : "px-4"
                  } ${
                    item.active
                      ? "bg-slate-800 text-white hover:bg-slate-700"
                      : "text-slate-300 hover:text-white hover:bg-slate-800"
                  }`}
                  onClick={() => {
                    router.push(item.href);
                    setSidebarOpen(false);
                  }}
                >
                  <item.icon className="h-4 w-4 flex-shrink-0" />
                  {!sidebarCollapsed && (
                    <span className="ml-3">{item.label}</span>
                  )}
                </Button>
              ))}
            </nav>

            {/* User Profile */}
            <div
              className={`p-4 border-t border-slate-700 ${
                sidebarCollapsed ? "px-2" : ""
              }`}
            >
              <div
                className={`flex items-center ${
                  sidebarCollapsed ? "justify-center" : "space-x-3"
                }`}
              >
                <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-bold text-white">
                    {user?.first_name?.charAt(0) ||
                      user?.username?.charAt(0) ||
                      "A"}
                  </span>
                </div>
                {!sidebarCollapsed && (
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-white truncate">
                      {user?.first_name} {user?.last_name}
                    </div>
                    <Badge
                      variant="secondary"
                      className="text-xs bg-slate-700 text-slate-300 hover:bg-slate-600"
                    >
                      {user?.role_name}
                    </Badge>
                  </div>
                )}
              </div>
            </div>

            {/* Footer - Copyright */}
            {!sidebarCollapsed && (
              <div className="mt-auto border-t border-slate-700/60 p-4 bg-slate-900/50">
                <p className="text-xs text-slate-400 text-center leading-relaxed">
                  © 2025 Engin DALGA
                  <br />
                  MAKÜ Yüksek Lisans Ödevi
                </p>
              </div>
            )}
          </div>
        </aside>

        {/* Main content area */}
        <div
          className={`flex flex-col min-h-screen transition-all duration-300 ease-in-out ${
            sidebarCollapsed ? "lg:ml-16" : "lg:ml-64"
          }`}
        >
          {/* Header */}
          <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/95">
            <div className="flex h-16 items-center px-3 sm:px-4 lg:px-6 gap-2 sm:gap-4">
              {/* Mobile menu button */}
              <Button
                variant="ghost"
                size="icon"
                className="lg:hidden min-h-[44px] min-w-[44px]"
                onClick={() => setSidebarOpen(true)}
              >
                <Menu className="h-6 w-6" />
              </Button>

              {/* Page title */}
              <div className="flex-1">
                {title && (
                  <h1 className="text-lg sm:text-xl lg:text-2xl font-bold text-foreground tracking-tight">
                    {title}
                  </h1>
                )}
              </div>

              {/* Header actions */}
              <div className="flex items-center space-x-2 sm:space-x-4">
                {/* Notifications */}
                <Button variant="ghost" size="icon" className="relative min-h-[44px] min-w-[44px]">
                  <Bell className="h-5 w-5" />
                  <span className="absolute -top-1 -right-1 h-3 w-3 bg-destructive rounded-full"></span>
                </Button>

                {/* User menu */}
                <div className="relative">
                  <Button
                    variant="ghost"
                    className="pl-2 sm:pl-3 pr-2 min-h-[44px]"
                    onClick={() => setUserMenuOpen(!userMenuOpen)}
                  >
                    <div className="flex items-center space-x-2">
                      <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-sm font-bold text-primary-foreground">
                          {user?.first_name?.charAt(0) ||
                            user?.username?.charAt(0) ||
                            "A"}
                        </span>
                      </div>
                      <div className="text-left hidden sm:block">
                        <div className="text-sm font-medium">
                          {user?.first_name} {user?.last_name}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {user?.role_name}
                        </div>
                      </div>
                    </div>
                  </Button>

                  {/* Dropdown Menu */}
                  {userMenuOpen && (
                    <>
                      <div
                        className="fixed inset-0 z-40"
                        onClick={() => setUserMenuOpen(false)}
                      />
                      <Card className="absolute right-0 top-full mt-2 w-48 sm:w-56 z-50">
                        <CardContent className="p-0">
                          <div className="px-4 py-3 border-b">
                            <p className="text-sm font-semibold">
                              {user?.username}
                            </p>
                            <p className="text-xs text-muted-foreground truncate">
                              {user?.email}
                            </p>
                          </div>

                          <Button
                            variant="ghost"
                            className="w-full justify-start px-4 py-2"
                            onClick={() => {
                              router.push("/profile");
                              setUserMenuOpen(false);
                            }}
                          >
                            <User className="w-4 h-4 mr-2" />
                            Profil
                          </Button>

                          <Button
                            variant="ghost"
                            className="w-full justify-start px-4 py-2"
                            onClick={() => {
                              router.push("/dashboard");
                              setUserMenuOpen(false);
                            }}
                          >
                            <Home className="w-4 h-4 mr-2" />
                            Ana Sayfa
                          </Button>

                          <div className="border-t my-1"></div>

                          <Button
                            variant="ghost"
                            className="w-full justify-start px-4 py-2 text-destructive hover:text-destructive"
                            onClick={handleLogout}
                          >
                            <LogOut className="w-4 h-4 mr-2" />
                            Çıkış Yap
                          </Button>
                        </CardContent>
                      </Card>
                    </>
                  )}
                </div>
              </div>
            </div>
          </header>

          {/* Page content */}
          <main className="flex-1 p-3 sm:p-4 lg:p-6 bg-slate-50">{children}</main>
        </div>
      </div>
    </NotificationProvider>
  );
}
