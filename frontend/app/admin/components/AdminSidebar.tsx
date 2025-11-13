"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";

interface AdminSidebarProps {
  onClose?: () => void;
  isMinimized?: boolean;
  onToggleMinimize?: () => void;
}

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ReactNode;
}

interface NavigationSection {
  name: string;
  items: NavigationItem[];
  icon: React.ReactNode;
}

const singleNavigation = [
  {
    name: "Kontrol Paneli",
    href: "/admin",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z"
        />
      </svg>
    ),
  },
  {
    name: "Belge Merkezi",
    href: "/admin/markdown",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
      </svg>
    ),
  },
];

const sectionNavigation: NavigationSection[] = [
  {
    name: "Kullanıcı yönetimi",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
        />
      </svg>
    ),
    items: [
      {
        name: "Kullanıcılar",
        href: "/admin/users",
        icon: (
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
              d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z"
            />
          </svg>
        ),
      },
      {
        name: "Roller",
        href: "/admin/roles",
        icon: (
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
              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
            />
          </svg>
        ),
      },
      {
        name: "Oturumlar",
        href: "/admin/sessions",
        icon: (
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
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
            />
          </svg>
        ),
      },
    ],
  },
  {
    name: "Testler",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
        />
      </svg>
    ),
    items: [
      {
        name: "RAG Testler",
        href: "/admin/rag-tests",
        icon: (
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
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
            />
          </svg>
        ),
      },
    ],
  },
];

export default function AdminSidebar({
  onClose,
  isMinimized = false,
  onToggleMinimize,
}: AdminSidebarProps) {
  const pathname = usePathname();
  const { user } = useAuth();
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set()
  );

  const handleLinkClick = () => {
    if (onClose) onClose();
  };

  const toggleSection = (sectionName: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionName)) {
      newExpanded.delete(sectionName);
    } else {
      newExpanded.add(sectionName);
    }
    setExpandedSections(newExpanded);
  };

  const isItemActive = (href: string) => {
    return (
      pathname === href || (pathname?.startsWith(href) && href !== "/admin")
    );
  };

  const isSectionActive = (section: NavigationSection) => {
    return section.items.some((item) => isItemActive(item.href));
  };

  return (
    <div
      className={`flex flex-col h-full bg-slate-900 text-slate-200 border-r border-slate-800 transition-all duration-300 ${
        isMinimized ? "w-16" : "w-64"
      }`}
    >
      {/* Logo/Brand */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-slate-800">
        <div className="flex items-center space-x-3">
          <div className="relative w-10 h-10 bg-gradient-to-br from-indigo-500 via-purple-600 to-pink-500 rounded-xl flex items-center justify-center shadow-md">
            <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent rounded-xl"></div>
            <span className="text-white font-bold text-sm relative z-10">
              {isMinimized ? "R" : "RA"}
            </span>
          </div>
          {!isMinimized && (
            <div className="flex flex-col">
              <span className="text-lg font-bold text-slate-100">
                RAG Sistemi
              </span>
              <span className="text-xs text-slate-400 font-medium">
                Yönetim Paneli
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-1">
          {/* Minimize/Maximize button for desktop */}
          {onToggleMinimize && (
            <button
              type="button"
              className="hidden lg:flex -mr-1 h-8 w-8 rounded-lg items-center justify-center text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-all duration-200"
              onClick={onToggleMinimize}
              title={isMinimized ? "Genişlet" : "Küçült"}
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
                  d={
                    isMinimized
                      ? "M13 7l5 5-5 5M6 12h12"
                      : "M11 17l-5-5 5-5M18 12H6"
                  }
                />
              </svg>
            </button>
          )}

          {/* Close button for mobile */}
          <button
            type="button"
            className="lg:hidden -mr-2 h-10 w-10 rounded-xl flex items-center justify-center text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-all duration-200"
            onClick={onClose}
          >
            <span className="sr-only">Kenar çubuğunu kapat</span>
            <svg
              className="w-6 h-6"
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

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-3 overflow-y-auto">
        {/* Single Navigation Items */}
        {singleNavigation.map((item) => {
          const isActive = isItemActive(item.href);

          return (
            <Link
              key={item.name}
              href={item.href}
              onClick={handleLinkClick}
              title={isMinimized ? item.name : undefined}
              className={`
                group relative flex items-center ${isMinimized ? "justify-center px-3" : "px-4"} py-3 text-sm font-semibold rounded-xl transition-colors duration-200
                ${isActive ? "bg-slate-800 text-white" : "text-slate-300 hover:bg-slate-800 hover:text-white"}
              `}
            >
              <span
                className={`${
                  isMinimized ? "" : "mr-4"
                } transition-all duration-200 ${
                  isActive
                    ? "text-white/90"
                    : "text-gray-500 dark:text-gray-400 group-hover:text-indigo-500 dark:group-hover:text-indigo-400"
                }`}
              >
                {item.icon}
              </span>
              {!isMinimized && (
                <span className="font-medium tracking-wide">{item.name}</span>
              )}

              {/* Active indicator */}
              {isActive && <div className={`absolute ${isMinimized ? "right-1" : "right-3"} w-2 h-2 bg-indigo-400 rounded-full`}></div>}

              {/* Hover glow effect */}
              {!isActive && <div className="absolute inset-0 rounded-xl"></div>}
            </Link>
          );
        })}

        {/* Section Navigation Items */}
        {sectionNavigation.map((section) => {
          const isExpanded = expandedSections.has(section.name) && !isMinimized;
          const sectionActive = isSectionActive(section);

          if (isMinimized) {
            // In minimized mode, show section items as individual buttons
            return (
              <div key={section.name} className="space-y-1">
                {section.items.map((item) => {
                  const isActive = isItemActive(item.href);
                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      onClick={handleLinkClick}
                      title={item.name}
                      className={`
                        group relative flex items-center justify-center px-3 py-3 text-sm font-semibold rounded-xl transition-all duration-200 transform hover:scale-[1.02] hover:shadow-lg
                        ${
                          isActive
                            ? "bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/25 dark:shadow-indigo-500/20"
                            : "text-gray-700 hover:bg-white/70 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-white hover:shadow-md backdrop-blur-sm border border-transparent hover:border-gray-200/50 dark:hover:border-gray-600/30"
                        }
                      `}
                    >
                      <span
                        className={`transition-all duration-200 ${
                          isActive
                            ? "text-white/90"
                            : "text-gray-500 dark:text-gray-400 group-hover:text-indigo-500 dark:group-hover:text-indigo-400"
                        }`}
                      >
                        {item.icon}
                      </span>

                      {/* Active indicator */}
                      {isActive && (
                        <div className="absolute right-1 w-2 h-2 bg-white/80 rounded-full shadow-sm animate-pulse"></div>
                      )}

                      {/* Hover glow effect */}
                      {!isActive && (
                        <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-indigo-500/0 via-purple-500/0 to-pink-500/0 group-hover:from-indigo-500/5 group-hover:via-purple-500/5 group-hover:to-pink-500/5 transition-all duration-300"></div>
                      )}
                    </Link>
                  );
                })}
              </div>
            );
          }

          return (
            <div key={section.name} className="space-y-1">
              {/* Section Header */}
              <button
                onClick={() => toggleSection(section.name)}
                className={`
                  group relative flex items-center w-full px-4 py-3 text-sm font-semibold rounded-xl transition-all duration-200 transform hover:scale-[1.02] hover:shadow-lg
                  ${
                    sectionActive
                      ? "bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/25 dark:shadow-indigo-500/20"
                      : "text-gray-700 hover:bg-white/70 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-white hover:shadow-md backdrop-blur-sm border border-transparent hover:border-gray-200/50 dark:hover:border-gray-600/30"
                  }
                `}
              >
                <span
                  className={`mr-4 transition-all duration-200 ${
                    sectionActive
                      ? "text-white/90"
                      : "text-gray-500 dark:text-gray-400 group-hover:text-indigo-500 dark:group-hover:text-indigo-400"
                  }`}
                >
                  {section.icon}
                </span>
                <span className="font-medium tracking-wide flex-1 text-left">
                  {section.name}
                </span>

                {/* Chevron icon */}
                <svg
                  className={`w-4 h-4 transition-transform duration-200 ${
                    isExpanded ? "rotate-180" : ""
                  } ${
                    sectionActive
                      ? "text-white/90"
                      : "text-gray-500 dark:text-gray-400"
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>

                {/* Active indicator */}
                {sectionActive && (
                  <div className="absolute right-10 w-2 h-2 bg-white/80 rounded-full shadow-sm animate-pulse"></div>
                )}

                {/* Hover glow effect */}
                {!sectionActive && (
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-indigo-500/0 via-purple-500/0 to-pink-500/0 group-hover:from-indigo-500/5 group-hover:via-purple-500/5 group-hover:to-pink-500/5 transition-all duration-300"></div>
                )}
              </button>

              {/* Section Items */}
              {isExpanded && (
                <div className="ml-4 space-y-1">
                  {section.items.map((item) => {
                    const isActive = isItemActive(item.href);

                    return (
                      <Link
                        key={item.name}
                        href={item.href}
                        onClick={handleLinkClick}
                        className={`
                          group relative flex items-center px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 transform hover:scale-[1.02]
                          ${
                            isActive
                              ? "bg-gradient-to-r from-indigo-400 to-purple-500 text-white shadow-md shadow-indigo-400/20"
                              : "text-gray-600 hover:bg-white/60 hover:text-gray-800 dark:text-gray-400 dark:hover:bg-gray-800/40 dark:hover:text-white hover:shadow-sm backdrop-blur-sm border border-transparent hover:border-gray-200/40 dark:hover:border-gray-600/20"
                          }
                        `}
                      >
                        <span
                          className={`mr-3 transition-all duration-200 ${
                            isActive
                              ? "text-white/90"
                              : "text-gray-400 dark:text-gray-500 group-hover:text-indigo-400 dark:group-hover:text-indigo-300"
                          }`}
                        >
                          {item.icon}
                        </span>
                        <span className="font-medium">{item.name}</span>

                        {/* Active indicator */}
                        {isActive && (
                          <div className="absolute right-2 w-1.5 h-1.5 bg-white/80 rounded-full shadow-sm animate-pulse"></div>
                        )}

                        {/* Hover glow effect */}
                        {!isActive && (
                          <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-indigo-400/0 via-purple-400/0 to-pink-400/0 group-hover:from-indigo-400/5 group-hover:via-purple-400/5 group-hover:to-pink-400/5 transition-all duration-300"></div>
                        )}
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </nav>

      {/* User info directly below navigation */}
      <div className="border-t border-gray-200/50 dark:border-gray-700/50 p-3 bg-gradient-to-t from-gray-50/50 to-transparent dark:from-gray-800/30">
        {!isMinimized ? (
          <div className="flex items-center space-x-3 p-2 rounded-lg bg-white/30 dark:bg-gray-800/30 backdrop-blur-sm">
            <div className="relative w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">
                {user?.first_name?.charAt(0) ||
                  user?.username?.charAt(0) ||
                  "S"}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-xs font-semibold text-gray-900 dark:text-white truncate">
                {user?.first_name && user?.last_name
                  ? `${user.first_name} ${user.last_name}`
                  : "System Administrator"}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                {user?.email || "admin@rag-assistant.local"}
              </div>
              <div className="flex items-center space-x-1 mt-0.5">
                <div className="w-1.5 h-1.5 bg-green-400 rounded-full"></div>
                <span className="text-xs text-green-600 dark:text-green-400">
                  Aktif
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex justify-center">
            <div className="relative w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">
                {user?.first_name?.charAt(0) ||
                  user?.username?.charAt(0) ||
                  "S"}
              </span>
              <div className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-green-400 border-2 border-white dark:border-gray-800 rounded-full"></div>
            </div>
          </div>
        )}
      </div>

      {/* Footer - Copyright */}
      {!isMinimized && (
        <div className="mt-auto border-t border-slate-800/60 p-4 bg-slate-900/50">
          <p className="text-xs text-slate-400 text-center leading-relaxed">
            © 2025 Engin DALGA
            <br />
            MAKÜ Yüksek Lisans Ödevi
          </p>
        </div>
      )}
    </div>
  );
}
