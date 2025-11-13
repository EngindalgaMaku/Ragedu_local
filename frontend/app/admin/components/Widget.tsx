"use client";

import React from "react";

interface WidgetProps {
  title: string;
  children: React.ReactNode;
  className?: string;
  actions?: React.ReactNode;
  loading?: boolean;
  error?: string | null;
  onRefresh?: () => void;
}

export default function Widget({
  title,
  children,
  className = "",
  actions,
  loading = false,
  error,
  onRefresh,
}: WidgetProps) {
  return (
    <div
      className={`admin-surface rounded-lg transition-all duration-300 overflow-hidden ${className}`}
    >
      {/* Widget Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/60 bg-slate-800/50 backdrop-blur">
        <h3 className="text-sm font-bold text-slate-100 tracking-tight">
          {title}
        </h3>

        <div className="flex items-center space-x-2">
          {/* Refresh Button */}
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="p-1.5 text-slate-300 hover:text-white hover:bg-slate-700/60 rounded transition-colors duration-200"
              disabled={loading}
            >
              <svg
                className={`w-4 h-4 ${loading ? "animate-spin" : ""}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
            </button>
          )}

          {/* Custom Actions */}
          {actions}
        </div>
      </div>

      {/* Widget Content */}
      <div className="p-4">
        {error ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <svg
                className="mx-auto h-12 w-12 text-red-400 mb-3"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="text-base font-medium text-red-400 mb-2">{error}</p>
              {onRefresh && (
                <button
                  onClick={onRefresh}
                  className="mt-3 px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                >
                  Tekrar Dene
                </button>
              )}
            </div>
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="flex flex-col items-center gap-2">
              <div className="animate-spin rounded-full h-8 w-8 border-4 border-gray-200 border-t-indigo-600"></div>
              <p className="text-xs text-gray-500 dark:text-gray-400">Yükleniyor...</p>
            </div>
          </div>
        ) : (
          children
        )}
      </div>
    </div>
  );
}
