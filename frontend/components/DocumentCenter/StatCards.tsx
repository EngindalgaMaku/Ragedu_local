"use client";
import React from "react";
import { MarkdownCategory } from "@/lib/api";

interface StatCardsProps {
  totalFiles: number;
  selectedFiles: number;
  totalSessions: number;
  categories: MarkdownCategory[];
}

const DocumentIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
    />
  </svg>
);

const UploadIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
    />
  </svg>
);

const SessionIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
    />
  </svg>
);

export function StatCards({
  totalFiles,
  selectedFiles,
  totalSessions,
  categories,
}: StatCardsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="bg-white/70 backdrop-blur-sm p-6 rounded-xl border border-white/50 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-100 text-blue-600 rounded-lg">
            <DocumentIcon />
          </div>
          <div>
            <p className="text-2xl font-bold text-gray-800">{totalFiles}</p>
            <p className="text-sm text-gray-600">Toplam Markdown Dosyası</p>
          </div>
        </div>
      </div>

      <div className="bg-white/70 backdrop-blur-sm p-6 rounded-xl border border-white/50 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-purple-100 text-purple-600 rounded-lg">
            <UploadIcon />
          </div>
          <div>
            <p className="text-2xl font-bold text-gray-800">{selectedFiles}</p>
            <p className="text-sm text-gray-600">Seçili Dosya</p>
          </div>
        </div>
      </div>

      <div className="bg-white/70 backdrop-blur-sm p-6 rounded-xl border border-white/50 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-green-100 text-green-600 rounded-lg">
            <SessionIcon />
          </div>
          <div>
            <p className="text-2xl font-bold text-gray-800">{totalSessions}</p>
            <p className="text-sm text-gray-600">Aktif Oturum</p>
          </div>
        </div>
      </div>
    </div>
  );
}
