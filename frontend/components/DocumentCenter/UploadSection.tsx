"use client";
import React, { useState } from "react";

interface UploadSectionProps {
  onOpenUploadModal?: () => void;
}

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

export function UploadSection({ onOpenUploadModal }: UploadSectionProps) {
  return (
    <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-8 text-white mb-8 shadow-xl">
      <div className="flex flex-col md:flex-row items-center justify-between">
        <div className="flex items-center gap-6 mb-6 md:mb-0">
          <div className="p-4 bg-white/20 rounded-2xl backdrop-blur-sm">
            <DocumentIcon />
          </div>
          <div>
            <h3 className="text-2xl font-bold mb-2">
              PDF'den Markdown'a Dönüştür
            </h3>
            <p className="text-blue-100">
              PDF dosyalarını otomatik olarak Markdown formatına dönüştürün ve
              kategorize edin
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          {onOpenUploadModal && (
            <button
              onClick={onOpenUploadModal}
              className="bg-white text-blue-600 px-6 py-3 rounded-xl font-semibold hover:bg-blue-50 transition-all transform hover:scale-105 shadow-lg flex items-center gap-3 min-w-[200px] justify-center"
            >
              <UploadIcon />
              <span>PDF Yükle</span>
            </button>
          )}

          <div className="text-center">
            <p className="text-xs text-blue-200">
              PDF dosyalarını Markdown'a dönüştürür
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
