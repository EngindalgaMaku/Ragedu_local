"use client";

import React, { Suspense } from "react";
import dynamic from "next/dynamic";
import remarkGfm from "remark-gfm";

// Lazy-load react-markdown to reduce render blocking on chat page
const ReactMarkdown = dynamic(() => import("react-markdown"), {
  ssr: false,
  loading: () => <div className="h-6 w-28 animate-pulse rounded bg-muted" />,
});

interface MarkdownViewerProps {
  content: string;
  isLoading?: boolean;
  error?: string | null;
}

export default function MarkdownViewer({
  content,
  isLoading = false,
  error = null,
}: MarkdownViewerProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="relative mx-auto w-12 h-12 mb-4">
            <div className="absolute inset-0 rounded-full border-4 border-gray-200"></div>
            <div className="absolute inset-0 rounded-full border-4 border-blue-500 border-t-transparent animate-spin"></div>
          </div>
          <p className="text-gray-600 animate-pulse">
            Markdown içeriği yükleniyor...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-center">
          <svg
            className="w-5 h-5 text-red-500 mr-2 flex-shrink-0"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
          <div>
            <h3 className="text-red-800 font-medium">Hata</h3>
            <p className="text-red-700 text-sm mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!content || content.trim() === "") {
    return (
      <div className="text-center py-12 text-gray-500">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
          <svg
            className="w-8 h-8 text-gray-400"
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
        </div>
        <p className="text-sm">İçerik bulunamadı</p>
      </div>
    );
  }

  return (
    <Suspense fallback={<div className="h-6 w-28 animate-pulse rounded bg-muted" />}>
      <div className="prose prose-lg max-w-none prose-headings:font-bold prose-headings:text-foreground prose-a:text-primary hover:prose-a:text-primary/80 prose-strong:text-foreground prose-blockquote:border-l-primary prose-blockquote:bg-primary/5 prose-blockquote:text-muted-foreground prose-code:bg-muted prose-code:text-foreground prose-code:rounded-md prose-code:px-1.5 prose-code:py-1 prose-pre:bg-secondary prose-pre:text-secondary-foreground prose-pre:rounded-xl">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </Suspense>
  );
}
