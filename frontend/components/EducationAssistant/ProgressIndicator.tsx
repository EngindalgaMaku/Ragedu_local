"use client";

import React from "react";

interface ProgressIndicatorProps {
  isLoading: boolean;
  loadingMessage?: string;
  progress?: number; // 0-100
  stage?: string;
  type?: "spinner" | "bar" | "dots" | "pulse";
  size?: "sm" | "md" | "lg";
  className?: string;
}

export default function ProgressIndicator({
  isLoading,
  loadingMessage = "İşlem devam ediyor...",
  progress,
  stage,
  type = "spinner",
  size = "md",
  className = "",
}: ProgressIndicatorProps) {
  if (!isLoading) {
    return null;
  }

  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-6 h-6",
    lg: "w-8 h-8",
  };

  const containerSizeClasses = {
    sm: "p-2",
    md: "p-4",
    lg: "p-6",
  };

  const SpinnerIndicator = () => (
    <div className="flex items-center justify-center space-x-3">
      <div className={`${sizeClasses[size]} animate-spin`}>
        <svg
          className="w-full h-full text-blue-600"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      </div>
      <div className="text-gray-700">
        <div className="font-medium">{loadingMessage}</div>
        {stage && <div className="text-sm text-gray-500">{stage}</div>}
      </div>
    </div>
  );

  const ProgressBar = () => (
    <div className="w-full space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-700">
          {loadingMessage}
        </span>
        {progress !== undefined && (
          <span className="text-sm text-gray-500">{Math.round(progress)}%</span>
        )}
      </div>

      <div className="w-full bg-gray-200 rounded-full h-2.5">
        <div
          className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2.5 rounded-full transition-all duration-300 ease-out"
          style={{
            width: progress !== undefined ? `${progress}%` : "0%",
            animation:
              progress === undefined
                ? "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite"
                : undefined,
          }}
        />
      </div>

      {stage && (
        <div className="text-xs text-gray-500 text-center">{stage}</div>
      )}
    </div>
  );

  const DotsIndicator = () => (
    <div className="flex items-center justify-center space-x-2">
      <div className="text-gray-700">
        <div className="font-medium">{loadingMessage}</div>
        {stage && <div className="text-sm text-gray-500">{stage}</div>}
      </div>
      <div className="flex space-x-1">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className={`${sizeClasses[size]} bg-blue-500 rounded-full animate-bounce`}
            style={{
              animationDelay: `${i * 0.1}s`,
              animationDuration: "0.6s",
            }}
          />
        ))}
      </div>
    </div>
  );

  const PulseIndicator = () => (
    <div className="flex items-center justify-center space-x-3">
      <div
        className={`${sizeClasses[size]} bg-blue-500 rounded-full animate-pulse`}
      />
      <div className="text-gray-700">
        <div className="font-medium">{loadingMessage}</div>
        {stage && <div className="text-sm text-gray-500">{stage}</div>}
      </div>
    </div>
  );

  const renderIndicator = () => {
    switch (type) {
      case "bar":
        return <ProgressBar />;
      case "dots":
        return <DotsIndicator />;
      case "pulse":
        return <PulseIndicator />;
      case "spinner":
      default:
        return <SpinnerIndicator />;
    }
  };

  return (
    <div
      className={`bg-white border border-gray-200 rounded-lg shadow-sm ${containerSizeClasses[size]} ${className}`}
    >
      {renderIndicator()}
    </div>
  );
}

// Specialized components for common use cases
export const QueryLoadingIndicator = ({
  isLoading,
  message = "Sorgunuz işleniyor...",
}: {
  isLoading: boolean;
  message?: string;
}) => (
  <ProgressIndicator
    isLoading={isLoading}
    loadingMessage={message}
    type="spinner"
    size="md"
    stage="Kaynaklardan bilgi toplanıyor"
    className="bg-blue-50 border-blue-200"
  />
);

export const SuggestionsLoadingIndicator = ({
  isLoading,
}: {
  isLoading: boolean;
}) => (
  <ProgressIndicator
    isLoading={isLoading}
    loadingMessage="Öneriler hazırlanıyor..."
    type="dots"
    size="sm"
    stage="AI önerileri oluşturuluyor"
    className="bg-green-50 border-green-200"
  />
);

export const ModelLoadingIndicator = ({
  isLoading,
  modelName,
}: {
  isLoading: boolean;
  modelName?: string;
}) => (
  <ProgressIndicator
    isLoading={isLoading}
    loadingMessage="Model yükleniyor..."
    type="pulse"
    size="md"
    stage={modelName ? `${modelName} hazırlanıyor` : "Model başlatılıyor"}
    className="bg-purple-50 border-purple-200"
  />
);

// Progress stages for different operations
export const RAGProcessingIndicator = ({
  isLoading,
  stage: currentStage,
}: {
  isLoading: boolean;
  stage?: number;
}) => {
  const stages = [
    "Sorgu analiz ediliyor...",
    "Dokümanlar taranıyor...",
    "En uygun kaynaklar seçiliyor...",
    "AI yanıtı oluşturuluyor...",
    "Sonuç hazırlanıyor...",
  ];

  const progress =
    currentStage !== undefined
      ? ((currentStage + 1) / stages.length) * 100
      : undefined;
  const stageMessage =
    currentStage !== undefined && currentStage < stages.length
      ? stages[currentStage]
      : undefined;

  return (
    <ProgressIndicator
      isLoading={isLoading}
      loadingMessage="RAG İşlemi"
      progress={progress}
      stage={stageMessage}
      type="bar"
      size="lg"
      className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200"
    />
  );
};

// Full-screen overlay loading
export const FullScreenLoader = ({
  isLoading,
  message = "Yükleniyor...",
}: {
  isLoading: boolean;
  message?: string;
}) => {
  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-8 shadow-2xl">
        <ProgressIndicator
          isLoading={true}
          loadingMessage={message}
          type="spinner"
          size="lg"
          className="border-0 shadow-none"
        />
      </div>
    </div>
  );
};

// Inline loading for buttons
export const ButtonLoader = ({
  isLoading,
  size = "sm",
}: {
  isLoading: boolean;
  size?: "sm" | "md";
}) => {
  if (!isLoading) return null;

  const spinnerSize = size === "sm" ? "w-4 h-4" : "w-5 h-5";

  return (
    <div className={`${spinnerSize} animate-spin`}>
      <svg
        className="w-full h-full text-current"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
    </div>
  );
};

// Skeleton loading for content
export const ContentSkeleton = ({
  lines = 3,
  className = "",
}: {
  lines?: number;
  className?: string;
}) => (
  <div className={`animate-pulse space-y-3 ${className}`}>
    {Array.from({ length: lines }, (_, i) => (
      <div
        key={i}
        className="h-4 bg-gray-200 rounded-lg"
        style={{
          width: i === lines - 1 ? "75%" : "100%",
        }}
      />
    ))}
  </div>
);

// Chat bubble skeleton
export const ChatSkeleton = () => (
  <div className="space-y-4 animate-pulse">
    <div className="flex items-start gap-3">
      <div className="w-8 h-8 bg-gray-200 rounded-full" />
      <div className="flex-1 space-y-2">
        <div className="h-4 bg-gray-200 rounded w-3/4" />
        <div className="h-4 bg-gray-200 rounded w-1/2" />
        <div className="h-4 bg-gray-200 rounded w-5/6" />
      </div>
    </div>
  </div>
);
