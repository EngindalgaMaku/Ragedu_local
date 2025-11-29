"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Loader2,
  FileText,
  Package,
  Settings,
  User,
  MessageSquare,
  Activity,
} from "lucide-react";

export interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg" | "xl";
  variant?: "spinner" | "pulse" | "dots" | "progress";
  className?: string;
  text?: string;
  progress?: number;
  showIcon?: boolean;
}

export interface SkeletonProps {
  className?: string;
  variant?: "text" | "avatar" | "card" | "button" | "input" | "badge";
  lines?: number;
  animate?: boolean;
}

// Basic loading spinner component
export default function LoadingSpinner({
  size = "md",
  variant = "spinner",
  className,
  text,
  progress,
  showIcon = true,
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-6 h-6",
    lg: "w-8 h-8",
    xl: "w-12 h-12",
  };

  const textSizeClasses = {
    sm: "text-sm",
    md: "text-base",
    lg: "text-lg",
    xl: "text-xl",
  };

  if (variant === "progress" && typeof progress === "number") {
    return (
      <div className={cn("flex flex-col items-center gap-3", className)}>
        <Progress value={progress} className="w-full max-w-md" />
        {text && (
          <p className={cn("text-muted-foreground", textSizeClasses[size])}>
            {text} {Math.round(progress)}%
          </p>
        )}
      </div>
    );
  }

  if (variant === "dots") {
    return (
      <div className={cn("flex items-center gap-2", className)}>
        <div className="flex items-center gap-1">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className={cn(
                "rounded-full bg-primary animate-pulse",
                size === "sm"
                  ? "w-1 h-1"
                  : size === "md"
                  ? "w-2 h-2"
                  : size === "lg"
                  ? "w-3 h-3"
                  : "w-4 h-4"
              )}
              style={{
                animationDelay: `${i * 0.2}s`,
                animationDuration: "1.5s",
              }}
            />
          ))}
        </div>
        {text && (
          <span
            className={cn("text-muted-foreground ml-2", textSizeClasses[size])}
          >
            {text}
          </span>
        )}
      </div>
    );
  }

  if (variant === "pulse") {
    return (
      <div className={cn("flex items-center gap-2", className)}>
        <div
          className={cn(
            "rounded-full bg-primary animate-pulse",
            sizeClasses[size]
          )}
        />
        {text && (
          <span className={cn("text-muted-foreground", textSizeClasses[size])}>
            {text}
          </span>
        )}
      </div>
    );
  }

  // Default spinner variant
  return (
    <div className={cn("flex items-center gap-2", className)}>
      {showIcon && (
        <Loader2
          className={cn("animate-spin text-primary", sizeClasses[size])}
        />
      )}
      {text && (
        <span className={cn("text-muted-foreground", textSizeClasses[size])}>
          {text}
        </span>
      )}
    </div>
  );
}

// Skeleton component for loading states
export function Skeleton({
  className,
  variant = "text",
  lines = 1,
  animate = true,
  ...props
}: SkeletonProps & React.HTMLAttributes<HTMLDivElement>) {
  const baseClasses = cn(
    "bg-muted rounded-md",
    animate && "animate-pulse",
    className
  );

  switch (variant) {
    case "avatar":
      return (
        <div className={cn(baseClasses, "w-10 h-10 rounded-full")} {...props} />
      );

    case "card":
      return (
        <div className={cn(baseClasses, "p-4 space-y-3")} {...props}>
          <div className="h-4 bg-muted rounded w-3/4" />
          <div className="space-y-2">
            <div className="h-3 bg-muted rounded" />
            <div className="h-3 bg-muted rounded w-5/6" />
          </div>
        </div>
      );

    case "button":
      return (
        <div className={cn(baseClasses, "h-9 w-20 rounded-md")} {...props} />
      );

    case "input":
      return (
        <div className={cn(baseClasses, "h-9 w-full rounded-md")} {...props} />
      );

    case "badge":
      return (
        <div className={cn(baseClasses, "h-5 w-12 rounded-full")} {...props} />
      );

    default: // text variant
      return (
        <div className="space-y-2" {...props}>
          {Array.from({ length: lines }).map((_, i) => (
            <div
              key={i}
              className={cn(
                baseClasses,
                "h-4",
                i === lines - 1 && lines > 1 ? "w-3/4" : "w-full"
              )}
            />
          ))}
        </div>
      );
  }
}

// Specialized loading states for session components
export function DocumentsLoadingSkeleton() {
  return (
    <div className="space-y-6 p-6">
      {/* Header skeleton */}
      <div className="space-y-2">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-4 w-64" />
      </div>

      {/* Stats cards skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <Skeleton className="h-4 w-20" />
                <Skeleton variant="avatar" className="w-4 h-4" />
              </div>
            </CardHeader>
            <CardContent>
              <Skeleton className="h-8 w-12 mb-1" />
              <Skeleton className="h-3 w-24" />
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Content skeleton */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Skeleton variant="card" className="h-64" />
        <Skeleton variant="card" className="h-64" />
      </div>
    </div>
  );
}

export function ChunksLoadingSkeleton() {
  return (
    <div className="space-y-4 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-8 w-32" />
          <Skeleton className="h-4 w-48" />
        </div>
        <Skeleton variant="button" />
      </div>

      {/* Filters */}
      <div className="flex gap-2">
        <Skeleton variant="input" className="w-64" />
        <Skeleton variant="button" />
        <Skeleton variant="button" />
      </div>

      {/* Chunk list */}
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <Card key={i} className="p-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Skeleton variant="avatar" className="w-6 h-6" />
                  <Skeleton className="h-5 w-32" />
                </div>
                <div className="flex gap-2">
                  <Skeleton variant="badge" />
                  <Skeleton variant="badge" />
                </div>
              </div>
              <Skeleton lines={3} />
              <div className="flex items-center gap-2">
                <Skeleton className="h-3 w-16" />
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-24" />
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

export function RagSettingsLoadingSkeleton() {
  return (
    <div className="space-y-6 p-6">
      <div className="space-y-2">
        <Skeleton className="h-8 w-40" />
        <Skeleton className="h-4 w-56" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Settings */}
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-32" />
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Skeleton className="h-4 w-24" />
              <Skeleton variant="input" />
            </div>
            <div className="space-y-2">
              <Skeleton className="h-4 w-32" />
              <Skeleton variant="input" />
            </div>
            <div className="space-y-2">
              <Skeleton className="h-4 w-28" />
              <Skeleton variant="input" />
            </div>
          </CardContent>
        </Card>

        {/* Advanced Settings */}
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-36" />
          </CardHeader>
          <CardContent className="space-y-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="flex items-center justify-between">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-5 w-10 rounded-full" />
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export function SessionSettingsLoadingSkeleton() {
  return (
    <div className="space-y-6 p-6">
      <div className="space-y-2">
        <Skeleton className="h-8 w-44" />
        <Skeleton className="h-4 w-60" />
      </div>

      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-6 w-48" />
              <Skeleton className="h-4 w-64" />
            </CardHeader>
            <CardContent className="space-y-3">
              {Array.from({ length: 3 }).map((_, j) => (
                <div key={j} className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Skeleton className="h-4 w-40" />
                    <Skeleton className="h-3 w-56" />
                  </div>
                  <Skeleton className="h-5 w-10 rounded-full" />
                </div>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

export function TopicsLoadingSkeleton() {
  return (
    <div className="space-y-4 p-6">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-8 w-28" />
          <Skeleton className="h-4 w-52" />
        </div>
        <Skeleton variant="button" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <Card key={i} className="p-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Skeleton className="h-5 w-24" />
                <Skeleton variant="badge" />
              </div>
              <Skeleton lines={2} />
              <div className="flex items-center gap-2">
                <Skeleton className="h-3 w-12" />
                <Skeleton className="h-3 w-16" />
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

export function InteractionsLoadingSkeleton() {
  return (
    <div className="space-y-4 p-6">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-8 w-36" />
          <Skeleton className="h-4 w-48" />
        </div>
        <div className="flex gap-2">
          <Skeleton variant="input" className="w-48" />
          <Skeleton variant="button" />
        </div>
      </div>

      <div className="space-y-3">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i} className="p-4">
            <div className="space-y-4">
              <div className="flex items-start justify-between">
                <div className="space-y-2 flex-1">
                  <div className="flex items-center gap-2">
                    <Skeleton variant="avatar" className="w-6 h-6" />
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-4 w-20" />
                  </div>
                  <Skeleton lines={2} />
                </div>
                <Skeleton variant="badge" />
              </div>

              <div className="border-l-2 border-muted ml-3 pl-4 space-y-2">
                <Skeleton lines={3} />
              </div>

              <div className="flex items-center gap-4 text-sm">
                <Skeleton className="h-3 w-24" />
                <Skeleton className="h-3 w-16" />
                <Skeleton className="h-3 w-20" />
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

// Loading overlay component
export function LoadingOverlay({
  isLoading,
  text = "Yükleniyor...",
  children,
}: {
  isLoading: boolean;
  text?: string;
  children: React.ReactNode;
}) {
  if (!isLoading) {
    return <>{children}</>;
  }

  return (
    <div className="relative">
      <div className="absolute inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center">
        <LoadingSpinner text={text} size="lg" />
      </div>
      <div className="opacity-50 pointer-events-none">{children}</div>
    </div>
  );
}

// Full page loading component
export function PageLoadingSpinner({
  text = "Sayfa yükleniyor...",
}: {
  text?: string;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center space-y-4">
        <LoadingSpinner size="xl" />
        <p className="text-muted-foreground text-lg">{text}</p>
      </div>
    </div>
  );
}
