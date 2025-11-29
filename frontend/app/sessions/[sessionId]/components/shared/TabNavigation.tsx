"use client";

import React from "react";
import { useRouter, usePathname, useParams } from "next/navigation";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  FileText,
  Package,
  Settings,
  User,
  MessageSquare,
  Activity,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useSessionData } from "../../hooks/useSessionData";

export type TabId =
  | "documents"
  | "chunks"
  | "rag-settings"
  | "session-settings"
  | "topics"
  | "interactions";

export interface TabItem {
  id: TabId;
  label: string;
  icon: React.ElementType;
  badge?: number | string;
  badgeVariant?: "default" | "secondary" | "destructive" | "outline";
  disabled?: boolean;
  description?: string;
}

interface TabNavigationProps {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  children?: React.ReactNode;
  className?: string;
  orientation?: "horizontal" | "vertical";
  size?: "sm" | "md" | "lg";
  showBadges?: boolean;
  showIcons?: boolean;
}

const DEFAULT_TABS: TabItem[] = [
  {
    id: "documents",
    label: "Belgeler",
    icon: FileText,
    description: "Dosya yükleme ve işleme",
  },
  {
    id: "chunks",
    label: "Parçalar",
    icon: Package,
    description: "Belge parçası yönetimi",
  },
  {
    id: "rag-settings",
    label: "RAG Ayarları",
    icon: Settings,
    description: "Model ve ayar yapılandırması",
  },
  {
    id: "session-settings",
    label: "Oturum Ayarları",
    icon: User,
    description: "Eğitsel özellik ayarları",
  },
  {
    id: "topics",
    label: "Konular",
    icon: MessageSquare,
    description: "APRAG konu yönetimi",
  },
  {
    id: "interactions",
    label: "Etkileşimler",
    icon: Activity,
    description: "Öğrenci sorgu geçmişi",
  },
];

export default function TabNavigation({
  activeTab,
  onTabChange,
  children,
  className,
  orientation = "horizontal",
  size = "md",
  showBadges = true,
  showIcons = true,
}: TabNavigationProps) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useParams();
  const sessionId = params?.sessionId as string;

  // Get data counts for badges from useSessionData
  const {
    chunksCount,
    interactionsCount,
    hasSession,
    hasChunks,
    hasRagConfig,
    sessionError,
    chunksError,
    ragError,
  } = useSessionData();

  // Enhanced tabs with dynamic badges and states
  const tabs: TabItem[] = DEFAULT_TABS.map((tab) => {
    let badge: number | string | undefined;
    let badgeVariant: TabItem["badgeVariant"] = "secondary";
    let disabled = false;

    switch (tab.id) {
      case "documents":
        if (!hasSession) disabled = true;
        if (sessionError) badgeVariant = "destructive";
        break;

      case "chunks":
        badge = chunksCount > 0 ? chunksCount : undefined;
        if (!hasChunks) disabled = false; // Allow access to upload files
        if (chunksError) badgeVariant = "destructive";
        break;

      case "rag-settings":
        if (!hasRagConfig) badge = "!";
        if (ragError) badgeVariant = "destructive";
        break;

      case "session-settings":
        if (!hasSession) disabled = true;
        break;

      case "topics":
        if (!hasChunks) disabled = true;
        break;

      case "interactions":
        badge = interactionsCount > 0 ? interactionsCount : undefined;
        if (!hasSession) disabled = true;
        break;
    }

    return {
      ...tab,
      badge: showBadges ? badge : undefined,
      badgeVariant,
      disabled,
    };
  });

  const handleTabChange = (value: string) => {
    const tabId = value as TabId;
    onTabChange(tabId);

    // Update URL if needed
    if (sessionId) {
      const newPath = `/sessions/${sessionId}?tab=${tabId}`;
      if (pathname !== newPath.split("?")[0]) {
        router.push(newPath, { scroll: false });
      }
    }
  };

  const tabSizeClasses = {
    sm: "text-xs px-2 py-1",
    md: "text-sm px-3 py-2",
    lg: "text-base px-4 py-3",
  };

  const orientationClasses = {
    horizontal: "flex-row",
    vertical: "flex-col w-full",
  };

  return (
    <div className={cn("w-full", className)}>
      <Tabs
        value={activeTab}
        onValueChange={handleTabChange}
        className="w-full"
      >
        <TabsList
          className={cn(
            "grid w-full bg-muted/30 p-1",
            orientation === "horizontal"
              ? "grid-cols-6 gap-1"
              : "grid-rows-6 gap-1 h-auto",
            orientationClasses[orientation]
          )}
        >
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            const hasError = tab.badgeVariant === "destructive";

            return (
              <TabsTrigger
                key={tab.id}
                value={tab.id}
                disabled={tab.disabled}
                className={cn(
                  "relative flex items-center justify-center gap-2 transition-all",
                  tabSizeClasses[size],
                  orientation === "vertical" && "justify-start",
                  isActive && "bg-background shadow-sm",
                  tab.disabled && "opacity-50 cursor-not-allowed",
                  hasError && "text-destructive"
                )}
                title={tab.description}
              >
                {showIcons && (
                  <Icon
                    className={cn(
                      "shrink-0",
                      size === "sm"
                        ? "w-3 h-3"
                        : size === "md"
                        ? "w-4 h-4"
                        : "w-5 h-5"
                    )}
                  />
                )}

                <span
                  className={cn(
                    "truncate",
                    orientation === "horizontal" &&
                      size === "sm" &&
                      "hidden sm:inline"
                  )}
                >
                  {tab.label}
                </span>

                {tab.badge && (
                  <Badge
                    variant={tab.badgeVariant}
                    className={cn(
                      "ml-1 h-5 min-w-5 text-xs px-1",
                      size === "sm" && "h-4 min-w-4 text-xs",
                      orientation === "horizontal" && "absolute -top-1 -right-1"
                    )}
                  >
                    {tab.badge}
                  </Badge>
                )}
              </TabsTrigger>
            );
          })}
        </TabsList>

        {/* Tab Content */}
        {children && <div className="mt-4">{children}</div>}
      </Tabs>
    </div>
  );
}

// Mobile-optimized tab navigation with scroll
export function MobileTabNavigation({
  activeTab,
  onTabChange,
  className,
}: {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  className?: string;
}) {
  const {
    chunksCount,
    interactionsCount,
    hasChunks,
    sessionError,
    chunksError,
    ragError,
  } = useSessionData();

  const tabs = DEFAULT_TABS.map((tab) => {
    let badge: number | string | undefined;
    let badgeVariant: TabItem["badgeVariant"] = "secondary";

    switch (tab.id) {
      case "chunks":
        badge = chunksCount > 0 ? chunksCount : undefined;
        if (chunksError) badgeVariant = "destructive";
        break;
      case "interactions":
        badge = interactionsCount > 0 ? interactionsCount : undefined;
        break;
      case "rag-settings":
        if (ragError) badgeVariant = "destructive";
        break;
    }

    return { ...tab, badge, badgeVariant };
  });

  return (
    <div className={cn("w-full overflow-hidden", className)}>
      <div className="flex overflow-x-auto scrollbar-hide gap-1 p-2 bg-muted/30 rounded-lg">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={cn(
                "relative flex flex-col items-center gap-1 px-3 py-2 rounded-md transition-all shrink-0 min-w-16",
                isActive
                  ? "bg-background shadow-sm text-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
            >
              <Icon className="w-4 h-4" />
              <span className="text-xs truncate max-w-12">{tab.label}</span>

              {tab.badge && (
                <Badge
                  variant={tab.badgeVariant}
                  className="absolute -top-1 -right-1 h-4 min-w-4 text-xs px-1"
                >
                  {tab.badge}
                </Badge>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Compact tab navigation for small spaces
export function CompactTabNavigation({
  activeTab,
  onTabChange,
  visibleTabs = ["documents", "chunks", "rag-settings"],
  className,
}: {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  visibleTabs?: TabId[];
  className?: string;
}) {
  const tabs = DEFAULT_TABS.filter((tab) => visibleTabs.includes(tab.id));

  return (
    <div className={cn("flex items-center gap-1", className)}>
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeTab === tab.id;

        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={cn(
              "p-2 rounded-md transition-colors",
              isActive
                ? "bg-background shadow-sm text-foreground"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
            )}
            title={tab.label}
          >
            <Icon className="w-4 h-4" />
          </button>
        );
      })}
    </div>
  );
}
