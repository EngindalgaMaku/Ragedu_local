"use client";

import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  FileText,
  Package,
  Settings,
  Upload,
  Search,
  MessageSquare,
  Activity,
  Plus,
  RefreshCw,
  AlertCircle,
  Inbox,
  Database,
  Brain,
  Users,
  BookOpen,
  Sparkles,
  ArrowRight,
  ExternalLink,
} from "lucide-react";

export interface EmptyStateProps {
  variant?: "default" | "card" | "compact" | "centered" | "illustration";
  type?: EmptyStateType;
  icon?: React.ElementType;
  title?: string;
  description?: string;
  actions?: EmptyStateAction[];
  className?: string;
  size?: "sm" | "md" | "lg";
  showBorder?: boolean;
  animated?: boolean;
}

export interface EmptyStateAction {
  label: string;
  onClick: () => void;
  variant?: "default" | "outline" | "secondary" | "ghost";
  icon?: React.ElementType;
  primary?: boolean;
  disabled?: boolean;
  external?: boolean;
}

export type EmptyStateType =
  | "documents"
  | "chunks"
  | "rag-settings"
  | "session-settings"
  | "topics"
  | "interactions"
  | "search"
  | "generic"
  | "loading"
  | "error";

const EMPTY_STATE_CONFIGS: Record<
  EmptyStateType,
  {
    icon: React.ElementType;
    title: string;
    description: string;
    suggestions?: string[];
  }
> = {
  documents: {
    icon: FileText,
    title: "Henüz belge yüklenmemiş",
    description: "Sisteminizi kullanmaya başlamak için belgelerinizi yükleyin.",
    suggestions: [
      "PDF, DOCX, TXT formatında dosyalar yükleyebilirsiniz",
      "Birden fazla dosyayı aynı anda seçebilirsiniz",
      "Yüklenen belgeler otomatik olarak işlenir",
    ],
  },
  chunks: {
    icon: Package,
    title: "Belge parçası bulunamadı",
    description: "Belgeleriniz işlendikten sonra parçalar burada görünecek.",
    suggestions: [
      "Önce belge yüklemeniz gerekiyor",
      "Belge işleme süreci birkaç dakika sürebilir",
      "İşlenen parçaları filtreleyebilir ve düzenleyebilirsiniz",
    ],
  },
  "rag-settings": {
    icon: Settings,
    title: "RAG ayarları yapılandırılmamış",
    description: "Model seçimi ve parametre ayarlarını yapılandırın.",
    suggestions: [
      "Model sağlayıcısını seçin",
      "Embedding modelini belirleyin",
      "Arama parametrelerini optimize edin",
    ],
  },
  "session-settings": {
    icon: Users,
    title: "Oturum ayarları varsayılan değerlerde",
    description: "Eğitsel özellikleri özelleştirin ve yapılandırın.",
    suggestions: [
      "Kişiselleştirilmiş yanıtları etkinleştirin",
      "Bloom taksonomisi desteğini açın",
      "Bilişsel yük analizini aktifleştirin",
    ],
  },
  topics: {
    icon: MessageSquare,
    title: "Henüz konu tanımlanmamış",
    description: "APRAG sistem konuları belgelerinizden otomatik çıkarılır.",
    suggestions: [
      "Belgeler yüklendikten sonra konular otomatik oluşturulur",
      "Manuel konu ekleyebilir ve düzenleyebilirsiniz",
      "Konuları kategorilere ayırabilirsiniz",
    ],
  },
  interactions: {
    icon: Activity,
    title: "Henüz etkileşim kaydı yok",
    description: "Öğrenci sorguları ve yanıtları burada görüntülenecek.",
    suggestions: [
      "Soru-cevap etkileşimleri otomatik kaydedilir",
      "Etkileşim geçmişini filtreleyebilirsiniz",
      "Öğrenci performansını analiz edebilirsiniz",
    ],
  },
  search: {
    icon: Search,
    title: "Arama sonucu bulunamadı",
    description: "Farklı anahtar kelimeler veya filtreler deneyin.",
    suggestions: [
      "Daha genel terimler kullanın",
      "Filtreleri kontrol edin",
      "Yazım hatalarını kontrol edin",
    ],
  },
  generic: {
    icon: Inbox,
    title: "İçerik bulunamadı",
    description: "Henüz burada görüntülenecek bir şey yok.",
  },
  loading: {
    icon: RefreshCw,
    title: "İçerik yükleniyor...",
    description: "Lütfen bekleyin, veriler getiriliyor.",
  },
  error: {
    icon: AlertCircle,
    title: "Bir hata oluştu",
    description: "İçerik yüklenirken bir sorun oluştu.",
  },
};

export default function EmptyState({
  variant = "default",
  type = "generic",
  icon: CustomIcon,
  title: customTitle,
  description: customDescription,
  actions = [],
  className,
  size = "md",
  showBorder = false,
  animated = true,
}: EmptyStateProps) {
  const config = EMPTY_STATE_CONFIGS[type];
  const Icon = CustomIcon || config.icon;
  const title = customTitle || config.title;
  const description = customDescription || config.description;

  const sizeClasses = {
    sm: {
      icon: "w-8 h-8",
      title: "text-lg",
      description: "text-sm",
      spacing: "space-y-3",
      padding: "p-4",
    },
    md: {
      icon: "w-12 h-12",
      title: "text-xl",
      description: "text-base",
      spacing: "space-y-4",
      padding: "p-6",
    },
    lg: {
      icon: "w-16 h-16",
      title: "text-2xl",
      description: "text-lg",
      spacing: "space-y-6",
      padding: "p-8",
    },
  };

  const classes = sizeClasses[size];

  if (variant === "card") {
    return (
      <Card className={cn("w-full", showBorder && "border-dashed", className)}>
        <CardContent
          className={cn("text-center", classes.padding, classes.spacing)}
        >
          <div className="flex flex-col items-center">
            <div
              className={cn(
                "rounded-full bg-muted p-4 mb-4",
                animated && type === "loading" && "animate-spin"
              )}
            >
              <Icon className={cn(classes.icon, "text-muted-foreground")} />
            </div>

            <h3 className={cn("font-semibold text-foreground", classes.title)}>
              {title}
            </h3>

            <p
              className={cn(
                "text-muted-foreground max-w-md",
                classes.description
              )}
            >
              {description}
            </p>

            {config.suggestions && config.suggestions.length > 0 && (
              <div className="mt-4 space-y-2 text-left">
                <h4 className="text-sm font-medium text-foreground">
                  Öneriler:
                </h4>
                <ul className="text-xs text-muted-foreground space-y-1">
                  {config.suggestions.map((suggestion, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <div className="w-1 h-1 bg-muted-foreground rounded-full mt-2 shrink-0" />
                      <span>{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {actions.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-6">
                {actions.map((action, index) => (
                  <Button
                    key={index}
                    variant={
                      action.variant || (action.primary ? "default" : "outline")
                    }
                    size={size === "lg" ? "default" : "sm"}
                    onClick={action.onClick}
                    disabled={action.disabled}
                    className="gap-2"
                  >
                    {action.icon && <action.icon className="w-4 h-4" />}
                    {action.label}
                    {action.external && <ExternalLink className="w-3 h-3" />}
                  </Button>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (variant === "compact") {
    return (
      <div
        className={cn(
          "flex items-center gap-3 p-4 rounded-lg bg-muted/30 border border-dashed",
          className
        )}
      >
        <Icon className={cn("w-6 h-6 text-muted-foreground shrink-0")} />
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm text-foreground truncate">
            {title}
          </p>
          <p className="text-xs text-muted-foreground truncate">
            {description}
          </p>
        </div>
        {actions.length > 0 && (
          <div className="flex gap-2 shrink-0">
            {actions.slice(0, 2).map((action, index) => (
              <Button
                key={index}
                variant={action.variant || "outline"}
                size="sm"
                onClick={action.onClick}
                disabled={action.disabled}
              >
                {action.icon && <action.icon className="w-3 h-3 mr-1" />}
                {action.label}
              </Button>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (variant === "centered") {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center min-h-64 text-center",
          classes.padding,
          classes.spacing,
          className
        )}
      >
        <div
          className={cn(
            "rounded-full bg-muted/50 p-6 mb-6",
            animated && type === "loading" && "animate-pulse"
          )}
        >
          <Icon className={cn(classes.icon, "text-muted-foreground")} />
        </div>

        <h2 className={cn("font-semibold text-foreground", classes.title)}>
          {title}
        </h2>

        <p
          className={cn(
            "text-muted-foreground max-w-lg mx-auto",
            classes.description
          )}
        >
          {description}
        </p>

        {actions.length > 0 && (
          <div className="flex flex-wrap gap-3 mt-8 justify-center">
            {actions.map((action, index) => (
              <Button
                key={index}
                variant={
                  action.variant || (action.primary ? "default" : "outline")
                }
                size={size === "sm" ? "sm" : "default"}
                onClick={action.onClick}
                disabled={action.disabled}
                className="gap-2"
              >
                {action.icon && <action.icon className="w-4 h-4" />}
                {action.label}
                {action.external && <ExternalLink className="w-3 h-3" />}
              </Button>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (variant === "illustration") {
    return (
      <div className={cn("text-center py-12 px-6", className)}>
        {/* Illustration Background */}
        <div className="relative mx-auto mb-8 w-32 h-32">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-primary/5 rounded-full" />
          <div className="absolute inset-2 bg-gradient-to-br from-background to-muted/20 rounded-full border border-border/20" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Icon className="w-12 h-12 text-primary/60" />
          </div>
          {animated && (
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent rounded-full animate-pulse" />
          )}
        </div>

        <h2 className="text-2xl font-semibold text-foreground mb-3">{title}</h2>

        <p className="text-muted-foreground text-lg mb-8 max-w-md mx-auto">
          {description}
        </p>

        {actions.length > 0 && (
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            {actions.map((action, index) => (
              <Button
                key={index}
                variant={
                  action.variant || (action.primary ? "default" : "outline")
                }
                size="lg"
                onClick={action.onClick}
                disabled={action.disabled}
                className="gap-2 min-w-32"
              >
                {action.icon && <action.icon className="w-5 h-5" />}
                {action.label}
                {action.external && <ExternalLink className="w-4 h-4" />}
              </Button>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Default variant
  return (
    <div
      className={cn(
        "text-center",
        classes.padding,
        classes.spacing,
        showBorder && "border border-dashed border-border rounded-lg",
        className
      )}
    >
      <Icon
        className={cn(
          classes.icon,
          "mx-auto mb-4 text-muted-foreground",
          animated && type === "loading" && "animate-spin"
        )}
      />

      <h3 className={cn("font-semibold text-foreground", classes.title)}>
        {title}
      </h3>

      <p className={cn("text-muted-foreground", classes.description)}>
        {description}
      </p>

      {actions.length > 0 && (
        <div className="flex flex-wrap gap-2 justify-center mt-6">
          {actions.map((action, index) => (
            <Button
              key={index}
              variant={
                action.variant || (action.primary ? "default" : "outline")
              }
              size={size === "lg" ? "default" : "sm"}
              onClick={action.onClick}
              disabled={action.disabled}
              className="gap-2"
            >
              {action.icon && <action.icon className="w-4 h-4" />}
              {action.label}
              {action.external && <ExternalLink className="w-3 h-3" />}
            </Button>
          ))}
        </div>
      )}
    </div>
  );
}

// Specialized empty states for each tab
export function DocumentsEmptyState({ onUpload }: { onUpload?: () => void }) {
  return (
    <EmptyState
      type="documents"
      variant="card"
      size="md"
      actions={
        onUpload
          ? [
              {
                label: "Belge Yükle",
                onClick: onUpload,
                icon: Upload,
                primary: true,
              },
            ]
          : undefined
      }
    />
  );
}

export function ChunksEmptyState({ onUpload }: { onUpload?: () => void }) {
  return (
    <EmptyState
      type="chunks"
      variant="centered"
      size="lg"
      actions={
        onUpload
          ? [
              {
                label: "İlk Belgeyi Yükle",
                onClick: onUpload,
                icon: Plus,
                primary: true,
              },
            ]
          : undefined
      }
    />
  );
}

export function RagSettingsEmptyState({
  onConfigure,
}: {
  onConfigure?: () => void;
}) {
  return (
    <EmptyState
      type="rag-settings"
      variant="card"
      actions={
        onConfigure
          ? [
              {
                label: "Ayarları Yapılandır",
                onClick: onConfigure,
                icon: Settings,
                primary: true,
              },
            ]
          : undefined
      }
    />
  );
}

export function SessionSettingsEmptyState({
  onConfigure,
}: {
  onConfigure?: () => void;
}) {
  return (
    <EmptyState
      type="session-settings"
      variant="compact"
      actions={
        onConfigure
          ? [
              {
                label: "Özelleştir",
                onClick: onConfigure,
                icon: Sparkles,
                primary: true,
              },
            ]
          : undefined
      }
    />
  );
}

export function TopicsEmptyState({ onRefresh }: { onRefresh?: () => void }) {
  return (
    <EmptyState
      type="topics"
      variant="illustration"
      size="lg"
      animated
      actions={
        onRefresh
          ? [
              {
                label: "Konuları Yenile",
                onClick: onRefresh,
                icon: RefreshCw,
                variant: "outline",
              },
            ]
          : undefined
      }
    />
  );
}

export function InteractionsEmptyState({ onStart }: { onStart?: () => void }) {
  return (
    <EmptyState
      type="interactions"
      variant="card"
      actions={
        onStart
          ? [
              {
                label: "İlk Soruyu Sor",
                onClick: onStart,
                icon: MessageSquare,
                primary: true,
              },
            ]
          : undefined
      }
    />
  );
}

export function SearchEmptyState({ onClear }: { onClear?: () => void }) {
  return (
    <EmptyState
      type="search"
      variant="compact"
      actions={
        onClear
          ? [
              {
                label: "Filtreleri Temizle",
                onClick: onClear,
                icon: RefreshCw,
                variant: "outline",
              },
            ]
          : undefined
      }
    />
  );
}

// Loading states
export function LoadingEmptyState({ message }: { message?: string }) {
  return (
    <EmptyState
      type="loading"
      variant="centered"
      animated
      description={message}
    />
  );
}

// Error states
export function ErrorEmptyState({
  onRetry,
  message,
}: {
  onRetry?: () => void;
  message?: string;
}) {
  return (
    <EmptyState
      type="error"
      variant="card"
      description={message}
      actions={
        onRetry
          ? [
              {
                label: "Yeniden Dene",
                onClick: onRetry,
                icon: RefreshCw,
                primary: true,
              },
            ]
          : undefined
      }
    />
  );
}
