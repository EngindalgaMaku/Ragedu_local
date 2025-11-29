"use client";

/**
 * IntegratedExample.tsx - Demo component showing how shared components work together
 * This is an example implementation demonstrating the integration of all Phase 3 components
 */

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// Import shared components and hooks
import TabNavigation from "./TabNavigation";
import LoadingSpinner, {
  LoadingOverlay,
  DocumentsLoadingSkeleton,
} from "./LoadingSpinner";
import { ErrorBoundary, TabErrorBoundary } from "./ErrorBoundary";
import EmptyState, {
  DocumentsEmptyState,
  ChunksEmptyState,
} from "./EmptyState";
import { useTabNavigation } from "../../hooks/useTabNavigation";
import { useSessionData } from "../../hooks/useSessionData";
import { useErrorHandler } from "../../hooks/useErrorHandler";

// Tab content components example
function DocumentsTabContent() {
  const { hasChunks, chunksCount, loading } = useSessionData();

  if (loading) {
    return <DocumentsLoadingSkeleton />;
  }

  if (!hasChunks) {
    return (
      <DocumentsEmptyState onUpload={() => console.log("Upload triggered")} />
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium">Documents Tab Active</h3>
        <p className="text-muted-foreground">
          {chunksCount} belge parçası bulundu
        </p>
      </div>
    </div>
  );
}

function ChunksTabContent() {
  const { hasChunks, chunksCount, chunksLoading } = useSessionData();

  if (chunksLoading) {
    return (
      <LoadingOverlay isLoading={true} text="Parçalar yükleniyor...">
        <div className="h-64" />
      </LoadingOverlay>
    );
  }

  if (!hasChunks) {
    return (
      <ChunksEmptyState
        onUpload={() => console.log("Upload from chunks tab")}
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium">Chunks Tab Active</h3>
        <p className="text-muted-foreground">
          {chunksCount} parça görüntüleniyor
        </p>
      </div>
    </div>
  );
}

function RagSettingsTabContent() {
  const { hasRagConfig, ragLoading } = useSessionData();

  if (ragLoading) {
    return (
      <LoadingSpinner
        variant="spinner"
        size="lg"
        text="RAG ayarları yükleniyor..."
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium">RAG Settings Tab Active</h3>
        <p className="text-muted-foreground">
          {hasRagConfig ? "Yapılandırma mevcut" : "Yapılandırma gerekli"}
        </p>
      </div>
    </div>
  );
}

function SessionSettingsTabContent() {
  return (
    <div className="space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium">Session Settings Tab Active</h3>
        <p className="text-muted-foreground">Oturum ayarları yapılandırması</p>
      </div>
    </div>
  );
}

function TopicsTabContent() {
  const { hasChunks } = useSessionData();

  if (!hasChunks) {
    return (
      <EmptyState
        type="topics"
        variant="centered"
        actions={[
          {
            label: "Belge Yükle",
            onClick: () => console.log("Upload from topics"),
            primary: true,
          },
        ]}
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium">Topics Tab Active</h3>
        <p className="text-muted-foreground">Konular ve APRAG yönetimi</p>
      </div>
    </div>
  );
}

function InteractionsTabContent() {
  const { interactionsCount } = useSessionData();

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium">Interactions Tab Active</h3>
        <p className="text-muted-foreground">
          {interactionsCount > 0
            ? `${interactionsCount} etkileşim kaydı`
            : "Henüz etkileşim yok"}
        </p>
      </div>
    </div>
  );
}

// Main integrated component
export default function IntegratedExample() {
  const { activeTab, setActiveTab } = useTabNavigation();
  const { loading, error } = useSessionData();
  const { handleError } = useErrorHandler();

  // Tab content renderer
  const renderTabContent = () => {
    switch (activeTab) {
      case "documents":
        return <DocumentsTabContent />;
      case "chunks":
        return <ChunksTabContent />;
      case "rag-settings":
        return <RagSettingsTabContent />;
      case "session-settings":
        return <SessionSettingsTabContent />;
      case "topics":
        return <TopicsTabContent />;
      case "interactions":
        return <InteractionsTabContent />;
      default:
        return (
          <EmptyState
            type="error"
            variant="card"
            title="Geçersiz Tab"
            description="Seçilen tab bulunamadı."
          />
        );
    }
  };

  // Show page-level loading for initial load
  if (loading && !activeTab) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner
          size="xl"
          text="Sayfa yükleniyor..."
          variant="spinner"
        />
      </div>
    );
  }

  return (
    <ErrorBoundary
      variant="full"
      onError={(error, errorInfo) => {
        handleError(error, {
          category: "session",
          component: "IntegratedExample",
        });
      }}
    >
      <div className="min-h-screen bg-background">
        {/* Header with navigation */}
        <div className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h1 className="text-2xl font-bold">Session Detail</h1>
                <p className="text-muted-foreground">
                  Phase 3 bileşen entegrasyonu örneği
                </p>
              </div>
              {error && (
                <div className="text-sm text-destructive">Hata: {error}</div>
              )}
            </div>

            {/* Tab Navigation */}
            <TabNavigation
              activeTab={activeTab}
              onTabChange={setActiveTab}
              size="md"
              showBadges={true}
              showIcons={true}
            />
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 py-8">
          <TabErrorBoundary>
            <LoadingOverlay isLoading={loading}>
              <Card className="min-h-96">
                <CardHeader>
                  <CardTitle className="capitalize">
                    {activeTab.replace("-", " ")} İçeriği
                  </CardTitle>
                </CardHeader>
                <CardContent>{renderTabContent()}</CardContent>
              </Card>
            </LoadingOverlay>
          </TabErrorBoundary>
        </div>

        {/* Mobile Navigation (for screens < md) */}
        <div className="md:hidden fixed bottom-0 left-0 right-0 border-t border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 p-2">
          <TabNavigation
            activeTab={activeTab}
            onTabChange={setActiveTab}
            orientation="horizontal"
            size="sm"
            className="justify-center"
          />
        </div>
      </div>
    </ErrorBoundary>
  );
}

// Usage example for individual components
export function ComponentShowcase() {
  return (
    <div className="space-y-8 p-8">
      <h2 className="text-2xl font-bold mb-6">Shared Components Showcase</h2>

      {/* Loading States */}
      <Card>
        <CardHeader>
          <CardTitle>Loading States</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4">
            <LoadingSpinner size="sm" text="Small" />
            <LoadingSpinner size="md" text="Medium" />
            <LoadingSpinner size="lg" text="Large" />
          </div>

          <div className="flex items-center gap-4">
            <LoadingSpinner variant="dots" text="Dots" />
            <LoadingSpinner variant="pulse" text="Pulse" />
            <LoadingSpinner variant="progress" progress={65} text="Progress" />
          </div>
        </CardContent>
      </Card>

      {/* Empty States */}
      <Card>
        <CardHeader>
          <CardTitle>Empty States</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <EmptyState
            type="documents"
            variant="compact"
            actions={[
              {
                label: "Action",
                onClick: () => console.log("Action clicked"),
              },
            ]}
          />

          <EmptyState type="search" variant="default" showBorder={true} />
        </CardContent>
      </Card>

      {/* Error Examples */}
      <Card>
        <CardHeader>
          <CardTitle>Error Handling</CardTitle>
        </CardHeader>
        <CardContent>
          <TabErrorBoundary>
            <button
              onClick={() => {
                throw new Error("Test error");
              }}
              className="px-4 py-2 bg-destructive text-destructive-foreground rounded-md"
            >
              Trigger Error (Wrapped in ErrorBoundary)
            </button>
          </TabErrorBoundary>
        </CardContent>
      </Card>
    </div>
  );
}
