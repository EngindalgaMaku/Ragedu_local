/**
 * InteractionsTab Component
 * Main interactions tab with statistics, filtering, and interaction management
 */

import React, { useState, useEffect } from "react";
import {
  Search,
  Filter,
  RefreshCw,
  BarChart3,
  MessageSquare,
  Clock,
  Users,
  BookOpen,
  Grid3X3,
  List,
  Calendar,
  TrendingUp,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import LoadingSpinner from "../shared/LoadingSpinner";
import { useInteractions } from "../../hooks/useInteractions";
import InteractionCard, { InteractionCardMobile } from "./InteractionCard";
import InteractionsPagination from "./InteractionsPagination";
import InteractionsEmpty from "./InteractionsEmpty";
import InteractionModal from "./InteractionModal";

interface InteractionsTabProps {
  sessionId: string;
  apragEnabled?: boolean;
}

type ViewMode = "list" | "grid";

export function InteractionsTab({
  sessionId,
  apragEnabled = false,
}: InteractionsTabProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [showFilters, setShowFilters] = useState(false);

  // Use the interactions hook
  const {
    // State
    interactions,
    loading,
    error,
    currentPage,
    totalPages,
    totalInteractions,
    perPage,
    searchTerm,
    feedbackFilter,
    dateRange,
    modalOpen,
    selectedInteraction,
    stats,

    // Actions
    fetchInteractions,
    refreshInteractions,
    setCurrentPage,
    setPerPage,
    setSearchTerm,
    setFeedbackFilter,
    setDateRange,
    clearFilters,
    openModal,
    closeModal,
    clearError,
  } = useInteractions({
    sessionId,
    initialPerPage: 10,
    autoRefresh: false,
  });

  // Auto-refresh every 30 seconds when tab is active
  useEffect(() => {
    let interval: NodeJS.Timeout;

    // Set up auto-refresh if APRAG is enabled
    if (apragEnabled) {
      interval = setInterval(() => {
        refreshInteractions();
      }, 30000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [apragEnabled, refreshInteractions]);

  // Check if any filters are active
  const hasActiveFilters: boolean = Boolean(
    searchTerm || (feedbackFilter && feedbackFilter !== "all") || !!dateRange
  );

  // Render statistics overview
  const renderStatsOverview = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <MessageSquare className="w-4 h-4 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Toplam Soru</p>
              <p className="text-xl font-bold text-blue-600">
                {stats.totalQueries}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <Clock className="w-4 h-4 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Ortalama Süre</p>
              <p className="text-xl font-bold text-green-600">
                {stats.averageResponseTime > 0
                  ? `${Math.round(stats.averageResponseTime)}ms`
                  : "--"}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <BookOpen className="w-4 h-4 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Kapsanan Konular</p>
              <p className="text-xl font-bold text-purple-600">
                {Object.keys(stats.topicsDistribution).length}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-orange-100 rounded-lg">
              <TrendingUp className="w-4 h-4 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Bu Sayfa</p>
              <p className="text-xl font-bold text-orange-600">
                {Math.min(perPage, interactions.length)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  // Render filter controls
  const renderFilterControls = () => (
    <Card className="mb-6">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Filtreler ve Arama</CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowFilters(!showFilters)}
          >
            <Filter className="w-4 h-4 mr-1" />
            {showFilters ? "Gizle" : "Göster"}
          </Button>
        </div>
      </CardHeader>

      {showFilters && (
        <CardContent className="space-y-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Soru veya cevap içerisinde ara..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-3 py-2 border border-border rounded-lg bg-background focus:ring-2 focus:ring-primary focus:border-primary"
            />
          </div>

          {/* Filter row */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Date range would go here if implemented */}
            <div className="flex items-center gap-2">
              <Calendar className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">
                Tarih filtreleri yakında eklenecek
              </span>
            </div>
          </div>

          {/* Active filters */}
          {hasActiveFilters && (
            <div className="flex items-center gap-2 pt-2 border-t border-border">
              <span className="text-sm text-muted-foreground">
                Aktif filtreler:
              </span>
              {searchTerm && (
                <Badge variant="secondary">Arama: {searchTerm}</Badge>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={clearFilters}
                className="text-xs"
              >
                Temizle
              </Button>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );

  // Render view controls
  const renderViewControls = () => (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">
          {totalInteractions} etkileşim
        </span>
        {loading && (
          <div className="w-4 h-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
        )}
      </div>

      <div className="flex items-center gap-2">
        {/* Refresh */}
        <Button
          variant="ghost"
          size="sm"
          onClick={refreshInteractions}
          disabled={loading}
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        </Button>

        {/* View mode toggle */}
        <div className="flex items-center rounded-lg border border-border">
          <Button
            variant={viewMode === "list" ? "default" : "ghost"}
            size="sm"
            onClick={() => setViewMode("list")}
            className="rounded-r-none"
          >
            <List className="w-4 h-4" />
          </Button>
          <Button
            variant={viewMode === "grid" ? "default" : "ghost"}
            size="sm"
            onClick={() => setViewMode("grid")}
            className="rounded-l-none"
          >
            <Grid3X3 className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );

  // Main content rendering
  if (!apragEnabled) {
    return (
      <div className="p-6">
        <Alert>
          <AlertDescription>
            APRAG özelliği bu oturum için etkinleştirilmemiş. Öğrenci
            etkileşimlerini görmek için APRAG'ı etkinleştirin.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Error display */}
      {error && (
        <Alert variant="destructive" className="mb-4">
          <AlertDescription className="flex items-center justify-between">
            {error}
            <Button variant="ghost" size="sm" onClick={clearError}>
              ×
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Statistics Overview */}
      {renderStatsOverview()}

      {/* Filter Controls */}
      {renderFilterControls()}

      {/* View Controls */}
      {renderViewControls()}

      {/* Main Content */}
      {loading && interactions.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <LoadingSpinner />
          <span className="ml-3 text-muted-foreground">
            Etkileşimler yükleniyor...
          </span>
        </div>
      ) : interactions.length === 0 ? (
        <InteractionsEmpty
          hasFilters={hasActiveFilters}
          onClearFilters={clearFilters}
        />
      ) : (
        <>
          {/* Interactions List/Grid */}
          {viewMode === "list" ? (
            <div className="space-y-4">
              {interactions.map((interaction, index) => {
                const globalIndex = (currentPage - 1) * perPage + index + 1;

                // Use mobile view on small screens
                return (
                  <div key={interaction.interaction_id} className="block">
                    <div className="hidden sm:block">
                      <InteractionCard
                        interaction={interaction}
                        index={globalIndex}
                        onViewDetails={openModal}
                        compact={true}
                      />
                    </div>
                    <div className="block sm:hidden">
                      <InteractionCardMobile
                        interaction={interaction}
                        index={globalIndex}
                        onViewDetails={openModal}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {interactions.map((interaction, index) => {
                const globalIndex = (currentPage - 1) * perPage + index + 1;

                return (
                  <InteractionCard
                    key={interaction.interaction_id}
                    interaction={interaction}
                    index={globalIndex}
                    onViewDetails={openModal}
                    compact={true}
                  />
                );
              })}
            </div>
          )}

          {/* Pagination */}
          <InteractionsPagination
            currentPage={currentPage}
            totalPages={totalPages}
            totalItems={totalInteractions}
            itemsPerPage={perPage}
            loading={loading}
            onPageChange={setCurrentPage}
            onItemsPerPageChange={setPerPage}
          />
        </>
      )}

      {/* Modal */}
      <InteractionModal
        interaction={selectedInteraction}
        open={modalOpen}
        onOpenChange={closeModal}
      />
    </div>
  );
}

export default InteractionsTab;
