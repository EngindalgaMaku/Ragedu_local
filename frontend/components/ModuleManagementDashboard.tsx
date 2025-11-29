"use client";

import React, { useState, useEffect } from "react";
import {
  Module,
  ModuleProgress,
  TeacherModuleDashboard,
  ModuleEditForm,
  ModuleAnalytics,
  getDifficultyColor,
  getDifficultyLabel,
  getCompletionStatusColor,
  getCompletionStatusLabel,
} from "@/types/modules";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  LayoutDashboard,
  Users,
  BookOpen,
  TrendingUp,
  Settings,
  Eye,
  Edit,
  Trash2,
  Plus,
  BarChart3,
  Clock,
  Target,
  CheckCircle,
  AlertTriangle,
  Award,
  Activity,
  Calendar,
  Filter,
  Download,
  RefreshCw,
} from "lucide-react";

interface ModuleManagementDashboardProps {
  sessionId: string;
  modules: Module[];
  onModuleUpdate?: (moduleId: number, updates: Partial<Module>) => void;
  onModuleDelete?: (moduleId: number) => void;
}

const ModuleManagementDashboard: React.FC<ModuleManagementDashboardProps> = ({
  sessionId,
  modules,
  onModuleUpdate,
  onModuleDelete,
}) => {
  // State Management
  const [dashboardData, setDashboardData] =
    useState<TeacherModuleDashboard | null>(null);
  const [selectedModule, setSelectedModule] = useState<Module | null>(null);
  const [editingModule, setEditingModule] = useState<Module | null>(null);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [filterDifficulty, setFilterDifficulty] = useState<string>("all");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [sortBy, setSortBy] = useState<
    "order" | "title" | "difficulty" | "created"
  >("order");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<{
    [moduleId: number]: ModuleAnalytics;
  }>({});

  // Load dashboard data
  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`/api/aprag/modules/dashboard/${sessionId}`);
      if (!response.ok) {
        throw new Error("Dashboard verileri yüklenemedi");
      }

      const data: TeacherModuleDashboard = await response.json();
      setDashboardData(data);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Load module analytics
  const fetchModuleAnalytics = async (moduleId: number) => {
    try {
      const response = await fetch(`/api/aprag/modules/${moduleId}/analytics`);
      if (response.ok) {
        const data: ModuleAnalytics = await response.json();
        setAnalytics((prev) => ({ ...prev, [moduleId]: data }));
      }
    } catch (error) {
      console.error("Analytics loading failed:", error);
    }
  };

  // Update module
  const handleUpdateModule = async (
    moduleId: number,
    updates: ModuleEditForm
  ) => {
    try {
      setError(null);

      const response = await fetch(`/api/aprag/modules/${moduleId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        throw new Error("Modül güncellenemedi");
      }

      if (onModuleUpdate) {
        onModuleUpdate(moduleId, updates);
      }

      setEditingModule(null);
      await fetchDashboardData();
    } catch (error: any) {
      setError(error.message);
    }
  };

  // Delete module
  const handleDeleteModule = async (moduleId: number, moduleTitle: string) => {
    if (
      !confirm(
        `"${moduleTitle}" modülünü silmek istediğinizden emin misiniz? Bu işlem geri alınamaz.`
      )
    ) {
      return;
    }

    try {
      setError(null);

      const response = await fetch(`/api/aprag/modules/${moduleId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Modül silinemedi");
      }

      if (onModuleDelete) {
        onModuleDelete(moduleId);
      }

      await fetchDashboardData();
    } catch (error: any) {
      setError(error.message);
    }
  };

  // Filter and sort modules
  const filteredAndSortedModules = React.useMemo(() => {
    let filtered = modules;

    // Apply filters
    if (filterDifficulty !== "all") {
      filtered = filtered.filter(
        (m) => m.difficulty_level === filterDifficulty
      );
    }

    if (filterStatus !== "all") {
      filtered = filtered.filter((m) => {
        if (filterStatus === "active") return m.is_active;
        if (filterStatus === "inactive") return !m.is_active;
        return true;
      });
    }

    // Apply sorting
    return filtered.sort((a, b) => {
      switch (sortBy) {
        case "title":
          return a.module_title.localeCompare(b.module_title, "tr");
        case "difficulty":
          const difficultyOrder = { beginner: 1, intermediate: 2, advanced: 3 };
          return (
            difficultyOrder[a.difficulty_level] -
            difficultyOrder[b.difficulty_level]
          );
        case "created":
          return (
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
          );
        case "order":
        default:
          return a.module_order - b.module_order;
      }
    });
  }, [modules, filterDifficulty, filterStatus, sortBy]);

  // Load data on mount
  useEffect(() => {
    if (sessionId && modules.length > 0) {
      fetchDashboardData();
    }
  }, [sessionId, modules.length]);

  return (
    <div className="space-y-6">
      {/* Dashboard Overview */}
      {dashboardData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-100 text-blue-600 rounded-lg flex items-center justify-center">
                <BookOpen className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">
                  {dashboardData.total_modules}
                </p>
                <p className="text-sm text-muted-foreground">Toplam Modül</p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-100 text-green-600 rounded-lg flex items-center justify-center">
                <Users className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">
                  {dashboardData.student_engagement.active_students}
                </p>
                <p className="text-sm text-muted-foreground">Aktif Öğrenci</p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-100 text-purple-600 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">
                  {(
                    dashboardData.student_engagement.average_completion_rate *
                    100
                  ).toFixed(1)}
                  %
                </p>
                <p className="text-sm text-muted-foreground">
                  Tamamlanma Oranı
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-orange-100 text-orange-600 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">
                  {(
                    dashboardData.quality_metrics.average_confidence * 100
                  ).toFixed(1)}
                  %
                </p>
                <p className="text-sm text-muted-foreground">Ortalama Güven</p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Filters and Controls */}
      <Card className="p-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <LayoutDashboard className="w-5 h-5" />
              Modül Yönetimi ({filteredAndSortedModules.length})
            </h2>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <select
              value={filterDifficulty}
              onChange={(e) => setFilterDifficulty(e.target.value)}
              className="px-3 py-1.5 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-purple-500"
            >
              <option value="all">Tüm Zorluklar</option>
              <option value="beginner">Başlangıç</option>
              <option value="intermediate">Orta</option>
              <option value="advanced">İleri</option>
            </select>

            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-1.5 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-purple-500"
            >
              <option value="all">Tüm Durumlar</option>
              <option value="active">Aktif</option>
              <option value="inactive">Pasif</option>
            </select>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-3 py-1.5 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-purple-500"
            >
              <option value="order">Sıraya Göre</option>
              <option value="title">Ada Göre</option>
              <option value="difficulty">Zorluğa Göre</option>
              <option value="created">Tarihe Göre</option>
            </select>

            <div className="flex border border-border rounded-lg overflow-hidden">
              <button
                onClick={() => setViewMode("grid")}
                className={`px-3 py-1.5 text-sm ${
                  viewMode === "grid"
                    ? "bg-purple-600 text-white"
                    : "bg-background hover:bg-muted"
                }`}
              >
                Grid
              </button>
              <button
                onClick={() => setViewMode("list")}
                className={`px-3 py-1.5 text-sm ${
                  viewMode === "list"
                    ? "bg-purple-600 text-white"
                    : "bg-background hover:bg-muted"
                }`}
              >
                Liste
              </button>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={fetchDashboardData}
              className="flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Yenile
            </Button>
          </div>
        </div>
      </Card>

      {/* Error Display */}
      {error && (
        <Card className="p-4 border-red-200 bg-red-50 dark:bg-red-900/20 dark:border-red-800">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            <p className="text-red-800 dark:text-red-200">{error}</p>
          </div>
        </Card>
      )}

      {/* Modules Display */}
      {loading ? (
        <Card className="p-8">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-2 border-purple-600 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-muted-foreground">Yükleniyor...</p>
          </div>
        </Card>
      ) : filteredAndSortedModules.length === 0 ? (
        <Card className="p-8">
          <div className="text-center">
            <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">
              Modül Bulunamadı
            </h3>
            <p className="text-muted-foreground">
              {modules.length === 0
                ? "Henüz hiç modül oluşturulmamış"
                : "Filtrelere uygun modül bulunamadı"}
            </p>
          </div>
        </Card>
      ) : (
        <div
          className={
            viewMode === "grid"
              ? "grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6"
              : "space-y-4"
          }
        >
          {filteredAndSortedModules.map((module) => (
            <Card
              key={module.module_id}
              className="p-6 hover:shadow-lg transition-shadow"
            >
              <div className="space-y-4">
                {/* Module Header */}
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-1 bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300 text-xs font-medium rounded">
                        Modül {module.module_order}
                      </span>
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded ${getDifficultyColor(
                          module.difficulty_level
                        )}`}
                      >
                        {getDifficultyLabel(module.difficulty_level)}
                      </span>
                      {module.is_active ? (
                        <span className="px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 text-xs font-medium rounded">
                          Aktif
                        </span>
                      ) : (
                        <span className="px-2 py-1 bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300 text-xs font-medium rounded">
                          Pasif
                        </span>
                      )}
                    </div>

                    <h3 className="text-lg font-semibold text-foreground mb-2">
                      {module.module_title}
                    </h3>

                    {module.module_description && (
                      <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                        {module.module_description}
                      </p>
                    )}
                  </div>

                  <div className="flex items-center gap-1 ml-3">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setSelectedModule(module)}
                      title="Detayları Görüntüle"
                    >
                      <Eye className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setEditingModule(module)}
                      title="Düzenle"
                    >
                      <Edit className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        handleDeleteModule(
                          module.module_id,
                          module.module_title
                        )
                      }
                      className="text-red-600 hover:text-red-700"
                      title="Sil"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                {/* Module Stats */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">
                      {module.estimated_duration_minutes
                        ? `${Math.round(
                            module.estimated_duration_minutes / 60
                          )}s`
                        : "Süre belirsiz"}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <Target className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">
                      {module.learning_objectives?.length || 0} hedef
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">
                      {new Date(module.created_at).toLocaleDateString("tr-TR")}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">
                      {new Date(module.updated_at).toLocaleDateString("tr-TR")}
                    </span>
                  </div>
                </div>

                {/* Learning Objectives Preview */}
                {module.learning_objectives &&
                  module.learning_objectives.length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-foreground mb-2">
                        Öğrenme Hedefleri:
                      </p>
                      <div className="space-y-1">
                        {module.learning_objectives
                          .slice(0, 2)
                          .map((objective, idx) => (
                            <div
                              key={idx}
                              className="text-xs text-muted-foreground flex items-start gap-2"
                            >
                              <span className="text-purple-500">•</span>
                              <span className="line-clamp-1">
                                {objective.length > 80
                                  ? objective.substring(0, 80) + "..."
                                  : objective}
                              </span>
                            </div>
                          ))}
                        {module.learning_objectives.length > 2 && (
                          <p className="text-xs text-muted-foreground italic">
                            +{module.learning_objectives.length - 2} hedef
                            daha...
                          </p>
                        )}
                      </div>
                    </div>
                  )}

                {/* Quick Actions */}
                <div className="flex items-center gap-2 pt-2 border-t border-border">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => fetchModuleAnalytics(module.module_id)}
                    className="flex items-center gap-2 text-xs"
                  >
                    <BarChart3 className="w-3 h-3" />
                    Analitik
                  </Button>

                  <Button
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-2 text-xs"
                  >
                    <Users className="w-3 h-3" />
                    Öğrenciler
                  </Button>

                  {module.extraction_metadata && (
                    <span className="text-xs text-muted-foreground ml-auto">
                      Güven:{" "}
                      {(
                        module.extraction_metadata.confidence_score * 100
                      ).toFixed(1)}
                      %
                    </span>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Module Detail Modal */}
      {selectedModule && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-card border border-border rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-border">
              <div>
                <h3 className="text-xl font-semibold text-foreground">
                  {selectedModule.module_title}
                </h3>
                <p className="text-sm text-muted-foreground">
                  Modül {selectedModule.module_order} •{" "}
                  {getDifficultyLabel(selectedModule.difficulty_level)}
                </p>
              </div>
              <Button variant="outline" onClick={() => setSelectedModule(null)}>
                Kapat
              </Button>
            </div>

            <div className="p-6 overflow-y-auto space-y-6">
              {/* Module Description */}
              {selectedModule.module_description && (
                <div>
                  <h4 className="text-sm font-medium text-foreground mb-2">
                    Açıklama
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    {selectedModule.module_description}
                  </p>
                </div>
              )}

              {/* Learning Objectives */}
              {selectedModule.learning_objectives &&
                selectedModule.learning_objectives.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-foreground mb-3">
                      Öğrenme Hedefleri (
                      {selectedModule.learning_objectives.length})
                    </h4>
                    <ul className="space-y-2">
                      {selectedModule.learning_objectives.map(
                        (objective, idx) => (
                          <li
                            key={idx}
                            className="text-sm text-muted-foreground flex items-start gap-2"
                          >
                            <span className="text-purple-500 mt-1">•</span>
                            <span>{objective}</span>
                          </li>
                        )
                      )}
                    </ul>
                  </div>
                )}

              {/* Module Metadata */}
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-medium text-foreground mb-3">
                    Modül Bilgileri
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">
                        Tahmini Süre:
                      </span>
                      <span>
                        {selectedModule.estimated_duration_minutes
                          ? `${Math.round(
                              selectedModule.estimated_duration_minutes / 60
                            )} saat`
                          : "Belirsiz"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Durum:</span>
                      <span
                        className={
                          selectedModule.is_active
                            ? "text-green-600"
                            : "text-gray-600"
                        }
                      >
                        {selectedModule.is_active ? "Aktif" : "Pasif"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">
                        Oluşturulma:
                      </span>
                      <span>
                        {new Date(selectedModule.created_at).toLocaleString(
                          "tr-TR"
                        )}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">
                        Güncellenme:
                      </span>
                      <span>
                        {new Date(selectedModule.updated_at).toLocaleString(
                          "tr-TR"
                        )}
                      </span>
                    </div>
                  </div>
                </div>

                {selectedModule.extraction_metadata && (
                  <div>
                    <h4 className="text-sm font-medium text-foreground mb-3">
                      Çıkarım Bilgileri
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">
                          Güven Skoru:
                        </span>
                        <span>
                          {(
                            selectedModule.extraction_metadata
                              .confidence_score * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">
                          Konu Sayısı:
                        </span>
                        <span>
                          {selectedModule.extraction_metadata.topic_count}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">
                          Kelime Sayısı:
                        </span>
                        <span>
                          {selectedModule.extraction_metadata.word_count?.toLocaleString(
                            "tr-TR"
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Yöntem:</span>
                        <span>
                          {selectedModule.extraction_metadata.extraction_method}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Edit Module Modal */}
      {editingModule && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-card border border-border rounded-lg shadow-xl max-w-2xl w-full">
            <div className="flex items-center justify-between px-6 py-4 border-b border-border">
              <h3 className="text-lg font-semibold text-foreground">
                Modül Düzenle
              </h3>
              <Button variant="outline" onClick={() => setEditingModule(null)}>
                İptal
              </Button>
            </div>

            <form
              className="p-6 space-y-4"
              onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.currentTarget);
                const updates: ModuleEditForm = {
                  module_title: formData.get("title") as string,
                  module_description: formData.get("description") as string,
                  difficulty_level: formData.get("difficulty") as any,
                  estimated_duration_minutes:
                    parseInt(formData.get("duration") as string) || 0,
                  learning_objectives: (formData.get("objectives") as string)
                    .split("\n")
                    .filter((obj) => obj.trim())
                    .map((obj) => obj.trim()),
                  prerequisites: [], // Could be enhanced to edit prerequisites
                  is_active: formData.get("active") === "on",
                };
                handleUpdateModule(editingModule.module_id, updates);
              }}
            >
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Modül Başlığı
                </label>
                <input
                  type="text"
                  name="title"
                  defaultValue={editingModule.module_title}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Açıklama
                </label>
                <textarea
                  name="description"
                  defaultValue={editingModule.module_description || ""}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500 resize-none"
                  rows={3}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Zorluk Seviyesi
                  </label>
                  <select
                    name="difficulty"
                    defaultValue={editingModule.difficulty_level}
                    className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="beginner">Başlangıç</option>
                    <option value="intermediate">Orta</option>
                    <option value="advanced">İleri</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Tahmini Süre (dakika)
                  </label>
                  <input
                    type="number"
                    name="duration"
                    defaultValue={
                      editingModule.estimated_duration_minutes || ""
                    }
                    className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500"
                    min={0}
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Öğrenme Hedefleri (her satırda bir hedef)
                </label>
                <textarea
                  name="objectives"
                  defaultValue={
                    editingModule.learning_objectives?.join("\n") || ""
                  }
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500 resize-none"
                  rows={4}
                  placeholder="Öğrenci bu modül sonunda...&#10;Temel kavramları anlayabilecek&#10;Örnekleri çözebilecek"
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  name="active"
                  id="active"
                  defaultChecked={editingModule.is_active}
                  className="rounded border-border"
                />
                <label htmlFor="active" className="text-sm text-foreground">
                  Modül aktif
                </label>
              </div>

              <div className="flex items-center gap-2 pt-4 border-t border-border">
                <Button
                  type="submit"
                  className="bg-purple-600 hover:bg-purple-700"
                >
                  Güncelle
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setEditingModule(null)}
                >
                  İptal
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModuleManagementDashboard;
