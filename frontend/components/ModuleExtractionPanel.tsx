"use client";

import React, { useState, useEffect } from "react";
import {
  ModuleExtractionRequest,
  ModuleExtractionResponse,
  Module,
  ModuleListResponse,
  ModuleExtractionJob,
  CurriculumPromptForm,
  DEFAULT_EXTRACTION_CONFIG,
  ModuleOperationStatus,
  getDifficultyColor,
  getDifficultyLabel,
} from "@/types/modules";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  BookOpen,
  Settings,
  Play,
  RefreshCw,
  CheckCircle,
  Clock,
  AlertCircle,
  Users,
  Target,
  BarChart3,
  PlusCircle,
  Edit,
  Trash2,
  Eye,
  Search,
} from "lucide-react";
import { createModuleFromRAG } from "@/lib/api";

interface ModuleExtractionPanelProps {
  sessionId?: string;
  sessionName?: string;
  onModuleChange?: (modules: Module[]) => void;
}

const ModuleExtractionPanel: React.FC<ModuleExtractionPanelProps> = ({
  sessionId: initialSessionId,
  sessionName: initialSessionName,
  onModuleChange,
}) => {
  // State Management
  const [selectedSessionId] = useState<string>(
    initialSessionId || ""
  );
  const [selectedSessionName, setSelectedSessionName] = useState<string>(
    initialSessionName || ""
  );
  const [modules, setModules] = useState<Module[]>([]);
  const [extractionJob, setExtractionJob] =
    useState<ModuleExtractionJob | null>(null);
  const [operationStatus, setOperationStatus] = useState<ModuleOperationStatus>(
    {
      operation: "extract",
      status: "idle",
    }
  );

  // Form State
  const [showExtractionForm, setShowExtractionForm] = useState(false);
  const [showRAGModuleForm, setShowRAGModuleForm] = useState(false);
  const [ragModuleTitle, setRAGModuleTitle] = useState("");
  const [ragModuleDescription, setRAGModuleDescription] = useState("");
  const [curriculumForm, setCurriculumForm] = useState<CurriculumPromptForm>({
    curriculum_prompt:
      "Türkiye eğitim sisteminde lise düzeyi güncel eğitim müfredatını göz önüne alarak mevcut konuları ders modüllerine ayırır mısın",
    subject_area: "Genel",
    grade_level: "Lise",
    education_system: "Türkiye Millî Eğitim Bakanlığı",
    max_modules: DEFAULT_EXTRACTION_CONFIG.max_modules,
    difficulty_distribution: {
      ...DEFAULT_EXTRACTION_CONFIG.difficulty_distribution,
    },
    include_assessments: DEFAULT_EXTRACTION_CONFIG.include_assessments,
    curriculum_alignment: DEFAULT_EXTRACTION_CONFIG.curriculum_alignment,
  });

  // UI State
  const [expandedModules, setExpandedModules] = useState<Set<number>>(
    new Set()
  );
  const [selectedModule, setSelectedModule] = useState<Module | null>(null);
  const [moduleTopics, setModuleTopics] = useState<Map<number, any[]>>(new Map());
  const [loadingTopics, setLoadingTopics] = useState<Set<number>>(new Set());


  // Fetch session name if not provided
  const fetchSessionName = async (sessionId: string) => {
    if (selectedSessionName) {
      return; // Already have session name
    }

    try {
      const response = await fetch(`/api/sessions/${sessionId}`);
      if (response.ok) {
        const sessionData = await response.json();
        if (sessionData.name || sessionData.session_name) {
          setSelectedSessionName(sessionData.name || sessionData.session_name);
        }
      }
    } catch (error) {
      console.error("Error fetching session name:", error);
    }
  };

  // Fetch existing modules for session
  const fetchModules = async () => {
    if (!selectedSessionId) {
      return;
    }

    try {
      setOperationStatus({ operation: "extract", status: "loading" });

      // Fetch session name if not available
      if (!selectedSessionName) {
        await fetchSessionName(selectedSessionId);
      }

      const response = await fetch(
        `/api/aprag/modules/session/${selectedSessionId}`
      );
      if (!response.ok) {
        throw new Error("Modüller yüklenemedi");
      }

      const data: ModuleListResponse = await response.json();
      if (data.success) {
        setModules(data.modules);
        if (onModuleChange) {
          onModuleChange(data.modules);
        }

        // Check if there's an active extraction job
        if (data.extraction_job && data.extraction_job.status !== "completed") {
          pollJobStatus(data.extraction_job.job_id);
        }
      }

      setOperationStatus({ operation: "extract", status: "success" });
    } catch (error: any) {
      setOperationStatus({
        operation: "extract",
        status: "error",
        error: {
          error_type: "api_error",
          error_code: "FETCH_MODULES_FAILED",
          message: error.message,
        },
      });
    }
  };

  // Fetch topics for a specific module
  const fetchModuleTopics = async (moduleId: number) => {
    if (loadingTopics.has(moduleId)) {
      return; // Already loading
    }

    try {
      setLoadingTopics(prev => new Set(prev).add(moduleId));
      
      const response = await fetch(`/api/aprag/modules/${moduleId}`);
      if (!response.ok) {
        throw new Error("Modül detayları yüklenemedi");
      }

      const data = await response.json();
      if (data.success && data.module && data.module.topics) {
        setModuleTopics(prev => {
          const newMap = new Map(prev);
          newMap.set(moduleId, data.module.topics);
          return newMap;
        });
      }
    } catch (error) {
      console.error(`Error fetching topics for module ${moduleId}:`, error);
    } finally {
      setLoadingTopics(prev => {
        const newSet = new Set(prev);
        newSet.delete(moduleId);
        return newSet;
      });
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
      setOperationStatus({
        operation: "extract",
        status: "loading",
        message: "Modül siliniyor...",
      });

      const response = await fetch(`/api/aprag/modules/${moduleId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail || errorData.message || "Modül silinemedi"
        );
      }

      // Refresh modules list
      await fetchModules();

      setOperationStatus({
        operation: "extract",
        status: "success",
        message: `"${moduleTitle}" modülü başarıyla silindi`,
      });
    } catch (error: any) {
      setOperationStatus({
        operation: "extract",
        status: "error",
        error: {
          error_type: "api_error",
          error_code: "DELETE_MODULE_FAILED",
          message: error.message || "Modül silinirken bir hata oluştu",
        },
      });
    }
  };

  // Create module from RAG
  const handleCreateModuleFromRAG = async () => {
    if (!ragModuleTitle.trim()) {
      setOperationStatus({
        operation: "extract",
        status: "error",
        error: {
          error_type: "validation_error",
          error_code: "EMPTY_MODULE_TITLE",
          message: "Lütfen modül ismi girin",
        },
      });
      return;
    }

    if (!selectedSessionId) {
      setOperationStatus({
        operation: "extract",
        status: "error",
        error: {
          error_type: "validation_error",
          error_code: "NO_SESSION",
          message: "Lütfen bir ders oturumu seçin",
        },
      });
      return;
    }

    try {
      setOperationStatus({
        operation: "extract",
        status: "loading",
        message: "RAG ile modül oluşturuluyor...",
      });

      const result = await createModuleFromRAG({
        session_id: selectedSessionId,
        module_title: ragModuleTitle,
        module_description: ragModuleDescription || undefined,
        top_k: 20,
        similarity_threshold: 0.6,
        use_hybrid_search: true,
      });

      // Reset form
      setRAGModuleTitle("");
      setRAGModuleDescription("");
      setShowRAGModuleForm(false);

      // Refresh modules list
      await fetchModules();

      setOperationStatus({
        operation: "extract",
        status: "success",
        message: `${result.message} (${result.topics_added} konu eklendi)`,
      });
    } catch (error: any) {
      setOperationStatus({
        operation: "extract",
        status: "error",
        error: {
          error_type: "api_error",
          error_code: "CREATE_MODULE_FROM_RAG_FAILED",
          message: error.message || "Modül oluşturulurken bir hata oluştu",
        },
      });
    }
  };

  // Extract modules with curriculum prompt
  const handleExtractModules = async () => {
    try {
      setOperationStatus({
        operation: "extract",
        status: "loading",
        progress: 0,
      });

      if (!selectedSessionId) {
        throw new Error("Lütfen bir ders oturumu seçin");
      }

      const request: ModuleExtractionRequest = {
        session_id: selectedSessionId,
        curriculum_prompt: curriculumForm.curriculum_prompt,
        extraction_config: {
          max_modules: curriculumForm.max_modules,
          difficulty_distribution: curriculumForm.difficulty_distribution,
          include_assessments: curriculumForm.include_assessments,
          curriculum_alignment: curriculumForm.curriculum_alignment,
          force_refresh: true,
        },
      };

      const response = await fetch("/api/aprag/modules/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error("Modül çıkarımı başlatılamadı");
      }

      const data: ModuleExtractionResponse = await response.json();
      if (data.success) {
        setOperationStatus({
          operation: "extract",
          status: "loading",
          progress: 10,
          message: `Modül çıkarımı başlatıldı (Job ID: ${data.job_id})`,
        });

        // Start polling for job status
        pollJobStatus(data.job_id);
        setShowExtractionForm(false);
      } else {
        throw new Error("Modül çıkarımı başarısız");
      }
    } catch (error: any) {
      setOperationStatus({
        operation: "extract",
        status: "error",
        error: {
          error_type: "extraction_failed",
          error_code: "MODULE_EXTRACTION_FAILED",
          message: error.message,
        },
      });
    }
  };

  // Poll job status
  const pollJobStatus = (jobId: number) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/aprag/modules/job/${jobId}/status`);
        if (!response.ok) {
          console.warn("Job status polling failed:", response.status);
          return;
        }

        const job: ModuleExtractionJob = await response.json();
        setExtractionJob(job);

        if (job.status === "completed") {
          clearInterval(pollInterval);
          setOperationStatus({
            operation: "extract",
            status: "success",
            progress: 100,
            message: `✅ ${
              job.result_summary?.modules_extracted || 0
            } modül başarıyla çıkarıldı!`,
          });
          await fetchModules();
        } else if (job.status === "failed") {
          clearInterval(pollInterval);
          setOperationStatus({
            operation: "extract",
            status: "error",
            error: {
              error_type: "extraction_failed",
              error_code: "JOB_FAILED",
              message:
                job.error_details?.error_message ||
                "Modül çıkarımı başarısız oldu",
            },
          });
        } else if (job.status === "in_progress") {
          const progress = Math.min(
            90,
            30 +
              ((Date.now() - new Date(job.created_at).getTime()) / 1000 / 60) *
                10
          );
          setOperationStatus({
            operation: "extract",
            status: "loading",
            progress: Math.round(progress),
            message: "Modüller çıkarılıyor ve düzenleniyor...",
          });
        }
      } catch (error) {
        console.error("Job polling error:", error);
      }
    }, 3000);

    // Timeout after 10 minutes
    setTimeout(() => {
      clearInterval(pollInterval);
      if (operationStatus.status === "loading") {
        setOperationStatus({
          operation: "extract",
          status: "error",
          error: {
            error_type: "extraction_failed",
            error_code: "TIMEOUT",
            message: "İşlem çok uzun sürdü, lütfen sayfayı yenileyin",
          },
        });
      }
    }, 600000);
  };

  // Toggle module expansion
  const toggleModule = async (moduleId: number) => {
    const newExpanded = new Set(expandedModules);
    if (newExpanded.has(moduleId)) {
      newExpanded.delete(moduleId);
    } else {
      newExpanded.add(moduleId);
      // Fetch module topics when expanding
      if (!moduleTopics.has(moduleId)) {
        await fetchModuleTopics(moduleId);
      }
    }
    setExpandedModules(newExpanded);
  };

  // Load session name and modules on mount if session is available
  useEffect(() => {
    if (selectedSessionId) {
      // Fetch session name if not provided
      if (!selectedSessionName) {
        fetchSessionName(selectedSessionId);
      }
      fetchModules();
    }
  }, [selectedSessionId]);

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-foreground">
                Modül Çıkarımı ve Yönetimi
              </h2>
              <p className="text-sm text-muted-foreground">
                Eğitim içeriklerini modüllere ayırarak organize edin
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowRAGModuleForm(!showRAGModuleForm)}
              className="flex items-center gap-2"
              disabled={!selectedSessionId}
            >
              <Search className="w-4 h-4" />
              RAG ile Modül Oluştur
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowExtractionForm(!showExtractionForm)}
              className="flex items-center gap-2"
              disabled={!selectedSessionId}
            >
              <Settings className="w-4 h-4" />
              Çıkarım Ayarları
            </Button>

            <Button
              onClick={handleExtractModules}
              disabled={
                operationStatus.status === "loading" || !selectedSessionId
              }
              className="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
            >
              {operationStatus.status === "loading" ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Modülleri Çıkar
            </Button>
          </div>
        </div>

        {/* Active Session Info */}
        {selectedSessionId && (
          <div className="border-t border-border pt-4">
            <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
              <p className="text-sm text-purple-700 dark:text-purple-300">
                <span className="font-medium">Aktif Oturum:</span>{" "}
                {selectedSessionName || "Yükleniyor..."}
              </p>
              <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
                Bu oturumdaki konulardan eğitim modülleri çıkarılacak
              </p>
            </div>
          </div>
        )}

        {/* Status Display */}
        {operationStatus.status !== "idle" && (
          <div
            className={`p-4 rounded-lg border ${
              operationStatus.status === "error"
                ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
                : operationStatus.status === "success"
                ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                : "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800"
            }`}
          >
            <div className="flex items-center gap-3">
              {operationStatus.status === "loading" && (
                <RefreshCw className="w-5 h-5 animate-spin text-blue-600" />
              )}
              {operationStatus.status === "success" && (
                <CheckCircle className="w-5 h-5 text-green-600" />
              )}
              {operationStatus.status === "error" && (
                <AlertCircle className="w-5 h-5 text-red-600" />
              )}

              <div className="flex-1">
                <p
                  className={`text-sm font-medium ${
                    operationStatus.status === "error"
                      ? "text-red-800 dark:text-red-200"
                      : operationStatus.status === "success"
                      ? "text-green-800 dark:text-green-200"
                      : "text-blue-800 dark:text-blue-200"
                  }`}
                >
                  {operationStatus.message ||
                    (operationStatus.status === "loading"
                      ? "İşleniyor..."
                      : operationStatus.status === "error"
                      ? "Hata oluştu"
                      : "İşlem tamamlandı")}
                </p>

                {operationStatus.progress !== undefined &&
                  operationStatus.status === "loading" && (
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${operationStatus.progress}%` }}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {operationStatus.progress}% tamamlandı
                      </p>
                    </div>
                  )}

                {operationStatus.error && (
                  <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                    {operationStatus.error.message}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* RAG Module Creation Form */}
      {showRAGModuleForm && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Search className="w-5 h-5" />
            RAG ile Modül Oluştur
          </h3>
          <p className="text-sm text-muted-foreground mb-4">
            Modül ismini girin, sistem RAG ile ilgili konuları bulup modüle ekleyecek.
          </p>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Modül İsmi <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={ragModuleTitle}
                onChange={(e) => setRAGModuleTitle(e.target.value)}
                className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                placeholder="Örnek: Hücre Yapısı ve Organeller"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Modül Açıklaması (Opsiyonel)
              </label>
              <textarea
                value={ragModuleDescription}
                onChange={(e) => setRAGModuleDescription(e.target.value)}
                className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground resize-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                rows={2}
                placeholder="Modül hakkında kısa bir açıklama..."
              />
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                onClick={handleCreateModuleFromRAG}
                disabled={!ragModuleTitle.trim() || operationStatus.status === "loading"}
                className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
              >
                {operationStatus.status === "loading" ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
                Modülü Oluştur
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setShowRAGModuleForm(false);
                  setRAGModuleTitle("");
                  setRAGModuleDescription("");
                }}
              >
                İptal
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Extraction Configuration Form */}
      {showExtractionForm && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Çıkarım Yapılandırması
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-foreground mb-2">
                Müfredat Yönergesi
              </label>
              <textarea
                value={curriculumForm.curriculum_prompt}
                onChange={(e) =>
                  setCurriculumForm((prev) => ({
                    ...prev,
                    curriculum_prompt: e.target.value,
                  }))
                }
                className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground resize-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                rows={3}
                placeholder="Örnek: Türkiye eğitim sisteminde lise düzeyi güncel eğitim müfredatını göz önüne alarak..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Konu Alanı
              </label>
              <select
                value={curriculumForm.subject_area}
                onChange={(e) =>
                  setCurriculumForm((prev) => ({
                    ...prev,
                    subject_area: e.target.value,
                  }))
                }
                className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              >
                <option value="Genel">Genel</option>
                <option value="Matematik">Matematik</option>
                <option value="Fen Bilimleri">Fen Bilimleri</option>
                <option value="Sosyal Bilimler">Sosyal Bilimler</option>
                <option value="Türk Dili ve Edebiyatı">
                  Türk Dili ve Edebiyatı
                </option>
                <option value="Yabancı Dil">Yabancı Dil</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Sınıf Düzeyi
              </label>
              <select
                value={curriculumForm.grade_level}
                onChange={(e) =>
                  setCurriculumForm((prev) => ({
                    ...prev,
                    grade_level: e.target.value,
                  }))
                }
                className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              >
                <option value="İlkokul">İlkokul (1-4)</option>
                <option value="Ortaokul">Ortaokul (5-8)</option>
                <option value="Lise">Lise (9-12)</option>
                <option value="Üniversite">Üniversite</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Maksimum Modül Sayısı
              </label>
              <input
                type="number"
                min={1}
                max={20}
                value={curriculumForm.max_modules}
                onChange={(e) =>
                  setCurriculumForm((prev) => ({
                    ...prev,
                    max_modules: parseInt(e.target.value) || 8,
                  }))
                }
                className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Zorluk Dağılımı (%)
              </label>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs w-16">Başlangıç:</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={curriculumForm.difficulty_distribution.beginner}
                    onChange={(e) =>
                      setCurriculumForm((prev) => ({
                        ...prev,
                        difficulty_distribution: {
                          ...prev.difficulty_distribution,
                          beginner: parseInt(e.target.value),
                        },
                      }))
                    }
                    className="flex-1"
                  />
                  <span className="text-xs w-8">
                    {curriculumForm.difficulty_distribution.beginner}%
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs w-16">Orta:</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={curriculumForm.difficulty_distribution.intermediate}
                    onChange={(e) =>
                      setCurriculumForm((prev) => ({
                        ...prev,
                        difficulty_distribution: {
                          ...prev.difficulty_distribution,
                          intermediate: parseInt(e.target.value),
                        },
                      }))
                    }
                    className="flex-1"
                  />
                  <span className="text-xs w-8">
                    {curriculumForm.difficulty_distribution.intermediate}%
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs w-16">İleri:</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={curriculumForm.difficulty_distribution.advanced}
                    onChange={(e) =>
                      setCurriculumForm((prev) => ({
                        ...prev,
                        difficulty_distribution: {
                          ...prev.difficulty_distribution,
                          advanced: parseInt(e.target.value),
                        },
                      }))
                    }
                    className="flex-1"
                  />
                  <span className="text-xs w-8">
                    {curriculumForm.difficulty_distribution.advanced}%
                  </span>
                </div>
              </div>
            </div>

            <div className="md:col-span-2">
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={curriculumForm.include_assessments}
                    onChange={(e) =>
                      setCurriculumForm((prev) => ({
                        ...prev,
                        include_assessments: e.target.checked,
                      }))
                    }
                    className="rounded border-border"
                  />
                  <span className="text-sm text-foreground">
                    Değerlendirme önerilerini dahil et
                  </span>
                </label>

                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={curriculumForm.curriculum_alignment}
                    onChange={(e) =>
                      setCurriculumForm((prev) => ({
                        ...prev,
                        curriculum_alignment: e.target.checked,
                      }))
                    }
                    className="rounded border-border"
                  />
                  <span className="text-sm text-foreground">
                    Müfredat uyumluluğunu kontrol et
                  </span>
                </label>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 mt-6">
            <Button
              onClick={handleExtractModules}
              disabled={operationStatus.status === "loading"}
              className="bg-purple-600 hover:bg-purple-700"
            >
              Çıkarımı Başlat
            </Button>

            <Button
              variant="outline"
              onClick={() => setShowExtractionForm(false)}
            >
              İptal
            </Button>
          </div>
        </Card>
      )}

      {/* Modules List */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <Target className="w-5 h-5" />
            Çıkarılan Modüller ({modules.length})
          </h3>

          {modules.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={fetchModules}
              className="flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Yenile
            </Button>
          )}
        </div>

        {modules.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center mx-auto mb-4">
              <BookOpen className="w-8 h-8 text-gray-400" />
            </div>
            <h4 className="text-lg font-medium text-foreground mb-2">
              Henüz modül çıkarılmamış
            </h4>
            <p className="text-sm text-muted-foreground mb-4">
              Eğitim içeriklerinizi modüllere ayırmak için çıkarım işlemini
              başlatın
            </p>
            <Button
              onClick={() => setShowExtractionForm(true)}
              className="bg-purple-600 hover:bg-purple-700"
            >
              <PlusCircle className="w-4 h-4 mr-2" />
              Modül Çıkarımını Başlat
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            {modules
              .sort((a, b) => a.module_order - b.module_order)
              .map((module) => {
                const isExpanded = expandedModules.has(module.module_id);

                return (
                  <div
                    key={module.module_id}
                    className="border border-border rounded-lg p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
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

                          {module.estimated_duration_minutes && (
                            <span className="px-2 py-1 bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300 text-xs font-medium rounded flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {Math.round(
                                module.estimated_duration_minutes / 60
                              )}
                              s
                            </span>
                          )}

                          {module.is_active && (
                            <span className="px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 text-xs font-medium rounded">
                              Aktif
                            </span>
                          )}
                        </div>

                        <h4 className="text-base font-semibold text-foreground mb-1">
                          {module.module_title}
                        </h4>

                        {module.module_description && (
                          <p className="text-sm text-muted-foreground mb-3">
                            {module.module_description}
                          </p>
                        )}

                        {module.learning_objectives &&
                          module.learning_objectives.length > 0 && (
                            <div className="mb-3">
                              <p className="text-xs font-medium text-foreground mb-1">
                                Öğrenme Hedefleri:
                              </p>
                              <div className="flex flex-wrap gap-1">
                                {module.learning_objectives
                                  .slice(0, 3)
                                  .map((objective, idx) => (
                                    <span
                                      key={idx}
                                      className="px-2 py-1 bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 text-xs rounded"
                                    >
                                      {objective.length > 50
                                        ? objective.substring(0, 50) + "..."
                                        : objective}
                                    </span>
                                  ))}
                                {module.learning_objectives.length > 3 && (
                                  <span className="px-2 py-1 bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400 text-xs rounded">
                                    +{module.learning_objectives.length - 3}{" "}
                                    daha
                                  </span>
                                )}
                              </div>
                            </div>
                          )}

                        {isExpanded && (
                          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg space-y-3">
                            <div>
                              <h5 className="text-sm font-medium text-foreground mb-2">
                                Modül Detayları
                              </h5>
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <span className="text-muted-foreground">
                                    Oluşturulma:
                                  </span>
                                  <span className="ml-2">
                                    {new Date(
                                      module.created_at
                                    ).toLocaleDateString("tr-TR")}
                                  </span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">
                                    Güncellenme:
                                  </span>
                                  <span className="ml-2">
                                    {new Date(
                                      module.updated_at
                                    ).toLocaleDateString("tr-TR")}
                                  </span>
                                </div>
                              </div>
                            </div>

                            {module.extraction_metadata && (
                              <div>
                                <h5 className="text-sm font-medium text-foreground mb-2">
                                  Çıkarım Bilgileri
                                </h5>
                                <div className="grid grid-cols-2 gap-4 text-sm">
                                  <div>
                                    <span className="text-muted-foreground">
                                      Güven Skoru:
                                    </span>
                                    <span className="ml-2">
                                      {(
                                        module.extraction_metadata
                                          .confidence_score * 100
                                      ).toFixed(1)}
                                      %
                                    </span>
                                  </div>
                                  <div>
                                    <span className="text-muted-foreground">
                                      Konu Sayısı:
                                    </span>
                                    <span className="ml-2">
                                      {module.extraction_metadata.topic_count}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            )}

                            {module.learning_objectives &&
                              module.learning_objectives.length > 3 && (
                                <div>
                                  <h5 className="text-sm font-medium text-foreground mb-2">
                                    Tüm Öğrenme Hedefleri
                                  </h5>
                                  <ul className="space-y-1 text-sm text-muted-foreground">
                                    {module.learning_objectives.map(
                                      (objective, idx) => (
                                        <li
                                          key={idx}
                                          className="flex items-start gap-2"
                                        >
                                          <span className="text-purple-500">
                                            •
                                          </span>
                                          <span>{objective}</span>
                                        </li>
                                      )
                                    )}
                                  </ul>
                                </div>
                              )}

                            {/* Module Topics */}
                            <div>
                              <h5 className="text-sm font-medium text-foreground mb-2">
                                Modüldeki Konular ({moduleTopics.get(module.module_id)?.length || module.extraction_metadata?.topic_count || 0})
                              </h5>
                              {loadingTopics.has(module.module_id) ? (
                                <div className="text-sm text-muted-foreground">
                                  Konular yükleniyor...
                                </div>
                              ) : moduleTopics.get(module.module_id) && moduleTopics.get(module.module_id)!.length > 0 ? (
                                <div className="space-y-2">
                                  {moduleTopics.get(module.module_id)!.map((topic: any, idx: number) => (
                                    <div
                                      key={topic.topic_id || idx}
                                      className="flex items-center gap-2 p-2 bg-white dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600"
                                    >
                                      <span className="text-xs font-medium text-purple-600 dark:text-purple-400">
                                        #{topic.topic_order_in_module || idx + 1}
                                      </span>
                                      <span className="text-sm text-foreground flex-1">
                                        {topic.topic_title || `Konu ${idx + 1}`}
                                      </span>
                                      {topic.importance_level && (
                                        <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
                                          {typeof topic.importance_level === 'string' 
                                            ? topic.importance_level 
                                            : topic.importance_level === 1 
                                            ? 'Yüksek' 
                                            : topic.importance_level === 2 
                                            ? 'Orta' 
                                            : 'Düşük'}
                                        </span>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-sm text-muted-foreground">
                                  Bu modülde henüz konu bulunmuyor.
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="flex items-center gap-2 ml-4">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={async () => {
                            const newExpanded = new Set(expandedModules);
                            if (newExpanded.has(module.module_id)) {
                              newExpanded.delete(module.module_id);
                            } else {
                              newExpanded.add(module.module_id);
                              // Fetch module topics when expanding
                              if (!moduleTopics.has(module.module_id)) {
                                await fetchModuleTopics(module.module_id);
                              }
                            }
                            setExpandedModules(newExpanded);
                          }}
                        >
                          <Eye className="w-4 h-4 mr-1" />
                          {isExpanded ? "Gizle" : "Detay"}
                        </Button>

                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedModule(module)}
                        >
                          <Edit className="w-4 h-4" />
                        </Button>

                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() =>
                            handleDeleteModule(
                              module.module_id,
                              module.module_title || `Modül ${module.module_order}`
                            )
                          }
                          className="text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        )}
      </Card>

      {/* Job Status Card (if job is running) */}
      {extractionJob && extractionJob.status !== "completed" && (
        <Card className="p-4 border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20">
          <div className="flex items-center gap-3">
            <RefreshCw className="w-5 h-5 animate-spin text-blue-600" />
            <div className="flex-1">
              <h4 className="text-sm font-medium text-blue-900 dark:text-blue-200">
                Çıkarım İşlemi Devam Ediyor
              </h4>
              <p className="text-xs text-blue-700 dark:text-blue-300">
                Job ID: {extractionJob.job_id} • Durum: {extractionJob.status} •
                Başlangıç:{" "}
                {new Date(extractionJob.created_at).toLocaleString("tr-TR")}
              </p>
              {extractionJob.result_summary && (
                <div className="mt-2 text-xs text-blue-600 dark:text-blue-400">
                  İşlenen: {extractionJob.result_summary.topics_organized} konu
                  • Çıkarılan: {extractionJob.result_summary.modules_extracted}{" "}
                  modül • Süre:{" "}
                  {extractionJob.result_summary.processing_time_ms}ms
                </div>
              )}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default ModuleExtractionPanel;
