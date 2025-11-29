"use client";

import React, { useState, useEffect } from "react";
import {
  Module,
  ModuleQualityReview,
  getDifficultyColor,
  getDifficultyLabel,
} from "@/types/modules";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Star,
  Eye,
  Edit,
  MessageSquare,
  BookOpen,
  Target,
  BarChart3,
  Lightbulb,
  Flag,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Send,
  Filter,
  Search,
  Calendar,
  User,
} from "lucide-react";

interface ModuleQualityReviewInterfaceProps {
  sessionId: string;
  modules: Module[];
  onReviewComplete?: (moduleId: number, review: ModuleQualityReview) => void;
}

interface ReviewCriteria {
  objective_clarity: number;
  content_alignment: number;
  difficulty_appropriateness: number;
  progression_logic: number;
  assessment_quality: number;
}

interface ReviewSuggestion {
  category:
    | "content"
    | "structure"
    | "difficulty"
    | "objectives"
    | "assessment";
  suggestion: string;
  priority: "high" | "medium" | "low";
}

const ModuleQualityReviewInterface: React.FC<
  ModuleQualityReviewInterfaceProps
> = ({ sessionId, modules, onReviewComplete }) => {
  // State Management
  const [pendingReviews, setPendingReviews] = useState<Module[]>([]);
  const [completedReviews, setCompletedReviews] = useState<
    ModuleQualityReview[]
  >([]);
  const [selectedModule, setSelectedModule] = useState<Module | null>(null);
  const [currentReview, setCurrentReview] =
    useState<Partial<ModuleQualityReview> | null>(null);
  const [reviewCriteria, setReviewCriteria] = useState<ReviewCriteria>({
    objective_clarity: 5,
    content_alignment: 5,
    difficulty_appropriateness: 5,
    progression_logic: 5,
    assessment_quality: 5,
  });
  const [reviewComments, setReviewComments] = useState("");
  const [suggestions, setSuggestions] = useState<ReviewSuggestion[]>([]);
  const [filterStatus, setFilterStatus] = useState<string>("pending");
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load reviews data
  const fetchReviews = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`/api/aprag/modules/reviews/${sessionId}`);
      if (!response.ok) {
        throw new Error("İncelemeler yüklenemedi");
      }

      const data = await response.json();

      // Separate pending and completed reviews
      const pending = modules.filter(
        (module) =>
          !data.reviews.some(
            (review: ModuleQualityReview) =>
              review.module_id === module.module_id
          )
      );

      setPendingReviews(pending);
      setCompletedReviews(data.reviews || []);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Start reviewing a module
  const startReview = (module: Module) => {
    setSelectedModule(module);
    setCurrentReview({
      module_id: module.module_id,
      review_status: "pending",
      quality_score: 0,
      review_comments: "",
      reviewer_id: 1, // This should come from auth context
      review_criteria: reviewCriteria,
      suggested_improvements: [],
    });
    setReviewComments("");
    setSuggestions([]);
  };

  // Add suggestion
  const addSuggestion = (
    category: ReviewSuggestion["category"],
    suggestion: string,
    priority: ReviewSuggestion["priority"]
  ) => {
    if (suggestion.trim()) {
      setSuggestions((prev) => [
        ...prev,
        { category, suggestion: suggestion.trim(), priority },
      ]);
    }
  };

  // Remove suggestion
  const removeSuggestion = (index: number) => {
    setSuggestions((prev) => prev.filter((_, i) => i !== index));
  };

  // Calculate quality score based on criteria
  const calculateQualityScore = () => {
    const scores = Object.values(reviewCriteria);
    const average =
      scores.reduce((sum, score) => sum + score, 0) / scores.length;
    return Math.round(average * 10); // Convert to 0-100 scale
  };

  // Submit review
  const submitReview = async (
    status: "approved" | "needs_revision" | "rejected"
  ) => {
    if (!selectedModule || !currentReview) return;

    try {
      setError(null);

      const qualityScore = calculateQualityScore();

      const reviewData: Partial<ModuleQualityReview> = {
        ...currentReview,
        review_status: status,
        quality_score: qualityScore,
        review_comments: reviewComments,
        review_criteria: reviewCriteria,
        suggested_improvements: suggestions,
        reviewed_at: new Date().toISOString(),
      };

      const response = await fetch(
        `/api/aprag/modules/${selectedModule.module_id}/review`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(reviewData),
        }
      );

      if (!response.ok) {
        throw new Error("İnceleme gönderilemedi");
      }

      const result = await response.json();

      if (onReviewComplete && result.review) {
        onReviewComplete(selectedModule.module_id, result.review);
      }

      // Reset state
      setSelectedModule(null);
      setCurrentReview(null);
      setReviewComments("");
      setSuggestions([]);

      // Refresh reviews
      await fetchReviews();
    } catch (error: any) {
      setError(error.message);
    }
  };

  // Filter modules based on search and status
  const filteredModules = React.useMemo(() => {
    let modules_to_filter =
      filterStatus === "pending"
        ? pendingReviews
        : filterStatus === "completed"
        ? (completedReviews
            .map((r) => modules.find((m) => m.module_id === r.module_id))
            .filter(Boolean) as Module[])
        : modules;

    if (searchQuery.trim()) {
      modules_to_filter = modules_to_filter.filter(
        (module) =>
          module.module_title
            .toLowerCase()
            .includes(searchQuery.toLowerCase()) ||
          module.module_description
            ?.toLowerCase()
            .includes(searchQuery.toLowerCase())
      );
    }

    return modules_to_filter;
  }, [pendingReviews, completedReviews, modules, filterStatus, searchQuery]);

  // Load data on mount
  useEffect(() => {
    if (sessionId && modules.length > 0) {
      fetchReviews();
    }
  }, [sessionId, modules.length]);

  // Get review status color
  const getReviewStatusColor = (status: string) => {
    switch (status) {
      case "approved":
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
      case "needs_revision":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300";
      case "rejected":
        return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300";
    }
  };

  const getReviewStatusLabel = (status: string) => {
    switch (status) {
      case "approved":
        return "Onaylandı";
      case "needs_revision":
        return "Düzeltme Gerek";
      case "rejected":
        return "Reddedildi";
      default:
        return "Bekliyor";
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";
      case "medium":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300";
      case "low":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
              <CheckCircle className="w-6 h-6 text-green-600" />
              Modül Kalite İncelemesi
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Çıkarılan modüllerin kalitesini değerlendirin ve onaylayın
            </p>
          </div>

          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Clock className="w-4 h-4" />
            <span>{pendingReviews.length} bekleyen</span>
            <span>•</span>
            <CheckCircle className="w-4 h-4" />
            <span>{completedReviews.length} tamamlanan</span>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Modül ara..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500"
            />
          </div>

          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500"
          >
            <option value="pending">Bekleyen İncelemeler</option>
            <option value="completed">Tamamlanan İncelemeler</option>
            <option value="all">Tüm Modüller</option>
          </select>
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

      {/* Modules List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-foreground">
            Modül Listesi ({filteredModules.length})
          </h3>

          {loading ? (
            <Card className="p-6">
              <div className="text-center">
                <div className="animate-spin h-6 w-6 border-2 border-purple-600 border-t-transparent rounded-full mx-auto mb-2"></div>
                <p className="text-sm text-muted-foreground">Yükleniyor...</p>
              </div>
            </Card>
          ) : filteredModules.length === 0 ? (
            <Card className="p-6">
              <div className="text-center">
                <BookOpen className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-sm text-muted-foreground">
                  {filterStatus === "pending"
                    ? "Bekleyen inceleme yok"
                    : "Modül bulunamadı"}
                </p>
              </div>
            </Card>
          ) : (
            <div className="space-y-3">
              {filteredModules.map((module) => {
                const completedReview = completedReviews.find(
                  (r) => r.module_id === module.module_id
                );
                const isPending = !completedReview;

                return (
                  <Card
                    key={module.module_id}
                    className={`p-4 cursor-pointer transition-all hover:shadow-md ${
                      selectedModule?.module_id === module.module_id
                        ? "ring-2 ring-purple-500"
                        : ""
                    }`}
                    onClick={() =>
                      !selectedModule && isPending && startReview(module)
                    }
                  >
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
                          {completedReview && (
                            <span
                              className={`px-2 py-1 text-xs font-medium rounded ${getReviewStatusColor(
                                completedReview.review_status
                              )}`}
                            >
                              {getReviewStatusLabel(
                                completedReview.review_status
                              )}
                            </span>
                          )}
                        </div>

                        <h4 className="text-base font-semibold text-foreground mb-1">
                          {module.module_title}
                        </h4>

                        {module.module_description && (
                          <p className="text-sm text-muted-foreground line-clamp-2 mb-2">
                            {module.module_description}
                          </p>
                        )}

                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Target className="w-3 h-3" />
                            {module.learning_objectives?.length || 0} hedef
                          </span>
                          {completedReview && (
                            <>
                              <span className="flex items-center gap-1">
                                <Star className="w-3 h-3" />
                                {completedReview.quality_score}/100
                              </span>
                              <span className="flex items-center gap-1">
                                <Calendar className="w-3 h-3" />
                                {new Date(
                                  completedReview.reviewed_at
                                ).toLocaleDateString("tr-TR")}
                              </span>
                            </>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2 ml-3">
                        {isPending ? (
                          <Button
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              startReview(module);
                            }}
                            className="bg-purple-600 hover:bg-purple-700"
                          >
                            <Eye className="w-4 h-4 mr-1" />
                            İncele
                          </Button>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedModule(module);
                            }}
                          >
                            <Eye className="w-4 h-4" />
                          </Button>
                        )}
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          )}
        </div>

        {/* Review Panel */}
        <div className="space-y-4">
          {selectedModule ? (
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-foreground">
                  Modül İncelemesi
                </h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setSelectedModule(null);
                    setCurrentReview(null);
                  }}
                >
                  Kapat
                </Button>
              </div>

              <div className="space-y-6">
                {/* Module Info */}
                <div className="p-4 bg-muted/30 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-2 py-1 bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300 text-xs font-medium rounded">
                      Modül {selectedModule.module_order}
                    </span>
                    <span
                      className={`px-2 py-1 text-xs font-medium rounded ${getDifficultyColor(
                        selectedModule.difficulty_level
                      )}`}
                    >
                      {getDifficultyLabel(selectedModule.difficulty_level)}
                    </span>
                  </div>
                  <h4 className="text-base font-semibold text-foreground mb-2">
                    {selectedModule.module_title}
                  </h4>
                  {selectedModule.module_description && (
                    <p className="text-sm text-muted-foreground">
                      {selectedModule.module_description}
                    </p>
                  )}
                </div>

                {/* Learning Objectives */}
                {selectedModule.learning_objectives &&
                  selectedModule.learning_objectives.length > 0 && (
                    <div>
                      <h5 className="text-sm font-medium text-foreground mb-2">
                        Öğrenme Hedefleri (
                        {selectedModule.learning_objectives.length})
                      </h5>
                      <ul className="space-y-1">
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

                {currentReview && (
                  <>
                    {/* Quality Criteria */}
                    <div>
                      <h5 className="text-sm font-medium text-foreground mb-3">
                        Kalite Kriterleri (1-10 arası puanlayın)
                      </h5>
                      <div className="space-y-3">
                        {Object.entries(reviewCriteria).map(([key, value]) => {
                          const labels = {
                            objective_clarity: "Hedef Netliği",
                            content_alignment: "İçerik Uyumluluğu",
                            difficulty_appropriateness: "Zorluk Uygunluğu",
                            progression_logic: "İlerleme Mantığı",
                            assessment_quality: "Değerlendirme Kalitesi",
                          };

                          return (
                            <div
                              key={key}
                              className="flex items-center justify-between"
                            >
                              <span className="text-sm text-foreground">
                                {labels[key as keyof typeof labels]}
                              </span>
                              <div className="flex items-center gap-2">
                                <input
                                  type="range"
                                  min={1}
                                  max={10}
                                  value={value}
                                  onChange={(e) =>
                                    setReviewCriteria((prev) => ({
                                      ...prev,
                                      [key]: parseInt(e.target.value),
                                    }))
                                  }
                                  className="w-20"
                                />
                                <span className="w-6 text-sm font-medium text-foreground">
                                  {value}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-sm">
                        <span className="font-medium text-blue-900 dark:text-blue-200">
                          Toplam Puan: {calculateQualityScore()}/100
                        </span>
                      </div>
                    </div>

                    {/* Review Comments */}
                    <div>
                      <h5 className="text-sm font-medium text-foreground mb-2">
                        İnceleme Yorumları
                      </h5>
                      <textarea
                        value={reviewComments}
                        onChange={(e) => setReviewComments(e.target.value)}
                        placeholder="Modül hakkında detaylı yorumlarınızı yazın..."
                        className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-purple-500 resize-none"
                        rows={4}
                      />
                    </div>

                    {/* Suggestions */}
                    <div>
                      <h5 className="text-sm font-medium text-foreground mb-2">
                        İyileştirme Önerileri
                      </h5>

                      {/* Add Suggestion Form */}
                      <div className="p-3 border border-border rounded-lg mb-3">
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-2">
                          <select
                            id="suggestion-category"
                            className="px-2 py-1 text-sm border border-border rounded bg-background"
                            defaultValue="content"
                          >
                            <option value="content">İçerik</option>
                            <option value="structure">Yapı</option>
                            <option value="difficulty">Zorluk</option>
                            <option value="objectives">Hedefler</option>
                            <option value="assessment">Değerlendirme</option>
                          </select>
                          <select
                            id="suggestion-priority"
                            className="px-2 py-1 text-sm border border-border rounded bg-background"
                            defaultValue="medium"
                          >
                            <option value="high">Yüksek</option>
                            <option value="medium">Orta</option>
                            <option value="low">Düşük</option>
                          </select>
                          <Button
                            size="sm"
                            onClick={() => {
                              const category = (
                                document.getElementById(
                                  "suggestion-category"
                                ) as HTMLSelectElement
                              ).value as ReviewSuggestion["category"];
                              const priority = (
                                document.getElementById(
                                  "suggestion-priority"
                                ) as HTMLSelectElement
                              ).value as ReviewSuggestion["priority"];
                              const suggestion = (
                                document.getElementById(
                                  "suggestion-text"
                                ) as HTMLTextAreaElement
                              ).value;
                              if (suggestion.trim()) {
                                addSuggestion(category, suggestion, priority);
                                (
                                  document.getElementById(
                                    "suggestion-text"
                                  ) as HTMLTextAreaElement
                                ).value = "";
                              }
                            }}
                            className="w-full"
                          >
                            <Lightbulb className="w-4 h-4 mr-1" />
                            Ekle
                          </Button>
                        </div>
                        <textarea
                          id="suggestion-text"
                          placeholder="İyileştirme önerinizi yazın..."
                          className="w-full px-2 py-1 text-sm border border-border rounded bg-background resize-none"
                          rows={2}
                        />
                      </div>

                      {/* Suggestions List */}
                      {suggestions.length > 0 && (
                        <div className="space-y-2">
                          {suggestions.map((suggestion, idx) => (
                            <div
                              key={idx}
                              className="p-2 border border-border rounded flex items-start justify-between"
                            >
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="px-1 py-0.5 bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 text-xs rounded">
                                    {suggestion.category}
                                  </span>
                                  <span
                                    className={`px-1 py-0.5 text-xs rounded ${getPriorityColor(
                                      suggestion.priority
                                    )}`}
                                  >
                                    {suggestion.priority}
                                  </span>
                                </div>
                                <p className="text-sm text-foreground">
                                  {suggestion.suggestion}
                                </p>
                              </div>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => removeSuggestion(idx)}
                                className="ml-2"
                              >
                                <XCircle className="w-4 h-4" />
                              </Button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Review Actions */}
                    <div className="flex items-center gap-2 pt-4 border-t border-border">
                      <Button
                        onClick={() => submitReview("approved")}
                        className="flex-1 bg-green-600 hover:bg-green-700"
                      >
                        <ThumbsUp className="w-4 h-4 mr-2" />
                        Onayla
                      </Button>
                      <Button
                        onClick={() => submitReview("needs_revision")}
                        variant="outline"
                        className="flex-1 border-yellow-300 text-yellow-700 hover:bg-yellow-50"
                      >
                        <RotateCcw className="w-4 h-4 mr-2" />
                        Düzeltme İste
                      </Button>
                      <Button
                        onClick={() => submitReview("rejected")}
                        variant="outline"
                        className="flex-1 border-red-300 text-red-700 hover:bg-red-50"
                      >
                        <ThumbsDown className="w-4 h-4 mr-2" />
                        Reddet
                      </Button>
                    </div>
                  </>
                )}

                {/* Show completed review details */}
                {!currentReview &&
                  (() => {
                    const completedReview = completedReviews.find(
                      (r) => r.module_id === selectedModule.module_id
                    );
                    if (!completedReview) return null;

                    return (
                      <div className="space-y-4">
                        <div className="p-4 bg-muted/30 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <span
                              className={`px-2 py-1 text-sm font-medium rounded ${getReviewStatusColor(
                                completedReview.review_status
                              )}`}
                            >
                              {getReviewStatusLabel(
                                completedReview.review_status
                              )}
                            </span>
                            <span className="text-sm text-muted-foreground">
                              Puan: {completedReview.quality_score}/100
                            </span>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            İnceleme:{" "}
                            {new Date(
                              completedReview.reviewed_at
                            ).toLocaleString("tr-TR")}
                          </p>
                        </div>

                        {completedReview.review_comments && (
                          <div>
                            <h5 className="text-sm font-medium text-foreground mb-2">
                              Yorumlar
                            </h5>
                            <p className="text-sm text-muted-foreground">
                              {completedReview.review_comments}
                            </p>
                          </div>
                        )}

                        {completedReview.suggested_improvements.length > 0 && (
                          <div>
                            <h5 className="text-sm font-medium text-foreground mb-2">
                              Öneriler
                            </h5>
                            <div className="space-y-2">
                              {completedReview.suggested_improvements.map(
                                (suggestion, idx) => (
                                  <div
                                    key={idx}
                                    className="p-2 border border-border rounded"
                                  >
                                    <div className="flex items-center gap-2 mb-1">
                                      <span className="px-1 py-0.5 bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 text-xs rounded">
                                        {suggestion.category}
                                      </span>
                                      <span
                                        className={`px-1 py-0.5 text-xs rounded ${getPriorityColor(
                                          suggestion.priority
                                        )}`}
                                      >
                                        {suggestion.priority}
                                      </span>
                                    </div>
                                    <p className="text-sm text-foreground">
                                      {suggestion.suggestion}
                                    </p>
                                  </div>
                                )
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })()}
              </div>
            </Card>
          ) : (
            <Card className="p-8">
              <div className="text-center">
                <MessageSquare className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-foreground mb-2">
                  İnceleme Seçin
                </h3>
                <p className="text-sm text-muted-foreground">
                  İncelemek istediğiniz modüle tıklayın
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModuleQualityReviewInterface;
