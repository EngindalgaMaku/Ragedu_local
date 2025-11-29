"use client";

import React, { useState, useEffect } from "react";
import {
  Module,
  ModuleProgress,
  StudentModuleOverview as StudentModuleOverviewType,
  getCompletionStatusColor,
  getCompletionStatusLabel,
  getDifficultyColor,
  getDifficultyLabel,
  calculateModuleProgress,
} from "@/types/modules";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  BookOpen,
  Target,
  Clock,
  Star,
  Award,
  TrendingUp,
  PlayCircle,
  PauseCircle,
  CheckCircle,
  Lock,
  Unlock,
  ChevronRight,
  BarChart3,
  Brain,
  Zap,
  Calendar,
  User,
  Trophy,
  Medal,
  Gift,
  Flame,
  RefreshCw,
} from "lucide-react";

interface StudentModuleOverviewProps {
  sessionId: string;
  userId: number;
  onModuleSelect?: (module: Module) => void;
}

const StudentModuleOverview: React.FC<StudentModuleOverviewProps> = ({
  sessionId,
  userId,
  onModuleSelect,
}) => {
  // State Management
  const [overviewData, setOverviewData] =
    useState<StudentModuleOverviewType | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModuleId, setSelectedModuleId] = useState<number | null>(null);

  // Load student module overview
  const fetchOverviewData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `/api/aprag/modules/student-overview/${userId}/${sessionId}`
      );
      if (!response.ok) {
        throw new Error("Modül genel bakış verileri yüklenemedi");
      }

      const data: StudentModuleOverviewType = await response.json();
      setOverviewData(data);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Start or continue module
  const startModule = async (moduleId: number) => {
    try {
      const response = await fetch(`/api/aprag/modules/${moduleId}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      });

      if (response.ok) {
        await fetchOverviewData();
        if (onModuleSelect) {
          const module = overviewData?.available_modules.find(
            (m) => m.module.module_id === moduleId
          )?.module;
          if (module) {
            onModuleSelect(module);
          }
        }
      }
    } catch (error) {
      console.error("Module start failed:", error);
    }
  };

  // Get achievement icon
  const getAchievementIcon = (type: string) => {
    switch (type) {
      case "module_completed":
        return Trophy;
      case "objective_mastered":
        return Medal;
      case "streak":
        return Flame;
      case "excellence":
        return Star;
      default:
        return Gift;
    }
  };

  // Get mastery level display
  const getMasteryLevelDisplay = (level: string) => {
    switch (level) {
      case "advanced":
        return {
          label: "İleri Seviye",
          color:
            "text-purple-600 bg-purple-100 dark:bg-purple-900/30 dark:text-purple-300",
          icon: Brain,
        };
      case "proficient":
        return {
          label: "Yeterli",
          color:
            "text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300",
          icon: Award,
        };
      case "developing":
        return {
          label: "Gelişiyor",
          color:
            "text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-300",
          icon: TrendingUp,
        };
      case "beginner":
        return {
          label: "Başlangıç",
          color:
            "text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-300",
          icon: User,
        };
      default:
        return {
          label: "Belirsiz",
          color:
            "text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-300",
          icon: User,
        };
    }
  };

  // Load data on mount
  useEffect(() => {
    if (sessionId && userId) {
      fetchOverviewData();
    }
  }, [sessionId, userId]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Card className="p-8">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-2 border-blue-600 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-muted-foreground">
              Modül bilgileri yükleniyor...
            </p>
          </div>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="p-6 border-red-200 bg-red-50 dark:bg-red-900/20 dark:border-red-800">
        <div className="text-center">
          <div className="w-12 h-12 bg-red-100 text-red-600 rounded-lg flex items-center justify-center mx-auto mb-4">
            <BookOpen className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-medium text-red-900 dark:text-red-200 mb-2">
            Modüller Yüklenemedi
          </h3>
          <p className="text-red-700 dark:text-red-300 mb-4">{error}</p>
          <Button
            onClick={fetchOverviewData}
            variant="outline"
            className="border-red-300 text-red-700"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Yeniden Dene
          </Button>
        </div>
      </Card>
    );
  }

  if (!overviewData) {
    return (
      <Card className="p-8">
        <div className="text-center">
          <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-foreground mb-2">
            Modül Verisi Bulunamadı
          </h3>
          <p className="text-muted-foreground">
            Bu oturum için henüz modül oluşturulmamış
          </p>
        </div>
      </Card>
    );
  }

  const masteryDisplay = getMasteryLevelDisplay(
    overviewData.learning_path.mastery_level
  );
  const MasteryIcon = masteryDisplay.icon;

  return (
    <div className="space-y-6">
      {/* Learning Progress Header */}
      <Card className="p-6 bg-gradient-to-r from-blue-500 to-purple-600 text-white">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h2 className="text-2xl font-bold mb-2">Öğrenme Yolculuğun</h2>
            <div className="flex items-center gap-4 mb-4">
              <div className="flex items-center gap-2">
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${masteryDisplay.color
                    .replace("text-", "text-")
                    .replace("bg-", "bg-white/20 text-white")}`}
                >
                  <MasteryIcon className="w-4 h-4 mr-1 inline" />
                  {masteryDisplay.label}
                </span>
              </div>
              <div className="text-sm">
                <span className="opacity-90">
                  {overviewData.learning_path.completed_modules}/
                  {overviewData.learning_path.total_modules} modül tamamlandı
                </span>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Genel İlerleme</span>
                <span className="text-sm font-bold">
                  {Math.round(
                    (overviewData.learning_path.completed_modules /
                      overviewData.learning_path.total_modules) *
                      100
                  )}
                  %
                </span>
              </div>
              <div className="w-full bg-white/20 rounded-full h-3">
                <div
                  className="bg-white rounded-full h-3 transition-all duration-500"
                  style={{
                    width: `${
                      (overviewData.learning_path.completed_modules /
                        overviewData.learning_path.total_modules) *
                      100
                    }%`,
                  }}
                />
              </div>
            </div>
          </div>

          <div className="text-right">
            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mb-2">
              <BarChart3 className="w-8 h-8" />
            </div>
            <p className="text-sm opacity-90">Aktif Öğrenci</p>
          </div>
        </div>
      </Card>

      {/* Current Module */}
      {overviewData.current_module && (
        <Card className="p-6 border-2 border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <PlayCircle className="w-5 h-5 text-blue-600" />
                <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-200">
                  Devam Eden Modül
                </h3>
              </div>

              <h4 className="text-xl font-bold text-foreground mb-2">
                Modül {overviewData.current_module.module.module_order}:{" "}
                {overviewData.current_module.module.module_title}
              </h4>

              {overviewData.current_module.module.module_description && (
                <p className="text-muted-foreground mb-4">
                  {overviewData.current_module.module.module_description}
                </p>
              )}

              <div className="flex items-center gap-4 mb-4">
                <span
                  className={`px-2 py-1 text-xs font-medium rounded ${getDifficultyColor(
                    overviewData.current_module.module.difficulty_level
                  )}`}
                >
                  {getDifficultyLabel(
                    overviewData.current_module.module.difficulty_level
                  )}
                </span>
                <span
                  className={`px-2 py-1 text-xs font-medium rounded ${getCompletionStatusColor(
                    overviewData.current_module.progress.completion_status
                  )}`}
                >
                  {getCompletionStatusLabel(
                    overviewData.current_module.progress.completion_status
                  )}
                </span>
                {overviewData.current_module.module
                  .estimated_duration_minutes && (
                  <span className="flex items-center gap-1 text-sm text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    {Math.round(
                      overviewData.current_module.module
                        .estimated_duration_minutes / 60
                    )}{" "}
                    saat
                  </span>
                )}
              </div>

              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-foreground">
                    Modül İlerlemesi
                  </span>
                  <span className="text-sm font-bold text-blue-600">
                    {overviewData.current_module.progress.progress_percentage.toFixed(
                      0
                    )}
                    %
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${overviewData.current_module.progress.progress_percentage}%`,
                    }}
                  />
                </div>
              </div>

              {overviewData.current_module.current_objective && (
                <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border">
                  <p className="text-sm font-medium text-foreground mb-1">
                    Şu anki hedef:
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {
                      overviewData.current_module.current_objective
                        .objective_text
                    }
                  </p>
                  <div className="mt-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                    <div
                      className="bg-green-500 h-1 rounded-full"
                      style={{
                        width: `${overviewData.current_module.current_objective.progress_percentage}%`,
                      }}
                    />
                  </div>
                </div>
              )}
            </div>

            <Button
              onClick={() =>
                startModule(overviewData.current_module!.module.module_id)
              }
              className="bg-blue-600 hover:bg-blue-700 ml-4"
            >
              <PlayCircle className="w-4 h-4 mr-2" />
              Devam Et
            </Button>
          </div>
        </Card>
      )}

      {/* Available Modules */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            Mevcut Modüller ({overviewData.available_modules.length})
          </h3>

          {overviewData.learning_path.next_recommended && (
            <div className="flex items-center gap-2 text-sm text-green-600">
              <Zap className="w-4 h-4" />
              <span>
                Önerilen: Modül{" "}
                {overviewData.learning_path.next_recommended.module_order}
              </span>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {overviewData.available_modules.map(
            ({
              module,
              progress,
              is_unlocked,
              prerequisites_met,
              estimated_time_remaining,
            }) => {
              const progressData = progress
                ? calculateModuleProgress(progress)
                : null;
              const isCompleted =
                progress?.completion_status === "completed" ||
                progress?.completion_status === "mastered";
              const isInProgress =
                progress?.completion_status === "in_progress";
              const isRecommended =
                overviewData.learning_path.next_recommended?.module_id ===
                module.module_id;

              return (
                <Card
                  key={module.module_id}
                  className={`p-4 transition-all hover:shadow-md cursor-pointer ${
                    !is_unlocked ? "opacity-60" : ""
                  } ${
                    isRecommended
                      ? "ring-2 ring-green-500 bg-green-50 dark:bg-green-900/10"
                      : ""
                  } ${
                    selectedModuleId === module.module_id
                      ? "ring-2 ring-blue-500"
                      : ""
                  }`}
                  onClick={() => {
                    if (is_unlocked) {
                      setSelectedModuleId(module.module_id);
                      if (onModuleSelect) {
                        onModuleSelect(module);
                      }
                    }
                  }}
                >
                  <div className="space-y-3">
                    {/* Module Header */}
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-1 bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300 text-xs font-medium rounded">
                          Modül {module.module_order}
                        </span>
                        {isRecommended && (
                          <span className="px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 text-xs font-medium rounded flex items-center gap-1">
                            <Zap className="w-3 h-3" />
                            Önerilen
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-1">
                        {isCompleted ? (
                          <CheckCircle className="w-5 h-5 text-green-600" />
                        ) : isInProgress ? (
                          <PlayCircle className="w-5 h-5 text-blue-600" />
                        ) : is_unlocked ? (
                          <Unlock className="w-5 h-5 text-gray-600" />
                        ) : (
                          <Lock className="w-5 h-5 text-gray-400" />
                        )}
                      </div>
                    </div>

                    {/* Module Info */}
                    <div>
                      <h4 className="text-sm font-semibold text-foreground mb-1 line-clamp-2">
                        {module.module_title}
                      </h4>

                      <div className="flex items-center gap-2 mb-2">
                        <span
                          className={`px-2 py-0.5 text-xs font-medium rounded ${getDifficultyColor(
                            module.difficulty_level
                          )}`}
                        >
                          {getDifficultyLabel(module.difficulty_level)}
                        </span>
                        {progress && (
                          <span
                            className={`px-2 py-0.5 text-xs font-medium rounded ${getCompletionStatusColor(
                              progress.completion_status
                            )}`}
                          >
                            {getCompletionStatusLabel(
                              progress.completion_status
                            )}
                          </span>
                        )}
                      </div>

                      {module.module_description && (
                        <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                          {module.module_description}
                        </p>
                      )}
                    </div>

                    {/* Progress */}
                    {progress && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">
                            İlerleme
                          </span>
                          <span className="text-xs font-medium text-foreground">
                            {progress.progress_percentage.toFixed(0)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                          <div
                            className={`h-1.5 rounded-full transition-all duration-300 ${
                              isCompleted
                                ? "bg-green-500"
                                : isInProgress
                                ? "bg-blue-500"
                                : "bg-gray-400"
                            }`}
                            style={{
                              width: `${progress.progress_percentage}%`,
                            }}
                          />
                        </div>
                        {progressData && (
                          <div className="flex items-center justify-between text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Target className="w-3 h-3" />
                              {progressData.objectivesCompleted}/
                              {progressData.totalObjectives}
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {progressData.timeSpent}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Module Stats */}
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Target className="w-3 h-3" />
                        {module.learning_objectives?.length || 0} hedef
                      </span>
                      {module.estimated_duration_minutes && (
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {Math.round(module.estimated_duration_minutes / 60)}s
                        </span>
                      )}
                    </div>

                    {/* Prerequisites Warning */}
                    {!prerequisites_met && !is_unlocked && (
                      <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded text-xs text-yellow-800 dark:text-yellow-300">
                        <div className="flex items-center gap-1">
                          <Lock className="w-3 h-3" />
                          <span>Önkoşullar tamamlanmalı</span>
                        </div>
                      </div>
                    )}

                    {/* Action Button */}
                    <Button
                      size="sm"
                      disabled={!is_unlocked}
                      onClick={(e) => {
                        e.stopPropagation();
                        startModule(module.module_id);
                      }}
                      className={`w-full ${
                        isCompleted
                          ? "bg-green-600 hover:bg-green-700"
                          : isInProgress
                          ? "bg-blue-600 hover:bg-blue-700"
                          : "bg-purple-600 hover:bg-purple-700"
                      }`}
                    >
                      {isCompleted ? (
                        <>
                          <CheckCircle className="w-4 h-4 mr-1" />
                          İncele
                        </>
                      ) : isInProgress ? (
                        <>
                          <PlayCircle className="w-4 h-4 mr-1" />
                          Devam Et
                        </>
                      ) : is_unlocked ? (
                        <>
                          <PlayCircle className="w-4 h-4 mr-1" />
                          Başla
                        </>
                      ) : (
                        <>
                          <Lock className="w-4 h-4 mr-1" />
                          Kilitli
                        </>
                      )}
                    </Button>
                  </div>
                </Card>
              );
            }
          )}
        </div>
      </Card>

      {/* Achievements */}
      {overviewData.achievements.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Award className="w-5 h-5" />
            Son Başarılar
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {overviewData.achievements.slice(0, 6).map((achievement, idx) => {
              const AchievementIcon = getAchievementIcon(
                achievement.achievement_type
              );

              return (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-3 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800"
                >
                  <div className="w-10 h-10 bg-yellow-500 text-white rounded-full flex items-center justify-center flex-shrink-0">
                    <AchievementIcon className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">
                      {achievement.title}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(achievement.earned_at).toLocaleDateString(
                        "tr-TR"
                      )}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>

          {overviewData.achievements.length > 6 && (
            <div className="text-center mt-4">
              <Button variant="outline" size="sm">
                Tüm Başarıları Gör ({overviewData.achievements.length})
                <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
            </div>
          )}
        </Card>
      )}
    </div>
  );
};

export default StudentModuleOverview;
