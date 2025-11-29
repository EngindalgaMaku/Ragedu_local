"use client";

import React, { useState, useEffect } from "react";
import {
  Module,
  ModuleProgress,
  ModuleAnalytics,
  LearningPathAnalytics,
  getCompletionStatusColor,
  getCompletionStatusLabel,
  getDifficultyLabel,
  calculateModuleProgress,
} from "@/types/modules";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  TrendingUp,
  Users,
  Clock,
  Target,
  Award,
  BarChart3,
  PieChart,
  Activity,
  Calendar,
  Filter,
  Download,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  CheckCircle,
  PlayCircle,
  PauseCircle,
  User,
  BookOpen,
  Star,
  Zap,
  Brain,
} from "lucide-react";

interface ModuleProgressMonitoringDashboardProps {
  sessionId: string;
  modules: Module[];
}

interface StudentProgress {
  user_id: number;
  user_name: string;
  user_email: string;
  total_modules: number;
  completed_modules: number;
  in_progress_modules: number;
  not_started_modules: number;
  overall_progress_percentage: number;
  total_time_spent_minutes: number;
  last_activity: string;
  module_progress: ModuleProgress[];
  current_module?: {
    module: Module;
    progress: ModuleProgress;
  };
  achievements: Array<{
    type: string;
    title: string;
    earned_at: string;
  }>;
  learning_velocity: number; // modules per week
  mastery_level: "beginner" | "developing" | "proficient" | "advanced";
}

interface ProgressStats {
  total_students: number;
  active_students: number;
  average_completion_rate: number;
  total_time_spent_hours: number;
  modules_completed: number;
  students_struggling: number;
  students_excelling: number;
  completion_by_module: { [moduleId: number]: number };
  difficulty_progression: {
    beginner_completion: number;
    intermediate_completion: number;
    advanced_completion: number;
  };
  engagement_trends: {
    daily_active_users: number[];
    weekly_progress: number[];
    module_completion_times: { [moduleId: number]: number };
  };
}

const ModuleProgressMonitoringDashboard: React.FC<
  ModuleProgressMonitoringDashboardProps
> = ({ sessionId, modules }) => {
  // State Management
  const [studentProgress, setStudentProgress] = useState<StudentProgress[]>([]);
  const [progressStats, setProgressStats] = useState<ProgressStats | null>(
    null
  );
  const [learningPathAnalytics, setLearningPathAnalytics] =
    useState<LearningPathAnalytics | null>(null);
  const [selectedModule, setSelectedModule] = useState<number | null>(null);
  const [selectedStudent, setSelectedStudent] =
    useState<StudentProgress | null>(null);
  const [expandedStudents, setExpandedStudents] = useState<Set<number>>(
    new Set()
  );
  const [timeFilter, setTimeFilter] = useState<"week" | "month" | "all">(
    "week"
  );
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<
    "name" | "progress" | "activity" | "completion"
  >("progress");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch progress data
  const fetchProgressData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch student progress
      const progressResponse = await fetch(
        `/api/aprag/modules/progress/${sessionId}`
      );
      if (!progressResponse.ok) {
        throw new Error("Öğrenci ilerlemesi yüklenemedi");
      }
      const progressData = await progressResponse.json();
      setStudentProgress(progressData.students || []);

      // Fetch progress statistics
      const statsResponse = await fetch(
        `/api/aprag/modules/progress/stats/${sessionId}?timeframe=${timeFilter}`
      );
      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setProgressStats(statsData);
      }

      // Fetch learning path analytics
      const analyticsResponse = await fetch(
        `/api/aprag/modules/analytics/learning-path/${sessionId}`
      );
      if (analyticsResponse.ok) {
        const analyticsData = await analyticsResponse.json();
        setLearningPathAnalytics(analyticsData);
      }
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Export progress report
  const exportProgressReport = async () => {
    try {
      const response = await fetch(
        `/api/aprag/modules/progress/export/${sessionId}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            format: "excel",
            timeframe: timeFilter,
            include_individual_progress: true,
          }),
        }
      );

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `modül-ilerleme-raporu-${
          new Date().toISOString().split("T")[0]
        }.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error("Export failed:", error);
    }
  };

  // Filter and sort students
  const filteredAndSortedStudents = React.useMemo(() => {
    let filtered = studentProgress;

    // Apply status filter
    if (statusFilter !== "all") {
      filtered = filtered.filter((student) => {
        switch (statusFilter) {
          case "completed":
            return student.completed_modules === student.total_modules;
          case "in_progress":
            return student.in_progress_modules > 0;
          case "struggling":
            return (
              student.overall_progress_percentage < 30 &&
              student.in_progress_modules > 0
            );
          case "excelling":
            return student.overall_progress_percentage > 80;
          default:
            return true;
        }
      });
    }

    // Apply sorting
    return filtered.sort((a, b) => {
      switch (sortBy) {
        case "name":
          return a.user_name.localeCompare(b.user_name, "tr");
        case "progress":
          return b.overall_progress_percentage - a.overall_progress_percentage;
        case "activity":
          return (
            new Date(b.last_activity).getTime() -
            new Date(a.last_activity).getTime()
          );
        case "completion":
          return b.completed_modules - a.completed_modules;
        default:
          return 0;
      }
    });
  }, [studentProgress, statusFilter, sortBy]);

  // Toggle student expansion
  const toggleStudent = (userId: number) => {
    const newExpanded = new Set(expandedStudents);
    if (newExpanded.has(userId)) {
      newExpanded.delete(userId);
    } else {
      newExpanded.add(userId);
    }
    setExpandedStudents(newExpanded);
  };

  // Get module completion color
  const getModuleCompletionColor = (completionRate: number) => {
    if (completionRate >= 80)
      return "text-green-600 bg-green-100 dark:bg-green-900/30";
    if (completionRate >= 60)
      return "text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30";
    if (completionRate >= 30)
      return "text-orange-600 bg-orange-100 dark:bg-orange-900/30";
    return "text-red-600 bg-red-100 dark:bg-red-900/30";
  };

  // Get mastery level display
  const getMasteryLevelDisplay = (level: string) => {
    switch (level) {
      case "advanced":
        return {
          label: "İleri",
          color: "text-purple-600 bg-purple-100 dark:bg-purple-900/30",
          icon: Star,
        };
      case "proficient":
        return {
          label: "Yeterli",
          color: "text-blue-600 bg-blue-100 dark:bg-blue-900/30",
          icon: Award,
        };
      case "developing":
        return {
          label: "Gelişiyor",
          color: "text-green-600 bg-green-100 dark:bg-green-900/30",
          icon: TrendingUp,
        };
      case "beginner":
        return {
          label: "Başlangıç",
          color: "text-gray-600 bg-gray-100 dark:bg-gray-800",
          icon: PlayCircle,
        };
      default:
        return {
          label: "Belirsiz",
          color: "text-gray-600 bg-gray-100 dark:bg-gray-800",
          icon: User,
        };
    }
  };

  // Load data on mount and filter changes
  useEffect(() => {
    if (sessionId && modules.length > 0) {
      fetchProgressData();
    }
  }, [sessionId, modules.length, timeFilter]);

  return (
    <div className="space-y-6">
      {/* Statistics Overview */}
      {progressStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-100 text-blue-600 rounded-lg flex items-center justify-center">
                <Users className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">
                  {progressStats.total_students}
                </p>
                <p className="text-sm text-muted-foreground">Toplam Öğrenci</p>
                <p className="text-xs text-green-600">
                  {progressStats.active_students} aktif
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-100 text-green-600 rounded-lg flex items-center justify-center">
                <Target className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">
                  {(progressStats.average_completion_rate * 100).toFixed(1)}%
                </p>
                <p className="text-sm text-muted-foreground">
                  Ortalama Tamamlama
                </p>
                <p className="text-xs text-blue-600">
                  {progressStats.modules_completed} modül
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-100 text-purple-600 rounded-lg flex items-center justify-center">
                <Clock className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">
                  {Math.round(progressStats.total_time_spent_hours)}s
                </p>
                <p className="text-sm text-muted-foreground">Toplam Çalışma</p>
                <p className="text-xs text-purple-600">
                  {Math.round(
                    progressStats.total_time_spent_hours /
                      progressStats.total_students
                  )}
                  s/öğrenci
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-orange-100 text-orange-600 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">
                  {progressStats.students_excelling}
                </p>
                <p className="text-sm text-muted-foreground">
                  Başarılı Öğrenci
                </p>
                <p className="text-xs text-red-600">
                  {progressStats.students_struggling} zorlananlar
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Module Completion Overview */}
      {modules.length > 0 && progressStats && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Modül Tamamlanma Oranları
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {modules.map((module) => {
              const completionRate =
                (progressStats.completion_by_module?.[module.module_id] || 0) *
                100;
              return (
                <div
                  key={module.module_id}
                  className="p-4 border border-border rounded-lg"
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-foreground truncate">
                      Modül {module.module_order}: {module.module_title}
                    </h4>
                    <span
                      className={`px-2 py-1 text-xs font-medium rounded ${getModuleCompletionColor(
                        completionRate
                      )}`}
                    >
                      {completionRate.toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-2">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${completionRate}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{getDifficultyLabel(module.difficulty_level)}</span>
                    <span>
                      {Math.round(
                        (progressStats.completion_by_module?.[
                          module.module_id
                        ] || 0) * progressStats.total_students
                      )}{" "}
                      öğrenci
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Controls and Filters */}
      <Card className="p-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <Users className="w-5 h-5" />
              Öğrenci İlerlemeleri ({filteredAndSortedStudents.length})
            </h2>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <select
              value={timeFilter}
              onChange={(e) => setTimeFilter(e.target.value as any)}
              className="px-3 py-1.5 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-purple-500"
            >
              <option value="week">Bu Hafta</option>
              <option value="month">Bu Ay</option>
              <option value="all">Tüm Zamanlar</option>
            </select>

            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-3 py-1.5 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-purple-500"
            >
              <option value="all">Tüm Öğrenciler</option>
              <option value="completed">Tamamlayanlar</option>
              <option value="in_progress">Devam Edenler</option>
              <option value="struggling">Zorlunanlar</option>
              <option value="excelling">Başarılılar</option>
            </select>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-3 py-1.5 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-purple-500"
            >
              <option value="progress">İlerlemeye Göre</option>
              <option value="name">Ada Göre</option>
              <option value="activity">Aktiviteye Göre</option>
              <option value="completion">Tamamlamaya Göre</option>
            </select>

            <Button
              variant="outline"
              size="sm"
              onClick={exportProgressReport}
              className="flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Rapor Al
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={fetchProgressData}
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
            <AlertCircle className="w-5 h-5 text-red-600" />
            <p className="text-red-800 dark:text-red-200">{error}</p>
          </div>
        </Card>
      )}

      {/* Student Progress List */}
      {loading ? (
        <Card className="p-8">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-2 border-purple-600 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-muted-foreground">
              İlerleme verileri yükleniyor...
            </p>
          </div>
        </Card>
      ) : filteredAndSortedStudents.length === 0 ? (
        <Card className="p-8">
          <div className="text-center">
            <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">
              Öğrenci Bulunamadı
            </h3>
            <p className="text-muted-foreground">
              {studentProgress.length === 0
                ? "Henüz hiç öğrenci modüllere başlamamış"
                : "Filtrelere uygun öğrenci bulunamadı"}
            </p>
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {filteredAndSortedStudents.map((student) => {
            const isExpanded = expandedStudents.has(student.user_id);
            const masteryDisplay = getMasteryLevelDisplay(
              student.mastery_level
            );
            const MasteryIcon = masteryDisplay.icon;

            return (
              <Card key={student.user_id} className="overflow-hidden">
                <div
                  className="p-4 cursor-pointer hover:bg-muted/30 transition-colors"
                  onClick={() => toggleStudent(student.user_id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="flex items-center gap-2">
                        {isExpanded ? (
                          <ChevronDown className="w-4 h-4 text-muted-foreground" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-muted-foreground" />
                        )}
                        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                          <span className="text-white text-sm font-bold">
                            {student.user_name.charAt(0).toUpperCase()}
                          </span>
                        </div>
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="text-base font-semibold text-foreground truncate">
                            {student.user_name}
                          </h3>
                          <span
                            className={`px-2 py-1 text-xs font-medium rounded flex items-center gap-1 ${masteryDisplay.color}`}
                          >
                            <MasteryIcon className="w-3 h-3" />
                            {masteryDisplay.label}
                          </span>
                        </div>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span>
                            {student.completed_modules}/{student.total_modules}{" "}
                            modül
                          </span>
                          <span>
                            {student.overall_progress_percentage.toFixed(1)}%
                            tamamlandı
                          </span>
                          <span>
                            Son aktivite:{" "}
                            {new Date(student.last_activity).toLocaleDateString(
                              "tr-TR"
                            )}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                            style={{
                              width: `${student.overall_progress_percentage}%`,
                            }}
                          />
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          {student.total_time_spent_minutes > 0
                            ? `${Math.round(
                                student.total_time_spent_minutes / 60
                              )}s`
                            : "0s"}
                        </p>
                      </div>

                      <div className="flex items-center gap-1">
                        {student.completed_modules === student.total_modules ? (
                          <CheckCircle className="w-5 h-5 text-green-600" />
                        ) : student.in_progress_modules > 0 ? (
                          <PlayCircle className="w-5 h-5 text-blue-600" />
                        ) : (
                          <PauseCircle className="w-5 h-5 text-gray-400" />
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Expanded Student Details */}
                {isExpanded && (
                  <div className="px-4 pb-4 border-t border-border bg-muted/20">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
                      {/* Module Progress */}
                      <div>
                        <h4 className="text-sm font-medium text-foreground mb-3 flex items-center gap-2">
                          <BookOpen className="w-4 h-4" />
                          Modül İlerlemesi
                        </h4>
                        <div className="space-y-2 max-h-60 overflow-y-auto">
                          {student.module_progress.map((progress) => {
                            const module = modules.find(
                              (m) => m.module_id === progress.module_id
                            );
                            if (!module) return null;

                            const progressData =
                              calculateModuleProgress(progress);

                            return (
                              <div
                                key={progress.progress_id}
                                className="flex items-center justify-between p-2 bg-background rounded border"
                              >
                                <div className="flex-1">
                                  <p className="text-sm font-medium text-foreground truncate">
                                    Modül {module.module_order}:{" "}
                                    {module.module_title}
                                  </p>
                                  <div className="flex items-center gap-2 mt-1">
                                    <span
                                      className={`px-2 py-0.5 text-xs font-medium rounded ${getCompletionStatusColor(
                                        progress.completion_status
                                      )}`}
                                    >
                                      {getCompletionStatusLabel(
                                        progress.completion_status
                                      )}
                                    </span>
                                    <span className="text-xs text-muted-foreground">
                                      {progressData.timeSpent}
                                    </span>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                                    <div
                                      className="bg-blue-500 h-1 rounded-full"
                                      style={{
                                        width: `${progress.progress_percentage}%`,
                                      }}
                                    />
                                  </div>
                                  <p className="text-xs text-muted-foreground mt-1">
                                    {progress.progress_percentage.toFixed(0)}%
                                  </p>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Student Stats & Achievements */}
                      <div>
                        <h4 className="text-sm font-medium text-foreground mb-3 flex items-center gap-2">
                          <Award className="w-4 h-4" />
                          İstatistikler & Başarılar
                        </h4>

                        <div className="space-y-3">
                          {/* Learning Stats */}
                          <div className="p-3 bg-background rounded border">
                            <div className="grid grid-cols-2 gap-3 text-sm">
                              <div className="flex items-center justify-between">
                                <span className="text-muted-foreground">
                                  Öğrenme Hızı:
                                </span>
                                <span className="font-medium flex items-center gap-1">
                                  <Zap className="w-3 h-3 text-yellow-500" />
                                  {student.learning_velocity.toFixed(1)}/hafta
                                </span>
                              </div>
                              <div className="flex items-center justify-between">
                                <span className="text-muted-foreground">
                                  Aktif Gün:
                                </span>
                                <span className="font-medium">
                                  {Math.ceil(
                                    (Date.now() -
                                      new Date(
                                        student.last_activity
                                      ).getTime()) /
                                      (1000 * 60 * 60 * 24)
                                  )}{" "}
                                  gün önce
                                </span>
                              </div>
                              <div className="flex items-center justify-between">
                                <span className="text-muted-foreground">
                                  Devam Eden:
                                </span>
                                <span className="font-medium text-blue-600">
                                  {student.in_progress_modules} modül
                                </span>
                              </div>
                              <div className="flex items-center justify-between">
                                <span className="text-muted-foreground">
                                  Başlanmamış:
                                </span>
                                <span className="font-medium text-gray-600">
                                  {student.not_started_modules} modül
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Current Module */}
                          {student.current_module && (
                            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
                              <p className="text-sm font-medium text-blue-900 dark:text-blue-200 mb-1">
                                Şu anki modül:
                              </p>
                              <p className="text-sm text-blue-800 dark:text-blue-300">
                                Modül{" "}
                                {student.current_module.module.module_order}:{" "}
                                {student.current_module.module.module_title}
                              </p>
                              <div className="mt-2 w-full bg-blue-200 dark:bg-blue-800 rounded-full h-1">
                                <div
                                  className="bg-blue-500 h-1 rounded-full"
                                  style={{
                                    width: `${student.current_module.progress.progress_percentage}%`,
                                  }}
                                />
                              </div>
                            </div>
                          )}

                          {/* Achievements */}
                          {student.achievements.length > 0 && (
                            <div>
                              <p className="text-sm font-medium text-foreground mb-2">
                                Son Başarılar:
                              </p>
                              <div className="space-y-1">
                                {student.achievements
                                  .slice(0, 3)
                                  .map((achievement, idx) => (
                                    <div
                                      key={idx}
                                      className="flex items-center gap-2 text-sm"
                                    >
                                      <Star className="w-3 h-3 text-yellow-500" />
                                      <span className="text-muted-foreground">
                                        {achievement.title} •{" "}
                                        {new Date(
                                          achievement.earned_at
                                        ).toLocaleDateString("tr-TR")}
                                      </span>
                                    </div>
                                  ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}

      {/* Learning Path Analytics */}
      {learningPathAnalytics && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Öğrenme Yolu Analitikleri
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <h4 className="text-sm font-medium text-purple-900 dark:text-purple-200 mb-2">
                Yol Tamamlama
              </h4>
              <p className="text-2xl font-bold text-purple-600">
                {(
                  learningPathAnalytics.student_journey.path_completion_rate *
                  100
                ).toFixed(1)}
                %
              </p>
              <p className="text-sm text-purple-700 dark:text-purple-300">
                Ortalama{" "}
                {Math.round(
                  learningPathAnalytics.student_journey.average_journey_time /
                    60
                )}{" "}
                saat
              </p>
            </div>

            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <h4 className="text-sm font-medium text-green-900 dark:text-green-200 mb-2">
                Adaptasyon Etkisi
              </h4>
              <p className="text-2xl font-bold text-green-600">
                {
                  learningPathAnalytics.personalization_impact
                    .adaptive_adjustments
                }
              </p>
              <p className="text-sm text-green-700 dark:text-green-300">
                Kişiselleştirme ayarı
              </p>
            </div>

            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
              <h4 className="text-sm font-medium text-orange-900 dark:text-orange-200 mb-2">
                Darboğaz Modüller
              </h4>
              <p className="text-2xl font-bold text-orange-600">
                {
                  learningPathAnalytics.module_relationships.bottleneck_modules
                    .length
                }
              </p>
              <p className="text-sm text-orange-700 dark:text-orange-300">
                İyileştirme gereken
              </p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default ModuleProgressMonitoringDashboard;
