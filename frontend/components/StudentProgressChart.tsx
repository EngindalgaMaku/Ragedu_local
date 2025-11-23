"use client";

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  TrendingUp,
  TrendingDown,
  Users,
  Clock,
  Target,
  Activity,
  BarChart3,
  LineChart,
  PieChart,
} from "lucide-react";

interface StudentProgress {
  user_id: string;
  topic_id: number;
  topic_title: string;
  mastery_level: number;
  completion_percentage: number;
  time_spent_minutes: number;
  learning_velocity: number;
  progress_trend: "improving" | "stable" | "declining" | "new";
  avg_understanding: number;
  total_interactions: number;
}

interface StudentProgressChartProps {
  sessionId: string;
  data?: any;
  userId?: string;
}

const StudentProgressChart: React.FC<StudentProgressChartProps> = ({
  sessionId,
  data,
  userId,
}) => {
  const [progressData, setProgressData] = useState<StudentProgress[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"overview" | "individual">(
    "overview"
  );
  const [selectedStudent, setSelectedStudent] = useState<string>("");
  const [sortBy, setSortBy] = useState<"mastery" | "understanding" | "time">(
    "mastery"
  );

  // Fetch progress data
  const fetchProgressData = async (targetUserId?: string) => {
    try {
      setLoading(true);
      setError(null);

      let url: string;
      if (targetUserId) {
        url = `/api/aprag/analytics/topics/student-progress/${targetUserId}?session_id=${sessionId}`;
      } else {
        // For overview, we need to get mastery overview first to find users
        url = `/api/aprag/analytics/topics/mastery-overview/${sessionId}`;
      }

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        if (targetUserId) {
          setProgressData(result.progress_data || []);
        } else {
          // For overview mode, we'll use the topics data to show aggregate progress
          setProgressData(result.topics || []);
        }
      } else {
        throw new Error("Failed to load progress data");
      }
    } catch (e: any) {
      setError(e.message || "Failed to load progress data");
      console.error("Progress data error:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (data) {
      // Use provided data
      setProgressData(data.topics || []);
    } else if (sessionId) {
      if (viewMode === "individual" && selectedStudent) {
        fetchProgressData(selectedStudent);
      } else {
        fetchProgressData();
      }
    }
  }, [sessionId, data, viewMode, selectedStudent]);

  // Get trend icon
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving":
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case "declining":
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      case "stable":
        return <Target className="h-4 w-4 text-blue-600" />;
      default:
        return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case "improving":
        return "text-green-600 bg-green-50 border-green-200";
      case "declining":
        return "text-red-600 bg-red-50 border-red-200";
      case "stable":
        return "text-blue-600 bg-blue-50 border-blue-200";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  // Sort data
  const sortedData = [...progressData].sort((a, b) => {
    switch (sortBy) {
      case "mastery":
        return (b.mastery_level || 0) - (a.mastery_level || 0);
      case "understanding":
        return (b.avg_understanding || 0) - (a.avg_understanding || 0);
      case "time":
        return (b.time_spent_minutes || 0) - (a.time_spent_minutes || 0);
      default:
        return 0;
    }
  });

  // Calculate progress distribution
  const progressDistribution = {
    excellent: progressData.filter((p) => (p.avg_understanding || 0) >= 4.0)
      .length,
    good: progressData.filter(
      (p) =>
        (p.avg_understanding || 0) >= 3.0 && (p.avg_understanding || 0) < 4.0
    ).length,
    fair: progressData.filter(
      (p) =>
        (p.avg_understanding || 0) >= 2.0 && (p.avg_understanding || 0) < 3.0
    ).length,
    poor: progressData.filter((p) => (p.avg_understanding || 0) < 2.0).length,
  };

  // Calculate trend distribution
  const trendDistribution = {
    improving: progressData.filter((p) => p.progress_trend === "improving")
      .length,
    stable: progressData.filter((p) => p.progress_trend === "stable").length,
    declining: progressData.filter((p) => p.progress_trend === "declining")
      .length,
    new: progressData.filter((p) => p.progress_trend === "new").length,
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Öğrenci İlerleme Analizi</CardTitle>
          <CardDescription>
            Öğrencilerin konu bazlı öğrenme ilerlemesi
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent"></div>
            <span className="ml-3 text-sm text-muted-foreground">
              Yükleniyor...
            </span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Öğrenci İlerleme Analizi</CardTitle>
          <CardDescription>
            Öğrencilerin konu bazlı öğrenme ilerlemesi
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <Button
              onClick={() => fetchProgressData()}
              variant="outline"
              size="sm"
            >
              Yeniden Dene
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Progress Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Performance Distribution */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">
              Performans Dağılımı
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-green-600">Mükemmel (4.0+)</span>
                <span className="font-medium">
                  {progressDistribution.excellent}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-blue-600">İyi (3.0+)</span>
                <span className="font-medium">{progressDistribution.good}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-yellow-600">Orta (2.0+)</span>
                <span className="font-medium">{progressDistribution.fair}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-red-600">Zayıf (&lt;2.0)</span>
                <span className="font-medium">{progressDistribution.poor}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Trend Distribution */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">
              İlerleme Eğilimleri
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-1">
                  <TrendingUp className="h-3 w-3 text-green-600" />
                  <span>Gelişen</span>
                </div>
                <span className="font-medium">
                  {trendDistribution.improving}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-1">
                  <Target className="h-3 w-3 text-blue-600" />
                  <span>Stabil</span>
                </div>
                <span className="font-medium">{trendDistribution.stable}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-1">
                  <TrendingDown className="h-3 w-3 text-red-600" />
                  <span>Azalan</span>
                </div>
                <span className="font-medium">
                  {trendDistribution.declining}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-1">
                  <Activity className="h-3 w-3 text-gray-600" />
                  <span>Yeni</span>
                </div>
                <span className="font-medium">{trendDistribution.new}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Average Metrics */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">
              Ortalama Metrikler
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Yeterlilik Seviyesi</span>
                <span className="font-medium">
                  {progressData.length > 0
                    ? (
                        progressData.reduce(
                          (sum, p) => sum + (p.mastery_level || 0),
                          0
                        ) / progressData.length
                      ).toFixed(2)
                    : "0.00"}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Anlama Skoru</span>
                <span className="font-medium">
                  {progressData.length > 0
                    ? (
                        progressData.reduce(
                          (sum, p) => sum + (p.avg_understanding || 0),
                          0
                        ) / progressData.length
                      ).toFixed(2)
                    : "0.00"}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Toplam Etkileşim</span>
                <span className="font-medium">
                  {progressData.reduce(
                    (sum, p) => sum + (p.total_interactions || 0),
                    0
                  )}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Time Metrics */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Zaman Analizi</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Toplam Süre</span>
                <span className="font-medium">
                  {Math.round(
                    progressData.reduce(
                      (sum, p) => sum + (p.time_spent_minutes || 0),
                      0
                    ) / 60
                  )}
                  h
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Ortalama/Konu</span>
                <span className="font-medium">
                  {progressData.length > 0
                    ? Math.round(
                        progressData.reduce(
                          (sum, p) => sum + (p.time_spent_minutes || 0),
                          0
                        ) / progressData.length
                      )
                    : 0}
                  m
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Öğrenen Hız</span>
                <span className="font-medium">
                  {progressData.length > 0
                    ? (
                        progressData.reduce(
                          (sum, p) => sum + (p.learning_velocity || 0),
                          0
                        ) / progressData.length
                      ).toFixed(1)
                    : "0.0"}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Progress Details */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Detaylı İlerleme Analizi
              </CardTitle>
              <CardDescription>
                Konu bazlı öğrenci performans detayları
              </CardDescription>
            </div>

            {/* Sort Controls */}
            <div className="flex gap-2">
              <Button
                variant={sortBy === "mastery" ? "default" : "outline"}
                size="sm"
                onClick={() => setSortBy("mastery")}
                className="text-xs"
              >
                Yeterlilik
              </Button>
              <Button
                variant={sortBy === "understanding" ? "default" : "outline"}
                size="sm"
                onClick={() => setSortBy("understanding")}
                className="text-xs"
              >
                Anlama
              </Button>
              <Button
                variant={sortBy === "time" ? "default" : "outline"}
                size="sm"
                onClick={() => setSortBy("time")}
                className="text-xs"
              >
                Süre
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {sortedData.length === 0 ? (
            <div className="text-center py-8">
              <LineChart className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-sm text-muted-foreground">
                Henüz ilerleme verisi bulunmamaktadır
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {sortedData.slice(0, 15).map((progress, index) => (
                <div
                  key={`${progress.topic_id}-${progress.user_id || index}`}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/30 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3">
                      <div className="flex-1">
                        <h4 className="font-medium text-sm text-foreground truncate">
                          {progress.topic_title}
                        </h4>
                        {progress.user_id && (
                          <p className="text-xs text-muted-foreground">
                            Öğrenci ID: {progress.user_id}
                          </p>
                        )}
                      </div>

                      {progress.progress_trend && (
                        <Badge
                          variant="outline"
                          className={`text-xs ${getTrendColor(
                            progress.progress_trend
                          )}`}
                        >
                          {getTrendIcon(progress.progress_trend)}
                          <span className="ml-1">
                            {progress.progress_trend === "improving"
                              ? "Gelişen"
                              : progress.progress_trend === "declining"
                              ? "Azalan"
                              : progress.progress_trend === "stable"
                              ? "Stabil"
                              : "Yeni"}
                          </span>
                        </Badge>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-6 text-sm">
                    {/* Mastery Level */}
                    <div className="text-center">
                      <div className="font-semibold text-blue-600">
                        {(progress.mastery_level || 0).toFixed(1)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Yeterlilik
                      </div>
                    </div>

                    {/* Understanding Score */}
                    <div className="text-center">
                      <div className="font-semibold text-green-600">
                        {(progress.avg_understanding || 0).toFixed(1)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Anlama
                      </div>
                    </div>

                    {/* Time Spent */}
                    <div className="text-center">
                      <div className="font-semibold text-purple-600">
                        {Math.round(progress.time_spent_minutes || 0)}m
                      </div>
                      <div className="text-xs text-muted-foreground">Süre</div>
                    </div>

                    {/* Interactions */}
                    <div className="text-center">
                      <div className="font-semibold text-orange-600">
                        {progress.total_interactions || 0}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Etkileşim
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {sortedData.length > 15 && (
                <div className="text-center pt-4">
                  <p className="text-sm text-muted-foreground">
                    {sortedData.length - 15} adet daha fazla kayıt bulunmaktadır
                  </p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default StudentProgressChart;
