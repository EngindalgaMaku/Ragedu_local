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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  BarChart3,
  TrendingUp,
  Users,
  Target,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  BookOpen,
  Brain,
  TrendingDown,
} from "lucide-react";

import TopicMasteryHeatmap from "./TopicMasteryHeatmap";
import StudentProgressChart from "./StudentProgressChart";
import TopicRecommendations from "./TopicRecommendations";
import TopicDifficultyAnalysis from "./TopicDifficultyAnalysis";

interface TopicAnalyticsDashboardProps {
  sessionId: string;
}

interface SessionOverview {
  success: boolean;
  session_id: string;
  generated_at: string;
  mastery_overview: any;
  difficulty_analysis: any;
  recommendations: any;
  executive_summary: {
    total_topics: number;
    engagement_rate: number;
    avg_understanding: number;
    topics_needing_attention: number;
    session_health_score: number;
    immediate_actions_needed: number;
    key_strengths: number;
    overall_status: "excellent" | "good" | "needs_improvement";
  };
}

const TopicAnalyticsDashboard: React.FC<TopicAnalyticsDashboardProps> = ({
  sessionId,
}) => {
  const [overview, setOverview] = useState<SessionOverview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [refreshing, setRefreshing] = useState(false);

  // Fetch session overview data
  const fetchOverview = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `/api/aprag/analytics/topics/session-overview/${sessionId}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        setOverview(data);
      } else {
        throw new Error("Failed to load analytics data");
      }
    } catch (e: any) {
      setError(e.message || "Failed to load topic analytics");
      console.error("Topic analytics error:", e);
    } finally {
      setLoading(false);
    }
  };

  // Refresh data
  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchOverview();
    setRefreshing(false);
  };

  // Load data on mount
  useEffect(() => {
    if (sessionId) {
      fetchOverview();
    }
  }, [sessionId]);

  // Get status color and icon
  const getStatusInfo = (status: string) => {
    switch (status) {
      case "excellent":
        return {
          color: "text-green-600",
          bgColor: "bg-green-50 border-green-200",
          icon: CheckCircle,
        };
      case "good":
        return {
          color: "text-blue-600",
          bgColor: "bg-blue-50 border-blue-200",
          icon: TrendingUp,
        };
      case "needs_improvement":
        return {
          color: "text-amber-600",
          bgColor: "bg-amber-50 border-amber-200",
          icon: AlertTriangle,
        };
      default:
        return {
          color: "text-gray-600",
          bgColor: "bg-gray-50 border-gray-200",
          icon: Target,
        };
    }
  };

  const formatPercentage = (value: number) => `${value.toFixed(1)}%`;
  const formatScore = (value: number) => value.toFixed(2);

  if (loading && !overview) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-2 border-primary border-t-transparent mb-4"></div>
        <p className="text-sm text-muted-foreground">
          Analitik veriler yükleniyor...
        </p>
      </div>
    );
  }

  if (error && !overview) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Hata</AlertTitle>
        <AlertDescription>
          {error}
          <Button
            variant="outline"
            size="sm"
            onClick={fetchOverview}
            className="mt-2 ml-2"
          >
            Yeniden Dene
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (!overview) {
    return (
      <div className="text-center py-12">
        <BookOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <p className="text-sm text-muted-foreground">
          Henüz analitik verisi bulunmuyor
        </p>
      </div>
    );
  }

  const { executive_summary } = overview;
  const statusInfo = getStatusInfo(executive_summary.overall_status);
  const StatusIcon = statusInfo.icon;

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <Brain className="h-6 w-6 text-primary" />
            Konu Seviyesi Analytics Dashboard
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Session: {sessionId} • Son güncelleme:{" "}
            {new Date(overview.generated_at).toLocaleString("tr-TR")}
          </p>
        </div>

        <Button
          onClick={handleRefresh}
          disabled={refreshing}
          variant="outline"
          size="sm"
          className="flex items-center gap-2"
        >
          <RefreshCw
            className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`}
          />
          Yenile
        </Button>
      </div>

      {/* Executive Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Overall Status */}
        <Card className={`${statusInfo.bgColor} border-2`}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Genel Durum
            </CardTitle>
            <div className="flex items-center gap-2">
              <StatusIcon className={`h-5 w-5 ${statusInfo.color}`} />
              <span
                className={`text-lg font-bold ${statusInfo.color} capitalize`}
              >
                {executive_summary.overall_status === "excellent"
                  ? "Mükemmel"
                  : executive_summary.overall_status === "good"
                  ? "İyi"
                  : "Geliştirilmeli"}
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Sağlık Skoru:{" "}
              {formatScore(executive_summary.session_health_score)}/3.0
            </p>
          </CardContent>
        </Card>

        {/* Topics Overview */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Konu İstatistikleri
            </CardTitle>
            <div className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-blue-600" />
              <span className="text-lg font-bold">
                {executive_summary.total_topics}
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Katılım Oranı:{" "}
              {formatPercentage(executive_summary.engagement_rate)}
            </p>
          </CardContent>
        </Card>

        {/* Understanding Score */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Ortalama Anlama
            </CardTitle>
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5 text-green-600" />
              <span className="text-lg font-bold">
                {formatScore(executive_summary.avg_understanding)}/5.0
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <Badge
              variant={
                executive_summary.avg_understanding >= 4.0
                  ? "default"
                  : executive_summary.avg_understanding >= 3.0
                  ? "secondary"
                  : "destructive"
              }
            >
              {executive_summary.avg_understanding >= 4.0
                ? "Yüksek"
                : executive_summary.avg_understanding >= 3.0
                ? "Orta"
                : "Düşük"}
            </Badge>
          </CardContent>
        </Card>

        {/* Action Items */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Eylem Gerektiren
            </CardTitle>
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-600" />
              <span className="text-lg font-bold">
                {executive_summary.immediate_actions_needed}
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Güçlü Alanlar: {executive_summary.key_strengths}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Analytics Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview" className="text-xs sm:text-sm">
            <BarChart3 className="h-4 w-4 mr-1" />
            Genel Bakış
          </TabsTrigger>
          <TabsTrigger value="mastery" className="text-xs sm:text-sm">
            <Target className="h-4 w-4 mr-1" />
            Yeterlilik Haritası
          </TabsTrigger>
          <TabsTrigger value="progress" className="text-xs sm:text-sm">
            <TrendingUp className="h-4 w-4 mr-1" />
            İlerleme
          </TabsTrigger>
          <TabsTrigger value="recommendations" className="text-xs sm:text-sm">
            <Brain className="h-4 w-4 mr-1" />
            Öneriler
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6 mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Session Health */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Session Sağlık Durumu
                </CardTitle>
                <CardDescription>Konu bazlı performans analizi</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Kritik Konular</span>
                    <Badge variant="destructive">
                      {overview.recommendations.critical_topics?.length || 0}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">
                      Orta Düzey Sorunlar
                    </span>
                    <Badge variant="secondary">
                      {overview.recommendations.moderate_issues?.length || 0}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Güçlü Alanlar</span>
                    <Badge variant="default">
                      {overview.recommendations.strong_topics?.length || 0}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Hızlı Eylemler
                </CardTitle>
                <CardDescription>
                  Acil müdahale gerektiren konular
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {overview.recommendations.action_items?.immediate_attention
                    ?.length > 0 ? (
                    overview.recommendations.action_items.immediate_attention.map(
                      (topic: string, index: number) => (
                        <div
                          key={index}
                          className="flex items-center gap-2 text-sm"
                        >
                          <AlertTriangle className="h-3 w-3 text-red-500" />
                          <span className="flex-1 truncate">{topic}</span>
                        </div>
                      )
                    )
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      Acil müdahale gerektiren konu bulunmamaktadır.
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Difficulty Analysis Preview */}
          <TopicDifficultyAnalysis
            sessionId={sessionId}
            data={overview.difficulty_analysis}
            compact={true}
          />
        </TabsContent>

        {/* Mastery Heatmap Tab */}
        <TabsContent value="mastery" className="mt-6">
          <TopicMasteryHeatmap
            sessionId={sessionId}
            data={overview.mastery_overview}
          />
        </TabsContent>

        {/* Progress Charts Tab */}
        <TabsContent value="progress" className="mt-6">
          <StudentProgressChart
            sessionId={sessionId}
            data={overview.mastery_overview}
          />
        </TabsContent>

        {/* Recommendations Tab */}
        <TabsContent value="recommendations" className="mt-6">
          <TopicRecommendations
            sessionId={sessionId}
            data={overview.recommendations}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default TopicAnalyticsDashboard;
