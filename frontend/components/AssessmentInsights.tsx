/**
 * Assessment Insights Dashboard Component (ADIM 3)
 *
 * Teacher dashboard for progressive assessment insights:
 * - Learning progression indicators
 * - Confidence trends
 * - Application readiness
 * - Concept mastery levels
 * - Intervention recommendations
 *
 * This component provides comprehensive analytics for educators.
 */

import React, { useState, useEffect } from "react";
import { getApiUrl } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";

// Simple chart placeholders to avoid recharts dependency
const ResponsiveContainer: React.FC<{
  width: string;
  height: number;
  children: React.ReactNode;
}> = ({ children }) => (
  <div className="w-full" style={{ height: "300px" }}>
    {children}
  </div>
);

const PieChart: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="flex items-center justify-center h-full">{children}</div>
);

const Pie: React.FC<any> = ({ data }) => (
  <div className="text-center">
    <div className="text-lg font-semibold mb-4">Güven Seviyesi Dağılımı</div>
    <div className="grid grid-cols-2 gap-4">
      {data?.map((item: any, index: number) => (
        <div key={index} className="flex items-center gap-2">
          <div
            className="w-4 h-4 rounded-full"
            style={{
              backgroundColor: [
                "#ef4444",
                "#f59e0b",
                "#3b82f6",
                "#22c55e",
                "#16a34a",
              ][index],
            }}
          />
          <span className="text-sm">
            {item.name}: {item.value}
          </span>
        </div>
      ))}
    </div>
  </div>
);

const Cell: React.FC<any> = () => null;
const Tooltip: React.FC<any> = () => null;
import {
  Brain,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Target,
  Users,
  BookOpen,
  Lightbulb,
  RefreshCw,
  Download,
} from "lucide-react";

interface AssessmentInsightsProps {
  sessionId?: string;
  userId?: string;
  className?: string;
}

interface StudentInsights {
  user_id: string;
  student_name?: string;
  session_id: string;
  confidence_trend: string;
  application_readiness: string;
  concept_mastery_level: number;
  weak_areas: string[];
  confusion_patterns: string[];
  knowledge_gaps: string[];
  recommended_topics: string[];
  suggested_exercises: string[];
  next_learning_steps: string[];
  total_assessments: number;
  follow_up_completion_rate: number;
  deep_analysis_trigger_rate: number;
  average_confidence: number;
  needs_immediate_intervention: boolean;
  intervention_reasons: string[];
}

interface SessionAnalytics {
  session_id: string;
  total_students: number;
  total_assessments: number;
  average_confidence: number;
  intervention_needed_count: number;
  completion_rates: {
    follow_up: number;
    deep_analysis: number;
  };
  common_weak_areas: string[];
  trend_analysis: string;
}

const AssessmentInsights: React.FC<AssessmentInsightsProps> = ({
  sessionId,
  userId,
  className,
}) => {
  const [insights, setInsights] = useState<StudentInsights[]>([]);
  const [sessionAnalytics, setSessionAnalytics] =
    useState<SessionAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState("overview");

  // API base URL
  const apiBaseUrl =
    process.env.NEXT_PUBLIC_APRAG_API_URL || "http://localhost:8007";

  useEffect(() => {
    loadInsights();
  }, [sessionId, userId]);

  const loadInsights = async () => {
    try {
      setLoading(true);
      setError(null);

      if (userId) {
        // Load individual student insights
        const response = await fetch(
          `${getApiUrl()}/aprag/progressive-assessment/insights/${userId}${
            sessionId ? `?session_id=${sessionId}` : ""
          }`,
          {
            headers: {
              "Content-Type": "application/json",
            },
          }
        );

        if (!response.ok) {
          throw new Error("Failed to load student insights");
        }

        const data = await response.json();
        setInsights([data]);
      } else if (sessionId) {
        // Load session-wide analytics
        // This would be a new endpoint we'd need to implement
        // For now, we'll simulate the data structure
        setSessionAnalytics({
          session_id: sessionId,
          total_students: 25,
          total_assessments: 75,
          average_confidence: 3.4,
          intervention_needed_count: 5,
          completion_rates: {
            follow_up: 0.8,
            deep_analysis: 0.3,
          },
          common_weak_areas: [
            "Matematiksel formüller",
            "Pratik uygulama",
            "Temel kavramlar",
          ],
          trend_analysis: "improving",
        });
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Veri yüklenirken hata oluştu"
      );
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving":
        return <TrendingUp className="h-4 w-4 text-green-500" />;
      case "declining":
        return <TrendingDown className="h-4 w-4 text-red-500" />;
      default:
        return <Target className="h-4 w-4 text-gray-500" />;
    }
  };

  const getConfidenceBadgeVariant = (confidence: number) => {
    if (confidence >= 4) return "default";
    if (confidence >= 3) return "secondary";
    return "destructive";
  };

  const getMasteryColor = (level: number) => {
    if (level >= 0.8) return "#22c55e"; // green
    if (level >= 0.6) return "#3b82f6"; // blue
    if (level >= 0.4) return "#f59e0b"; // yellow
    return "#ef4444"; // red
  };

  const confidenceData =
    insights.length > 0
      ? [
          {
            name: "Çok Düşük (1)",
            value: insights.filter((i) => i.average_confidence < 1.5).length,
          },
          {
            name: "Düşük (2)",
            value: insights.filter(
              (i) => i.average_confidence >= 1.5 && i.average_confidence < 2.5
            ).length,
          },
          {
            name: "Orta (3)",
            value: insights.filter(
              (i) => i.average_confidence >= 2.5 && i.average_confidence < 3.5
            ).length,
          },
          {
            name: "Yüksek (4)",
            value: insights.filter(
              (i) => i.average_confidence >= 3.5 && i.average_confidence < 4.5
            ).length,
          },
          {
            name: "Çok Yüksek (5)",
            value: insights.filter((i) => i.average_confidence >= 4.5).length,
          },
        ]
      : [];

  const COLORS = ["#ef4444", "#f59e0b", "#3b82f6", "#22c55e", "#16a34a"];

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center p-6">
          <RefreshCw className="h-6 w-6 animate-spin mr-2" />
          <span>İnsights yükleniyor...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className={className}>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6" />
          <h2 className="text-2xl font-bold">Aşamalı Değerlendirme İnsights</h2>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={loadInsights} size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Yenile
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Rapor İndir
          </Button>
        </div>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Genel Bakış</TabsTrigger>
          <TabsTrigger value="students">Öğrenci Detayları</TabsTrigger>
          <TabsTrigger value="analytics">Analitik</TabsTrigger>
          <TabsTrigger value="interventions">Müdahaleler</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Toplam Öğrenci
                </CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {sessionAnalytics?.total_students || insights.length}
                </div>
                <p className="text-xs text-muted-foreground">
                  Bu oturumda aktif
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Ortalama Güven
                </CardTitle>
                <Target className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {sessionAnalytics?.average_confidence.toFixed(1) ||
                    (
                      insights.reduce(
                        (acc, i) => acc + i.average_confidence,
                        0
                      ) / insights.length
                    ).toFixed(1)}
                </div>
                <p className="text-xs text-muted-foreground">5 üzerinden</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Müdahale Gereken
                </CardTitle>
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-600">
                  {sessionAnalytics?.intervention_needed_count ||
                    insights.filter((i) => i.needs_immediate_intervention)
                      .length}
                </div>
                <p className="text-xs text-muted-foreground">Öğrenci sayısı</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Tamamlanma Oranı
                </CardTitle>
                <CheckCircle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(
                    (sessionAnalytics?.completion_rates.follow_up ||
                      insights.reduce(
                        (acc, i) => acc + i.follow_up_completion_rate,
                        0
                      ) / insights.length) * 100
                  ).toFixed(0)}
                  %
                </div>
                <p className="text-xs text-muted-foreground">
                  Takip değerlendirmesi
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Confidence Distribution Chart */}
          {confidenceData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Güven Seviyesi Dağılımı</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={confidenceData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({
                        name,
                        value,
                        percent,
                      }: {
                        name: string;
                        value: number;
                        percent: number;
                      }) =>
                        `${name}: ${value} (${(percent * 100).toFixed(0)}%)`
                      }
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {confidenceData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={COLORS[index % COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="students" className="space-y-4">
          {insights.map((student, index) => (
            <Card key={student.user_id || index}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">
                    {student.student_name || `Öğrenci ${student.user_id}`}
                  </CardTitle>
                  <div className="flex items-center space-x-2">
                    {getTrendIcon(student.confidence_trend)}
                    <Badge
                      variant={getConfidenceBadgeVariant(
                        student.average_confidence
                      )}
                    >
                      Güven: {student.average_confidence.toFixed(1)}
                    </Badge>
                    {student.needs_immediate_intervention && (
                      <Badge variant="destructive">Müdahale Gerekli</Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Mastery Level */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">
                      Kavram Hâkimiyeti
                    </span>
                    <span className="text-sm text-gray-600">
                      {(student.concept_mastery_level * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Progress
                    value={student.concept_mastery_level * 100}
                    className="h-2"
                    style={{
                      backgroundColor: getMasteryColor(
                        student.concept_mastery_level
                      ),
                    }}
                  />
                </div>

                {/* Assessment Stats */}
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-lg font-bold">
                      {student.total_assessments}
                    </div>
                    <div className="text-xs text-gray-600">
                      Toplam Değerlendirme
                    </div>
                  </div>
                  <div>
                    <div className="text-lg font-bold">
                      {(student.follow_up_completion_rate * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-600">Takip Tamamlama</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold">
                      {(student.deep_analysis_trigger_rate * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-600">Detaylı Analiz</div>
                  </div>
                </div>

                {/* Weak Areas */}
                {student.weak_areas.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium mb-2">Zayıf Alanlar:</h4>
                    <div className="flex flex-wrap gap-1">
                      {student.weak_areas.slice(0, 5).map((area, i) => (
                        <Badge key={i} variant="outline" className="text-xs">
                          {area}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Intervention Reasons */}
                {student.intervention_reasons.length > 0 && (
                  <Alert>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      <strong>Müdahale Nedenleri:</strong>
                      <ul className="list-disc list-inside mt-1">
                        {student.intervention_reasons.map((reason, i) => (
                          <li key={i} className="text-sm">
                            {reason}
                          </li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}

                {/* Next Steps */}
                {student.next_learning_steps.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium mb-2 flex items-center">
                      <Lightbulb className="h-4 w-4 mr-1" />
                      Önerilen Adımlar:
                    </h4>
                    <ul className="text-sm space-y-1">
                      {student.next_learning_steps
                        .slice(0, 3)
                        .map((step, i) => (
                          <li key={i} className="flex items-start">
                            <CheckCircle className="h-3 w-3 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                            {step}
                          </li>
                        ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Detaylı Analitik Raporu</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center text-gray-500 py-8">
                <BookOpen className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>Detaylı analitik rapor özelliği geliştiriliyor...</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="interventions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Müdahale Önerileri</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center text-gray-500 py-8">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>Otomatik müdahale önerisi sistemi geliştiriliyor...</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AssessmentInsights;
