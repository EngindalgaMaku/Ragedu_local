/**
 * Progressive Assessment Flow Component (ADIM 3)
 *
 * Implements the 3-stage progressive assessment system:
 * 1. Initial Response (Immediate) - Emoji feedback
 * 2. Follow-up Assessment (30 sec delay) - Confidence + Application
 * 3. Deeper Analysis (Optional, triggered by low scores) - Concept mapping + Clarification
 *
 * This component manages the entire progressive flow and provides deeper learning insights.
 */

import React, { useState, useEffect, useCallback } from "react";
// Mock useSession hook since next-auth is not available
const useSession = () => ({
  data: {
    accessToken: "mock-token", // This should be replaced with actual token handling
    user: { id: "user1" },
  },
});
import axios from "axios";
import { getApiUrl } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Clock, CheckCircle, AlertCircle, Brain } from "lucide-react";
import FollowUpQuestions from "./FollowUpQuestions";
import DeepAnalysisModal from "./DeepAnalysisModal";

interface ProgressiveAssessmentFlowProps {
  interactionId: number;
  userId: string;
  sessionId: string;
  emojiScore?: number;
  emojiFeedback?: string;
  onAssessmentComplete?: (stage: string, data: any) => void;
}

interface AssessmentStage {
  id: "initial" | "follow_up" | "deep_analysis";
  name: string;
  description: string;
  completed: boolean;
  available: boolean;
  delay: number; // seconds
}

interface TriggerData {
  trigger_follow_up: boolean;
  trigger_deep_analysis: boolean;
  emoji_feedback: string;
  emoji_score: number;
  interaction_id: number;
  user_id: string;
  session_id: string;
}

const ProgressiveAssessmentFlow: React.FC<ProgressiveAssessmentFlowProps> = ({
  interactionId,
  userId,
  sessionId,
  emojiScore,
  emojiFeedback,
  onAssessmentComplete,
}) => {
  const { data: session } = useSession();
  const [stages, setStages] = useState<AssessmentStage[]>([
    {
      id: "initial",
      name: "İlk Değerlendirme",
      description: "Emoji ile hızlı geri bildirim",
      completed: !!emojiFeedback,
      available: true,
      delay: 0,
    },
    {
      id: "follow_up",
      name: "Takip Değerlendirmesi",
      description: "Güven seviyesi ve uygulama anlayışı",
      completed: false,
      available: false,
      delay: 30,
    },
    {
      id: "deep_analysis",
      name: "Detaylı Analiz",
      description: "Kavram haritalaması ve açıklama talepleri",
      completed: false,
      available: false,
      delay: 0,
    },
  ]);

  const [currentStage, setCurrentStage] = useState<
    "initial" | "follow_up" | "deep_analysis"
  >("initial");
  const [loading, setLoading] = useState(false);
  const [triggerData, setTriggerData] = useState<TriggerData | null>(null);
  const [followUpDelay, setFollowUpDelay] = useState(30);
  const [showFollowUp, setShowFollowUp] = useState(false);
  const [showDeepAnalysis, setShowDeepAnalysis] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<string[]>([]);

  // API base URL
  const apiBaseUrl =
    process.env.NEXT_PUBLIC_APRAG_API_URL || "http://localhost:8007";

  // Check trigger conditions when component mounts or emoji feedback changes
  useEffect(() => {
    if (interactionId && !triggerData) {
      checkProgressiveTrigger();
    }
  }, [interactionId, emojiFeedback]);

  // Handle follow-up delay countdown
  useEffect(() => {
    if (triggerData?.trigger_follow_up && followUpDelay > 0) {
      const timer = setInterval(() => {
        setFollowUpDelay((prev) => {
          if (prev <= 1) {
            setShowFollowUp(true);
            updateStageAvailability("follow_up", true);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [triggerData, followUpDelay]);

  const checkProgressiveTrigger = async () => {
    try {
      setLoading(true);
      const response = await axios.get(
        `${getApiUrl()}/aprag/progressive-assessment/check-trigger/${interactionId}`,
        {
          headers: {
            Authorization: `Bearer ${session?.accessToken}`,
            "Content-Type": "application/json",
          },
        }
      );

      const data: TriggerData = response.data;
      setTriggerData(data);

      // Update initial stage as completed if we have emoji feedback
      if (data.emoji_feedback) {
        updateStageCompletion("initial", true);
      }

      // Start follow-up countdown if triggered
      if (data.trigger_follow_up) {
        setFollowUpDelay(30); // Reset to 30 seconds
      }
    } catch (error) {
      console.error("Failed to check progressive trigger:", error);
      setError("Değerlendirme durumu kontrol edilemedi");
    } finally {
      setLoading(false);
    }
  };

  const updateStageCompletion = (stageId: string, completed: boolean) => {
    setStages((prev) =>
      prev.map((stage) =>
        stage.id === stageId ? { ...stage, completed } : stage
      )
    );
  };

  const updateStageAvailability = (stageId: string, available: boolean) => {
    setStages((prev) =>
      prev.map((stage) =>
        stage.id === stageId ? { ...stage, available } : stage
      )
    );
  };

  const handleFollowUpSubmit = async (followUpData: any) => {
    try {
      setLoading(true);
      const response = await axios.post(
        `${getApiUrl()}/aprag/progressive-assessment/follow-up`,
        {
          interaction_id: interactionId,
          user_id: userId,
          session_id: sessionId,
          ...followUpData,
        },
        {
          headers: {
            Authorization: `Bearer ${session?.accessToken}`,
            "Content-Type": "application/json",
          },
        }
      );

      const result = response.data;

      // Update stage completion
      updateStageCompletion("follow_up", true);
      setCurrentStage("follow_up");

      // Set recommendations
      if (result.recommended_actions) {
        setRecommendations(result.recommended_actions);
      }

      // Check if deep analysis should be available
      if (result.next_stage_available) {
        updateStageAvailability("deep_analysis", true);
        setShowDeepAnalysis(true);
      }

      // Notify parent component
      if (onAssessmentComplete) {
        onAssessmentComplete("follow_up", result);
      }
    } catch (error) {
      console.error("Failed to submit follow-up assessment:", error);
      setError("Takip değerlendirmesi gönderilemedi");
    } finally {
      setLoading(false);
    }
  };

  const handleDeepAnalysisSubmit = async (analysisData: any) => {
    try {
      setLoading(true);
      const response = await axios.post(
        `${getApiUrl()}/aprag/progressive-assessment/deep-analysis`,
        {
          interaction_id: interactionId,
          user_id: userId,
          session_id: sessionId,
          ...analysisData,
        },
        {
          headers: {
            Authorization: `Bearer ${session?.accessToken}`,
            "Content-Type": "application/json",
          },
        }
      );

      const result = response.data;

      // Update stage completion
      updateStageCompletion("deep_analysis", true);
      setCurrentStage("deep_analysis");

      // Set recommendations
      if (result.recommended_actions) {
        setRecommendations(result.recommended_actions);
      }

      // Hide deep analysis modal
      setShowDeepAnalysis(false);

      // Notify parent component
      if (onAssessmentComplete) {
        onAssessmentComplete("deep_analysis", result);
      }
    } catch (error) {
      console.error("Failed to submit deep analysis:", error);
      setError("Detaylı analiz gönderilemedi");
    } finally {
      setLoading(false);
    }
  };

  const renderStageIndicator = (stage: AssessmentStage) => {
    const getStageIcon = () => {
      if (stage.completed)
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      if (stage.available) return <Brain className="h-5 w-5 text-blue-500" />;
      return <Clock className="h-5 w-5 text-gray-400" />;
    };

    const getStageStatus = () => {
      if (stage.completed) return "completed";
      if (stage.available) return "available";
      return "waiting";
    };

    return (
      <div
        key={stage.id}
        className="flex items-center space-x-3 p-3 rounded-lg border"
      >
        {getStageIcon()}
        <div className="flex-1">
          <h4 className="font-medium">{stage.name}</h4>
          <p className="text-sm text-gray-600">{stage.description}</p>
        </div>
        <Badge
          variant={
            stage.completed
              ? "default"
              : stage.available
              ? "secondary"
              : "outline"
          }
        >
          {stage.completed
            ? "Tamamlandı"
            : stage.available
            ? "Hazır"
            : "Bekliyor"}
        </Badge>
      </div>
    );
  };

  if (loading && !triggerData) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center p-6">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          <span>Değerlendirme durumu kontrol ediliyor...</span>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Progressive Assessment Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>Aşamalı Değerlendirme</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {stages.map(renderStageIndicator)}
        </CardContent>
      </Card>

      {/* Follow-up Countdown */}
      {triggerData?.trigger_follow_up && followUpDelay > 0 && !showFollowUp && (
        <Alert>
          <Clock className="h-4 w-4" />
          <AlertDescription>
            Takip değerlendirmesi {followUpDelay} saniye sonra hazır olacak...
          </AlertDescription>
        </Alert>
      )}

      {/* Follow-up Questions */}
      {showFollowUp && !stages.find((s) => s.id === "follow_up")?.completed && (
        <FollowUpQuestions
          onSubmit={handleFollowUpSubmit}
          loading={loading}
          interactionId={interactionId}
        />
      )}

      {/* Deep Analysis Modal */}
      {showDeepAnalysis &&
        stages.find((s) => s.id === "deep_analysis")?.available && (
          <DeepAnalysisModal
            open={showDeepAnalysis}
            onClose={() => setShowDeepAnalysis(false)}
            onSubmit={handleDeepAnalysisSubmit}
            loading={loading}
            interactionId={interactionId}
          />
        )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Size Özel Öneriler</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {recommendations.map((recommendation, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                  <span className="text-sm">{recommendation}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Trigger Deep Analysis Button */}
      {stages.find((s) => s.id === "deep_analysis")?.available &&
        !stages.find((s) => s.id === "deep_analysis")?.completed &&
        !showDeepAnalysis && (
          <Card>
            <CardContent className="pt-6">
              <div className="text-center space-y-3">
                <p className="text-sm text-gray-600">
                  Daha detaylı yardım almak ister misiniz?
                </p>
                <Button
                  onClick={() => setShowDeepAnalysis(true)}
                  variant="outline"
                  className="w-full"
                >
                  <Brain className="h-4 w-4 mr-2" />
                  Detaylı Analiz Talep Et
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

      {/* Completion Message */}
      {stages.every((s) => s.completed || !s.available) && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>
            Değerlendirme süreci tamamlandı. Geri bildirimleriniz için
            teşekkürler!
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default ProgressiveAssessmentFlow;
