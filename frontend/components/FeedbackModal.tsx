"use client";
import React, { useState } from "react";

interface FeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (feedback: FeedbackData) => Promise<void>;
  interactionId: number;
  query: string;
  answer: string;
}

export interface FeedbackData {
  interaction_id: number;
  user_id: string;
  session_id: string;
  understanding_level?: number;
  answer_adequacy?: number;
  satisfaction_level?: number;
  difficulty_level?: number;
  topic_understood?: boolean;
  answer_helpful?: boolean;
  needs_more_explanation?: boolean;
  comment?: string;
}

export default function FeedbackModal({
  isOpen,
  onClose,
  onSubmit,
  interactionId,
  query,
  answer,
}: FeedbackModalProps) {
  const [understanding, setUnderstanding] = useState<number | undefined>();
  const [adequacy, setAdequacy] = useState<number | undefined>();
  const [satisfaction, setSatisfaction] = useState<number | undefined>();
  const [difficulty, setDifficulty] = useState<number | undefined>();
  const [topicUnderstood, setTopicUnderstood] = useState<boolean | undefined>();
  const [answerHelpful, setAnswerHelpful] = useState<boolean | undefined>();
  const [needsMoreExplanation, setNeedsMoreExplanation] = useState<boolean | undefined>();
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async () => {
    setError(null);
    
    // Get user info from localStorage or context
    const userId = localStorage.getItem("userId") || "anonymous";
    const sessionId = localStorage.getItem("currentSessionId") || "";
    
    if (!sessionId) {
      setError("Session ID bulunamadı");
      return;
    }

    const feedback: FeedbackData = {
      interaction_id: interactionId,
      user_id: userId,
      session_id: sessionId,
      understanding_level: understanding,
      answer_adequacy: adequacy,
      satisfaction_level: satisfaction,
      difficulty_level: difficulty,
      topic_understood: topicUnderstood,
      answer_helpful: answerHelpful,
      needs_more_explanation: needsMoreExplanation,
      comment: comment || undefined,
    };

    try {
      setSubmitting(true);
      await onSubmit(feedback);
      // Reset form
      setUnderstanding(undefined);
      setAdequacy(undefined);
      setSatisfaction(undefined);
      setDifficulty(undefined);
      setTopicUnderstood(undefined);
      setAnswerHelpful(undefined);
      setNeedsMoreExplanation(undefined);
      setComment("");
      onClose();
    } catch (e: any) {
      setError(e.message || "Feedback gönderilemedi");
    } finally {
      setSubmitting(false);
    }
  };

  const StarRating = ({
    value,
    onChange,
    label,
  }: {
    value: number | undefined;
    onChange: (value: number) => void;
    label: string;
  }) => (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            onClick={() => onChange(star)}
            className={`text-2xl transition-transform hover:scale-110 ${
              value && star <= value
                ? "text-yellow-400"
                : "text-gray-300"
            }`}
          >
            ⭐
          </button>
        ))}
        {value && (
          <span className="ml-2 text-sm text-gray-600">
            ({value}/5)
          </span>
        )}
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white border-b border-gray-200 p-6 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-800">
            Geri Bildirim Ver
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6">
          {/* Question and Answer Preview */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <div className="mb-3">
              <p className="text-sm font-semibold text-gray-700 mb-1">Soru:</p>
              <p className="text-sm text-gray-800">{query}</p>
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-700 mb-1">Cevap:</p>
              <p className="text-sm text-gray-600 line-clamp-3">{answer}</p>
            </div>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {/* Star Ratings */}
          <StarRating
            value={understanding}
            onChange={setUnderstanding}
            label="Konuyu anlama seviyesi"
          />
          <StarRating
            value={adequacy}
            onChange={setAdequacy}
            label="Cevabın yeterliliği"
          />
          <StarRating
            value={satisfaction}
            onChange={setSatisfaction}
            label="Genel memnuniyet"
          />
          <StarRating
            value={difficulty}
            onChange={setDifficulty}
            label="Sorunun zorluk seviyesi"
          />

          {/* Boolean Questions */}
          <div className="mb-4 space-y-3">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <label className="text-sm font-medium text-gray-700">
                Konuyu anladım
              </label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setTopicUnderstood(true)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    topicUnderstood === true
                      ? "bg-green-500 text-white"
                      : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                >
                  Evet
                </button>
                <button
                  type="button"
                  onClick={() => setTopicUnderstood(false)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    topicUnderstood === false
                      ? "bg-red-500 text-white"
                      : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                >
                  Hayır
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <label className="text-sm font-medium text-gray-700">
                Cevap yararlıydı
              </label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setAnswerHelpful(true)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    answerHelpful === true
                      ? "bg-green-500 text-white"
                      : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                >
                  Evet
                </button>
                <button
                  type="button"
                  onClick={() => setAnswerHelpful(false)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    answerHelpful === false
                      ? "bg-red-500 text-white"
                      : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                >
                  Hayır
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <label className="text-sm font-medium text-gray-700">
                Daha fazla açıklama gerekli
              </label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setNeedsMoreExplanation(true)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    needsMoreExplanation === true
                      ? "bg-orange-500 text-white"
                      : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                >
                  Evet
                </button>
                <button
                  type="button"
                  onClick={() => setNeedsMoreExplanation(false)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    needsMoreExplanation === false
                      ? "bg-green-500 text-white"
                      : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                >
                  Hayır
                </button>
              </div>
            </div>
          </div>

          {/* Comment */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Yorum (isteğe bağlı)
            </label>
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Ek görüşlerinizi paylaşın..."
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={3}
            />
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
            >
              İptal
            </button>
            <button
              onClick={handleSubmit}
              disabled={submitting}
              className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {submitting ? "Gönderiliyor..." : "Gönder"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

