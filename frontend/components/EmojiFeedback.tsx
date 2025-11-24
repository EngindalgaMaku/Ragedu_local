"use client";

import React, { useState } from "react";
import { submitEmojiFeedback, EmojiFeedbackCreate } from "@/lib/api";
import { useAPRAGSettings } from "@/hooks/useAPRAGSettings";
import DetailedFeedbackModal from "./DetailedFeedbackModal";

interface EmojiFeedbackProps {
  interactionId: number;
  userId: string;
  sessionId: string;
  onFeedbackSubmitted?: () => void;
  compact?: boolean;
}

const EMOJI_OPTIONS = [
  {
    emoji: "👍" as const,
    name: "Tam Anladım",
    description: "Bu cevap sorumu tam karşılıyor ve net",
    shortDescription: "Tam Anladım",
    color: "bg-green-500 hover:bg-green-600",
    hoverColor: "hover:bg-green-700",
  },
  {
    emoji: "😊" as const,
    name: "Genel Anladım",
    description: "Cevap yardımcı ama bazı kısımları daha açık olabilir",
    shortDescription: "Genel Anladım",
    color: "bg-blue-500 hover:bg-blue-600",
    hoverColor: "hover:bg-blue-700",
  },
  {
    emoji: "😐" as const,
    name: "Kısmen Anladım",
    description: "Cevap karmaşık, ek açıklama lazım",
    shortDescription: "Kısmen Anladım",
    color: "bg-yellow-500 hover:bg-yellow-600",
    hoverColor: "hover:bg-yellow-700",
  },
  {
    emoji: "❌" as const,
    name: "Anlamadım",
    description: "Cevap soruma uygun değil veya çok karmaşık",
    shortDescription: "Anlamadım",
    color: "bg-red-500 hover:bg-red-600",
    hoverColor: "hover:bg-red-700",
  },
];

export default function EmojiFeedback({
  interactionId,
  userId,
  sessionId,
  onFeedbackSubmitted,
  compact = false,
}: EmojiFeedbackProps) {
  const [selectedEmoji, setSelectedEmoji] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDetailedModal, setShowDetailedModal] = useState(false);
  const [pendingEmoji, setPendingEmoji] = useState<
    "😊" | "👍" | "😐" | "❌" | null
  >(null);
  const { isEnabled, features } = useAPRAGSettings(sessionId);

  // Don't render if APRAG or emoji feedback is disabled
  if (!isEnabled || !features.feedback_collection) {
    return null;
  }

  const handleEmojiClick = async (
    emoji: "😊" | "👍" | "😐" | "❌",
    skipDetailed: boolean = false
  ) => {
    if (submitting || selectedEmoji) return;

    // If we want to show detailed feedback and it's not being skipped
    if (!skipDetailed && !compact) {
      setPendingEmoji(emoji);
      setShowDetailedModal(true);
      return;
    }

    // Direct emoji submission (for compact mode or when detailed is skipped)
    setSubmitting(true);
    setError(null);

    try {
      const feedback: EmojiFeedbackCreate = {
        interaction_id: interactionId,
        user_id: userId,
        session_id: sessionId,
        emoji,
      };

      await submitEmojiFeedback(feedback);
      setSelectedEmoji(emoji);
      setShowSuccess(true);

      // Auto-hide success message after 2 seconds
      setTimeout(() => {
        setShowSuccess(false);
      }, 2000);

      if (onFeedbackSubmitted) {
        onFeedbackSubmitted();
      }
    } catch (err: any) {
      setError(err.message || "Geri bildirim gönderilemedi");
      setSubmitting(false);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDetailedModalClose = () => {
    setShowDetailedModal(false);
    setPendingEmoji(null);
  };

  const handleDetailedFeedbackSubmitted = () => {
    // Mark as completed after detailed feedback
    setSelectedEmoji(pendingEmoji);
    setShowSuccess(true);
    setPendingEmoji(null);

    // Auto-hide success message after 2 seconds
    setTimeout(() => {
      setShowSuccess(false);
    }, 2000);

    if (onFeedbackSubmitted) {
      onFeedbackSubmitted();
    }
  };

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        {!selectedEmoji ? (
          <>
            <span className="text-xs text-gray-500 mr-1">
              Bu cevap ne kadar yararlıydı?
            </span>
            {EMOJI_OPTIONS.map((option) => (
              <button
                key={option.emoji}
                onClick={() => handleEmojiClick(option.emoji)}
                disabled={submitting}
                title={`${option.name}: ${option.description}`}
                className={`
                  text-2xl p-2 rounded-lg transition-all transform hover:scale-110 hover:shadow-lg
                  ${
                    submitting
                      ? "opacity-50 cursor-not-allowed"
                      : `hover:bg-gray-100 ${option.hoverColor}`
                  }
                  relative group
                `}
              >
                {option.emoji}
                <div
                  className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2
                               bg-gray-800 text-white text-xs rounded-lg px-3 py-2
                               opacity-0 group-hover:opacity-100 transition-opacity duration-200
                               whitespace-nowrap z-50 pointer-events-none"
                >
                  <div className="text-center">
                    <div className="font-semibold">{option.name}</div>
                    <div className="text-xs opacity-90 mt-1">
                      {option.description}
                    </div>
                  </div>
                  <div
                    className="absolute top-full left-1/2 transform -translate-x-1/2
                                 border-4 border-transparent border-t-gray-800"
                  ></div>
                </div>
              </button>
            ))}
          </>
        ) : (
          <div className="flex items-center gap-2 text-sm text-green-600 font-medium animate-fadeIn">
            <span className="text-2xl">{selectedEmoji}</span>
            <span>Teşekkürler!</span>
          </div>
        )}

        {/* Detailed Feedback Modal */}
        <DetailedFeedbackModal
          isOpen={showDetailedModal}
          onClose={handleDetailedModalClose}
          interactionId={interactionId}
          userId={userId}
          sessionId={sessionId}
          initialEmoji={pendingEmoji || undefined}
          onFeedbackSubmitted={handleDetailedFeedbackSubmitted}
        />
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-gray-800 mb-2">
          Cevap Değerlendirmesi
        </h3>
        <p className="text-sm text-gray-600 mb-1">
          Bu cevap sorunu ne kadar iyi yanıtladı?
        </p>
        <p className="text-xs text-gray-500">
          Açıklama ne kadar net ve anlaşılır? Değerlendirmen öğrenme deneyimini
          iyileştirecek.
        </p>
      </div>

      {!selectedEmoji ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {EMOJI_OPTIONS.map((option) => (
            <button
              key={option.emoji}
              onClick={() => handleEmojiClick(option.emoji)}
              disabled={submitting}
              title={option.description}
              className={`
                ${option.color} ${option.hoverColor} text-white rounded-lg p-4
                transition-all transform hover:scale-105 hover:shadow-lg
                disabled:opacity-50 disabled:cursor-not-allowed
                flex flex-col items-center gap-3 relative group
                border-2 border-transparent hover:border-opacity-30
              `}
            >
              <span className="text-4xl">{option.emoji}</span>
              <div className="text-center">
                <div className="text-sm font-semibold mb-1">{option.name}</div>
                <div className="text-xs opacity-90 leading-tight">
                  {option.description}
                </div>
              </div>

              {/* Enhanced hover effect */}
              <div
                className="absolute inset-0 bg-white bg-opacity-10 rounded-lg
                             opacity-0 group-hover:opacity-100 transition-opacity duration-200"
              ></div>
            </button>
          ))}
        </div>
      ) : (
        <div className="text-center py-6 animate-fadeIn">
          <div className="text-6xl mb-3">{selectedEmoji}</div>
          <p className="text-green-600 font-semibold text-lg mb-2">
            Geri bildiriminiz kaydedildi. Teşekkürler!
          </p>
          <p className="text-sm text-gray-600 mb-1">
            Bu değerlendirme, cevapların kalitesini artırmamıza yardımcı olacak.
          </p>
          <p className="text-xs text-gray-500">
            Öğrenme deneyiminizi daha da iyileştirmek için çalışmaya devam
            ediyoruz.
          </p>

          {/* Option to provide detailed feedback after quick emoji */}
          {!showSuccess && (
            <button
              onClick={() => {
                setPendingEmoji(selectedEmoji as any);
                setShowDetailedModal(true);
              }}
              className="mt-3 text-xs text-blue-600 hover:text-blue-800 underline"
            >
              Daha detaylı değerlendirme yap
            </button>
          )}
        </div>
      )}

      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800">
          <div className="flex items-center gap-2">
            <span className="text-red-500">⚠️</span>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Detailed Feedback Modal */}
      <DetailedFeedbackModal
        isOpen={showDetailedModal}
        onClose={handleDetailedModalClose}
        interactionId={interactionId}
        userId={userId}
        sessionId={sessionId}
        initialEmoji={pendingEmoji || undefined}
        onFeedbackSubmitted={handleDetailedFeedbackSubmitted}
      />
    </div>
  );
}

// Quick inline emoji feedback (for chat messages)
interface QuickEmojiFeedbackProps {
  interactionId: number;
  userId: string;
  sessionId: string;
  initialEmoji?: string; // Emoji feedback from chat message
  onFeedbackSubmitted?: () => void;
}

export function QuickEmojiFeedback({
  interactionId,
  userId,
  sessionId,
  initialEmoji,
  onFeedbackSubmitted,
}: QuickEmojiFeedbackProps) {
  const [selectedEmoji, setSelectedEmoji] = useState<string | null>(initialEmoji || null);
  const [submitting, setSubmitting] = useState(false);
  const [showDetailedModal, setShowDetailedModal] = useState(false);
  const [pendingEmoji, setPendingEmoji] = useState<
    "😊" | "👍" | "😐" | "❌" | null
  >(null);

  // Update selectedEmoji when initialEmoji changes (from chat history)
  React.useEffect(() => {
    if (initialEmoji) {
      setSelectedEmoji(initialEmoji);
    }
  }, [initialEmoji]);

  const handleEmojiClick = async (emoji: "😊" | "👍" | "😐" | "❌") => {
    if (submitting || selectedEmoji) return;

    setSubmitting(true);

    try {
      const result = await submitEmojiFeedback({
        interaction_id: interactionId,
        user_id: userId,
        session_id: sessionId,
        emoji,
      });
      console.log("✅ Emoji feedback submitted:", result);
      setSelectedEmoji(emoji);
      
      // Update chat message with emoji feedback
      try {
        const { getStudentChatHistory, saveStudentChatMessage } = await import("@/lib/api");
        const chatHistory = await getStudentChatHistory(sessionId);
        const messageToUpdate = chatHistory.find(
          (msg) => msg.aprag_interaction_id === interactionId
        );
        if (messageToUpdate) {
          // CRITICAL: Preserve ALL fields including topic and suggestions
          await saveStudentChatMessage({
            user: messageToUpdate.user,
            bot: messageToUpdate.bot,
            sources: messageToUpdate.sources || [],
            durationMs: messageToUpdate.durationMs || 0,
            session_id: messageToUpdate.session_id,
            suggestions: messageToUpdate.suggestions || [], // Preserve suggestions
            aprag_interaction_id: messageToUpdate.aprag_interaction_id,
            emoji_feedback: emoji, // Update emoji
            topic: messageToUpdate.topic || undefined, // Preserve topic
          });
          console.log("✅ Chat message updated with emoji feedback", { 
            emoji,
            topic: messageToUpdate.topic,
            hasSuggestions: !!messageToUpdate.suggestions 
          });
        } else {
          console.warn("⚠️ Message not found for emoji update", { interactionId });
        }
      } catch (updateErr) {
        console.error("❌ Failed to update chat message with emoji:", updateErr);
        // Non-critical, don't fail the whole operation
      }
      
      if (onFeedbackSubmitted) {
        onFeedbackSubmitted();
      }
    } catch (err: any) {
      console.error("❌ Failed to submit emoji feedback:", err);
      alert(`Geri bildirim gönderilemedi: ${err.message || "Bilinmeyen hata"}`);
      setSubmitting(false);
      return;
    } finally {
      setSubmitting(false);
    }
  };

  const handleDetailedModalClose = () => {
    setShowDetailedModal(false);
    setPendingEmoji(null);
  };

  const handleDetailedFeedbackSubmitted = () => {
    setSelectedEmoji(pendingEmoji);
    setPendingEmoji(null);
    if (onFeedbackSubmitted) {
      onFeedbackSubmitted();
    }
  };

  if (selectedEmoji) {
    return (
      <div className="inline-flex items-center gap-1 text-xs text-gray-500">
        <span className="text-base">{selectedEmoji}</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 w-full">
      {/* Emoji buttons row with labels */}
      <div className="flex items-center justify-between bg-white rounded-lg px-2 py-3 border border-gray-200">
        {EMOJI_OPTIONS.map((option) => (
          <button
            key={option.emoji}
            onClick={() => handleEmojiClick(option.emoji)}
            disabled={submitting}
            className="flex-1 flex flex-col items-center gap-1.5 p-2 rounded-lg 
                     hover:bg-gray-50 transition-all transform hover:scale-110
                     disabled:opacity-50 disabled:cursor-not-allowed
                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1
                     active:scale-95"
            aria-label={option.name}
            title={option.description}
          >
            <span className="text-3xl leading-none">{option.emoji}</span>
            <span className="text-[10px] font-medium text-gray-600 text-center leading-tight">
              {option.shortDescription}
            </span>
          </button>
        ))}
      </div>
      {/* Detailed Feedback Modal */}
      <DetailedFeedbackModal
        isOpen={showDetailedModal}
        onClose={handleDetailedModalClose}
        interactionId={interactionId}
        userId={userId}
        sessionId={sessionId}
        initialEmoji={pendingEmoji || undefined}
        onFeedbackSubmitted={handleDetailedFeedbackSubmitted}
      />
    </div>
  );
}
