"use client";

import React, { useState } from "react";
import { submitEmojiFeedback, EmojiFeedbackCreate } from "@/lib/api";
import { useAPRAGSettings } from "@/hooks/useAPRAGSettings";

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
    name: "mukemmel",
    description: "Mükemmel - Çok açıklayıcı",
    color: "bg-green-500 hover:bg-green-600",
  },
  {
    emoji: "😊" as const,
    name: "anladim",
    description: "Anladım - Cevap anlaşılır",
    color: "bg-blue-500 hover:bg-blue-600",
  },
  {
    emoji: "😐" as const,
    name: "karisik",
    description: "Karışık - Ek açıklama gerekli",
    color: "bg-yellow-500 hover:bg-yellow-600",
  },
  {
    emoji: "❌" as const,
    name: "anlamadim",
    description: "Anlamadım - Alternatif yaklaşım gerekli",
    color: "bg-red-500 hover:bg-red-600",
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
  const { isEnabled, features } = useAPRAGSettings(sessionId);

  // Don't render if APRAG or emoji feedback is disabled
  if (!isEnabled || !features.feedback_collection) {
    return null;
  }

  const handleEmojiClick = async (emoji: "😊" | "👍" | "😐" | "❌") => {
    if (submitting || selectedEmoji) return;

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

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        {!selectedEmoji ? (
          <>
            {EMOJI_OPTIONS.map((option) => (
              <button
                key={option.emoji}
                onClick={() => handleEmojiClick(option.emoji)}
                disabled={submitting}
                title={option.description}
                className={`
                  text-2xl p-2 rounded-lg transition-all transform hover:scale-110
                  ${submitting ? "opacity-50 cursor-not-allowed" : "hover:shadow-md"}
                `}
              >
                {option.emoji}
              </button>
            ))}
          </>
        ) : (
          <div className="flex items-center gap-2 text-sm text-green-600 font-medium animate-fadeIn">
            <span className="text-2xl">{selectedEmoji}</span>
            <span>Teşekkürler!</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-gray-800 mb-1">
          Bu cevap hakkında ne düşünüyorsun?
        </h3>
        <p className="text-xs text-gray-500">
          Hızlı geri bildirimle bize yardımcı ol
        </p>
      </div>

      {!selectedEmoji ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {EMOJI_OPTIONS.map((option) => (
            <button
              key={option.emoji}
              onClick={() => handleEmojiClick(option.emoji)}
              disabled={submitting}
              className={`
                ${option.color} text-white rounded-lg p-3 
                transition-all transform hover:scale-105 hover:shadow-lg
                disabled:opacity-50 disabled:cursor-not-allowed
                flex flex-col items-center gap-2
              `}
            >
              <span className="text-3xl">{option.emoji}</span>
              <span className="text-xs font-medium">{option.name}</span>
            </button>
          ))}
        </div>
      ) : (
        <div className="text-center py-4 animate-fadeIn">
          <div className="text-5xl mb-2">{selectedEmoji}</div>
          <p className="text-green-600 font-semibold">
            Geri bildiriminiz kaydedildi. Teşekkürler!
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Bu, öğrenme deneyiminizi iyileştirmemize yardımcı olacak
          </p>
        </div>
      )}

      {error && (
        <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-800">
          {error}
        </div>
      )}
    </div>
  );
}

// Quick inline emoji feedback (for chat messages)
interface QuickEmojiFeedbackProps {
  interactionId: number;
  userId: string;
  sessionId: string;
  onFeedbackSubmitted?: () => void;
}

export function QuickEmojiFeedback({
  interactionId,
  userId,
  sessionId,
  onFeedbackSubmitted,
}: QuickEmojiFeedbackProps) {
  const [selectedEmoji, setSelectedEmoji] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleEmojiClick = async (emoji: "😊" | "👍" | "😐" | "❌") => {
    if (submitting || selectedEmoji) return;

    setSubmitting(true);

    try {
      await submitEmojiFeedback({
        interaction_id: interactionId,
        user_id: userId,
        session_id: sessionId,
        emoji,
      });
      setSelectedEmoji(emoji);
      if (onFeedbackSubmitted) {
        onFeedbackSubmitted();
      }
    } catch (err) {
      console.error("Failed to submit emoji feedback:", err);
    } finally {
      setSubmitting(false);
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
    <div className="inline-flex items-center gap-1">
      {["👍", "😊", "😐", "❌"].map((emoji) => (
        <button
          key={emoji}
          onClick={() => handleEmojiClick(emoji as any)}
          disabled={submitting}
          title="Hızlı geri bildirim"
          className="text-lg hover:scale-125 transition-transform disabled:opacity-50"
        >
          {emoji}
        </button>
      ))}
    </div>
  );
}

