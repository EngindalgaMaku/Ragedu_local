"use client";

import React, { useState } from "react";
import { submitDetailedFeedback, MultiFeedbackCreate } from "@/lib/api";

interface DetailedFeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
  interactionId: number;
  userId: string;
  sessionId: string;
  initialEmoji?: "😊" | "👍" | "😐" | "❌";
  onFeedbackSubmitted?: () => void;
}

interface DimensionRating {
  understanding: number;
  relevance: number;
  clarity: number;
}

const DIMENSION_QUESTIONS = {
  understanding: {
    title: "Anlama Düzeyi",
    question: "Bu açıklamayı ne kadar anladın?",
    description: "Cevabın ne kadar kolay anlaşılır olduğunu değerlendir",
  },
  relevance: {
    title: "İlgili Olma",
    question: "Cevap soruna ne kadar uygun?",
    description:
      "Verilen cevabın sorunla ne kadar alakalı olduğunu değerlendir",
  },
  clarity: {
    title: "Açıklık",
    question: "Açıklama ne kadar netti?",
    description:
      "Cevabın ne kadar açık ve net bir şekilde sunulduğunu değerlendir",
  },
};

const RATING_LABELS = {
  1: "Çok Kötü",
  2: "Kötü",
  3: "Orta",
  4: "İyi",
  5: "Mükemmel",
};

const EMOJI_OPTIONS = [
  { emoji: "👍", name: "Tam Anladım", color: "text-green-500" },
  { emoji: "😊", name: "Genel Anladım", color: "text-blue-500" },
  { emoji: "😐", name: "Kısmen Anladım", color: "text-yellow-500" },
  { emoji: "❌", name: "Anlamadım", color: "text-red-500" },
];

export default function DetailedFeedbackModal({
  isOpen,
  onClose,
  interactionId,
  userId,
  sessionId,
  initialEmoji,
  onFeedbackSubmitted,
}: DetailedFeedbackModalProps) {
  const [ratings, setRatings] = useState<DimensionRating>({
    understanding: 3,
    relevance: 3,
    clarity: 3,
  });

  const [selectedEmoji, setSelectedEmoji] = useState<string | undefined>(
    initialEmoji
  );
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRatingChange = (
    dimension: keyof DimensionRating,
    value: number
  ) => {
    setRatings((prev) => ({
      ...prev,
      [dimension]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const feedback: MultiFeedbackCreate = {
        interaction_id: interactionId,
        user_id: userId,
        session_id: sessionId,
        understanding: ratings.understanding,
        relevance: ratings.relevance,
        clarity: ratings.clarity,
        emoji: selectedEmoji as any,
        comment: comment.trim() || undefined,
      };

      await submitDetailedFeedback(feedback);

      setShowSuccess(true);

      // Auto-close after success
      setTimeout(() => {
        setShowSuccess(false);
        onClose();
        if (onFeedbackSubmitted) {
          onFeedbackSubmitted();
        }
      }, 2000);
    } catch (err: any) {
      setError(err.message || "Geri bildirim gönderilemedi");
    } finally {
      setSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!submitting) {
      onClose();
    }
  };

  const overallScore =
    (ratings.understanding + ratings.relevance + ratings.clarity) / 3;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-800">
            Detaylı Değerlendirme
          </h2>
          <button
            onClick={handleClose}
            disabled={submitting}
            className="text-gray-400 hover:text-gray-600 disabled:opacity-50"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="p-6">
          {showSuccess ? (
            <div className="text-center py-8">
              <div className="text-6xl mb-4">✅</div>
              <p className="text-green-600 font-semibold text-lg mb-2">
                Detaylı geri bildiriminiz kaydedildi!
              </p>
              <p className="text-sm text-gray-600">
                Bu değerlendirme, cevapların kalitesini artırmamıza yardımcı
                olacak.
              </p>
            </div>
          ) : (
            <>
              {/* Instructions */}
              <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <h3 className="font-medium text-blue-800 mb-2">
                  Bu cevabı üç farklı boyutta değerlendir:
                </h3>
                <div className="text-sm text-blue-700 space-y-1">
                  <div>
                    • <strong>Anlama:</strong> Açıklamayı ne kadar anladın?
                  </div>
                  <div>
                    • <strong>İlgili Olma:</strong> Cevap soruna ne kadar uygun?
                  </div>
                  <div>
                    • <strong>Açıklık:</strong> Açıklama ne kadar net ve
                    anlaşılır?
                  </div>
                </div>
              </div>

              {/* Dimension Ratings */}
              <div className="space-y-6 mb-6">
                {Object.entries(DIMENSION_QUESTIONS).map(([key, config]) => (
                  <div
                    key={key}
                    className="border border-gray-200 rounded-lg p-4"
                  >
                    <div className="mb-3">
                      <h4 className="font-medium text-gray-800 mb-1">
                        {config.title}
                      </h4>
                      <p className="text-sm text-gray-600 mb-1">
                        {config.question}
                      </p>
                      <p className="text-xs text-gray-500">
                        {config.description}
                      </p>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex space-x-2">
                        {[1, 2, 3, 4, 5].map((value) => (
                          <button
                            key={value}
                            type="button"
                            onClick={() =>
                              handleRatingChange(
                                key as keyof DimensionRating,
                                value
                              )
                            }
                            className={`
                              w-10 h-10 rounded-full border-2 flex items-center justify-center
                              text-sm font-medium transition-all duration-200
                              ${
                                ratings[key as keyof DimensionRating] >= value
                                  ? "bg-blue-500 border-blue-500 text-white"
                                  : "border-gray-300 text-gray-500 hover:border-blue-300"
                              }
                            `}
                          >
                            {value}
                          </button>
                        ))}
                      </div>
                      <div className="text-sm font-medium text-gray-700">
                        {
                          RATING_LABELS[
                            ratings[
                              key as keyof DimensionRating
                            ] as keyof typeof RATING_LABELS
                          ]
                        }
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Overall Score */}
              <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-gray-800">Genel Puan:</span>
                  <div className="flex items-center space-x-2">
                    <div className="text-2xl font-bold text-blue-600">
                      {overallScore.toFixed(1)}
                    </div>
                    <div className="text-sm text-gray-500">/5.0</div>
                  </div>
                </div>
              </div>

              {/* Emoji Selection */}
              <div className="mb-6">
                <h4 className="font-medium text-gray-800 mb-3">
                  Emoji ile Özetle (İsteğe Bağlı):
                </h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {EMOJI_OPTIONS.map((option) => (
                    <button
                      key={option.emoji}
                      type="button"
                      onClick={() =>
                        setSelectedEmoji(
                          selectedEmoji === option.emoji
                            ? undefined
                            : option.emoji
                        )
                      }
                      className={`
                        p-3 rounded-lg border-2 transition-all duration-200
                        ${
                          selectedEmoji === option.emoji
                            ? "border-blue-500 bg-blue-50"
                            : "border-gray-200 hover:border-gray-300"
                        }
                      `}
                    >
                      <div className="text-2xl mb-1">{option.emoji}</div>
                      <div className="text-xs text-gray-600">{option.name}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Comment */}
              <div className="mb-6">
                <label className="block font-medium text-gray-800 mb-2">
                  Ek Yorumlar (İsteğe Bağlı):
                </label>
                <textarea
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="Bu cevap hakkında başka söylemek istediğin bir şey var mı?"
                  rows={3}
                  maxLength={1000}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                />
                <div className="text-xs text-gray-500 mt-1">
                  {comment.length}/1000 karakter
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800">
                  <div className="flex items-center gap-2">
                    <span className="text-red-500">⚠️</span>
                    <span>{error}</span>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={handleClose}
                  disabled={submitting}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50"
                >
                  İptal
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 
                           disabled:opacity-50 disabled:cursor-not-allowed transition-colors
                           flex items-center space-x-2"
                >
                  {submitting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Gönderiliyor...</span>
                    </>
                  ) : (
                    <span>Geri Bildirim Gönder</span>
                  )}
                </button>
              </div>
            </>
          )}
        </form>
      </div>
    </div>
  );
}
