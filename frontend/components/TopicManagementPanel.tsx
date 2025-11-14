"use client";

import React, { useState, useEffect } from "react";
import {
  extractTopics,
  getSessionTopics,
  updateTopic,
  Topic,
  TopicExtractionRequest,
} from "@/lib/api";
import { URLS } from "../config/ports";

interface TopicManagementPanelProps {
  sessionId: string;
  apragEnabled: boolean;
}

const TopicManagementPanel: React.FC<TopicManagementPanelProps> = ({
  sessionId,
  apragEnabled,
}) => {
  const [topics, setTopics] = useState<Topic[]>([]);
  const [loading, setLoading] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [editingTopic, setEditingTopic] = useState<Topic | null>(null);
  const [topicPage, setTopicPage] = useState(1);
  const [generatingQuestions, setGeneratingQuestions] = useState<number | null>(
    null
  );
  const [generatedQuestions, setGeneratedQuestions] = useState<{
    [key: number]: string[];
  }>({});
  const TOPICS_PER_PAGE = 20; // Increased from 10 to 20 to show more topics

  // Fetch topics
  const fetchTopics = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await getSessionTopics(sessionId);
      if (response.success) {
        console.log(
          `[TopicManagement] Fetched ${response.topics.length} topics for session ${sessionId}`
        );
        setTopics(response.topics);
      }
    } catch (e: any) {
      setError(e.message || "Konular yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  // Extract topics
  const handleExtractTopics = async () => {
    try {
      setExtracting(true);
      setError(null);
      const request: TopicExtractionRequest = {
        session_id: sessionId,
        extraction_method: "llm_analysis",
        options: {
          include_subtopics: true,
          min_confidence: 0.7,
          max_topics: 50,
        },
      };

      const response = await extractTopics(request);
      if (response.success) {
        setSuccess(
          `${response.total_topics} konu başarıyla çıkarıldı (${(
            response.extraction_time_ms / 1000
          ).toFixed(1)}s)`
        );
        await fetchTopics();
      }
    } catch (e: any) {
      setError(e.message || "Konu çıkarımı başarısız oldu");
    } finally {
      setExtracting(false);
    }
  };

  // Update topic
  const handleUpdateTopic = async (topicId: number, updates: any) => {
    try {
      setError(null);
      await updateTopic(topicId, updates);
      setSuccess("Konu başarıyla güncellendi");
      await fetchTopics();
      setEditingTopic(null);
    } catch (e: any) {
      setError(e.message || "Konu güncellenemedi");
    }
  };

  // Generate questions for a topic
  const handleGenerateQuestions = async (
    topicId: number,
    topicTitle: string
  ) => {
    try {
      setGeneratingQuestions(topicId);
      setError(null);

      const response = await fetch(
        `${URLS.API_GATEWAY}/api/aprag/topics/${topicId}/generate-questions`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            count: 10,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Sorular üretilemedi");
      }

      const data = await response.json();
      if (data.success && data.questions) {
        setGeneratedQuestions((prev) => ({
          ...prev,
          [topicId]: data.questions,
        }));
        setSuccess(
          `${topicTitle} konusu için ${data.questions.length} soru üretildi!`
        );
      }
    } catch (e: any) {
      setError(e.message || "Soru üretimi başarısız oldu");
    } finally {
      setGeneratingQuestions(null);
    }
  };

  // Load topics on mount
  useEffect(() => {
    if (sessionId && apragEnabled) {
      fetchTopics();
    }
  }, [sessionId, apragEnabled]);

  if (!apragEnabled) {
    return null;
  }

  // Get main topics (without parents) for pagination
  const mainTopics = topics
    .filter((t) => !t.parent_topic_id)
    .sort((a, b) => a.topic_order - b.topic_order);
  const totalPages = Math.ceil(mainTopics.length / TOPICS_PER_PAGE);
  const paginatedTopics = mainTopics.slice(
    (topicPage - 1) * TOPICS_PER_PAGE,
    topicPage * TOPICS_PER_PAGE
  );

  // Debug logging
  console.log(
    `[TopicManagement DEBUG] Total topics from API: ${topics.length}`
  );
  console.log(
    `[TopicManagement DEBUG] Main topics (no parent): ${mainTopics.length}`
  );
  console.log(
    `[TopicManagement DEBUG] Paginated topics showing: ${paginatedTopics.length}`
  );
  console.log(
    `[TopicManagement DEBUG] Current page: ${topicPage}/${totalPages}`
  );
  console.log(`[TopicManagement DEBUG] Topics per page: ${TOPICS_PER_PAGE}`);
  console.log(
    "[TopicManagement DEBUG] All topics:",
    topics.map((t) => ({
      id: t.topic_id,
      title: t.topic_title,
      parent: t.parent_topic_id,
    }))
  );
  console.log(
    "[TopicManagement DEBUG] Main topics:",
    mainTopics.map((t) => ({ id: t.topic_id, title: t.topic_title }))
  );
  console.log(
    "[TopicManagement DEBUG] Paginated topics:",
    paginatedTopics.map((t) => ({ id: t.topic_id, title: t.topic_title }))
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-semibold text-foreground">
            Konu Yönetimi
          </h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            {topics.length} konu
          </p>
        </div>
        <button
          onClick={handleExtractTopics}
          disabled={extracting}
          className="py-2 px-3 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors inline-flex items-center gap-2"
        >
          {extracting ? (
            <>
              <svg
                className="animate-spin h-4 w-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              <span>Çıkarılıyor...</span>
            </>
          ) : (
            <span>Konuları Çıkar</span>
          )}
        </button>
      </div>

      <div>
        {/* Messages */}
        {error && (
          <div className="mb-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
            <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {success && (
          <div className="mb-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md p-3">
            <p className="text-sm text-green-800 dark:text-green-200">
              {success}
            </p>
          </div>
        )}

        {/* Topics List */}
        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mb-3"></div>
            <p className="text-sm text-muted-foreground">Yükleniyor...</p>
          </div>
        ) : topics.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-sm text-muted-foreground mb-1">
              Henüz konu çıkarılmamış
            </p>
            <p className="text-xs text-muted-foreground">
              "Konuları Çıkar" butonuna tıklayarak başlayın
            </p>
          </div>
        ) : (
          <>
            <div className="space-y-3">
              {paginatedTopics.map((topic) => {
                const subtopics = topics.filter(
                  (t) => t.parent_topic_id === topic.topic_id
                );
                return (
                  <div
                    key={topic.topic_id}
                    className="border border-border rounded-lg p-4 hover:bg-muted/30 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-xs font-medium text-muted-foreground bg-muted px-2 py-0.5 rounded">
                            #{topic.topic_order}
                          </span>
                          <h3 className="text-base font-semibold text-foreground">
                            {topic.topic_title}
                          </h3>
                          {topic.estimated_difficulty && (
                            <span
                              className={`text-xs px-2 py-0.5 rounded ${
                                topic.estimated_difficulty === "beginner"
                                  ? "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300"
                                  : topic.estimated_difficulty ===
                                    "intermediate"
                                  ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300"
                                  : "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300"
                              }`}
                            >
                              {topic.estimated_difficulty === "beginner"
                                ? "Başlangıç"
                                : topic.estimated_difficulty === "intermediate"
                                ? "Orta"
                                : "İleri"}
                            </span>
                          )}
                        </div>
                        {topic.description && (
                          <p className="text-sm text-muted-foreground mt-1.5">
                            {topic.description}
                          </p>
                        )}
                        {topic.keywords && topic.keywords.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {topic.keywords.map((keyword, idx) => (
                              <span
                                key={idx}
                                className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded"
                              >
                                {keyword}
                              </span>
                            ))}
                          </div>
                        )}
                        {subtopics.length > 0 && (
                          <div className="mt-4 p-3 bg-muted/30 rounded-lg border">
                            <p className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                              📋 Alt Konular ({subtopics.length})
                            </p>
                            <div className="space-y-2">
                              {subtopics
                                .sort((a, b) => a.topic_order - b.topic_order)
                                .map((subtopic) => (
                                  <div
                                    key={subtopic.topic_id}
                                    className="flex items-start gap-3 p-2 bg-background rounded border hover:bg-muted/50 transition-colors"
                                  >
                                    <span className="text-xs font-medium text-muted-foreground bg-primary/10 px-2 py-1 rounded min-w-0 shrink-0">
                                      #{subtopic.topic_order}
                                    </span>
                                    <span className="text-sm font-medium text-foreground flex-1">
                                      {subtopic.topic_title}
                                    </span>
                                    {subtopic.keywords &&
                                      subtopic.keywords.length > 0 && (
                                        <div className="flex flex-wrap gap-1">
                                          {subtopic.keywords
                                            .slice(0, 2)
                                            .map((keyword, idx) => (
                                              <span
                                                key={idx}
                                                className="text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 px-1.5 py-0.5 rounded"
                                              >
                                                {keyword}
                                              </span>
                                            ))}
                                        </div>
                                      )}
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}

                        {/* Generated Questions */}
                        {generatedQuestions[topic.topic_id] && (
                          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                            <p className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-3 flex items-center gap-2">
                              🧠 Üretilen Sorular (
                              {generatedQuestions[topic.topic_id].length})
                            </p>
                            <div className="space-y-2">
                              {generatedQuestions[topic.topic_id].map(
                                (question, idx) => (
                                  <div
                                    key={idx}
                                    className="flex items-start gap-2 text-sm"
                                  >
                                    <span className="text-blue-600 dark:text-blue-400 font-medium min-w-0 shrink-0">
                                      {idx + 1}.
                                    </span>
                                    <span className="text-blue-800 dark:text-blue-200 flex-1">
                                      {question}
                                    </span>
                                  </div>
                                )
                              )}
                            </div>
                            <button
                              onClick={() => {
                                const questions = generatedQuestions[
                                  topic.topic_id
                                ]
                                  .map((q, i) => `${i + 1}. ${q}`)
                                  .join("\n\n");
                                navigator.clipboard.writeText(questions);
                                setSuccess("Sorular panoya kopyalandı!");
                              }}
                              className="mt-3 px-3 py-1.5 text-xs bg-blue-100 text-blue-800 dark:bg-blue-800/30 dark:text-blue-200 rounded hover:bg-blue-200 dark:hover:bg-blue-800/50 transition-colors"
                            >
                              📋 Panoya Kopyala
                            </button>
                          </div>
                        )}
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() =>
                            handleGenerateQuestions(
                              topic.topic_id,
                              topic.topic_title
                            )
                          }
                          disabled={generatingQuestions === topic.topic_id}
                          className="flex-shrink-0 px-3 py-1.5 text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 rounded hover:bg-green-200 dark:hover:bg-green-900/50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                          title="Bu konu için sorular üret"
                        >
                          {generatingQuestions === topic.topic_id ? (
                            <>
                              <svg
                                className="animate-spin h-3 w-3 mr-1"
                                fill="none"
                                viewBox="0 0 24 24"
                              >
                                <circle
                                  className="opacity-25"
                                  cx="12"
                                  cy="12"
                                  r="10"
                                  stroke="currentColor"
                                  strokeWidth="4"
                                ></circle>
                                <path
                                  className="opacity-75"
                                  fill="currentColor"
                                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                ></path>
                              </svg>
                              Üretiliyor...
                            </>
                          ) : (
                            <>🧠 Soru Üret</>
                          )}
                        </button>
                        <button
                          onClick={() => setEditingTopic(topic)}
                          className="flex-shrink-0 text-muted-foreground hover:text-foreground transition-colors"
                          title="Düzenle"
                        >
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                              d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                            />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
                <button
                  onClick={() => setTopicPage((p) => Math.max(1, p - 1))}
                  disabled={topicPage === 1}
                  className="py-1.5 px-3 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Önceki
                </button>
                <span className="text-sm text-muted-foreground">
                  Sayfa {topicPage} / {totalPages}
                </span>
                <button
                  onClick={() =>
                    setTopicPage((p) => Math.min(totalPages, p + 1))
                  }
                  disabled={topicPage >= totalPages}
                  className="py-1.5 px-3 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Sonraki
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Edit Modal */}
      {editingTopic && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-card border border-border rounded-lg shadow-xl max-w-md w-full p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">
              Konu Düzenle
            </h3>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.currentTarget);
                handleUpdateTopic(editingTopic.topic_id, {
                  topic_title: formData.get("title") as string,
                  description: formData.get("description") as string,
                  topic_order: parseInt(formData.get("order") as string),
                });
              }}
            >
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Konu Başlığı
                  </label>
                  <input
                    type="text"
                    name="title"
                    defaultValue={editingTopic.topic_title}
                    className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground text-sm focus:ring-2 focus:ring-primary focus:border-primary"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Açıklama
                  </label>
                  <textarea
                    name="description"
                    defaultValue={editingTopic.description || ""}
                    className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground text-sm focus:ring-2 focus:ring-primary focus:border-primary"
                    rows={3}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Sıra
                  </label>
                  <input
                    type="number"
                    name="order"
                    defaultValue={editingTopic.topic_order}
                    className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground text-sm focus:ring-2 focus:ring-primary focus:border-primary"
                    required
                  />
                </div>
              </div>
              <div className="flex gap-2 mt-6">
                <button
                  type="submit"
                  className="flex-1 py-2 px-4 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 transition-colors"
                >
                  Kaydet
                </button>
                <button
                  type="button"
                  onClick={() => setEditingTopic(null)}
                  className="flex-1 py-2 px-4 bg-secondary text-secondary-foreground rounded-md text-sm font-medium hover:bg-secondary/80 transition-colors"
                >
                  İptal
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default TopicManagementPanel;
