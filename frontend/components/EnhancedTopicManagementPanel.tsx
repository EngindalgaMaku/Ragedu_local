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

interface EnhancedTopicManagementPanelProps {
  sessionId: string;
  apragEnabled: boolean;
}

interface KnowledgeBase {
  knowledge_id: number;
  topic_id: number;
  topic_summary: string;
  key_concepts: Array<{
    term: string;
    definition: string;
    importance: string;
    category?: string;
  }>;
  learning_objectives: Array<{
    level: string;
    objective: string;
    assessment_method?: string;
  }>;
  examples: Array<{
    title: string;
    scenario: string;
    explanation: string;
    concepts_demonstrated: string[];
  }>;
  content_quality_score: number;
  is_validated: boolean;
}

interface QAPair {
  qa_id: number;
  question: string;
  answer: string;
  explanation?: string;
  difficulty_level: string;
  question_type: string;
  bloom_taxonomy_level: string;
  times_asked: number;
  average_student_rating?: number;
}

const EnhancedTopicManagementPanel: React.FC<
  EnhancedTopicManagementPanelProps
> = ({ sessionId, apragEnabled }) => {
  const [topics, setTopics] = useState<Topic[]>([]);
  const [loading, setLoading] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [editingTopic, setEditingTopic] = useState<Topic | null>(null);
  const [topicPage, setTopicPage] = useState(1);
  
  // KB-Enhanced RAG states
  const [extractingKB, setExtractingKB] = useState<number | null>(null);
  const [extractingKBBatch, setExtractingKBBatch] = useState(false);
  const [viewingKB, setViewingKB] = useState<number | null>(null);
  const [knowledgeBases, setKnowledgeBases] = useState<{[key: number]: KnowledgeBase}>({});
  const [generatingQA, setGeneratingQA] = useState<number | null>(null);
  const [qaPairs, setQAPairs] = useState<{[key: number]: QAPair[]}>({});
  const [expandedTopics, setExpandedTopics] = useState<Set<number>>(new Set());
  
  const TOPICS_PER_PAGE = 20;

  // Fetch topics
  const fetchTopics = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await getSessionTopics(sessionId);
      if (response.success) {
        console.log(
          `[TopicManagement] Fetched ${response.topics.length} topics`
        );
        setTopics(response.topics);
      }
    } catch (e: any) {
      setError(e.message || "Konular yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  // Extract topics (existing)
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

  // NEW: Extract Knowledge Base for single topic
  const handleExtractKnowledge = async (topicId: number, topicTitle: string) => {
    try {
      setExtractingKB(topicId);
      setError(null);

      const response = await fetch(
        `${URLS.API_GATEWAY}/api/aprag/knowledge/extract/${topicId}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic_id: topicId, force_refresh: false }),
        }
      );

      if (!response.ok) {
        throw new Error("Knowledge base çıkarılamadı");
      }

      const data = await response.json();
      if (data.success) {
        setSuccess(
          `${topicTitle} için bilgi tabanı oluşturuldu! ` +
          `(Özet: ${data.extracted_components.summary_length} kelime, ` +
          `Kavram: ${data.extracted_components.concepts_count}, ` +
          `Hedef: ${data.extracted_components.objectives_count})`
        );
        
        // Fetch the KB to display
        await fetchKnowledgeBase(topicId);
      }
    } catch (e: any) {
      setError(e.message || "Bilgi tabanı çıkarımı başarısız");
    } finally {
      setExtractingKB(null);
    }
  };

  // NEW: Extract Knowledge Base for all topics (batch)
  const handleExtractKnowledgeBatch = async () => {
    try {
      setExtractingKBBatch(true);
      setError(null);

      const response = await fetch(
        `${URLS.API_GATEWAY}/api/aprag/knowledge/extract-batch/${sessionId}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            force_refresh: false,
            extraction_config: {
              generate_qa_pairs: true,
              qa_pairs_per_topic: 15,
              extract_examples: true,
              extract_misconceptions: true,
            },
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Toplu bilgi tabanı çıkarımı başarısız");
      }

      const data = await response.json();
      if (data.success) {
        setSuccess(
          `${data.processed_successfully}/${data.total_topics} konu için bilgi tabanı oluşturuldu! ` +
          `${data.errors_count > 0 ? `(${data.errors_count} hata)` : ""}`
        );
      }
    } catch (e: any) {
      setError(e.message || "Toplu bilgi tabanı çıkarımı başarısız");
    } finally {
      setExtractingKBBatch(false);
    }
  };

  // NEW: Fetch Knowledge Base for a topic
  const fetchKnowledgeBase = async (topicId: number) => {
    try {
      const response = await fetch(
        `${URLS.API_GATEWAY}/api/aprag/knowledge/kb/${topicId}`
      );

      if (!response.ok) {
        if (response.status === 404) {
          // KB doesn't exist yet
          return;
        }
        throw new Error("Bilgi tabanı yüklenemedi");
      }

      const data = await response.json();
      if (data.success && data.knowledge_base) {
        setKnowledgeBases(prev => ({
          ...prev,
          [topicId]: data.knowledge_base
        }));
        
        // Also set QA pairs if available
        if (data.knowledge_base.qa_pairs) {
          setQAPairs(prev => ({
            ...prev,
            [topicId]: data.knowledge_base.qa_pairs
          }));
        }
      }
    } catch (e: any) {
      console.error("Error fetching KB:", e);
    }
  };

  // NEW: Generate QA Pairs (Enhanced)
  const handleGenerateQAPairs = async (topicId: number, topicTitle: string) => {
    try {
      setGeneratingQA(topicId);
      setError(null);

      const response = await fetch(
        `${URLS.API_GATEWAY}/api/aprag/knowledge/generate-qa/${topicId}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            topic_id: topicId,
            count: 15,
            difficulty_distribution: {
              beginner: 5,
              intermediate: 7,
              advanced: 3,
            },
          }),
        }
      );

      if (!response.ok) {
        throw new Error("QA pairs üretilemedi");
      }

      const data = await response.json();
      if (data.success && data.qa_pairs) {
        setQAPairs(prev => ({
          ...prev,
          [topicId]: data.qa_pairs
        }));
        setSuccess(
          `${topicTitle} için ${data.count} soru-cevap çifti oluşturuldu!`
        );
      }
    } catch (e: any) {
      setError(e.message || "QA üretimi başarısız");
    } finally {
      setGeneratingQA(null);
    }
  };

  // Toggle topic expansion
  const toggleTopicExpansion = (topicId: number) => {
    const newExpanded = new Set(expandedTopics);
    if (newExpanded.has(topicId)) {
      newExpanded.delete(topicId);
    } else {
      newExpanded.add(topicId);
      // Load KB if not loaded yet
      if (!knowledgeBases[topicId]) {
        fetchKnowledgeBase(topicId);
      }
    }
    setExpandedTopics(newExpanded);
  };

  // Update topic (existing)
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

  // Load topics on mount
  useEffect(() => {
    if (sessionId && apragEnabled) {
      fetchTopics();
    }
  }, [sessionId, apragEnabled]);

  // Auto-dismiss messages
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  if (!apragEnabled) {
    return null;
  }

  const mainTopics = topics
    .filter((t) => !t.parent_topic_id)
    .sort((a, b) => a.topic_order - b.topic_order);
  const totalPages = Math.ceil(mainTopics.length / TOPICS_PER_PAGE);
  const paginatedTopics = mainTopics.slice(
    (topicPage - 1) * TOPICS_PER_PAGE,
    topicPage * TOPICS_PER_PAGE
  );

  return (
    <div>
      {/* Header with Actions */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <span className="text-2xl">📚</span>
            Gelişmiş Konu Yönetimi
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            {topics.length} konu • KB-Enhanced RAG
          </p>
        </div>
        
        <div className="flex gap-2">
          {/* Extract Topics */}
          <button
            onClick={handleExtractTopics}
            disabled={extracting}
            className="py-2 px-4 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all inline-flex items-center gap-2 shadow-sm"
          >
            {extracting ? (
              <>
                <svg
                  className="animate-spin h-4 w-4"
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
              <>
                <span>📋</span>
                <span>Konuları Çıkar</span>
              </>
            )}
          </button>

          {/* Batch KB Extraction - NEW */}
          <button
            onClick={handleExtractKnowledgeBatch}
            disabled={extractingKBBatch || topics.length === 0}
            className="py-2 px-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg text-sm font-medium hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all inline-flex items-center gap-2 shadow-md"
            title="Tüm konular için bilgi tabanı, özet, ve soru-cevaplar oluştur"
          >
            {extractingKBBatch ? (
              <>
                <svg
                  className="animate-spin h-4 w-4"
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
                <span>KB Oluşturuluyor...</span>
              </>
            ) : (
              <>
                <span>🧠</span>
                <span>Bilgi Tabanı Oluştur</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="mb-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 animate-in fade-in slide-in-from-top-2">
          <div className="flex items-start gap-3">
            <span className="text-xl">❌</span>
            <p className="text-sm text-red-800 dark:text-red-200 flex-1">{error}</p>
            <button
              onClick={() => setError(null)}
              className="text-red-500 hover:text-red-700"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {success && (
        <div className="mb-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4 animate-in fade-in slide-in-from-top-2">
          <div className="flex items-start gap-3">
            <span className="text-xl">✅</span>
            <p className="text-sm text-green-800 dark:text-green-200 flex-1">
              {success}
            </p>
            <button
              onClick={() => setSuccess(null)}
              className="text-green-500 hover:text-green-700"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Topics List */}
      {loading ? (
        <div className="text-center py-16">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent mb-4"></div>
          <p className="text-sm text-muted-foreground">Yükleniyor...</p>
        </div>
      ) : topics.length === 0 ? (
        <div className="text-center py-16 bg-muted/20 rounded-lg border-2 border-dashed">
          <span className="text-6xl mb-4 block">📚</span>
          <p className="text-base text-muted-foreground mb-2 font-medium">
            Henüz konu çıkarılmamış
          </p>
          <p className="text-sm text-muted-foreground mb-4">
            "Konuları Çıkar" butonuna tıklayarak başlayın
          </p>
        </div>
      ) : (
        <>
          <div className="space-y-4">
            {paginatedTopics.map((topic) => {
              const subtopics = topics.filter(
                (t) => t.parent_topic_id === topic.topic_id
              );
              const isExpanded = expandedTopics.has(topic.topic_id);
              const kb = knowledgeBases[topic.topic_id];
              const qaList = qaPairs[topic.topic_id] || [];

              return (
                <div
                  key={topic.topic_id}
                  className="border border-border rounded-xl overflow-hidden hover:shadow-md transition-all bg-card"
                >
                  {/* Topic Header */}
                  <div className="p-5 bg-gradient-to-r from-muted/50 to-muted/30">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 flex-wrap mb-2">
                          <span className="text-xs font-bold text-primary bg-primary/10 px-3 py-1 rounded-full">
                            #{topic.topic_order}
                          </span>
                          <h3 className="text-lg font-bold text-foreground">
                            {topic.topic_title}
                          </h3>
                          {topic.estimated_difficulty && (
                            <span
                              className={`text-xs px-3 py-1 rounded-full font-medium ${
                                topic.estimated_difficulty === "beginner"
                                  ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
                                  : topic.estimated_difficulty === "intermediate"
                                  ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300"
                                  : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                              }`}
                            >
                              {topic.estimated_difficulty === "beginner"
                                ? "🟢 Başlangıç"
                                : topic.estimated_difficulty === "intermediate"
                                ? "🟡 Orta"
                                : "🔴 İleri"}
                            </span>
                          )}
                          {kb && (
                            <span className="text-xs px-3 py-1 rounded-full font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300">
                              ✨ KB Hazır
                            </span>
                          )}
                        </div>
                        
                        {topic.description && (
                          <p className="text-sm text-muted-foreground mt-2">
                            {topic.description}
                          </p>
                        )}
                        
                        {topic.keywords && topic.keywords.length > 0 && (
                          <div className="flex flex-wrap gap-2 mt-3">
                            {topic.keywords.map((keyword, idx) => (
                              <span
                                key={idx}
                                className="text-xs bg-primary/10 text-primary px-2.5 py-1 rounded-full font-medium"
                              >
                                🔑 {keyword}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Action Buttons */}
                      <div className="flex gap-2">
                        <button
                          onClick={() => toggleTopicExpansion(topic.topic_id)}
                          className="flex-shrink-0 p-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg transition-colors"
                          title={isExpanded ? "Daralt" : "Genişlet"}
                        >
                          <svg
                            className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                              d="M19 9l-7 7-7-7"
                            />
                          </svg>
                        </button>
                      </div>
                    </div>

                    {/* Subtopics */}
                    {subtopics.length > 0 && (
                      <div className="mt-4 p-3 bg-background/80 rounded-lg border border-border/50">
                        <p className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                          📋 Alt Konular ({subtopics.length})
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {subtopics
                            .sort((a, b) => a.topic_order - b.topic_order)
                            .map((subtopic) => (
                              <div
                                key={subtopic.topic_id}
                                className="flex items-center gap-2 p-2 bg-muted/50 rounded border border-border/50 text-sm"
                              >
                                <span className="text-xs font-medium text-muted-foreground bg-primary/10 px-2 py-1 rounded">
                                  #{subtopic.topic_order}
                                </span>
                                <span className="text-sm font-medium text-foreground flex-1">
                                  {subtopic.topic_title}
                                </span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Expanded Content - NEW */}
                  {isExpanded && (
                    <div className="p-5 space-y-4 bg-background">
                      {/* Action Buttons Row */}
                      <div className="flex gap-2 flex-wrap">
                        <button
                          onClick={() => handleExtractKnowledge(topic.topic_id, topic.topic_title)}
                          disabled={extractingKB === topic.topic_id}
                          className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all inline-flex items-center gap-2"
                        >
                          {extractingKB === topic.topic_id ? (
                            <>
                              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                              </svg>
                              <span>Oluşturuluyor...</span>
                            </>
                          ) : (
                            <>
                              <span>✨</span>
                              <span>Bilgi Tabanı Oluştur</span>
                            </>
                          )}
                        </button>

                        <button
                          onClick={() => handleGenerateQAPairs(topic.topic_id, topic.topic_title)}
                          disabled={generatingQA === topic.topic_id}
                          className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all inline-flex items-center gap-2"
                        >
                          {generatingQA === topic.topic_id ? (
                            <>
                              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                              </svg>
                              <span>Üretiliyor...</span>
                            </>
                          ) : (
                            <>
                              <span>🧠</span>
                              <span>Soru-Cevap Üret (15)</span>
                            </>
                          )}
                        </button>
                      </div>

                      {/* Knowledge Base Display - NEW */}
                      {kb && (
                        <div className="space-y-4">
                          {/* Summary */}
                          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                            <h4 className="text-sm font-bold text-blue-900 dark:text-blue-100 mb-2 flex items-center gap-2">
                              <span>📝</span>
                              Konu Özeti
                              <span className="text-xs font-normal text-blue-600 dark:text-blue-300">
                                (Kalite: {(kb.content_quality_score * 100).toFixed(0)}%)
                              </span>
                            </h4>
                            <p className="text-sm text-blue-800 dark:text-blue-200 leading-relaxed">
                              {kb.topic_summary}
                            </p>
                          </div>

                          {/* Key Concepts */}
                          {kb.key_concepts && kb.key_concepts.length > 0 && (
                            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                              <h4 className="text-sm font-bold text-purple-900 dark:text-purple-100 mb-3 flex items-center gap-2">
                                <span>💡</span>
                                Anahtar Kavramlar ({kb.key_concepts.length})
                              </h4>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                {kb.key_concepts.map((concept, idx) => (
                                  <div
                                    key={idx}
                                    className="p-3 bg-white dark:bg-gray-800 rounded border border-purple-200 dark:border-purple-800"
                                  >
                                    <div className="flex items-start justify-between gap-2 mb-1">
                                      <h5 className="text-sm font-semibold text-purple-900 dark:text-purple-100">
                                        {concept.term}
                                      </h5>
                                      {concept.importance === "high" && (
                                        <span className="text-xs bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300 px-2 py-0.5 rounded">
                                          Önemli
                                        </span>
                                      )}
                                    </div>
                                    <p className="text-xs text-purple-700 dark:text-purple-300">
                                      {concept.definition}
                                    </p>
                                    {concept.category && (
                                      <span className="text-xs text-purple-500 dark:text-purple-400 mt-1 block">
                                        🏷️ {concept.category}
                                      </span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Learning Objectives */}
                          {kb.learning_objectives && kb.learning_objectives.length > 0 && (
                            <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                              <h4 className="text-sm font-bold text-amber-900 dark:text-amber-100 mb-3 flex items-center gap-2">
                                <span>🎯</span>
                                Öğrenme Hedefleri ({kb.learning_objectives.length})
                              </h4>
                              <div className="space-y-2">
                                {kb.learning_objectives.map((obj, idx) => {
                                  const bloomIcons: {[key: string]: string} = {
                                    remember: "🧠",
                                    understand: "💭",
                                    apply: "⚡",
                                    analyze: "🔍",
                                    evaluate: "⚖️",
                                    create: "✨"
                                  };
                                  return (
                                    <div
                                      key={idx}
                                      className="flex items-start gap-3 p-2 bg-white dark:bg-gray-800 rounded"
                                    >
                                      <span className="text-lg">{bloomIcons[obj.level] || "📌"}</span>
                                      <div className="flex-1">
                                        <span className="text-xs font-medium text-amber-600 dark:text-amber-400 uppercase">
                                          {obj.level}
                                        </span>
                                        <p className="text-sm text-amber-900 dark:text-amber-100 mt-0.5">
                                          {obj.objective}
                                        </p>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}

                          {/* Examples */}
                          {kb.examples && kb.examples.length > 0 && (
                            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                              <h4 className="text-sm font-bold text-green-900 dark:text-green-100 mb-3 flex items-center gap-2">
                                <span>🌍</span>
                                Gerçek Hayat Örnekleri ({kb.examples.length})
                              </h4>
                              <div className="space-y-3">
                                {kb.examples.slice(0, 3).map((example, idx) => (
                                  <div
                                    key={idx}
                                    className="p-3 bg-white dark:bg-gray-800 rounded border border-green-200 dark:border-green-800"
                                  >
                                    <h5 className="text-sm font-semibold text-green-900 dark:text-green-100 mb-1">
                                      {example.title}
                                    </h5>
                                    <p className="text-xs text-green-700 dark:text-green-300 mb-2">
                                      {example.scenario}
                                    </p>
                                    <p className="text-xs text-green-600 dark:text-green-400 italic">
                                      💡 {example.explanation}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* QA Pairs Display - NEW */}
                      {qaList.length > 0 && (
                        <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
                          <div className="flex items-center justify-between mb-3">
                            <h4 className="text-sm font-bold text-indigo-900 dark:text-indigo-100 flex items-center gap-2">
                              <span>❓</span>
                              Soru-Cevap Bankası ({qaList.length})
                            </h4>
                            <button
                              onClick={() => {
                                const qaText = qaList
                                  .map(
                                    (qa, i) =>
                                      `${i + 1}. SORU: ${qa.question}\n   CEVAP: ${qa.answer}\n   AÇIKLAMA: ${qa.explanation || "-"}\n   ZORLUK: ${qa.difficulty_level} | TÜR: ${qa.question_type}`
                                  )
                                  .join("\n\n");
                                navigator.clipboard.writeText(qaText);
                                setSuccess("Tüm soru-cevaplar panoya kopyalandı!");
                              }}
                              className="px-3 py-1 text-xs bg-indigo-100 text-indigo-700 dark:bg-indigo-800/30 dark:text-indigo-200 rounded hover:bg-indigo-200 dark:hover:bg-indigo-800/50 transition-colors"
                            >
                              📋 Tümünü Kopyala
                            </button>
                          </div>
                          <div className="space-y-3 max-h-96 overflow-y-auto">
                            {qaList.map((qa, idx) => (
                              <div
                                key={qa.qa_id}
                                className="p-3 bg-white dark:bg-gray-800 rounded border border-indigo-200 dark:border-indigo-800"
                              >
                                <div className="flex items-start justify-between gap-2 mb-2">
                                  <div className="flex items-center gap-2">
                                    <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400">
                                      #{idx + 1}
                                    </span>
                                    <span
                                      className={`text-xs px-2 py-0.5 rounded font-medium ${
                                        qa.difficulty_level === "beginner"
                                          ? "bg-green-100 text-green-700"
                                          : qa.difficulty_level === "intermediate"
                                          ? "bg-yellow-100 text-yellow-700"
                                          : "bg-red-100 text-red-700"
                                      }`}
                                    >
                                      {qa.difficulty_level}
                                    </span>
                                    <span className="text-xs px-2 py-0.5 rounded font-medium bg-blue-100 text-blue-700">
                                      {qa.question_type}
                                    </span>
                                  </div>
                                  {qa.times_asked > 0 && (
                                    <span className="text-xs text-muted-foreground">
                                      🔥 {qa.times_asked}x soruldu
                                    </span>
                                  )}
                                </div>
                                <p className="text-sm font-semibold text-indigo-900 dark:text-indigo-100 mb-2">
                                  ❓ {qa.question}
                                </p>
                                <p className="text-sm text-indigo-800 dark:text-indigo-200 mb-1">
                                  ✅ {qa.answer}
                                </p>
                                {qa.explanation && (
                                  <p className="text-xs text-indigo-600 dark:text-indigo-400 italic mt-2 p-2 bg-indigo-100/50 dark:bg-indigo-900/30 rounded">
                                    💡 {qa.explanation}
                                  </p>
                                )}
                                {qa.average_student_rating && (
                                  <div className="flex items-center gap-1 mt-2">
                                    <span className="text-xs text-muted-foreground">Rating:</span>
                                    <span className="text-xs font-medium text-yellow-600">
                                      {"⭐".repeat(Math.round(qa.average_student_rating))}
                                    </span>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* No KB Yet Message */}
                      {!kb && (
                        <div className="p-6 bg-muted/30 rounded-lg border-2 border-dashed text-center">
                          <span className="text-4xl block mb-2">📚</span>
                          <p className="text-sm text-muted-foreground mb-3">
                            Bu konu için henüz bilgi tabanı oluşturulmamış
                          </p>
                          <button
                            onClick={() => handleExtractKnowledge(topic.topic_id, topic.topic_title)}
                            disabled={extractingKB === topic.topic_id}
                            className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 disabled:opacity-50 transition-all inline-flex items-center gap-2"
                          >
                            <span>✨</span>
                            <span>Şimdi Oluştur</span>
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-6 pt-4 border-t border-border">
              <button
                onClick={() => setTopicPage((p) => Math.max(1, p - 1))}
                disabled={topicPage === 1}
                className="py-2 px-4 text-sm bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                ← Önceki
              </button>
              <span className="text-sm text-muted-foreground font-medium">
                Sayfa {topicPage} / {totalPages}
              </span>
              <button
                onClick={() => setTopicPage((p) => Math.min(totalPages, p + 1))}
                disabled={topicPage >= totalPages}
                className="py-2 px-4 text-sm bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Sonraki →
              </button>
            </div>
          )}
        </>
      )}

      {/* Edit Modal - Existing */}
      {editingTopic && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in">
          <div className="bg-card border border-border rounded-xl shadow-2xl max-w-lg w-full p-6 animate-in slide-in-from-bottom-4">
            <h3 className="text-xl font-bold text-foreground mb-5 flex items-center gap-2">
              <span>✏️</span>
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
                    className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground text-sm focus:ring-2 focus:ring-primary focus:border-primary"
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
                    className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground text-sm focus:ring-2 focus:ring-primary focus:border-primary"
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
                    className="w-full px-3 py-2 border border-border rounded-lg bg-background text-foreground text-sm focus:ring-2 focus:ring-primary focus:border-primary"
                    required
                  />
                </div>
              </div>
              <div className="flex gap-3 mt-6">
                <button
                  type="submit"
                  className="flex-1 py-2.5 px-4 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
                >
                  💾 Kaydet
                </button>
                <button
                  type="button"
                  onClick={() => setEditingTopic(null)}
                  className="flex-1 py-2.5 px-4 bg-secondary text-secondary-foreground rounded-lg text-sm font-medium hover:bg-secondary/80 transition-colors"
                >
                  ✕ İptal
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhancedTopicManagementPanel;






