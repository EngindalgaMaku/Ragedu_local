"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import ModernAdminLayout from "../components/ModernAdminLayout";

interface AnswerMetrics {
  word_count: number;
  sentence_count: number;
  has_citation: boolean;
  keyword_coverage: number;
  context_relevance: number;
  completeness_score: number;
  overall_quality: number;
}

interface ComparativeAnalysis {
  similarity: number;
  length_difference: number;
  unique_to_answer2: number;
}

interface TestResult {
  test_id: string;
  query: string;
  expected_relevant: boolean;
  expected_result: string;
  actual_result: string;
  passed: boolean;
  documents_retrieved: number;
  execution_time_ms: number;
  category?: string;
  llm_model?: string;
  embedding_model?: string;
  session_id?: string;
  results_preview?: Array<{
    text: string;
    similarity_score: number;
    crag_score: number;
  }>;
  llm_answers?: {
    direct_llm: string;
    simple_rag: string;
    advanced_rag: string;
  };
  evaluation_metrics?: {
    direct_llm: AnswerMetrics;
    simple_rag: AnswerMetrics;
    advanced_rag: AnswerMetrics;
  };
  comparative_analysis?: {
    direct_vs_simple: ComparativeAnalysis;
    simple_vs_advanced: ComparativeAnalysis;
    direct_vs_advanced: ComparativeAnalysis;
  };
  timestamp?: string;
  crag_evaluation?: any;
  correction?: {
    original_answer: string;
    issues: string[];
    was_corrected: boolean;
  };
}

interface TestSummary {
  total_tests: number;
  passed: number;
  failed: number;
  success_rate: number;
  avg_execution_time_ms: number;
  false_positive_rate: number;
  false_negative_rate: number;
}

interface SystemStatus {
  chromadb_available: boolean;
  document_count: number;
  chunk_count: number;
  crag_enabled: boolean;
  crag_evaluator_available: boolean;
  test_generation_available: boolean;
}

interface Model {
  id: string;
  name: string;
  provider?: string;
  description?: string;
}

interface Session {
  id: string;
  title: string;
  description?: string;
  category?: string;
  created_by?: string;
  created_at?: string;
}

export default function RAGTestsPage() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<"manual" | "auto">("manual");
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(false);

  // Manual test state
  const [manualQuery, setManualQuery] = useState("");
  const [expectedRelevant, setExpectedRelevant] = useState(true);
  const [manualResult, setManualResult] = useState<TestResult | null>(null);

  // Model and session selection
  const [llmModels, setLlmModels] = useState<Model[]>([]);
  const [embeddingModels, setEmbeddingModels] = useState<Model[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedLlm, setSelectedLlm] = useState<string>("");
  const [selectedEmbedding, setSelectedEmbedding] = useState<string>("");
  const [selectedSession, setSelectedSession] = useState<string>("");
  const [sessionEmbeddingModel, setSessionEmbeddingModel] =
    useState<string>("");
  const [sessionStats, setSessionStats] = useState<{
    documents: number;
    chunks: number;
  } | null>(null);

  // Auto test state
  const [numTests, setNumTests] = useState(10);
  const [includeRelevant, setIncludeRelevant] = useState(true);
  const [includeIrrelevant, setIncludeIrrelevant] = useState(true);
  const [generatedTests, setGeneratedTests] = useState<any[]>([]);
  const [batchResults, setBatchResults] = useState<{
    summary: TestSummary;
    results: TestResult[];
  } | null>(null);

  const formatPercent = (value?: number | null) => {
    if (value === undefined || value === null || Number.isNaN(value)) {
      return "—";
    }
    return `${(value * 100).toFixed(1)}%`;
  };

  // Load system status
  useEffect(() => {
    loadSystemStatus();
    loadAvailableModels();
    loadAvailableSessions();
  }, []);

  // Auto-load embedding model when session changes
  useEffect(() => {
    if (selectedSession) {
      loadSessionDetails(selectedSession);
    } else {
      setSessionEmbeddingModel("");
      setSessionStats(null);
    }
  }, [selectedSession]);

  const loadSystemStatus = async () => {
    try {
      const response = await fetch("/api/admin/rag-tests/status", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setStatus(data.status);
      }
    } catch (error) {
      console.error("Failed to load system status:", error);
    }
  };

  const loadAvailableModels = async () => {
    try {
      console.log("🔍 Loading LLM models from /api/models");
      const llmResponse = await fetch("/api/models", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
      });

      console.log("📡 LLM models response status:", llmResponse.status);

      if (llmResponse.ok) {
        const llmData = await llmResponse.json();
        console.log("✅ LLM models loaded:", llmData);

        // Format LLM models
        const formattedLlmModels =
          llmData.models?.map((model: any) => ({
            id: model.id || model,
            name: model.name || model,
            provider: model.provider || "groq",
            description: model.description || "",
          })) || [];

        setLlmModels(formattedLlmModels);
        if (formattedLlmModels.length > 0) {
          setSelectedLlm(formattedLlmModels[0].id);
        }
      }

      // Load embedding models
      console.log("🔍 Loading embedding models from /api/models/embedding");
      const embResponse = await fetch("/api/models/embedding", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
      });

      console.log("📡 Embedding models response status:", embResponse.status);

      if (embResponse.ok) {
        const embData = await embResponse.json();
        console.log("✅ Embedding models loaded:", embData);

        // Combine Ollama and HuggingFace models
        const formattedEmbModels = [
          ...(embData.ollama?.map((m: string) => ({
            id: m,
            name: m,
            provider: "ollama",
          })) || []),
          ...(embData.huggingface?.map((m: any) => ({
            id: m.id,
            name: m.name,
            provider: "huggingface",
            description: m.description,
          })) || []),
        ];

        setEmbeddingModels(formattedEmbModels);
        if (formattedEmbModels.length > 0) {
          setSelectedEmbedding(formattedEmbModels[0].id);
        }
      }
    } catch (error) {
      console.error("❌ Error loading models:", error);
    }
  };

  const loadAvailableSessions = async () => {
    try {
      console.log("🔍 Loading sessions from /api/sessions");
      const response = await fetch("/api/sessions", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
      });

      console.log("📡 Sessions response status:", response.status);

      if (response.ok) {
        const data = await response.json();
        console.log("✅ Sessions loaded:", data);

        // Format sessions - API returns SessionMeta[] directly
        const formattedSessions = data.map((s: any) => ({
          id: s.session_id,
          title: s.name,
          description: s.description,
          category: s.category,
          created_by: s.created_by,
          created_at: s.created_at,
        }));

        setSessions(formattedSessions);

        // Auto-select first session if available
        if (formattedSessions.length > 0) {
          setSelectedSession(formattedSessions[0].id);
        }
      } else {
        const errorText = await response.text();
        console.error(
          "❌ Failed to load sessions:",
          response.status,
          errorText
        );
      }
    } catch (error) {
      console.error("❌ Error loading sessions:", error);
    }
  };

  const loadSessionDetails = async (sessionId: string) => {
    try {
      // Get session metadata to find embedding model
      const sessionResponse = await fetch(`/api/sessions/${sessionId}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
      });

      if (sessionResponse.ok) {
        const sessionData = await sessionResponse.json();
        const embeddingModel =
          sessionData.rag_settings?.embedding_model ||
          "nomic-embed-text-latest";
        setSessionEmbeddingModel(embeddingModel);
        setSelectedEmbedding(embeddingModel);

        // Update stats
        setSessionStats({
          documents: sessionData.document_count || 0,
          chunks: sessionData.total_chunks || 0,
        });

        console.log(
          `✅ Session ${sessionId} embedding model: ${embeddingModel}`
        );
      }
    } catch (error) {
      console.error("❌ Error loading session details:", error);
    }
  };

  const executeManualTest = async () => {
    if (!manualQuery.trim()) {
      alert("Lütfen bir test sorusu girin");
      return;
    }

    if (!selectedSession) {
      alert("Lütfen bir ders oturumu seçin");
      return;
    }

    setLoading(true);
    setManualResult(null);

    try {
      const response = await fetch("/api/admin/rag-tests/execute-manual", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
        body: JSON.stringify({
          query: manualQuery,
          expected_relevant: expectedRelevant,
          category: "manual",
          llm_model: selectedLlm || undefined,
          session_id: selectedSession,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setManualResult(data.test_result);
      } else {
        const errorData = await response.json().catch(() => null);
        const errorMessage =
          errorData?.detail || `Test çalıştırılamadı (${response.status})`;
        console.error("❌ Test failed:", response.status, errorData);
        alert(errorMessage);
      }
    } catch (error) {
      console.error("❌ Error executing manual test:", error);
      alert(
        `Bir hata oluştu: ${
          error instanceof Error ? error.message : "Bilinmeyen hata"
        }`
      );
    } finally {
      setLoading(false);
    }
  };

  const generateAutoTests = async () => {
    setLoading(true);
    setGeneratedTests([]);

    try {
      const response = await fetch("/api/admin/rag-tests/generate-auto-tests", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
        body: JSON.stringify({
          num_tests: numTests,
          include_relevant: includeRelevant,
          include_irrelevant: includeIrrelevant,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setGeneratedTests(data.test_queries);
      } else {
        alert("Testler oluşturulamadı");
      }
    } catch (error) {
      console.error("Error generating tests:", error);
      alert("Bir hata oluştu");
    } finally {
      setLoading(false);
    }
  };

  const executeBatchTests = async () => {
    if (generatedTests.length === 0) {
      alert("Önce testleri oluşturun");
      return;
    }

    setLoading(true);
    setBatchResults(null);

    try {
      const response = await fetch("/api/admin/rag-tests/execute-batch", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
        body: JSON.stringify(generatedTests),
      });

      if (response.ok) {
        const data = await response.json();
        setBatchResults({
          summary: data.summary,
          results: data.results,
        });
      } else {
        alert("Testler çalıştırılamadı");
      }
    } catch (error) {
      console.error("Error executing batch tests:", error);
      alert("Bir hata oluştu");
    } finally {
      setLoading(false);
    }
  };

  const loadSampleQueries = async (category: string) => {
    try {
      const response = await fetch(
        `/api/admin/rag-tests/sample-queries?category=${category}`,
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("access_token")}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        if (data.queries.length > 0) {
          const randomQuery =
            data.queries[Math.floor(Math.random() * data.queries.length)];
          setManualQuery(randomQuery.query);
          setExpectedRelevant(randomQuery.expected_relevant);
        }
      }
    } catch (error) {
      console.error("Error loading sample queries:", error);
    }
  };

  return (
    <ModernAdminLayout
      title="RAG Test Sistemi"
      description="RAG sisteminin kalitesini test edin ve CRAG evaluator performansını ölçün"
    >
      {(llmModels.length === 0 ||
        embeddingModels.length === 0 ||
        sessions.length === 0) && (
        <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            ⚠️ Model/oturum listesi yükleniyor... Eğer yüklenmediyse sayfayı
            yenileyin.
          </p>
        </div>
      )}

      {/* System Status - Session Specific */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                ChromaDB
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {status?.chromadb_available ? "✅" : "❌"}
              </p>
            </div>
            <div className="text-right">
              <p className="text-xs text-gray-500">Dökümanlar</p>
              <p className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                {sessionStats?.documents || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Chunk'lar
              </p>
              <p className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                {sessionStats?.chunks || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center">
                DYSK
                <span
                  className="ml-1 text-xs text-gray-400"
                  title="Doğrulayıcı Yeniden Sıralama Katmanı"
                >
                  (?)
                </span>
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {status?.crag_enabled ? "✅" : "❌"}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Evaluator
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {status?.crag_evaluator_available ? "✅" : "❌"}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex space-x-4 mb-6 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setActiveTab("manual")}
          className={`px-4 py-2 font-semibold border-b-2 transition-colors ${
            activeTab === "manual"
              ? "border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400"
              : "border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
          }`}
        >
          📝 Manuel Test
        </button>
        <button
          onClick={() => setActiveTab("auto")}
          className={`px-4 py-2 font-semibold border-b-2 transition-colors ${
            activeTab === "auto"
              ? "border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400"
              : "border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
          }`}
        >
          🤖 Otomatik Testler
        </button>
      </div>

      {/* Manual Test Tab */}
      {activeTab === "manual" && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Manuel Test Çalıştır
            </h2>

            <div className="space-y-4">
              {/* Model ve Oturum Seçimi */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    🤖 LLM Modeli
                  </label>
                  <select
                    value={selectedLlm}
                    onChange={(e) => setSelectedLlm(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">Varsayılan</option>
                    {llmModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name} ({model.provider})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    📏 Embedding Modeli
                  </label>
                  <input
                    type="text"
                    value={sessionEmbeddingModel || "Oturum seçin..."}
                    readOnly
                    disabled
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 cursor-not-allowed"
                    title="Embedding modeli, seçilen oturumdan otomatik olarak yüklenir ve değiştirilemez."
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    🔒 Sorgu, bu modelle vektöre dönüştürülecektir.
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    📚 Ders Oturumu *
                  </label>
                  <select
                    value={selectedSession}
                    onChange={(e) => setSelectedSession(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
                    required
                  >
                    {sessions.length === 0 && (
                      <option value="">Yükleniyor...</option>
                    )}
                    {sessions.map((session) => (
                      <option key={session.id} value={session.id}>
                        {session.title} ({session.category})
                      </option>
                    ))}
                  </select>
                  {sessionStats && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      📄 {sessionStats.documents} doküman, {sessionStats.chunks}{" "}
                      chunk
                    </p>
                  )}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Test Sorusu
                </label>
                <textarea
                  value={manualQuery}
                  onChange={(e) => setManualQuery(e.target.value)}
                  placeholder="Test sorusunu girin..."
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
                  rows={3}
                />
              </div>

              <div className="flex items-center space-x-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={expectedRelevant}
                    onChange={(e) => setExpectedRelevant(e.target.checked)}
                    className="w-4 h-4 text-indigo-600 rounded"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    Alakalı sonuç bekleniyor
                  </span>
                </label>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={executeManualTest}
                  disabled={loading || !manualQuery.trim()}
                  className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? "⏳ Çalıştırılıyor..." : "▶️ Testi Çalıştır"}
                </button>

                <button
                  onClick={() => loadSampleQueries("all")}
                  className="px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-white rounded-lg font-semibold transition-colors"
                >
                  🎲 Örnek Soru Yükle
                </button>
              </div>
            </div>

            {/* Manual Test Result */}
            {manualResult && (
              <div className="mt-6 p-4 rounded-lg border-2 border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                    Test Sonucu
                  </h3>
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-semibold ${
                      manualResult.passed
                        ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                        : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                    }`}
                  >
                    {manualResult.passed ? "✅ PASSED" : "❌ FAILED"}
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Beklenen
                    </p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {manualResult.expected_result.toUpperCase()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Gerçekleşen
                    </p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {manualResult.actual_result.toUpperCase()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Dökümanlar
                    </p>
                    <p className="text-lg font-semibold text-indigo-600 dark:text-indigo-400">
                      {manualResult.documents_retrieved}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Süre
                    </p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {(manualResult.execution_time_ms || 0).toFixed(2)}ms
                    </p>
                  </div>
                  {manualResult.llm_model && (
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        LLM Model
                      </p>
                      <p className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                        {manualResult.llm_model}
                      </p>
                    </div>
                  )}
                  {manualResult.embedding_model && (
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Embedding
                      </p>
                      <p className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                        {manualResult.embedding_model}
                      </p>
                    </div>
                  )}
                </div>

                {manualResult.results_preview &&
                  manualResult.results_preview.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                        Bulunan Dökümanlar:
                      </h4>
                      {manualResult.results_preview.map((result, idx) => (
                        <div
                          key={idx}
                          className="mb-2 p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
                        >
                          <p className="text-sm text-gray-800 dark:text-gray-200 mb-1">
                            {result.text}
                          </p>
                          <div className="flex space-x-4 text-xs text-gray-600 dark:text-gray-400">
                            <span>
                              Similarity:{" "}
                              {(result.similarity_score || 0).toFixed(3)}
                            </span>
                            <span>
                              DYSK Skoru: {(result.crag_score || 0).toFixed(4)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                {/* Comparative LLM Answers Section */}
                {manualResult.llm_answers && (
                  <div className="mt-6 space-y-4">
                    <div>
                      <h4 className="text-md font-bold text-gray-800 dark:text-gray-200 mb-2">
                        Karşılaştırmalı Yanıtlar
                      </h4>
                    </div>

                    {/* Advanced RAG (DYSK) Answer */}
                    <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-700">
                      <h5 className="font-semibold text-indigo-800 dark:text-indigo-300">
                        Gelişmiş RAG (DYSK) Yanıtı
                      </h5>
                      <p className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap mt-1">
                        {manualResult.llm_answers.advanced_rag}
                      </p>
                      
                      {/* Self-Correction Notice */}
                      {manualResult.correction && manualResult.correction.was_corrected && (
                        <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 dark:border-amber-600 rounded-r">
                          <div className="flex items-start gap-2">
                            <div className="text-amber-600 dark:text-amber-400 mt-0.5">
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                              </svg>
                            </div>
                            <div className="flex-1">
                              <h6 className="text-xs font-bold text-amber-800 dark:text-amber-300 mb-1">
                                ⚠️ Otomatik Doğrulama ve Düzeltme Uygulandı
                              </h6>
                              <p className="text-xs text-amber-700 dark:text-amber-400 mb-1">
                                Sistem ilk cevabında tutarsızlık tespit etti ve cevabı otomatik olarak düzeltti:
                              </p>
                              <ul className="list-disc list-inside text-xs text-amber-800 dark:text-amber-300 space-y-0.5 bg-amber-100/50 dark:bg-amber-900/30 p-2 rounded">
                                {manualResult.correction.issues.map((issue, idx) => (
                                  <li key={idx}>{issue}</li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {/* Verification Success Notice */}
                      {manualResult.correction && !manualResult.correction.was_corrected && manualResult.correction.issues && manualResult.correction.issues.length === 0 && (
                        <div className="mt-3 p-2 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 dark:border-green-600 rounded-r">
                          <div className="flex items-start gap-2">
                            <div className="text-green-600 dark:text-green-400">
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium text-green-800 dark:text-green-300">
                              ✅ Cevap doğrulandı - Tutarlılık kontrolü başarılı
                            </p>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Simple RAG Answer */}
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-700">
                      <h5 className="font-semibold text-blue-800 dark:text-blue-300">
                        Basit RAG Yanıtı (DYSK olmadan)
                      </h5>
                      <p className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap mt-1">
                        {manualResult.llm_answers.simple_rag}
                      </p>
                    </div>

                    {/* Direct LLM Answer */}
                    <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                      <h5 className="font-semibold text-gray-800 dark:text-gray-300">
                        Direkt LLM Yanıtı (RAG olmadan)
                      </h5>
                      <p className="text-sm text-gray-700 dark:text-gray-400 whitespace-pre-wrap mt-1">
                        {manualResult.llm_answers.direct_llm}
                      </p>
                    </div>
                  </div>
                )}

                {/* Evaluation Metrics */}
                {manualResult && manualResult.evaluation_metrics && (
                  <div className="mt-6">
                    <h4 className="text-md font-bold text-gray-800 dark:text-gray-200 mb-3">
                      📊 Değerlendirme Metrikleri
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {/* Direct LLM */}
                      <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
                        <h5 className="font-semibold text-gray-700 dark:text-gray-300 text-sm mb-2">
                          Direkt LLM
                        </h5>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Kalite:</span>
                            <span className="font-semibold text-gray-900 dark:text-white">
                              {(manualResult.evaluation_metrics.direct_llm.overall_quality * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Kelime:</span>
                            <span className="font-semibold">{manualResult.evaluation_metrics.direct_llm.word_count}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Kaynak:</span>
                            <span>{manualResult.evaluation_metrics.direct_llm.has_citation ? "✅" : "❌"}</span>
                          </div>
                        </div>
                      </div>

                      {/* Simple RAG */}
                      <div className="p-4 rounded-lg border border-blue-200 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/20">
                        <h5 className="font-semibold text-blue-700 dark:text-blue-300 text-sm mb-2">
                          Basit RAG
                        </h5>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Kalite:</span>
                            <span className="font-semibold text-blue-700 dark:text-blue-300">
                              {(manualResult.evaluation_metrics.simple_rag.overall_quality * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Bağlam Alakası:</span>
                            <span className="font-semibold">{(manualResult.evaluation_metrics.simple_rag.context_relevance * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Kaynak:</span>
                            <span>{manualResult.evaluation_metrics.simple_rag.has_citation ? "✅" : "❌"}</span>
                          </div>
                        </div>
                      </div>

                      {/* Advanced RAG */}
                      <div className="p-4 rounded-lg border border-indigo-200 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-900/20">
                        <h5 className="font-semibold text-indigo-700 dark:text-indigo-300 text-sm mb-2">
                          Gelişmiş RAG (DYSK)
                        </h5>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Kalite:</span>
                            <span className="font-semibold text-indigo-700 dark:text-indigo-300">
                              {(manualResult.evaluation_metrics.advanced_rag.overall_quality * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Bağlam Alakası:</span>
                            <span className="font-semibold">{(manualResult.evaluation_metrics.advanced_rag.context_relevance * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Kaynak:</span>
                            <span>{manualResult.evaluation_metrics.advanced_rag.has_citation ? "✅" : "❌"}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Scientific Report Button */}
                <div className="mt-6">
                  <button
                    onClick={() => copyScientificReport(manualResult)}
                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
                  >
                    📋 Bilimsel Raporu Kopyala
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Auto Test Tab */}
      {activeTab === "auto" && (
        <div className="space-y-6">
          {/* Test Generation */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Otomatik Test Oluştur
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Test Sayısı: {numTests}
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={numTests}
                  onChange={(e) => setNumTests(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="flex space-x-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeRelevant}
                    onChange={(e) => setIncludeRelevant(e.target.checked)}
                    className="w-4 h-4 text-indigo-600 rounded"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    Alakalı sorular ekle
                  </span>
                </label>

                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeIrrelevant}
                    onChange={(e) => setIncludeIrrelevant(e.target.checked)}
                    className="w-4 h-4 text-indigo-600 rounded"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    Alakasız sorular ekle
                  </span>
                </label>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={generateAutoTests}
                  disabled={loading || (!includeRelevant && !includeIrrelevant)}
                  className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? "⏳ Oluşturuluyor..." : "🎲 Testleri Oluştur"}
                </button>

                {generatedTests.length > 0 && (
                  <button
                    onClick={executeBatchTests}
                    disabled={loading}
                    className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {loading ? "⏳ Çalıştırılıyor..." : "▶️ Testleri Çalıştır"}
                  </button>
                )}
              </div>
            </div>

            {generatedTests.length > 0 && !batchResults && (
              <div className="mt-6">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  {generatedTests.length} test oluşturuldu
                </p>
                <div className="max-h-60 overflow-y-auto space-y-2">
                  {generatedTests.slice(0, 10).map((test, idx) => (
                    <div
                      key={idx}
                      className="p-2 bg-gray-50 dark:bg-gray-900 rounded text-sm text-gray-700 dark:text-gray-300"
                    >
                      <span
                        className={`inline-block px-2 py-1 rounded text-xs mr-2 ${
                          test.expected_relevant
                            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                            : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                        }`}
                      >
                        {test.expected_relevant ? "✅ Alakalı" : "❌ Alakasız"}
                      </span>
                      {test.query}
                    </div>
                  ))}
                  {generatedTests.length > 10 && (
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      +{generatedTests.length - 10} daha...
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Batch Results */}
          {batchResults && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                Test Sonuçları
              </h2>

              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="p-4 bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-900/20 dark:to-indigo-800/20 rounded-lg">
                  <p className="text-sm text-indigo-600 dark:text-indigo-400 mb-1">
                    Başarı Oranı
                  </p>
                  <p className="text-3xl font-bold text-indigo-900 dark:text-indigo-100">
                    {(batchResults.summary.success_rate || 0).toFixed(1)}%
                  </p>
                </div>

                <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
                  <p className="text-sm text-green-600 dark:text-green-400 mb-1">
                    Başarılı
                  </p>
                  <p className="text-3xl font-bold text-green-900 dark:text-green-100">
                    {batchResults.summary.passed}
                  </p>
                </div>

                <div className="p-4 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-lg">
                  <p className="text-sm text-red-600 dark:text-red-400 mb-1">
                    Başarısız
                  </p>
                  <p className="text-3xl font-bold text-red-900 dark:text-red-100">
                    {batchResults.summary.failed}
                  </p>
                </div>

                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
                  <p className="text-sm text-purple-600 dark:text-purple-400 mb-1">
                    Ort. Süre
                  </p>
                  <p className="text-3xl font-bold text-purple-900 dark:text-purple-100">
                    {(batchResults.summary.avg_execution_time_ms || 0).toFixed(
                      0
                    )}
                    ms
                  </p>
                </div>
              </div>

              {/* Detailed Results */}
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {batchResults.results.map((result, idx) => (
                  <div
                    key={result.test_id}
                    className={`p-3 rounded-lg border-2 ${
                      result.passed
                        ? "border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20"
                        : "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20"
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                          {idx + 1}. {result.query}
                        </p>
                        <div className="flex space-x-4 text-xs text-gray-600 dark:text-gray-400">
                          <span>Beklenen: {result.expected_result}</span>
                          <span>Gerçekleşen: {result.actual_result}</span>
                          <span>Döküman: {result.documents_retrieved}</span>
                          <span>
                            {(result.execution_time_ms || 0).toFixed(2)}ms
                          </span>
                        </div>
                      </div>
                      <span className="text-2xl">
                        {result.passed ? "✅" : "❌"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* AUTO TEST TAB */}
      {activeTab === "auto" && (
        <div>AUTO TEST CONTENT HERE</div>
      )}
    </ModernAdminLayout>
  );
}

function copyScientificReport(result: TestResult) {
  if (!result) return;

  const report = `
## RAG Test Sonuç Raporu

**Test Tarihi:** ${new Date(result.timestamp || Date.now()).toLocaleString(
    "tr-TR"
  )}
**Test ID:** ${result.test_id}

---

### 1. Test Parametreleri
- **Sorgu:** ${result.query}
- **Beklenen Sonuç:** ${result.expected_result.toUpperCase()}
- **Oturum ID:** ${result.session_id}
- **LLM Modeli:** ${result.llm_model}
- **Embedding Modeli:** ${result.embedding_model}

---

### 2. Doğrulayıcı Yeniden Sıralama (DYSK) Sonucu
- **Gerçekleşen Sonuç:** ${result.actual_result.toUpperCase()}
- **Test Durumu:** ${result.passed ? "BAŞARILI" : "BAŞARISIZ"}
- **Değerlendirme Süresi:** ${result.execution_time_ms.toFixed(2)} ms
- **Alaka Düzeyi Güven Skoru (Confidence):** ${(
    result.crag_evaluation?.confidence || 0
  ).toFixed(3)}
- **Getirilen Döküman Sayısı:** ${result.documents_retrieved}

---

### 3. Getirilen Döküman Analizi
${(result.results_preview || [])
  .map(
    (doc, index) => `
**Döküman ${index + 1}:**
- **DYSK Skoru (Alaka Düzeyi):** ${doc.crag_score.toFixed(4)}
- **Vektör Benzerlik Skoru:** ${doc.similarity_score.toFixed(4)}
- **İçerik Özeti:** ${doc.text.substring(0, 200)}...
`
  )
  .join("\n")}
---

### 4. Karşılaştırmalı Yanıtlar

**4.1. Gelişmiş RAG (DYSK) Yanıtı**
*Bu yanıt, DYSK katmanının alakasız dokümanları filtrelemesinden sonra kalan, yüksek kaliteli bağlam ile üretilmiştir.*
\`\`\`
${result.llm_answers?.advanced_rag || "Yanıt üretilmedi."}
\`\`\`

**4.2. Basit RAG Yanıtı (DYSK olmadan)**
*Bu yanıt, ilk vektör aramasından gelen tüm dokümanların (filtresiz) kullanılmasıyla üretilmiştir.*
\`\`\`
${result.llm_answers?.simple_rag || "Yanıt üretilmedi."}
\`\`\`

**4.3. Direkt LLM Yanıtı (RAG olmadan)**
*Bu yanıt, hiçbir doküman bağlamı olmadan, sadece LLM'in kendi bilgisiyle üretilmiştir.*
\`\`\`
${result.llm_answers?.direct_llm || "Yanıt üretilmedi."}
\`\`\`

---

### 5. Analiz ve Yorum
Bu test, RAG sisteminin "${
    result.query
  }" sorgusuna verdiği yanıtın kalitesini ölçmektedir. Doğrulayıcı Yeniden Sıralama Katmanı (DYSK), getirilen dökümanların alaka düzeyini ${(
    result.crag_evaluation?.confidence || 0
  ).toFixed(
    3
  )} güven skoru ile "${result.actual_result.toUpperCase()}" olarak belirlemiştir. Bu sonuç, beklenen "${result.expected_result.toUpperCase()}" hedefi ile ${
    result.passed ? "uyumludur" : "uyumlu değildir"
  }.
    `.trim();

  navigator.clipboard
    .writeText(report)
    .then(() => {
      alert("Bilimsel rapor panoya kopyalandı!");
    })
    .catch((err) => {
      console.error("Rapor kopyalanamadı: ", err);
      alert("Rapor kopyalanamadı.");
    });
}
