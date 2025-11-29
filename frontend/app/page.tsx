"use client";
import React, { useState, useEffect, useMemo, FormEvent } from "react";
import Link from "next/link";

// Utility function for Turkish date formatting
const formatTimestamp = (timestamp: string): string => {
  try {
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return "";

    return date.toLocaleString("tr-TR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch (error) {
    return "";
  }
};
import {
  listSessions,
  createSession,
  uploadDocument,
  ragQuery,
  generateSuggestions,
  listMarkdownFiles,
  addMarkdownDocumentsToSession,
  convertPdfToMarkdown,
  getMarkdownFileContent,
  listAvailableModels,
  listAvailableEmbeddingModels,
  deleteSession,
  updateSessionStatus,
  updateSessionName,
  listMarkdownCategories,
  createMarkdownCategory,
  deleteMarkdownCategory,
  assignMarkdownCategory,
  listMarkdownFilesWithCategories,
  SessionMeta,
  RAGSource,
  getRecentInteractions,
  getSession,
  saveSessionRagSettings,
  clearSessionInteractions,
  exportSession,
  importSessionFromFile,
  deleteMarkdownFile,
  deleteAllMarkdownFiles,
  getProfile,
  updateProfile,
  changePassword,
  UserProfile,
  generateCourseQuestions,
  submitFeedback,
  FeedbackCreate,
  getSessionInteractions,
  getTotalInteractions,
  getApiUrl,
  CorrectionDetails,
  MarkdownFileWithCategory,
  MarkdownCategory,
} from "@/lib/api";
import { Module } from "@/types/modules";
import FeedbackModal, { FeedbackData } from "@/components/FeedbackModal";
import { useStudentChat } from "@/hooks/useStudentChat";
import { useEducationAssistant } from "@/hooks/useEducationAssistant";
import Modal from "@/components/Modal";
import MarkdownViewer from "@/components/MarkdownViewer";
import ChangelogCard from "@/components/ChangelogCard";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import DocumentUploadModal from "@/components/DocumentUploadModal";
import EnhancedDocumentUploadModal from "@/components/EnhancedDocumentUploadModal";
import LogoutButton from "@/components/LogoutButton";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import TeacherLayout from "@/app/components/TeacherLayout";
import RecommendationPanel from "@/components/RecommendationPanel";
import TopicProgressCard from "@/components/TopicProgressCard";
import TopicAnalyticsDashboard from "@/components/TopicAnalyticsDashboard";
import ModuleExtractionPanel from "@/components/ModuleExtractionPanel";
import ModuleManagementDashboard from "@/components/ModuleManagementDashboard";
import ModuleProgressMonitoringDashboard from "@/components/ModuleProgressMonitoringDashboard";
import EmojiFeedback from "@/components/EmojiFeedback";
import EBARSStatusPanel from "@/components/EBARSStatusPanel";

// Type definition for RAG source (extend if RAGSource is not exported from api.ts)
interface ExtendedRAGSource {
  content?: string;
  text?: string;
  metadata?: any;
  score: number;
  crag_score?: number; // DYSK (Doğrulayıcı Yeniden Sıralama Katmanı) relevance score
}

// Dashboard Statistics Card Component
function StatsCard({
  title,
  value,
  description,
  icon,
  color = "primary",
}: {
  title: string;
  value: string | number;
  description?: string;
  icon: React.ReactNode;
  color?: string;
}) {
  const gradientClasses = {
    primary: "from-blue-500 to-blue-600",
    green: "from-green-500 to-green-600",
    blue: "from-sky-500 to-sky-600",
    purple: "from-purple-500 to-purple-600",
  };

  return (
    <div
      className={`rounded-xl p-6 shadow-lg text-white bg-gradient-to-br ${
        gradientClasses[color as keyof typeof gradientClasses]
      } transition-all duration-300 transform hover:scale-105 hover:shadow-2xl`}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-sm font-medium opacity-80">{title}</p>
          <p className="text-3xl font-bold">{value}</p>
          {description && <p className="text-xs opacity-70">{description}</p>}
        </div>
        <div className="p-3 rounded-lg bg-white/20">{icon}</div>
      </div>
    </div>
  );
}

// Enhanced Session Card Component
function SessionCard({
  session,
  onNavigate,
  onDelete,
  onToggleStatus,
  onOpenRecent,
  onUpdateName,
  index = 0,
}: {
  session: SessionMeta;
  onNavigate: (id: string) => void;
  onDelete: (id: string, name: string) => void;
  onToggleStatus: (id: string, currentStatus: string) => void;
  onOpenRecent: (id: string) => void;
  onUpdateName: (id: string, newName: string) => void;
  index?: number;
}) {
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState(session.name);
  const colorPalette = [
    { bg: "from-blue-500 to-indigo-600", text: "text-white" },
    { bg: "from-green-500 to-emerald-600", text: "text-white" },
    { bg: "from-purple-500 to-violet-600", text: "text-white" },
    { bg: "from-rose-500 to-red-600", text: "text-white" },
    { bg: "from-amber-500 to-orange-600", text: "text-white" },
    { bg: "from-cyan-500 to-sky-600", text: "text-white" },
  ];
  const c = colorPalette[index % colorPalette.length];

  const handleNameEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsEditingName(true);
    setEditedName(session.name);
  };

  const handleNameSave = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (editedName.trim() && editedName.trim() !== session.name) {
      onUpdateName(session.session_id, editedName.trim());
    }
    setIsEditingName(false);
  };

  const handleNameCancel = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditedName(session.name);
    setIsEditingName(false);
  };

  return (
    <div
      className={`relative group bg-gradient-to-br ${c.bg} ${c.text} rounded-xl sm:rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 cursor-pointer p-4 sm:p-6 animate-slide-up`}
      style={{ animationDelay: `${index * 0.1}s` }}
      onClick={() => onNavigate(session.session_id)}
    >
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          {isEditingName ? (
            <div
              className="flex items-center gap-2 mb-1"
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="text"
                value={editedName}
                onChange={(e) => setEditedName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleNameSave(e as any);
                  } else if (e.key === "Escape") {
                    handleNameCancel(e as any);
                  }
                }}
                className="flex-1 px-2 py-1 text-base sm:text-lg font-bold bg-white/20 border border-white/40 rounded text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-white/50"
                placeholder="Oturum adı"
                autoFocus
                onClick={(e) => e.stopPropagation()}
              />
              <button
                onClick={handleNameSave}
                className="px-2 py-1 bg-green-500 hover:bg-green-600 text-white rounded text-sm font-semibold transition-colors"
                title="Kaydet"
              >
                ✓
              </button>
              <button
                onClick={handleNameCancel}
                className="px-2 py-1 bg-red-500 hover:bg-red-600 text-white rounded text-sm font-semibold transition-colors"
                title="İptal"
              >
                ✕
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2 mb-1">
              <h3 className="text-base sm:text-lg font-bold truncate flex-1">
                {session.name}
              </h3>
              <button
                onClick={handleNameEdit}
                className="opacity-0 group-hover:opacity-100 transition-opacity px-2 py-1 text-xs bg-white/20 hover:bg-white/30 rounded text-white font-semibold"
                title="İsmi değiştir"
              >
                ✏️
              </button>
            </div>
          )}
          <p className="text-xs sm:text-sm opacity-80 mb-3 sm:mb-4 line-clamp-2">
            {session.description}
          </p>
          <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-xs sm:text-sm opacity-90">
            <div className="flex items-center gap-1.5">
              <DocumentIcon />
              <span>{session.document_count}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <ChartIcon />
              <span>{session.total_chunks}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <QueryIcon />
              <span>{session.query_count}</span>
            </div>
          </div>

          {/* RAG Settings Info */}
          {session.rag_settings && (
            <div className="mt-3 pt-3 border-t border-white/20">
              <div className="text-xs opacity-75 mb-2 font-semibold">
                ⚙️ Oturum Ayarları:
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs opacity-80">
                {session.rag_settings.embedding_model && (
                  <div className="flex items-center gap-1">
                    <span className="opacity-70">📊</span>
                    <span className="font-mono truncate">
                      {session.rag_settings.embedding_model}
                    </span>
                  </div>
                )}
                {session.rag_settings.chunk_strategy && (
                  <div className="flex items-center gap-1">
                    <span className="opacity-70">✂️</span>
                    <span className="truncate">
                      {session.rag_settings.chunk_strategy}
                    </span>
                  </div>
                )}
                {session.rag_settings.chunk_size && (
                  <div className="flex items-center gap-1">
                    <span className="opacity-70">📏</span>
                    <span>Size: {session.rag_settings.chunk_size}</span>
                  </div>
                )}
                {session.rag_settings.top_k && (
                  <div className="flex items-center gap-1">
                    <span className="opacity-70">🔍</span>
                    <span>Top-K: {session.rag_settings.top_k}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col items-end gap-2 sm:gap-3 min-w-fit">
          {/* Status Display */}
          <div
            className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              session.status === "active"
                ? "bg-white/30 text-white"
                : "bg-black/20 text-gray-200"
            }`}
          >
            <div
              className={`w-2 h-2 rounded-full mr-2 ${
                session.status === "active" ? "bg-green-300" : "bg-gray-400"
              }`}
            ></div>
            {session.status === "active" ? "Aktif" : "Pasif"}
          </div>

          {/* Action Buttons - Mobile Optimized */}
          <div className="flex flex-wrap items-center gap-1 sm:gap-2 bg-white/20 p-1.5 rounded-lg backdrop-blur-sm">
            {/* Toggle Status Button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleStatus(session.session_id, session.status);
              }}
              className={`px-3 sm:px-4 py-2 text-sm font-semibold rounded-lg transition-all min-h-[40px] shadow-md hover:shadow-lg ${
                session.status === "active"
                  ? "bg-red-600 text-white hover:bg-red-700 active:bg-red-800"
                  : "bg-green-600 text-white hover:bg-green-700 active:bg-green-800"
              }`}
              title={session.status === "active" ? "Deaktif et" : "Aktif et"}
            >
              <span className="flex items-center gap-1.5 whitespace-nowrap">
                {session.status === "active" ? (
                  <>
                    <span className="text-base">🔴</span>
                    <span className="hidden sm:inline">Deaktif Et</span>
                  </>
                ) : (
                  <>
                    <span className="text-base">🟢</span>
                    <span className="hidden sm:inline">Aktif Et</span>
                  </>
                )}
              </span>
            </button>

            {/* Recent Interactions Button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onOpenRecent(session.session_id);
              }}
              className="px-2 sm:px-3 py-1 text-xs font-medium rounded-md bg-white text-indigo-700 hover:bg-indigo-50 border border-indigo-200 min-h-[32px] whitespace-nowrap"
              title="Bu oturumun öğrenci sorguları"
            >
              👥 <span className="hidden sm:inline">Sorgular</span>
            </button>

            {/* Export Button */}
            <button
              onClick={async (e) => {
                e.stopPropagation();
                try {
                  const blob = await exportSession(session.session_id, "zip");
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `${session.name}.zip`;
                  document.body.appendChild(a);
                  a.click();
                  URL.revokeObjectURL(url);
                  a.remove();
                } catch (err) {
                  // no-op; error banner already handled upstream if needed
                }
              }}
              className="px-2 py-1 text-xs font-medium rounded-md bg-white text-purple-700 hover:bg-purple-50 border border-purple-200 min-h-[32px] whitespace-nowrap"
              title="Dışa Aktar"
            >
              ⬇️ <span className="hidden sm:inline">Export</span>
            </button>

            {/* Delete Button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(session.session_id, session.name);
              }}
              className="p-1.5 text-white hover:bg-red-500/80 rounded-md transition-colors min-h-[32px] min-w-[32px]"
              title="Oturumu sil"
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
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>
          </div>

          {/* Navigate Indicator */}
          <div className="flex items-center text-sm text-primary opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <span>Ayarları Görüntüle</span>
            <svg
              className="w-4 h-4 ml-1"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M17 8l4 4m0 0l-4 4m4-4H3"
              />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}

// Icons
const DocumentIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
    />
  </svg>
);

const SessionIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
    />
  </svg>
);

const QueryIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
    />
  </svg>
);

const ChartIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 00-2-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
    />
  </svg>
);

const UploadIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
    />
  </svg>
);

const MarkdownIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
    />
  </svg>
);

export default function HomePage() {
  const router = useRouter();
  const { user } = useAuth();
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionPage, setSessionPage] = useState(1);
  const SESSIONS_PER_PAGE = 5;
  type TabType = "dashboard" | "sessions" | "upload" | "analytics" | "modules" | "assistant" | "query";
  const [activeTab, setActiveTab] = useState<TabType>("dashboard");
  
  // Check URL params for tab on mount and when URL changes
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const tabParam = params.get("tab");
    if (tabParam === "query") {
      setActiveTab("query");
    }
  }, []);

  // Listen for URL changes
  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search);
      const tabParam = params.get("tab");
      if (tabParam === "query") {
        setActiveTab("query");
      }
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Module state
  const [modules, setModules] = useState<Module[]>([]);
  const [modulesLoading, setModulesLoading] = useState(false);

  // Form states
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [category, setCategory] = useState("research");
  const [file, setFile] = useState<File | null>(null);

  // Basic query states - kept for compatibility (not causing errors)
  const [query, setQuery] = useState("");
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(true);

  // Model selection states - minimal for compatibility
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>("groq");
  const [selectedQueryModel, setSelectedQueryModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelProviders, setModelProviders] = useState<Record<string, any>>({});
  const [useDirectLLM, setUseDirectLLM] = useState<boolean>(false);
  const [chainType, setChainType] = useState<"stuff" | "refine" | "map_reduce">(
    "stuff"
  );
  const [answerLength, setAnswerLength] = useState<
    "short" | "normal" | "detailed"
  >("normal");
  const [sessionRagSettings, setSessionRagSettings] = useState<any>(null);

  // Embedding model selection - minimal for compatibility
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] =
    useState<string>("nomic-embed-text");
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<any>(
    { ollama: [], huggingface: [] }
  );
  const [embeddingModelsLoading, setEmbeddingModelsLoading] = useState(false);

  // Modal states - minimal for compatibility
  const [sourceModalOpen, setSourceModalOpen] = useState(false);
  const [selectedSource, setSelectedSource] = useState<RAGSource | null>(null);
  const [uploadStats, setUploadStats] = useState<any>(null);

  // Missing state variables for feedback system
  const [selectedInteractionForFeedback, setSelectedInteractionForFeedback] =
    useState<{
      interactionId: string;
      query: string;
      answer: string;
    } | null>(null);
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false);

  // Missing embedding provider state
  const [selectedEmbeddingProvider, setSelectedEmbeddingProvider] =
    useState<string>("ollama");

  // Missing reranker states
  const [useRerankerService, setUseRerankerService] = useState<boolean>(false);
  const [selectedRerankerType, setSelectedRerankerType] =
    useState<string>("bge-reranker-v2-m3");

  // Markdown files states
  const [markdownFiles, setMarkdownFiles] = useState<
    MarkdownFileWithCategory[]
  >([]);
  const [filteredMarkdownFiles, setFilteredMarkdownFiles] = useState<
    MarkdownFileWithCategory[]
  >([]);
  const [selectedMarkdownFiles, setSelectedMarkdownFiles] = useState<string[]>(
    []
  );
  const [markdownLoading, setMarkdownLoading] = useState(false);

  // Category states
  const [categories, setCategories] = useState<MarkdownCategory[]>([]);
  const [selectedCategoryId, setSelectedCategoryId] = useState<number | null>(
    null
  );
  const [isCategoryModalOpen, setIsCategoryModalOpen] = useState(false);
  const [newCategoryName, setNewCategoryName] = useState("");
  const [isCategoryLoading, setIsCategoryLoading] = useState(false);
  const [addingDocuments, setAddingDocuments] = useState(false);
  const [markdownPage, setMarkdownPage] = useState(1);
  const MARKDOWN_FILES_PER_PAGE = 20;

  // PDF to Markdown conversion states
  const [isDocumentUploadModalOpen, setIsDocumentUploadModalOpen] =
    useState(false);
  const [success, setSuccess] = useState<string | null>(null);

  // Modal viewer states
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedFileContent, setSelectedFileContent] = useState<string>("");
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [isCreateSessionModalOpen, setIsCreateSessionModalOpen] =
    useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string>("");
  const [modalError, setModalError] = useState<string | null>(null);

  // Normalize role name (lowercase) for consistent comparison
  const userRole = user?.role_name?.toLowerCase() || "teacher";

  // APRAG enabled state
  const [apragEnabled, setApragEnabled] = useState<boolean>(false);

  // Redirect students to their dedicated panel
  useEffect(() => {
    if (user && userRole === "student") {
      router.push("/student");
    }
  }, [user, userRole, router]);

  // Check APRAG status on mount
  useEffect(() => {
    const checkApragStatus = async () => {
      try {
        const response = await fetch(`${getApiUrl()}/aprag/settings/status`);
        if (response.ok) {
          const data = await response.json();
          setApragEnabled(data.global_enabled || false);
        }
      } catch (err) {
        console.error("Failed to check APRAG status:", err);
        setApragEnabled(false);
      }
    };
    checkApragStatus();
    // Check every 30 seconds
    const interval = setInterval(checkApragStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Student-specific chat management with database persistence
  const isStudent =
    userRole === "student" ||
    user?.role_name === "student" ||
    user?.role_name === "Student";

  // Only use the hook effectively for students; pass empty sessionId otherwise to disable network calls
  const studentChatResult = useStudentChat({
    sessionId: isStudent && selectedSessionId ? selectedSessionId : "",
    autoSave: isStudent,
    maxMessages: 50,
  });

  // Extract values only for students, otherwise use defaults
  const {
    messages: studentMessages = [],
    sendMessage: sendStudentMessage = async () => {},
    isLoading: studentChatLoading = false,
    clearHistory: clearStudentHistory = async () => {},
    error: studentChatError = null,
  } = isStudent && selectedSessionId ? studentChatResult : {};

  // State for total queries (fetched from database)
  const [totalQueries, setTotalQueries] = useState<number>(0);
  const [queriesLoading, setQueriesLoading] = useState<boolean>(false);

  // Calculated metrics
  const totalSessions = sessions.length;
  const totalDocuments = sessions.reduce(
    (acc, s) => acc + (s.document_count || 0),
    0
  );
  const totalChunks = sessions.reduce(
    (acc, s) => acc + (s.total_chunks || 0),
    0
  );

  // Fetch total queries from database when sessions change
  useEffect(() => {
    const fetchTotalQueries = async () => {
      if (sessions.length === 0) {
        setTotalQueries(0);
        return;
      }

      setQueriesLoading(true);
      try {
        const sessionIds = sessions.map((s) => s.session_id);
        const result = await getTotalInteractions(sessionIds);
        setTotalQueries(result.total);
      } catch (error) {
        console.error("Failed to fetch total queries:", error);
        // Fallback to metadata count if API fails
        const fallbackCount = sessions.reduce(
          (acc, s) => acc + (s.query_count || 0),
          0
        );
        setTotalQueries(fallbackCount);
      } finally {
        setQueriesLoading(false);
      }
    };

    fetchTotalQueries();
  }, [sessions]);

  const [isRecentModalOpen, setIsRecentModalOpen] = useState(false);
  const [recentInteractions, setRecentInteractions] = useState<any[]>([]);
  const [recentLoading, setRecentLoading] = useState(false);
  const [recentPage, setRecentPage] = useState(1);
  const [recentTotal, setRecentTotal] = useState(0);
  const [recentLimit, setRecentLimit] = useState(20);
  const [recentSearch, setRecentSearch] = useState("");

  // Profile states
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [profileFormData, setProfileFormData] = useState({
    username: "",
    email: "",
    first_name: "",
    last_name: "",
  });
  const [passwordFormData, setPasswordFormData] = useState({
    old_password: "",
    new_password: "",
    confirm_password: "",
  });
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileSuccess, setProfileSuccess] = useState<string | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);

  // Student panel state management - Three-stage experience
  const [studentView, setStudentView] = useState<
    "course-selection" | "modules" | "chat"
  >("course-selection");
  const [selectedCourse, setSelectedCourse] = useState<SessionMeta | null>(
    null
  );
  const [selectedModule, setSelectedModule] = useState<any>(null);
  const [courseModules, setCourseModules] = useState<any[]>([]);
  const [loadingModules, setLoadingModules] = useState(false);

  // Dynamic course questions
  const [courseQuestions, setCourseQuestions] = useState<string[]>([]);
  const [loadingCourseQuestions, setLoadingCourseQuestions] = useState(false);

  // EBARS refresh trigger
  const [ebarsRefreshTrigger, setEbarsRefreshTrigger] = useState(0);

  // Scroll to top when new message arrives (en yeni mesaj üstte)
  useEffect(() => {
    if (chatHistory.length > 0 && isChatOpen) {
      const container = document.getElementById("chat-history-container");
      if (container) {
        // En yeni mesaj üstte gösterildiği için scroll'u üste çek
        container.scrollTop = 0;
      }
    }
  }, [chatHistory.length, isChatOpen]);

  async function loadRecentInteractions(page = 1, sid?: string) {
    try {
      setRecentLoading(true);
      const data = await getRecentInteractions({
        limit: recentLimit,
        page,
        q: recentSearch,
        session_id: sid || selectedSessionId || undefined,
      });
      setRecentInteractions(data.items || []);
      setRecentTotal(data.count || 0);
      setRecentPage(data.page || page);
    } catch (e: any) {
      setError(e.message || "Son öğrenci sorguları alınamadı");
    } finally {
      setRecentLoading(false);
    }
  }

  async function openRecentInteractions(sessionId?: string) {
    setIsRecentModalOpen(true);
    await loadRecentInteractions(1, sessionId);
  }

  // Fetch modules for selected session
  async function fetchModules(sessionId: string) {
    if (!sessionId) {
      setModules([]);
      return;
    }

    try {
      setModulesLoading(true);
      const response = await fetch(
        `${getApiUrl()}/api/v1/modules/session/${sessionId}`
      );
      if (response.ok) {
        const data = await response.json();
        setModules(data.modules || []);
      } else {
        setModules([]);
      }
    } catch (error) {
      console.error("Error fetching modules:", error);
      setModules([]);
    } finally {
      setModulesLoading(false);
    }
  }

  async function refreshSessions() {
    try {
      setLoading(true);
      setError(null);
      const data = await listSessions();
      setSessions(data);

      // Auto-select session only for teachers, not students
      if (
        !selectedSessionId &&
        data.length > 0 &&
        userRole !== "student" &&
        user?.role_name !== "student" &&
        user?.role_name !== "Student"
      ) {
        const sortedSessions = [...data].sort(
          (a, b) =>
            new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
        );
        setSelectedSessionId(sortedSessions[0].session_id);
      }
    } catch (e: any) {
      setError(e.message || "Oturumlar yüklenemedi");
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateSession(e: FormEvent) {
    e.preventDefault();
    try {
      setError(null);
      const session = await createSession({
        name,
        description,
        category,
      });
      await refreshSessions();
      setSelectedSessionId(session.session_id);
      setIsCreateSessionModalOpen(false);
    } catch (e: any) {
      setError(e.message || "Oluşturma başarısız");
    }
  }

  async function handleUpload(e: FormEvent) {
    e.preventDefault();
    if (!file || !selectedSessionId) return;

    const formData = new FormData();
    formData.append("session_id", selectedSessionId);
    formData.append("strategy", "semantic");
    formData.append("chunk_size", "1500");
    formData.append("chunk_overlap", "150");
    // Embedding model comes from session settings or teacher selection
    const embeddingModel =
      sessionRagSettings?.embedding_model ||
      selectedEmbeddingModel ||
      "nomic-embed-text";
    formData.append("embedding_model", embeddingModel);
    formData.append("file", file);

    try {
      setError(null);
      const result = await uploadDocument(formData);
      setUploadStats(result.stats || result);
      await refreshSessions();
    } catch (e: any) {
      setError(e.message || "Yükleme başarısız");
    }
  }

  async function handleQuery(e: FormEvent, suggestionText?: string) {
    e.preventDefault();
    // Direkt LLM modunda ders oturumu seçimi zorunlu değil
    if (!useDirectLLM && !selectedSessionId) return;
    const queryText = suggestionText || query;
    if (!queryText.trim()) return;
    // For teachers we require model; for students we use saved config
    if (userRole !== "student" && !selectedQueryModel) {
      setError("Lütfen bir model seçin");
      return;
    }

    const userMessage = queryText;
    if (!suggestionText) setQuery(""); // Only clear query if not a suggestion
    setError(null);

    // For students, use the database-backed chat system
    const isStudent =
      userRole === "student" ||
      user?.role_name === "student" ||
      user?.role_name === "Student";

    if (isStudent && selectedSessionId && sendStudentMessage) {
      try {
        await sendStudentMessage(userMessage, sessionRagSettings);

        // For students: Generate context-aware questions based on response (handled in useStudentChat hook)
        setTimeout(() => {
          loadContextAwareQuestions([]);
        }, 1000);
      } catch (e: any) {
        setError(e.message || "Sorgu başarısız oldu");
      }
      return;
    }

    // Teacher/traditional flow
    setIsQuerying(true);

    // Build conversation history BEFORE adding new message
    const conversationHistory = chatHistory.slice(-4).flatMap((msg) => {
      const messages: Array<{ role: "user" | "assistant"; content: string }> =
        [];
      if (msg.user && msg.user.trim() && msg.user !== "...") {
        messages.push({ role: "user", content: msg.user });
      }
      if (
        msg.bot &&
        msg.bot.trim() &&
        msg.bot !== "..." &&
        !msg.bot.startsWith("Hata:")
      ) {
        messages.push({ role: "assistant", content: msg.bot });
      }
      return messages;
    });

    setChatHistory((prev) => [
      ...prev,
      {
        user: userMessage,
        bot: "...",
        timestamp: new Date().toISOString(),
      },
    ]);

    // Start timing from when we actually send the request
    const startTime = Date.now();

    try {
      // Calculate max_tokens based on selected answer length
      const maxTokensMap = {
        short: 1024,
        normal: 2048,
        detailed: 4096,
      };
      const maxTokens = maxTokensMap[answerLength];

      const payload: any = {
        session_id: selectedSessionId || "direct-llm-session",
        query: userMessage,
        top_k: 5,
        use_rerank: sessionRagSettings?.use_rerank ?? false, // Use session settings, default false
        min_score: sessionRagSettings?.min_score ?? 0.5,
        max_context_chars: 8000,
        use_direct_llm: useDirectLLM,
        max_tokens: maxTokens,
        conversation_history:
          conversationHistory.length > 0 ? conversationHistory : undefined,
      };
      // Use session RAG settings if available, otherwise use defaults
      if (userRole !== "student") {
        payload.model =
          sessionRagSettings?.model || selectedQueryModel || undefined;
        payload.chain_type =
          sessionRagSettings?.chain_type || chainType || "stuff";
      }
      // IMPORTANT: Always use session's embedding_model if available
      // This ensures chat uses the same embedding model as documents
      if (sessionRagSettings?.embedding_model) {
        payload.embedding_model = sessionRagSettings.embedding_model;
      } else if (userRole !== "student" && selectedEmbeddingModel) {
        // Fallback to selected embedding model for teachers if session doesn't have one
        payload.embedding_model = selectedEmbeddingModel;
      }
      const result = await ragQuery(payload);

      // Calculate actual elapsed time from request start to response received
      const actualDurationMs = Date.now() - startTime;

      setChatHistory((prev) => {
        const newHistory = [...prev];
        newHistory[newHistory.length - 1].bot = result.answer;
        newHistory[newHistory.length - 1].sources = result.sources || [];
        // Do not block UI with suggestions; fill later asynchronously
        newHistory[newHistory.length - 1].suggestions = [];
        // Use actual client-side measured time instead of backend processing_time_ms
        newHistory[newHistory.length - 1].durationMs = actualDurationMs;
        newHistory[newHistory.length - 1].correction = result.correction; // Save correction details
        return newHistory;
      });

      // Try to get interaction ID from recent interactions (APRAG)
      // This is a workaround until API Gateway returns interaction_id in response
      if (selectedSessionId && !useDirectLLM) {
        setTimeout(async () => {
          try {
            const interactions = await getSessionInteractions(
              selectedSessionId,
              1,
              0
            );
            if (
              interactions.interactions &&
              interactions.interactions.length > 0
            ) {
              const latestInteraction = interactions.interactions[0];
              // Match by query and timestamp (within 5 seconds)
              const timeDiff = Math.abs(
                new Date(latestInteraction.timestamp).getTime() - Date.now()
              );
              if (latestInteraction.query === userMessage && timeDiff < 5000) {
                setChatHistory((prev) => {
                  const updated = [...prev];
                  if (updated.length > 0) {
                    updated[updated.length - 1].interactionId =
                      latestInteraction.interaction_id;
                  }
                  return updated;
                });
              }
            }
          } catch (e) {
            // Silently ignore - interaction ID is optional
            console.debug("Could not fetch interaction ID:", e);
          }
        }, 1000); // Wait 1 second for APRAG to log the interaction
      }

      // Fetch suggestions asynchronously (non-blocking)
      (async () => {
        try {
          const sugg = await generateSuggestions({
            question: userMessage,
            answer: result.answer,
            sources: result.sources || [],
          });
          if (Array.isArray(sugg) && sugg.length > 0) {
            setChatHistory((prev) => {
              const updated = [...prev];
              // Update last entry suggestions
              updated[updated.length - 1].suggestions = sugg;
              return updated;
            });
          }
        } catch (_err) {
          // Silently ignore suggestion errors
        }
      })();
    } catch (e: any) {
      const errorMessage = e.message || "Sorgu başarısız oldu";
      setError(errorMessage);
      setChatHistory((prev) => {
        const newHistory = [...prev];
        newHistory[newHistory.length - 1].bot = `Hata: ${errorMessage}`;
        return newHistory;
      });
    } finally {
      setIsQuerying(false);
    }
  }

  async function handleSuggestionClick(text: string) {
    // Send the clicked suggestion as a new query with conversation history
    const fakeEvent = { preventDefault: () => {} } as unknown as FormEvent;
    await handleQuery(fakeEvent, text); // Pass suggestion text directly
  }

  // Load dynamic course questions asynchronously
  async function loadCourseQuestions(sessionId: string) {
    try {
      setLoadingCourseQuestions(true);
      console.log(
        "[Course Questions] Loading initial dynamic questions for session:",
        sessionId
      );
      const questions = await generateCourseQuestions(sessionId, 5);
      console.log("[Course Questions] Generated questions:", questions);
      setCourseQuestions(questions || []);
    } catch (error: any) {
      console.error(
        "[Course Questions] Error loading course questions:",
        error
      );
      // Fallback to generic questions if API fails
      setCourseQuestions([
        "Bu dersin temel konuları neler?",
        "Kısa bir özet hazırla",
        "Test soruları oluştur",
        "Önemli kavramları listele",
      ]);
    } finally {
      setLoadingCourseQuestions(false);
    }
  }

  // Load context-aware questions based on response chunks
  async function loadContextAwareQuestions(responseChunks: any[]) {
    if (!selectedSessionId || !responseChunks || responseChunks.length === 0) {
      return;
    }

    try {
      console.log(
        "[Context Questions] Generating questions from response chunks:",
        responseChunks.length
      );
      const questions = await generateCourseQuestions(selectedSessionId, 3);
      console.log("[Context Questions] Generated new questions:", questions);

      // Add new questions to existing ones (avoid duplicates)
      setCourseQuestions((prev) => {
        const newQuestions = questions.filter((q) => !prev.includes(q));
        return [...prev, ...newQuestions].slice(0, 6); // Keep max 6 questions
      });
    } catch (error: any) {
      console.error(
        "[Context Questions] Error generating context-aware questions:",
        error
      );
    }
  }

  async function fetchAvailableEmbeddingModels() {
    try {
      setEmbeddingModelsLoading(true);
      console.log("[Embedding] Fetching embedding models...");
      const data = await listAvailableEmbeddingModels();
      console.log("[Embedding] Received data:", data);
      console.log("[Embedding] Ollama models:", data.ollama?.length || 0);
      console.log(
        "[Embedding] HuggingFace models:",
        data.huggingface?.length || 0
      );
      setAvailableEmbeddingModels(data);
      // Set default if not already set
      if (!selectedEmbeddingModel && data.ollama.length > 0) {
        setSelectedEmbeddingModel(data.ollama[0]);
      }
    } catch (e: any) {
      console.error("[Embedding] Error fetching embedding models:", e);
      setError(e.message || "Embedding modelleri yüklenemedi");
    } finally {
      setEmbeddingModelsLoading(false);
    }
  }

  async function fetchAvailableModels() {
    try {
      setModelsLoading(true);
      setError(null);
      const data = await listAvailableModels();

      // Debug: Log received models
      console.log("Fetched models data:", data);
      console.log("Available models:", data.models);
      console.log("Providers:", data.providers);

      setAvailableModels(data.models || []);
      setModelProviders(data.providers || {});

      // Set default model based on selected provider
      if (data.models && data.models.length > 0) {
        const providerModels = data.models.filter((m: any) => {
          if (typeof m === "string") return false;
          const modelProvider = (m.provider || "").toLowerCase();
          const selectedProviderLower = selectedProvider.toLowerCase();
          return (
            modelProvider === selectedProviderLower ||
            (selectedProviderLower === "huggingface" &&
              (modelProvider === "hf" || modelProvider === "huggingface"))
          );
        });

        console.log(`Models for provider ${selectedProvider}:`, providerModels);

        if (providerModels.length > 0) {
          const firstModel = providerModels[0];
          setSelectedQueryModel(
            typeof firstModel === "string" ? firstModel : firstModel.id
          );
        } else {
          // Fallback: try to find any model from selected provider
          const allProviderModels = data.models.filter((m: any) => {
            if (typeof m === "string") return false;
            return (
              (m.provider || "").toLowerCase() ===
              selectedProvider.toLowerCase()
            );
          });

          if (allProviderModels.length > 0) {
            setSelectedQueryModel(allProviderModels[0].id);
          } else {
            // Last fallback: use first available model
            const firstModel = data.models[0];
            setSelectedQueryModel(
              typeof firstModel === "string" ? firstModel : firstModel.id
            );
          }
        }
      }
    } catch (e: any) {
      console.error("Error fetching models:", e);
      setError(e.message || "Modeller yüklenemedi");
    } finally {
      setModelsLoading(false);
    }
  }

  // Filter models by selected provider - useMemo for performance
  const filteredModels = useMemo(() => {
    const filtered = availableModels.filter((model: any) => {
      // Skip string models (should not happen but handle gracefully)
      if (typeof model === "string") {
        return false;
      }

      // Get model provider and normalize
      const modelProvider = (model.provider || "").toLowerCase().trim();
      const selectedProviderLower = selectedProvider.toLowerCase().trim();

      // Direct match
      if (modelProvider === selectedProviderLower) {
        return true;
      }

      // Handle provider aliases
      if (
        selectedProviderLower === "huggingface" &&
        (modelProvider === "hf" || modelProvider === "huggingface")
      ) {
        return true;
      }
      if (selectedProviderLower === "hf" && modelProvider === "huggingface") {
        return true;
      }

      return false;
    });

    // Debug logging (only log when provider or models change)
    console.log(
      `[Filter] Provider: "${selectedProvider}", Found ${filtered.length} models`
    );

    return filtered;
  }, [availableModels, selectedProvider]);

  // Handle provider change
  const handleProviderChange = (provider: string) => {
    setSelectedProvider(provider);

    // Filter models for the new provider
    const providerModels = availableModels.filter((model: any) => {
      if (typeof model === "string") {
        return false;
      }
      const modelProvider = (model.provider || "").toLowerCase().trim();
      const providerLower = provider.toLowerCase().trim();

      // Direct match
      if (modelProvider === providerLower) {
        return true;
      }

      // Handle aliases
      if (
        providerLower === "huggingface" &&
        (modelProvider === "hf" || modelProvider === "huggingface")
      ) {
        return true;
      }
      if (providerLower === "hf" && modelProvider === "huggingface") {
        return true;
      }

      return false;
    });

    // Auto-select first model from new provider
    if (providerModels.length > 0) {
      const firstModel = providerModels[0];
      const modelId =
        typeof firstModel === "string"
          ? firstModel
          : firstModel.id || firstModel.name;
      setSelectedQueryModel(modelId);
    } else {
      setSelectedQueryModel("");
    }
  };

  // Fetch markdown files with categories
  async function fetchMarkdownFiles() {
    try {
      setMarkdownLoading(true);
      setError(null);

      // Fetch both files with categories and categories in parallel
      const [files, cats] = await Promise.all([
        listMarkdownFilesWithCategories(selectedCategoryId || undefined),
        listMarkdownCategories(),
      ]);

      setMarkdownFiles(files);
      setFilteredMarkdownFiles(files);
      setCategories(cats);
    } catch (e: any) {
      setError(e.message || "Markdown dosyaları yüklenemedi");
    } finally {
      setMarkdownLoading(false);
    }
  }

  // Handle category filter change
  const handleCategoryFilterChange = (categoryId: number | null) => {
    setSelectedCategoryId(categoryId);
    setMarkdownPage(1); // Reset to first page when changing filter
  };

  // Create a new category
  const handleCreateCategory = async () => {
    if (!newCategoryName.trim()) return;

    try {
      setIsCategoryLoading(true);
      const category = await createMarkdownCategory({
        name: newCategoryName.trim(),
        description: null,
      });

      setCategories([...categories, category]);
      setNewCategoryName("");
      setSuccess(`"${category.name}" kategorisi oluşturuldu`);
    } catch (e: any) {
      setError(e.message || "Kategori oluşturulurken hata oluştu");
    } finally {
      setIsCategoryLoading(false);
    }
  };

  // Delete a category
  const handleDeleteCategory = async (id: number) => {
    if (
      !confirm(
        "Bu kategoriyi silmek istediğinize emin misiniz? Bu işlem geri alınamaz."
      )
    ) {
      return;
    }

    try {
      setIsCategoryLoading(true);
      await deleteMarkdownCategory(id);
      setCategories(categories.filter((cat) => cat.id !== id));
      setSuccess("Kategori silindi");

      // Reset category filter if the deleted category was selected
      if (selectedCategoryId === id) {
        setSelectedCategoryId(null);
      }
    } catch (e: any) {
      setError(e.message || "Kategori silinirken hata oluştu");
    } finally {
      setIsCategoryLoading(false);
    }
  };

  // Assign selected files to a category
  const handleAssignCategory = async (categoryId: number | null) => {
    if (selectedMarkdownFiles.length === 0) return;

    try {
      setMarkdownLoading(true);
      await assignMarkdownCategory(selectedMarkdownFiles, categoryId);

      // Update local state
      const updatedFiles = markdownFiles.map((file) =>
        selectedMarkdownFiles.includes(file.filename)
          ? {
              ...file,
              category_id: categoryId,
              category_name: categoryId
                ? categories.find((c) => c.id === categoryId)?.name || null
                : null,
            }
          : file
      );

      setMarkdownFiles(updatedFiles);
      setFilteredMarkdownFiles(
        updatedFiles.filter(
          (file) =>
            !selectedCategoryId || file.category_id === selectedCategoryId
        )
      );

      setSuccess(
        `Seçili dosyalar ${
          categoryId ? "kategoriye eklendi" : "kategorisiz olarak işaretlendi"
        }`
      );
      setSelectedMarkdownFiles([]);
    } catch (e: any) {
      setError(e.message || "Kategori atanırken hata oluştu");
    } finally {
      setMarkdownLoading(false);
    }
  };

  async function handleAddMarkdownDocuments() {
    if (!selectedSessionId || selectedMarkdownFiles.length === 0) return;

    try {
      setAddingDocuments(true);
      setError(null);
      const result = await addMarkdownDocumentsToSession(
        selectedSessionId,
        selectedMarkdownFiles,
        sessionRagSettings?.embedding_model ||
          selectedEmbeddingModel ||
          "nomic-embed-text"
      );

      setUploadStats({
        processed_count: result.processed_count,
        chunks_created: result.total_chunks_added,
        message: result.message,
      });

      setSelectedMarkdownFiles([]);
      await refreshSessions();
    } catch (e: any) {
      setError(e.message || "Dokümanlar eklenemedi");
    } finally {
      setAddingDocuments(false);
    }
  }

  function handleMarkdownFileToggle(filename: string) {
    setSelectedMarkdownFiles((prev) =>
      prev.includes(filename)
        ? prev.filter((f) => f !== filename)
        : [...prev, filename]
    );
  }

  const handleDocumentUploadSuccess = (message: string) => {
    setSuccess(message);
  };

  const handleDocumentUploadError = (error: string) => {
    setError(error);
  };

  const handleViewMarkdownFile = async (filename: string) => {
    try {
      setIsLoadingContent(true);
      setIsModalOpen(true);
      setSelectedFileName(filename);
      setModalError(null);
      setSelectedFileContent("");

      const result = await getMarkdownFileContent(filename);
      setSelectedFileContent(result.content);
    } catch (e: any) {
      setModalError(e.message || "Dosya içeriği yüklenemedi");
    } finally {
      setIsLoadingContent(false);
    }
  };

  const handleDownloadMarkdownFile = async (filename: string) => {
    try {
      const result = await getMarkdownFileContent(filename);

      // Create a blob with the markdown content
      const blob = new Blob([result.content], { type: "text/markdown" });

      // Create a download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;

      // Trigger download
      document.body.appendChild(link);
      link.click();

      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message || "Dosya indirilemedi");
    }
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedFileContent("");
    setSelectedFileName("");
    setModalError(null);
    setIsLoadingContent(false);
  };

  const handleNavigateToSession = (sessionId: string) => {
    router.push(`/sessions/${sessionId}`);
  };

  const handleDeleteSession = async (
    sessionId: string,
    sessionName: string
  ) => {
    if (
      !confirm(
        `"${sessionName}" oturumunu silmek istediğinizden emin misiniz? Bu işlem geri alınamaz.`
      )
    ) {
      return;
    }

    try {
      setError(null);
      await deleteSession(sessionId);
      await refreshSessions();
      // If deleted session was selected, clear selection
      if (selectedSessionId === sessionId) {
        setSelectedSessionId("");
      }
    } catch (e: any) {
      setError(e.message || "Oturum silinirken hata oluştu");
    }
  };

  const handleToggleSessionStatus = async (
    sessionId: string,
    currentStatus: string
  ) => {
    const newStatus = currentStatus === "active" ? "inactive" : "active";

    try {
      setError(null);
      await updateSessionStatus(sessionId, newStatus);
      await refreshSessions();
    } catch (e: any) {
      setError(e.message || "Oturum durumu güncellenirken hata oluştu");
    }
  };

  const handleUpdateSessionName = async (
    sessionId: string,
    newName: string
  ) => {
    console.log(
      "🔧 [SESSION NAME UPDATE] Starting session name update process"
    );
    console.log("🔧 [SESSION NAME UPDATE] Session ID:", sessionId);
    console.log("🔧 [SESSION NAME UPDATE] New Name:", newName);
    console.log(
      "🔧 [SESSION NAME UPDATE] New Name (trimmed):",
      newName?.trim()
    );

    if (!newName || !newName.trim()) {
      console.error("❌ [SESSION NAME UPDATE] Validation failed: Empty name");
      setError("Oturum adı boş olamaz");
      return;
    }

    try {
      console.log("🔧 [SESSION NAME UPDATE] Starting API call...");
      setError(null);

      // Call the API function
      console.log("🔧 [SESSION NAME UPDATE] Calling updateSessionName API...");
      const apiResult = await updateSessionName(sessionId, newName.trim());
      console.log("✅ [SESSION NAME UPDATE] API call successful:", apiResult);

      // Refresh sessions list
      console.log("🔧 [SESSION NAME UPDATE] Refreshing sessions list...");
      await refreshSessions();
      console.log("✅ [SESSION NAME UPDATE] Sessions refreshed successfully");

      // Show success message
      console.log("🔧 [SESSION NAME UPDATE] Setting success message...");
      setSuccess("Oturum adı başarıyla güncellendi");
      setTimeout(() => setSuccess(null), 3000);
      console.log("✅ [SESSION NAME UPDATE] Process completed successfully!");
    } catch (e: any) {
      console.error("❌ [SESSION NAME UPDATE] Error occurred:", e);
      console.error("❌ [SESSION NAME UPDATE] Error message:", e.message);
      console.error("❌ [SESSION NAME UPDATE] Error stack:", e.stack);
      console.error(
        "❌ [SESSION NAME UPDATE] Full error object:",
        JSON.stringify(e, null, 2)
      );

      setError(e.message || "Oturum adı güncellenirken hata oluştu");
    }
  };

  const handleLogout = () => {
    router.push("/login");
  };

  // Source modal handlers
  const handleOpenSourceModal = (source: RAGSource) => {
    setSelectedSource(source);
    setSourceModalOpen(true);
  };

  const handleCloseSourceModal = () => {
    setSelectedSource(null);
    setSourceModalOpen(false);
  };

  useEffect(() => {
    refreshSessions();
  }, []);

  // Load session rag_settings when selectedSessionId changes
  useEffect(() => {
    const loadSessionRagSettings = async () => {
      if (!selectedSessionId) {
        setSessionRagSettings(null);
        return;
      }
      try {
        const session = await getSession(selectedSessionId);
        if (session?.rag_settings) {
          setSessionRagSettings(session.rag_settings);

          // Load embedding model from session settings
          if (session.rag_settings.embedding_model) {
            setSelectedEmbeddingModel(session.rag_settings.embedding_model);

            // Determine provider based on model name
            const modelName =
              session.rag_settings.embedding_model.toLowerCase();
            if (
              modelName.includes("huggingface") ||
              modelName.includes("sentence-transformers") ||
              modelName.includes("intfloat") ||
              modelName.includes("multilingual")
            ) {
              setSelectedEmbeddingProvider("huggingface");
            } else {
              setSelectedEmbeddingProvider("ollama");
            }
          }

          // Load other settings
          if (session.rag_settings.model) {
            setSelectedQueryModel(session.rag_settings.model);
          }
          if (session.rag_settings.chain_type) {
            setChainType(
              session.rag_settings.chain_type as
                | "stuff"
                | "refine"
                | "map_reduce"
            );
          }
          if (session.rag_settings.use_direct_llm !== undefined) {
            setUseDirectLLM(session.rag_settings.use_direct_llm);
          }

          // Load reranker settings
          if (session.rag_settings.use_reranker_service !== undefined) {
            setUseRerankerService(session.rag_settings.use_reranker_service);
          }
          if (session.rag_settings.reranker_type) {
            setSelectedRerankerType(session.rag_settings.reranker_type);
          }
        } else {
          setSessionRagSettings(null);
        }
      } catch (e: any) {
        console.error("Failed to load session rag settings:", e);
        setSessionRagSettings(null);
      }
    };
    loadSessionRagSettings();
  }, [selectedSessionId]);

  // Fetch categories - Robust version that doesn't break UI
  const fetchCategories = async () => {
    try {
      console.log("[Categories] Fetching categories...");
      const cats = await listMarkdownCategories();
      console.log("[Categories] Fetched categories:", cats);
      setCategories(cats || []);
    } catch (e: any) {
      console.error("[Categories] Error fetching categories:", e);
      // Don't set error state as it might break UI - just log the error
      // Initialize with empty array to ensure UI renders
      setCategories([]);
    }
  };

  useEffect(() => {
    if (activeTab === "upload") {
      fetchMarkdownFiles();
      fetchCategories();
    }
  }, [activeTab]);

  // Auto-load models for student panel
  useEffect(() => {
    if (
      (userRole === "student" ||
        user?.role_name === "student" ||
        user?.role_name === "Student") &&
      availableModels.length === 0 &&
      !modelsLoading
    ) {
      fetchAvailableModels();
    }
  }, [userRole, availableModels.length, modelsLoading]);

  // Periodic refresh for teacher view (update query counts)
  useEffect(() => {
    const isTeacher =
      userRole !== "student" &&
      user?.role_name !== "student" &&
      user?.role_name !== "Student";
    if (!isTeacher) return;
    const id = setInterval(() => {
      refreshSessions();
    }, 15000); // 15s
    return () => clearInterval(id);
  }, [userRole]);

  // Update browser tab title for student view
  useEffect(() => {
    if (typeof document === "undefined") return;
    const isStudent =
      userRole === "student" ||
      user?.role_name === "student" ||
      user?.role_name === "Student";
    if (!isStudent) return;
    const courseName = selectedCourse?.name?.trim();
    if (studentView === "chat" && courseName) {
      document.title = `${courseName} | Online Kütüphane`;
    } else {
      document.title = "Online Kütüphane";
    }
  }, [userRole, user?.role_name, studentView, selectedCourse?.name]);

  // Student view - Two-stage experience: Course Selection -> Chat Interface
  // Check both lowercase and original format for compatibility
  if (
    userRole === "student" ||
    user?.role_name === "student" ||
    user?.role_name === "Student"
  ) {
    console.log(
      "[DEBUG] Student panel - User role detected:",
      userRole,
      user?.role_name
    );
    console.log("[DEBUG] Student panel - Available sessions:", sessions.length);
    console.log("[DEBUG] Student panel - Current view:", studentView);
    console.log(
      "[DEBUG] Student panel - Selected course:",
      selectedCourse?.name
    );

    // Helper function to determine subject category and styling
    const getSubjectStyling = (session: SessionMeta) => {
      const name = session.name.toLowerCase();
      const category = session.category?.toLowerCase() || "";

      if (name.includes("biyoloji") || category.includes("biology")) {
        return {
          gradient: "from-green-400 via-emerald-500 to-teal-600",
          icon: "🧬",
          bgPattern:
            "bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50",
          accent: "border-green-200",
        };
      } else if (name.includes("kimya") || category.includes("chemistry")) {
        return {
          gradient: "from-purple-400 via-violet-500 to-indigo-600",
          icon: "⚗️",
          bgPattern:
            "bg-gradient-to-br from-purple-50 via-violet-50 to-indigo-50",
          accent: "border-purple-200",
        };
      } else if (name.includes("fizik") || category.includes("physics")) {
        return {
          gradient: "from-blue-400 via-sky-500 to-cyan-600",
          icon: "🧲",
          bgPattern: "bg-gradient-to-br from-blue-50 via-sky-50 to-cyan-50",
          accent: "border-blue-200",
        };
      } else if (
        name.includes("matematik") ||
        category.includes("mathematics")
      ) {
        return {
          gradient: "from-orange-400 via-amber-500 to-yellow-600",
          icon: "➗",
          bgPattern:
            "bg-gradient-to-br from-orange-50 via-amber-50 to-yellow-50",
          accent: "border-orange-200",
        };
      } else if (name.includes("tarih") || category.includes("history")) {
        return {
          gradient: "from-amber-400 via-orange-500 to-red-600",
          icon: "🏛️",
          bgPattern: "bg-gradient-to-br from-amber-50 via-orange-50 to-red-50",
          accent: "border-amber-200",
        };
      } else if (name.includes("coğrafya") || category.includes("geography")) {
        return {
          gradient: "from-teal-400 via-green-500 to-emerald-600",
          icon: "🗺️",
          bgPattern:
            "bg-gradient-to-br from-teal-50 via-green-50 to-emerald-50",
          accent: "border-teal-200",
        };
      } else if (
        name.includes("dil") ||
        name.includes("ingilizce") ||
        category.includes("language")
      ) {
        return {
          gradient: "from-rose-400 via-pink-500 to-fuchsia-600",
          icon: "🗣️",
          bgPattern: "bg-gradient-to-br from-rose-50 via-pink-50 to-fuchsia-50",
          accent: "border-rose-200",
        };
      } else {
        return {
          gradient: "from-indigo-400 via-blue-500 to-purple-600",
          icon: "📚",
          bgPattern:
            "bg-gradient-to-br from-indigo-50 via-blue-50 to-purple-50",
          accent: "border-indigo-200",
        };
      }
    };

    // Stage 1: Course Selection
    if (studentView === "course-selection") {
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
          <div className="w-full">
            {/* Header */}
            <div className="text-center mb-12">
              <div className="text-6xl mb-4 animate-bounce">🎓</div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent mb-4">
                Online Kütüphane
              </h1>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                Öğrenmek istediğiniz dersi seçin ve akıllı eğitim asistanınızla
                sohbete başlayın
              </p>
            </div>

            {/* Available Courses Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4 sm:gap-6 mb-8">
              {sessions
                .filter((s) => s.status === "active")
                .map((session, index) => {
                  const styling = getSubjectStyling(session);
                  return (
                    <div
                      key={session.session_id}
                      className={`group relative overflow-hidden rounded-2xl ${styling.bgPattern} ${styling.accent} border-2 cursor-pointer transform transition-all duration-300 hover:scale-105 hover:shadow-2xl animate-fade-in`}
                      style={{ animationDelay: `${index * 0.1}s` }}
                      onClick={() => {
                        setSelectedCourse(session);
                        setSelectedSessionId(session.session_id);
                        // Load session settings
                        getSession(session.session_id)
                          .then((s) => {
                            setSessionRagSettings(s.rag_settings || null);
                          })
                          .catch(() => {});
                        // Load dynamic course questions asynchronously (non-blocking)
                        setStudentView("chat");
                        // Load questions in background after navigation
                        setTimeout(() => {
                          loadCourseQuestions(session.session_id);
                        }, 100);
                      }}
                    >
                      {/* Background Pattern */}
                      <div className="absolute inset-0 opacity-10">
                        <div className="absolute inset-0 bg-gradient-to-br from-transparent via-white to-transparent transform rotate-45 translate-x-full group-hover:translate-x-[-100%] transition-transform duration-1000"></div>
                      </div>

                      {/* Course Card Content */}
                      <div className="relative p-6 h-full flex flex-col">
                        {/* Course Icon & Badge */}
                        <div className="flex items-center justify-between mb-4">
                          <div
                            className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${styling.gradient} flex items-center justify-center text-white text-2xl shadow-lg group-hover:rotate-12 transition-transform duration-300`}
                          >
                            {styling.icon}
                          </div>
                          <div className="flex flex-col items-end gap-2">
                            <span className="px-3 py-1 text-xs font-semibold bg-white/70 text-gray-700 rounded-full border backdrop-blur-sm">
                              {session.document_count} Materyal
                            </span>
                            <span className="px-3 py-1 text-xs font-semibold bg-green-100 text-green-700 rounded-full border border-green-200">
                              ✓ Aktif
                            </span>
                          </div>
                        </div>

                        {/* Course Info */}
                        <div className="flex-1">
                          <h3 className="text-xl font-bold text-gray-800 mb-3 group-hover:text-gray-900 transition-colors">
                            {session.name}
                          </h3>
                          <p className="text-gray-600 text-sm mb-4 line-clamp-3">
                            {session.description ||
                              "Bu derste kapsamlı materyaller ve etkileşimli öğrenme deneyimi sizi bekliyor."}
                          </p>

                          {/* RAG Settings Info */}
                          {session.rag_settings && (
                            <div className="mb-4 p-3 bg-white/50 rounded-lg border border-gray-200/50 backdrop-blur-sm">
                              <div className="text-xs text-gray-600 font-semibold mb-2 flex items-center gap-1">
                                <span>⚙️</span>
                                <span>Oturum Ayarları</span>
                              </div>
                              <div className="grid grid-cols-1 gap-2">
                                {session.rag_settings.embedding_model && (
                                  <div className="flex items-center justify-between text-xs">
                                    <span className="text-gray-500">
                                      Embedding:
                                    </span>
                                    <span className="font-mono text-gray-700 bg-white px-2 py-0.5 rounded">
                                      {session.rag_settings.embedding_model}
                                    </span>
                                  </div>
                                )}
                                {session.rag_settings.chunk_strategy && (
                                  <div className="flex items-center justify-between text-xs">
                                    <span className="text-gray-500">
                                      Chunk Stratejisi:
                                    </span>
                                    <span className="font-mono text-gray-700 bg-white px-2 py-0.5 rounded">
                                      {session.rag_settings.chunk_strategy}
                                    </span>
                                  </div>
                                )}
                                {session.rag_settings.chunk_size && (
                                  <div className="flex items-center justify-between text-xs">
                                    <span className="text-gray-500">
                                      Chunk Size:
                                    </span>
                                    <span className="font-mono text-gray-700 bg-white px-2 py-0.5 rounded">
                                      {session.rag_settings.chunk_size}
                                    </span>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Course Stats */}
                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 sm:gap-3 mb-4">
                            <div className="text-center p-3 bg-white/50 rounded-lg backdrop-blur-sm border border-white/50">
                              <div className="text-2xl font-bold text-gray-800">
                                {session.total_chunks}
                              </div>
                              <div className="text-xs text-gray-600">
                                İçerik Bölümü
                              </div>
                            </div>
                            <div className="text-center p-3 bg-white/50 rounded-lg backdrop-blur-sm border border-white/50">
                              <div className="text-2xl font-bold text-gray-800">
                                {session.query_count}
                              </div>
                              <div className="text-xs text-gray-600">
                                Yanıtlanan Soru
                              </div>
                            </div>
                            <div className="text-center p-3 bg-white/50 rounded-lg backdrop-blur-sm border border-white/50">
                              <div className="text-2xl font-bold text-gray-800">
                                {session.student_entry_count || 0}
                              </div>
                              <div className="text-xs text-gray-600">
                                Giren Öğrenci
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Enter Button */}
                        <div
                          className={`mt-4 p-4 rounded-xl bg-gradient-to-r ${styling.gradient} text-white shadow-lg group-hover:shadow-xl transition-all duration-300 text-center`}
                        >
                          <div className="flex items-center justify-center gap-2">
                            <span className="text-lg">🚀</span>
                            <span className="font-semibold">Derse Başla</span>
                            <svg
                              className="w-5 h-5 group-hover:translate-x-1 transition-transform"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M17 8l4 4m0 0l-4 4m4-4H3"
                              />
                            </svg>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
            </div>

            {/* Empty State */}
            {sessions.filter((s) => s.status === "active").length === 0 && (
              <div className="text-center py-16">
                <div className="text-6xl mb-4">📚</div>
                <h3 className="text-2xl font-bold text-gray-700 mb-4">
                  Henüz Aktif Ders Bulunmuyor
                </h3>
                <p className="text-gray-500 max-w-md mx-auto mb-8">
                  Öğretmeniniz henüz bir ders oturumu oluşturmamış. Lütfen daha
                  sonra tekrar kontrol edin.
                </p>
                <button
                  onClick={() => refreshSessions()}
                  className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-indigo-700 transition-all shadow-lg hover:shadow-xl"
                >
                  🔄 Dersleri Yenile
                </button>
              </div>
            )}
          </div>
        </div>
      );
    }

    // Stage 2: Clean Chat Interface
    if (studentView === "chat" && selectedCourse) {
      const styling = getSubjectStyling(selectedCourse);

      return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-100">
          <div className="w-full">
            {/* Clean Header with Back Navigation */}
            <div
              className={`${styling.bgPattern} ${styling.accent} border-2 rounded-2xl p-6 mb-6 shadow-lg`}
            >
              <div className="flex items-center justify-between mb-4">
                <button
                  onClick={() => {
                    setStudentView("course-selection");
                    setSelectedCourse(null);
                    setSelectedSessionId("");
                    // Clear student chat history if needed
                    if (clearStudentHistory) {
                      clearStudentHistory().catch(console.error);
                    }
                  }}
                  className="flex items-center gap-2 px-4 py-2 bg-white/70 hover:bg-white/90 rounded-xl transition-all border border-white/50 backdrop-blur-sm"
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
                      strokeWidth={2}
                      d="M15 19l-7-7 7-7"
                    />
                  </svg>
                  <span className="text-sm font-medium">Ders Listesi</span>
                </button>
              </div>

              <div className="flex items-center gap-4">
                <div
                  className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${styling.gradient} flex items-center justify-center text-white text-2xl shadow-lg`}
                >
                  {styling.icon}
                </div>
                <div className="flex-1">
                  <h1 className="text-2xl font-bold text-gray-800 mb-1">
                    {selectedCourse.name}
                  </h1>
                  <p className="text-gray-600 text-sm mb-2">
                    Eğitim Asistanınızla sohbet edin - Size yardımcı olmak için
                    buradayım!
                  </p>

                  {/* RAG Settings Display - Beautiful Format */}
                  {sessionRagSettings && (
                    <div className="mt-3 p-4 bg-white/80 rounded-xl backdrop-blur-sm border border-white/60 shadow-sm">
                      <div className="text-xs font-semibold text-gray-700 mb-3 flex items-center gap-2">
                        ⚙️{" "}
                        <span className="text-indigo-700">
                          Ders Yapılandırması
                        </span>
                      </div>
                      <div className="grid grid-cols-1 gap-3">
                        {/* Model Display - More spacious */}
                        <div className="flex items-center gap-2 p-2 bg-blue-50 rounded-lg">
                          <span className="w-3 h-3 bg-blue-500 rounded-full flex-shrink-0"></span>
                          <span className="text-gray-600 text-xs">Model:</span>
                          <span className="font-semibold text-blue-800 text-xs truncate">
                            {(() => {
                              const model = sessionRagSettings.model;
                              if (!model) return "AI Assistant";

                              // Clean model name display
                              let displayName = model.split("/").pop() || model;
                              displayName = displayName.replace(":latest", "");

                              // Format common model names nicely
                              if (displayName.includes("llama")) {
                                displayName = displayName
                                  .replace("llama-", "Llama-")
                                  .replace("-instant", " (Instant)")
                                  .replace("3.1", "3.1");
                              } else if (displayName.includes("mixtral")) {
                                displayName = displayName.replace(
                                  "mixtral",
                                  "Mixtral"
                                );
                              } else if (displayName.includes("gemma")) {
                                displayName = displayName.replace(
                                  "gemma",
                                  "Gemma"
                                );
                              }

                              return displayName;
                            })()}
                          </span>
                        </div>

                        {/* Chain Type */}
                        <div className="flex items-center gap-2 p-2 bg-green-50 rounded-lg">
                          <span className="w-3 h-3 bg-green-500 rounded-full flex-shrink-0"></span>
                          <span className="text-gray-600 text-xs">Zincir:</span>
                          <span className="font-semibold text-green-800 text-xs">
                            {sessionRagSettings.chain_type || "stuff"}
                          </span>
                        </div>

                        {/* Top-k */}
                        <div className="flex items-center gap-2 p-2 bg-purple-50 rounded-lg">
                          <span className="w-3 h-3 bg-purple-500 rounded-full flex-shrink-0"></span>
                          <span className="text-gray-600 text-xs">Top-k:</span>
                          <span className="font-semibold text-purple-800 text-xs">
                            {sessionRagSettings.top_k ?? 5}
                          </span>
                        </div>

                        {/* Rerank - Enhanced with boost visual */}
                        <div
                          className={`flex items-center gap-2 p-2 rounded-lg ${
                            sessionRagSettings.use_rerank
                              ? "bg-gradient-to-r from-orange-50 to-yellow-50 border border-orange-200"
                              : "bg-gray-50"
                          }`}
                        >
                          {sessionRagSettings.use_rerank ? (
                            <div className="flex items-center gap-1">
                              <span className="text-orange-500">🚀</span>
                              <span className="text-xs bg-gradient-to-r from-orange-500 to-yellow-500 bg-clip-text text-transparent font-bold">
                                BOOST
                              </span>
                            </div>
                          ) : (
                            <span className="w-3 h-3 bg-gray-400 rounded-full flex-shrink-0"></span>
                          )}
                          <span className="text-gray-600 text-xs">Rerank:</span>
                          <span
                            className={`font-semibold text-xs ${
                              sessionRagSettings.use_rerank
                                ? "text-orange-800"
                                : "text-gray-600"
                            }`}
                          >
                            {sessionRagSettings.use_rerank ? (
                              <span className="flex items-center gap-1">
                                Açık
                                <span className="text-orange-500">⚡</span>
                              </span>
                            ) : (
                              "Kapalı"
                            )}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* EBARS Status Panel - Only for students when EBARS is enabled */}
            {apragEnabled &&
              userRole === "student" &&
              selectedSessionId &&
              user?.username && (
                <div className="mb-4">
                  <EBARSStatusPanel
                    key={ebarsRefreshTrigger}
                    userId={user.username}
                    sessionId={selectedSessionId}
                    onFeedbackSubmitted={() => {
                      // Trigger refresh when feedback is submitted
                      setEbarsRefreshTrigger(prev => prev + 1);
                    }}
                  />
                </div>
              )}

            {/* Topic Progress Card - APRAG (only for students when enabled) */}
            {apragEnabled &&
              userRole === "student" &&
              selectedSessionId &&
              user?.username && (
                <div className="mb-4">
                  <TopicProgressCard
                    userId={user.username}
                    sessionId={selectedSessionId}
                    apragEnabled={apragEnabled}
                  />
                </div>
              )}

            {/* Clean Chat Interface */}
            <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
              {/* Chat Input - Top */}
              <div className="border-b border-gray-100 p-4 sm:p-6">
                <form
                  onSubmit={handleQuery}
                  className="flex flex-col sm:flex-row gap-3"
                >
                  <div className="flex-1 relative">
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Ders hakkında soru sorabilirsiniz..."
                      className="w-full px-4 sm:px-6 py-3 sm:py-4 border-2 border-gray-200 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all shadow-sm text-gray-800 placeholder-gray-400 text-sm sm:text-base"
                      disabled={isQuerying || studentChatLoading}
                    />
                    <div className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400">
                      💭
                    </div>
                  </div>
                  <button
                    type="submit"
                    disabled={isQuerying || studentChatLoading || !query.trim()}
                    className={`w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 rounded-2xl bg-gradient-to-r ${styling.gradient} text-white font-semibold hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md text-sm sm:text-base min-h-[44px]`}
                  >
                    {isQuerying || studentChatLoading ? (
                      <div className="flex items-center gap-2">
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        <span>Düşünüyor...</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <span>🚀</span>
                        <span>Sor</span>
                      </div>
                    )}
                  </button>
                </form>
              </div>

              {/* Chat History */}
              <div className="min-h-[50vh] max-h-[70vh] overflow-y-auto p-4 sm:p-6 space-y-4 sm:space-y-6">
                {studentMessages.length === 0 &&
                  !isQuerying &&
                  !studentChatLoading && (
                    <div className="text-center py-16">
                      <div className="text-6xl mb-4">🤖</div>
                      <h3 className="text-xl font-semibold text-gray-700 mb-2">
                        Merhaba! Size nasıl yardımcı olabilirim?
                      </h3>
                      <p className="text-gray-500 max-w-md mx-auto mb-6">
                        {selectedCourse.name} dersi hakkında sorularınızı
                        bekliyorum. Detaylı cevaplar verebilirim!
                      </p>

                      {/* Quick Start Suggestions - Dynamic course questions */}
                      <div className="space-y-4">
                        <div className="flex flex-wrap justify-center gap-3 max-w-2xl mx-auto">
                          {courseQuestions.map((suggestion, i) => (
                            <button
                              key={i}
                              onClick={() => {
                                if (isStudent && sendStudentMessage) {
                                  sendStudentMessage(
                                    suggestion,
                                    sessionRagSettings
                                  ).catch(console.error);
                                } else {
                                  handleSuggestionClick(suggestion);
                                }
                              }}
                              className="px-4 py-3 text-sm bg-gray-50 hover:bg-gray-100 text-gray-700 rounded-full border border-gray-200 hover:border-gray-300 transition-all transform hover:scale-105 min-h-[44px] min-w-[44px]"
                            >
                              {suggestion}
                            </button>
                          ))}
                        </div>

                        {/* Loading indicator - non-blocking */}
                        {loadingCourseQuestions && (
                          <div className="flex items-center justify-center gap-2 text-gray-500 mt-4">
                            <div className="w-4 h-4 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin"></div>
                            <span className="text-sm">
                              Daha fazla soru hazırlanıyor...
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                {/* Chat Messages - Reverse order (newest first) */}
                <div className="flex flex-col-reverse space-y-6 space-y-reverse">
                  {studentMessages
                    .slice()
                    .reverse()
                    .map((message, index) => (
                      <div key={message.id || index} className="space-y-4">
                        {/* Student Question */}
                        <div className="flex justify-end">
                          <div className="max-w-3xl bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-4 border-l-4 border-blue-400 shadow-sm">
                            <div className="flex items-start gap-3">
                              <span className="text-xl">👨‍🎓</span>
                              <div className="flex-1">
                                <div className="flex items-center justify-between mb-1">
                                  <p className="font-medium text-sm text-blue-700">
                                    Soru
                                  </p>
                                  {message.timestamp && (
                                    <p className="text-xs text-gray-500">
                                      {formatTimestamp(message.timestamp)}
                                    </p>
                                  )}
                                </div>
                                <p className="text-gray-800">{message.user}</p>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* AI Assistant Response */}
                        <div className="flex justify-start">
                          <div className="max-w-4xl bg-gradient-to-br from-white to-gray-50 border-2 border-indigo-100 rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                            {message.bot === "..." ? (
                              <div className="py-8">
                                <div className="flex items-center gap-4 mb-6">
                                  <div className="relative">
                                    <div
                                      className={`w-12 h-12 bg-gradient-to-br ${styling.gradient} rounded-full flex items-center justify-center animate-pulse`}
                                    >
                                      <span className="text-2xl text-white">
                                        🤖
                                      </span>
                                    </div>
                                    <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white animate-ping"></div>
                                  </div>
                                  <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-3">
                                      <div className="h-2 w-2 bg-blue-500 rounded-full animate-pulse"></div>
                                      <span className="text-sm font-medium text-gray-700">
                                        Cevap hazırlanıyor...
                                      </span>
                                    </div>
                                    <div className="space-y-3">
                                      <div className="flex items-center gap-2">
                                        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                          <div
                                            className="h-full bg-gradient-to-r from-blue-400 to-indigo-500 rounded-full animate-pulse"
                                            style={{ width: "70%" }}
                                          ></div>
                                        </div>
                                        <span className="text-xs text-gray-500">
                                          Kaynaklar taranıyor
                                        </span>
                                      </div>
                                      <div className="flex items-center gap-2">
                                        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                          <div
                                            className="h-full bg-gradient-to-r from-purple-400 to-pink-500 rounded-full animate-pulse"
                                            style={{
                                              width: "45%",
                                              animationDelay: "150ms",
                                            }}
                                          ></div>
                                        </div>
                                        <span className="text-xs text-gray-500">
                                          Cevap üretiliyor
                                        </span>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                                <div className="flex items-center gap-2 pl-16">
                                  <div className="flex gap-1">
                                    <div
                                      className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"
                                      style={{ animationDelay: "0ms" }}
                                    ></div>
                                    <div
                                      className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"
                                      style={{ animationDelay: "150ms" }}
                                    ></div>
                                    <div
                                      className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce"
                                      style={{ animationDelay: "300ms" }}
                                    ></div>
                                  </div>
                                  <span className="text-xs text-gray-400 italic">
                                    AI değerlendirme yapıyor...
                                  </span>
                                </div>
                              </div>
                            ) : (
                              <div className="flex items-start gap-4">
                                <div
                                  className={`flex-shrink-0 w-12 h-12 bg-gradient-to-br ${styling.gradient} rounded-full flex items-center justify-center text-white text-xl shadow-lg`}
                                >
                                  🎓
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-200">
                                    <p className="font-bold text-lg text-gray-800">
                                      Cevap
                                    </p>
                                    <div className="flex items-center gap-2">
                                      {message.timestamp && (
                                        <span className="text-xs text-gray-500">
                                          {formatTimestamp(message.timestamp)}
                                        </span>
                                      )}
                                      {message.durationMs != null && (
                                        <span className="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                                          ⏱️ {message.durationMs} ms
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                  <div className="prose prose-base max-w-none text-gray-700 leading-relaxed">
                                    {message.bot}
                                  </div>

                                  {/* Suggestions */}
                                  {Array.isArray(message.suggestions) &&
                                    message.suggestions.length > 0 && (
                                      <div className="mt-6 pt-4 border-t border-gray-100">
                                        <div className="flex items-center gap-2 mb-3">
                                          <svg
                                            className="w-4 h-4 text-indigo-600"
                                            fill="none"
                                            stroke="currentColor"
                                            viewBox="0 0 24 24"
                                          >
                                            <path
                                              strokeLinecap="round"
                                              strokeLinejoin="round"
                                              strokeWidth={2}
                                              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                                            />
                                          </svg>
                                          <span className="text-sm font-semibold text-gray-700">
                                            İlgili Sorular
                                          </span>
                                        </div>
                                        <div className="flex flex-wrap gap-2">
                                          {message.suggestions.map(
                                            (suggestion, i) => (
                                              <button
                                                key={i}
                                                onClick={() => {
                                                  if (
                                                    isStudent &&
                                                    sendStudentMessage
                                                  ) {
                                                    sendStudentMessage(
                                                      suggestion,
                                                      sessionRagSettings
                                                    ).catch(console.error);
                                                  } else {
                                                    handleSuggestionClick(
                                                      suggestion
                                                    );
                                                  }
                                                }}
                                                className={`px-4 py-3 text-sm bg-gradient-to-r ${styling.gradient} text-white rounded-full hover:shadow-md transition-all duration-200 transform hover:scale-105 min-h-[44px]`}
                                                title="Bu soruyu sor"
                                              >
                                                {suggestion}
                                              </button>
                                            )
                                          )}
                                        </div>
                                      </div>
                                    )}

                                  {/* Debug: Show feedback component status - ALWAYS VISIBLE */}
                                  <div className="mt-4 text-xs text-gray-500 p-3 bg-yellow-50 border border-yellow-200 rounded">
                                    <div className="font-semibold mb-1">
                                      🔍 Feedback Debug Info:
                                    </div>
                                    <div>
                                      aprag_interaction_id:{" "}
                                      {message.aprag_interaction_id
                                        ? `✅ ${message.aprag_interaction_id}`
                                        : "❌ missing"}
                                    </div>
                                    <div>
                                      user_id:{" "}
                                      {user?.id
                                        ? `✅ ${user.id}`
                                        : "❌ missing"}
                                    </div>
                                    <div>
                                      session_id:{" "}
                                      {selectedSessionId
                                        ? `✅ ${selectedSessionId}`
                                        : "❌ missing"}
                                    </div>
                                    <div>
                                      Will render EmojiFeedback:{" "}
                                      {message.aprag_interaction_id &&
                                      user?.id &&
                                      selectedSessionId
                                        ? "✅ YES"
                                        : "❌ NO"}
                                    </div>
                                  </div>

                                  {/* Emoji Feedback Component */}
                                  {message.aprag_interaction_id &&
                                    user?.id &&
                                    selectedSessionId && (
                                      <div className="mt-6 pt-4 border-t border-gray-100">
                                        <EmojiFeedback
                                          interactionId={
                                            message.aprag_interaction_id
                                          }
                                          userId={user.id.toString()}
                                          sessionId={selectedSessionId}
                                          compact={false}
                                          onFeedbackSubmitted={() => {
                                            // Refresh EBARS status panel
                                            setEbarsRefreshTrigger(prev => prev + 1);
                                            console.log("Feedback submitted, EBARS refreshed");
                                          }}
                                        />
                                      </div>
                                    )}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }
  }

  // Education Assistant Component for Teachers
  function EducationAssistantContent() {
    const assistant = useEducationAssistant();
    
    return (
      <div className="space-y-6">
        {/* Session Selection */}
        {!assistant.selectedSessionId && assistant.sessions.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <p className="text-yellow-800">
              ⚠️ Lütfen bir ders oturumu seçin.
            </p>
          </div>
        )}

        {/* Query Interface */}
        {assistant.selectedSessionId && (
          <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
            {/* Chat Input */}
            <div className="border-b border-gray-100 p-4 sm:p-6">
              <form
                onSubmit={assistant.handleQuery}
                className="flex flex-col sm:flex-row gap-3"
              >
                <div className="flex-1 relative">
                  <input
                    value={assistant.query}
                    onChange={(e) => assistant.setQuery(e.target.value)}
                    placeholder="Ders hakkında soru sorabilirsiniz..."
                    className="w-full px-4 sm:px-6 py-3 sm:py-4 border-2 border-gray-200 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all shadow-sm text-gray-800 placeholder-gray-400 text-sm sm:text-base"
                    disabled={assistant.isQuerying}
                  />
                </div>
                <button
                  type="submit"
                  disabled={assistant.isQuerying || !assistant.query.trim()}
                  className="w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 rounded-2xl bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md text-sm sm:text-base min-h-[44px]"
                >
                  {assistant.isQuerying ? (
                    <div className="flex items-center gap-2">
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Düşünüyor...</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <span>🚀</span>
                      <span>Sor</span>
                    </div>
                  )}
                </button>
              </form>
            </div>

            {/* Chat History */}
            <div className="min-h-[50vh] max-h-[70vh] overflow-y-auto p-4 sm:p-6 space-y-4 sm:space-y-6">
              {assistant.chatHistory.length === 0 && !assistant.isQuerying && (
                <div className="text-center py-16">
                  <div className="text-6xl mb-4">🤖</div>
                  <h3 className="text-xl font-semibold text-gray-700 mb-2">
                    Merhaba! Size nasıl yardımcı olabilirim?
                  </h3>
                  <p className="text-gray-500 max-w-md mx-auto mb-6">
                    Ders hakkında sorularınızı sorabilirsiniz. Detaylı cevaplar verebilirim!
                  </p>
                </div>
              )}

              {/* Chat Messages */}
              <div className="flex flex-col space-y-6">
                {assistant.chatHistory.map((message, index) => (
                  <div key={index} className="space-y-4">
                    {/* User Question */}
                    <div className="flex justify-end">
                      <div className="max-w-3xl bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-4 border-l-4 border-blue-400 shadow-sm">
                        <div className="flex items-start gap-3">
                          <span className="text-xl">👨‍🏫</span>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <p className="font-medium text-sm text-blue-700">
                                Soru
                              </p>
                              {message.timestamp && (
                                <p className="text-xs text-gray-500">
                                  {formatTimestamp(message.timestamp)}
                                </p>
                              )}
                            </div>
                            <p className="text-gray-800">{message.user}</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* AI Response */}
                    <div className="flex justify-start">
                      <div className="max-w-4xl bg-gradient-to-br from-white to-gray-50 border-2 border-indigo-100 rounded-2xl p-6 shadow-lg">
                        <div className="flex items-start gap-4">
                          <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center text-white text-xl shadow-lg">
                            🎓
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-200">
                              <p className="font-bold text-lg text-gray-800">
                                Cevap
                              </p>
                              <div className="flex items-center gap-2">
                                {message.timestamp && (
                                  <span className="text-xs text-gray-500">
                                    {formatTimestamp(message.timestamp)}
                                  </span>
                                )}
                                {message.durationMs != null && (
                                  <span className="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                                    ⏱️ {message.durationMs} ms
                                  </span>
                                )}
                              </div>
                            </div>
                            <div className="prose prose-base max-w-none text-gray-700 leading-relaxed">
                              {message.bot}
                            </div>

                            {/* Sources */}
                            {message.sources && message.sources.length > 0 && (
                              <div className="mt-6 pt-4 border-t border-gray-100">
                                <div className="flex items-center gap-2 mb-3">
                                  <span className="text-sm font-semibold text-gray-700">
                                    📚 Kaynaklar
                                  </span>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  {message.sources.map((source: any, i: number) => (
                                    <button
                                      key={i}
                                      onClick={() => assistant.handleOpenSourceModal(source)}
                                      className="px-3 py-2 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg border border-gray-200 hover:border-gray-300 transition-all"
                                    >
                                      {source.source || source.metadata?.source_file || `Kaynak ${i + 1}`}
                                      {source.score !== undefined && (
                                        <span className="ml-2 text-gray-500">
                                          ({Math.round(source.score * 100)}%)
                                        </span>
                                      )}
                                    </button>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Suggestions */}
                            {Array.isArray(message.suggestions) &&
                              message.suggestions.length > 0 && (
                                <div className="mt-6 pt-4 border-t border-gray-100">
                                  <div className="flex items-center gap-2 mb-3">
                                    <span className="text-sm font-semibold text-gray-700">
                                      💡 İlgili Sorular
                                    </span>
                                  </div>
                                  <div className="flex flex-wrap gap-2">
                                    {message.suggestions.map(
                                      (suggestion: string, i: number) => (
                                        <button
                                          key={i}
                                          onClick={() => assistant.handleSuggestionClick(suggestion)}
                                          className="px-4 py-3 text-sm bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-full hover:shadow-md transition-all duration-200 transform hover:scale-105 min-h-[44px]"
                                        >
                                          {suggestion}
                                        </button>
                                      )
                                    )}
                                  </div>
                                </div>
                              )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Source Modal */}
        <Modal
          isOpen={assistant.sourceModalOpen}
          onClose={assistant.handleCloseSourceModal}
          title="Kaynak Detayları"
          size="xl"
        >
          {assistant.selectedSource && (
            <div className="space-y-4">
              <div className="prose prose-sm max-w-none">
                {assistant.selectedSource.content}
              </div>
            </div>
          )}
        </Modal>
      </div>
    );
  }

  // Teacher view - full interface with sidebar layout
  return (
    <TeacherLayout activeTab={activeTab} onTabChange={setActiveTab}>
      {activeTab === "dashboard" && (
        <div className="space-y-8 -mt-4">
          {success && (
            <div className="mb-4 bg-green-50 border border-green-200 rounded-md p-4">
              <div className="text-green-800">{success}</div>
            </div>
          )}
          {/* Educational Stats Cards - Mobile Optimized */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 lg:gap-6 pt-4">
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-4 sm:p-6 rounded-xl sm:rounded-2xl text-white shadow-lg sm:shadow-xl transform hover:scale-105 transition-all">
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-blue-100 text-xs sm:text-sm font-medium truncate">
                    Ders Oturumları
                  </p>
                  <p className="text-2xl sm:text-3xl font-bold">
                    {totalSessions}
                  </p>
                  <p className="text-blue-200 text-xs truncate">
                    Aktif sınıflar
                  </p>
                </div>
                <div className="text-3xl sm:text-4xl opacity-80 ml-2 flex-shrink-0">
                  📚
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-500 to-green-600 p-4 sm:p-6 rounded-xl sm:rounded-2xl text-white shadow-lg sm:shadow-xl transform hover:scale-105 transition-all">
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-green-100 text-xs sm:text-sm font-medium truncate">
                    Ders Materyalleri
                  </p>
                  <p className="text-2xl sm:text-3xl font-bold">
                    {totalDocuments}
                  </p>
                  <p className="text-green-200 text-xs truncate">
                    Yüklenen belgeler
                  </p>
                </div>
                <div className="text-3xl sm:text-4xl opacity-80 ml-2 flex-shrink-0">
                  📄
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-500 to-purple-600 p-4 sm:p-6 rounded-xl sm:rounded-2xl text-white shadow-lg sm:shadow-xl transform hover:scale-105 transition-all">
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-purple-100 text-xs sm:text-sm font-medium truncate">
                    Bilgi Parçaları
                  </p>
                  <p className="text-2xl sm:text-3xl font-bold">
                    {totalChunks}
                  </p>
                  <p className="text-purple-200 text-xs truncate">
                    İşlenmiş içerik
                  </p>
                </div>
                <div className="text-3xl sm:text-4xl opacity-80 ml-2 flex-shrink-0">
                  🧩
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-orange-500 to-orange-600 p-4 sm:p-6 rounded-xl sm:rounded-2xl text-white shadow-lg sm:shadow-xl transform hover:scale-105 transition-all">
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-orange-100 text-xs sm:text-sm font-medium truncate">
                    Öğrenci Soruları
                  </p>
                  <p className="text-2xl sm:text-3xl font-bold">
                    {totalQueries}
                  </p>
                  <p className="text-orange-200 text-xs truncate">
                    Yanıtlanan sorular
                  </p>
                </div>
                <div className="text-3xl sm:text-4xl opacity-80 ml-2 flex-shrink-0">
                  ❓
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-8">
              <div className="bg-gradient-to-br from-white to-blue-50 p-8 rounded-2xl shadow-xl border border-blue-100">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 text-white rounded-xl shadow-lg">
                      <span className="text-2xl">📚</span>
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                        Son Ders Oturumları
                      </h2>
                      <p className="text-gray-600 text-sm">
                        Aktif sınıflarınızın durumu
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={refreshSessions}
                    className="bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white px-4 py-2 rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all transform hover:scale-105 flex items-center gap-2"
                    disabled={loading}
                  >
                    <span className="text-lg">🔄</span>
                    {loading ? "Yenileniyor..." : "Yenile"}
                  </button>
                </div>

                {sessions.length > 0 ? (
                  <div className="space-y-4">
                    {sessions.slice(0, 5).map((session, index) => (
                      <SessionCard
                        key={session.session_id}
                        session={session}
                        onNavigate={handleNavigateToSession}
                        onDelete={handleDeleteSession}
                        onToggleStatus={handleToggleSessionStatus}
                        onOpenRecent={(id) => openRecentInteractions(id)}
                        onUpdateName={handleUpdateSessionName}
                        index={index}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12 text-muted-foreground">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
                      <SessionIcon />
                    </div>
                    <p className="font-medium">Henüz ders oturumu bulunmuyor</p>
                    <p className="text-sm mt-1">
                      "Ders Oturumları" sekmesinden yeni bir oturum oluşturun.
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="lg:col-span-1">
              <ChangelogCard />
            </div>
          </div>
        </div>
      )}

      {activeTab === "sessions" && (
        <div className="space-y-4">
          {/* Compact Header Section */}
          <div className="flex items-center justify-between pb-3 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Ders Oturumları Yönetimi
              </h2>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                Sınıflarınızı organize edin ve ders materyallerinizi yönetin
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsCreateSessionModalOpen(true)}
                className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white px-4 py-2 rounded-lg font-medium shadow-sm hover:shadow-md transition-all flex items-center gap-2 text-sm"
              >
                <span>➕</span>
                <span>Yeni Oturum</span>
              </button>
              <label className="px-4 py-2 rounded-lg font-medium border border-purple-200 text-purple-700 bg-white hover:bg-purple-50 cursor-pointer shadow-sm text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-purple-400 dark:hover:bg-gray-700">
                İçe Aktar
                <input
                  type="file"
                  accept=".zip,.json"
                  className="hidden"
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    try {
                      setError(null);
                      const resp = await importSessionFromFile(f);
                      await refreshSessions();
                      setSelectedSessionId(resp.new_session_id);
                    } catch (err: any) {
                      setError(err.message || "İçe aktarma başarısız");
                    } finally {
                      e.currentTarget.value = "";
                    }
                  }}
                />
              </label>
            </div>
          </div>

          {/* Compact Info Section */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 flex items-start gap-2">
            <span className="text-lg">💡</span>
            <div className="flex-1">
              <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                Nasıl Çalışır?{" "}
              </span>
              <span className="text-sm text-blue-700 dark:text-blue-300">
                Ders oturumları oluşturarak farklı konuları organize
                edebilirsiniz. Her oturuma özel belgeler yükleyip,
                öğrencilerinizin o konuda sorular sormasını sağlayabilirsiniz.
              </span>
            </div>
          </div>

          <div className="bg-card p-4 rounded-lg shadow-sm">
            <div className="flex items-center gap-2 mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Tüm Ders Oturumları
              </h3>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                Aktif ve geçmiş sınıflarınız
              </span>
            </div>
            {sessions.length > 0 ? (
              <>
                <div className="space-y-4">
                  {sessions
                    .slice(
                      (sessionPage - 1) * SESSIONS_PER_PAGE,
                      sessionPage * SESSIONS_PER_PAGE
                    )
                    .map((session, index) => (
                      <SessionCard
                        key={session.session_id}
                        session={session}
                        onNavigate={handleNavigateToSession}
                        onDelete={handleDeleteSession}
                        onToggleStatus={handleToggleSessionStatus}
                        onOpenRecent={(id) => openRecentInteractions(id)}
                        onUpdateName={handleUpdateSessionName}
                        index={index}
                      />
                    ))}
                </div>
                <div className="flex items-center justify-between mt-6 pt-4 border-t border-border">
                  <button
                    onClick={() => setSessionPage((p) => Math.max(1, p - 1))}
                    disabled={sessionPage === 1}
                    className="btn btn-secondary"
                  >
                    Önceki
                  </button>
                  <span className="text-sm text-muted-foreground">
                    Sayfa {sessionPage} /{" "}
                    {Math.ceil(sessions.length / SESSIONS_PER_PAGE)}
                  </span>
                  <button
                    onClick={() =>
                      setSessionPage((p) =>
                        Math.min(
                          Math.ceil(sessions.length / SESSIONS_PER_PAGE),
                          p + 1
                        )
                      )
                    }
                    disabled={
                      sessionPage >=
                      Math.ceil(sessions.length / SESSIONS_PER_PAGE)
                    }
                    className="btn btn-secondary"
                  >
                    Sonraki
                  </button>
                </div>
              </>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
                  <SessionIcon />
                </div>
                <p className="font-medium">Henüz ders oturumu bulunmuyor</p>
                <p className="text-sm mt-1">
                  Başlamak için yeni bir ders oturumu oluşturun.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "upload" && (
        <div className="space-y-8">
          {/* Header Section */}
          <div className="bg-gradient-to-br from-blue-50 via-white to-purple-50 rounded-2xl p-8 border border-blue-100">
            <div className="flex flex-col lg:flex-row items-center justify-between gap-6 mb-6">
              <div className="flex items-center gap-4">
                <div className="p-4 bg-gradient-to-br from-blue-500 to-purple-600 text-white rounded-2xl shadow-lg">
                  <DocumentIcon />
                </div>
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    Belge Merkezi
                  </h2>
                  <p className="text-gray-600 mt-1">
                    Kapsamlı belge yönetimi ve markdown dosya işlemleri
                  </p>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row items-center gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-800">
                    Yeni Deneyim
                  </div>
                  <div className="text-sm text-gray-600">
                    Ayrı sayfada organize edildi
                  </div>
                </div>
              </div>
            </div>

            {success && (
              <div className="bg-green-50 border border-green-200 text-green-800 p-4 rounded-xl mb-6 flex items-center gap-3">
                <svg
                  className="w-6 h-6 text-green-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                {success}
              </div>
            )}

            {/* Information Card */}
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 mb-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-blue-100 rounded-full">
                  <svg
                    className="w-6 h-6 text-blue-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-blue-800 mb-2">
                    Belge Merkezi Ayrı Sayfaya Taşındı
                  </h3>
                  <p className="text-blue-700 mb-4">
                    Tüm belge yönetimi işlemleri artık daha organize ve
                    kullanıcı dostu bir ayrı sayfada bulunmaktadır. Bu sayede
                    daha iyi performans ve geliştirilmiş kullanıcı deneyimi
                    sunuyoruz.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                    <div className="flex items-center gap-2">
                      <span className="text-blue-600">✅</span>
                      <span className="text-sm text-blue-700">
                        Dosya yükleme ve dönüştürme
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-blue-600">✅</span>
                      <span className="text-sm text-blue-700">
                        Kategori yönetimi
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-blue-600">✅</span>
                      <span className="text-sm text-blue-700">
                        Markdown dosya listesi
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-blue-600">✅</span>
                      <span className="text-sm text-blue-700">
                        Oturuma belge ekleme
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Navigation Card */}
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl p-8 text-white shadow-xl">
              <div className="text-center">
                <h3 className="text-2xl font-bold mb-4">
                  Belge Merkezine Geçin
                </h3>
                <p className="text-indigo-100 mb-6 max-w-2xl mx-auto">
                  Tüm belge yönetimi işlemlerinizi modern ve organize bir
                  arayüzde gerçekleştirin. Markdown dosyalarınızı kategorilere
                  ayırın, oturumlara ekleyin ve kolayca yönetin.
                </p>

                <Link
                  href="/document-center"
                  className="inline-flex items-center gap-3 bg-white text-indigo-600 px-8 py-4 rounded-xl font-bold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 hover:bg-indigo-50"
                >
                  <span className="text-2xl">📁</span>
                  <span>Belge Merkezini Aç</span>
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M17 8l4 4m0 0l-4 4m4-4H3"
                    />
                  </svg>
                </Link>

                <div className="mt-4 text-sm text-indigo-200">
                  Yeni sekmede açmak için Ctrl+Click kullanın
                </div>
              </div>
            </div>
          </div>

          {/* Quick Action Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Upload Documents */}
            <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-blue-100 text-blue-600 rounded-lg">
                  <UploadIcon />
                </div>
                <h3 className="text-lg font-semibold text-gray-800">
                  Belge Yükle
                </h3>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                PDF dosyalarınızı yükleyin ve Markdown formatına dönüştürün
              </p>
              <button
                onClick={() => setIsDocumentUploadModalOpen(true)}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                Belge Yükle
              </button>
            </div>

            {/* Manage Categories */}
            <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-purple-100 text-purple-600 rounded-lg">
                  <span className="text-lg">📂</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-800">
                  Kategoriler
                </h3>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                Markdown dosyalarınızı organize etmek için kategoriler oluşturun
              </p>
              <Link
                href="/document-center"
                className="block w-full bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors font-medium text-center"
              >
                Kategorileri Yönet
              </Link>
            </div>

            {/* View Files */}
            <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-green-100 text-green-600 rounded-lg">
                  <MarkdownIcon />
                </div>
                <h3 className="text-lg font-semibold text-gray-800">
                  Dosyalar
                </h3>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                Mevcut Markdown dosyalarınızı görüntüleyin ve yönetin
              </p>
              <Link
                href="/document-center"
                className="block w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors font-medium text-center"
              >
                Dosyaları Görüntüle
              </Link>
            </div>
          </div>

          {/* Migration Notice */}
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-xl p-6">
            <div className="flex items-start gap-3">
              <span className="text-2xl">🚀</span>
              <div>
                <h3 className="text-lg font-semibold text-amber-800 mb-2">
                  Geliştirilmiş Belge Yönetimi
                </h3>
                <p className="text-amber-700 mb-3">
                  Belge merkezi, daha iyi performans ve kullanılabilirlik için
                  tamamen yeniden tasarlandı. Tüm mevcut özellikler korunurken,
                  yeni özellikler de eklendi:
                </p>
                <ul className="text-sm text-amber-700 space-y-1 mb-4">
                  <li>• Daha hızlı dosya yükleme ve işleme</li>
                  <li>• Gelişmiş kategori yönetimi</li>
                  <li>• Daha iyi dosya görüntüleme ve filtreleme</li>
                  <li>• Responsive tasarım ve mobil uyumluluk</li>
                  <li>• Sürükle-bırak dosya yükleme</li>
                </ul>
                <div className="text-sm text-amber-600 font-medium">
                  Tüm verileriniz güvende - hiçbir şey kaybolmadı!
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {false && (
        <div className="space-y-8">
          {/* Header Section */}
          <div className="bg-gradient-to-br from-blue-50 via-white to-indigo-50 rounded-2xl p-8 border border-blue-100">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <div className="p-4 bg-gradient-to-br from-blue-500 to-indigo-600 text-white rounded-2xl shadow-lg">
                  <span className="text-3xl">🎓</span>
                </div>
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    RAG Temelli Eğitim Asistanı
                  </h2>
                  <p className="text-gray-600 mt-1">
                    Ders materyalleri hakkında sorularınızı sorun
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Session Selection - RAG Settings moved to session detail page */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden p-6">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                🎯 Ders Oturumu Seçin
              </label>
              <div className="flex items-center gap-3">
                <select
                  value={selectedSessionId}
                  onChange={(e) => setSelectedSessionId(e.target.value)}
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all bg-white shadow-sm"
                >
                  <option value="">Ders Oturumu Seçin</option>
                  {sessions
                    .filter((s) => s.status === "active")
                    .map((session) => (
                      <option
                        key={session.session_id}
                        value={session.session_id}
                      >
                        📚 {session.name} ({session.document_count} doküman)
                      </option>
                    ))}
                </select>
                {selectedSessionId && (
                  <Link
                    href={`/sessions/${selectedSessionId}`}
                    className="px-4 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors font-medium whitespace-nowrap"
                  >
                    ⚙️ Ayarları Düzenle
                  </Link>
                )}
              </div>
              <p className="text-xs text-gray-500 mt-2">
                💡 RAG ayarlarını düzenlemek için oturum sayfasına gidin
              </p>
            </div>
          </div>

          {/* Chat Interface Accordion */}
          {(selectedSessionId || useDirectLLM) && (
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
              <button
                onClick={() => {
                  setIsChatOpen(!isChatOpen);
                }}
                className="w-full px-8 py-6 bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200 flex items-center justify-between hover:from-gray-100 hover:to-gray-200 transition-all"
              >
                <div className="flex items-center gap-3">
                  <span className="text-3xl">💬</span>
                  <div className="text-left">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">
                      Eğitim Asistanı Sohbeti
                    </h2>
                    {useDirectLLM ? (
                      <p className="text-gray-600">
                        <strong>🤖 Direkt LLM Modu:</strong> Genel sorularınız
                        için LLM kullanılıyor
                      </p>
                    ) : (
                      <p className="text-gray-600">
                        Seçili ders oturumu:{" "}
                        <strong>
                          {
                            sessions.find(
                              (s) => s.session_id === selectedSessionId
                            )?.name
                          }
                        </strong>
                      </p>
                    )}
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      {!!(sessionRagSettings?.model || selectedQueryModel) && (
                        <span className="px-2 py-0.5 rounded-full text-xs bg-indigo-50 text-indigo-700 border border-indigo-200">
                          Model:{" "}
                          {sessionRagSettings?.model || selectedQueryModel}
                        </span>
                      )}
                      {!!(
                        sessionRagSettings?.embedding_model ||
                        selectedEmbeddingModel
                      ) && (
                        <span className="px-2 py-0.5 rounded-full text-xs bg-purple-50 text-purple-700 border border-purple-200">
                          Embedding:{" "}
                          {sessionRagSettings?.embedding_model ||
                            selectedEmbeddingModel}
                        </span>
                      )}
                      {!!(sessionRagSettings?.chain_type || chainType) && (
                        <span className="px-2 py-0.5 rounded-full text-xs bg-blue-50 text-blue-700 border border-blue-200">
                          Chain: {sessionRagSettings?.chain_type || chainType}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <svg
                  className={`w-6 h-6 text-gray-600 transition-transform duration-200 ${
                    isChatOpen ? "rotate-180" : ""
                  }`}
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

              {isChatOpen && (
                <div className="p-8">
                  {/* Recommendation Panel - Only for Students and when APRAG is enabled */}
                  {apragEnabled &&
                    selectedSessionId &&
                    user?.username &&
                    userRole === "student" && (
                      <div className="mb-6">
                        <RecommendationPanel
                          userId={user?.username || "anonymous"}
                          sessionId={selectedSessionId}
                          onQuestionClick={(question) => {
                            setQuery(question);
                            // Auto-submit if desired
                            // handleQuery(new Event('submit') as any);
                          }}
                        />
                      </div>
                    )}

                  {/* Query Input - Moved to top */}
                  <form
                    onSubmit={handleQuery}
                    className="bg-white rounded-xl border border-gray-200 shadow-lg p-4 mb-6"
                  >
                    {/* Answer Length Selector */}
                    {userRole !== "student" && (
                      <div className="mb-3 flex items-center gap-2">
                        <span className="text-xs font-medium text-gray-600">
                          Cevap Uzunluğu:
                        </span>
                        <div className="flex gap-2">
                          <button
                            type="button"
                            onClick={() => setAnswerLength("short")}
                            className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                              answerLength === "short"
                                ? "bg-blue-500 text-white"
                                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                            }`}
                          >
                            🔸 Kısa
                          </button>
                          <button
                            type="button"
                            onClick={() => setAnswerLength("normal")}
                            className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                              answerLength === "normal"
                                ? "bg-blue-500 text-white"
                                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                            }`}
                          >
                            🔹 Normal
                          </button>
                          <button
                            type="button"
                            onClick={() => setAnswerLength("detailed")}
                            className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                              answerLength === "detailed"
                                ? "bg-blue-500 text-white"
                                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                            }`}
                          >
                            🔶 Detaylı
                          </button>
                        </div>
                      </div>
                    )}

                    <div className="flex items-center gap-3">
                      <div className="flex-1 relative">
                        <input
                          type="text"
                          value={query}
                          onChange={(e) => setQuery(e.target.value)}
                          placeholder="Ders materyalleri hakkında sorunuzu yazın..."
                          className="w-full p-4 pr-12 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-gray-800 placeholder-gray-400"
                          disabled={
                            isQuerying ||
                            (!sessionRagSettings?.model && !selectedQueryModel)
                          }
                        />
                        <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400">
                          💭
                        </div>
                      </div>
                      <button
                        type="submit"
                        disabled={
                          isQuerying ||
                          !query.trim() ||
                          (!sessionRagSettings?.model && !selectedQueryModel)
                        }
                        className="px-6 py-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg font-medium"
                      >
                        {isQuerying ? (
                          <>
                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                            <span>Düşünüyor...</span>
                          </>
                        ) : (
                          <>
                            <span className="text-lg">🚀</span>
                            <span>Sor</span>
                          </>
                        )}
                      </button>
                    </div>
                    <div className="mt-2 text-xs text-gray-500 flex items-center gap-1">
                      <span>💡</span>
                      <span>
                        İpucu: Spesifik sorular daha iyi sonuçlar verir
                      </span>
                    </div>
                  </form>

                  {chatHistory.length === 0 && (
                    <div className="-mt-2 mb-6 flex flex-wrap gap-2">
                      {[
                        "Kısa özet çıkar",
                        "Bu başlığın ana fikirleri neler?",
                        "Öğrenciye 3 soru hazırla",
                        "Kaynaklardan alıntıyla cevapla",
                      ].map((s, i) => (
                        <button
                          type="button"
                          key={i}
                          onClick={() => handleSuggestionClick(s)}
                          className="px-2.5 py-1.5 text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full hover:bg-indigo-100"
                          title="Öneriyi sor"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Chat History - Reversed order (newest on top) */}
                  <div
                    id="chat-history-container"
                    className="relative min-h-[50vh] max-h-[65vh] overflow-y-auto bg-gradient-to-b from-indigo-50/40 to-white rounded-xl border border-gray-200 p-3 sm:p-4 space-y-3 sm:space-y-4 flex flex-col"
                  >
                    {chatHistory.length === 0 && (
                      <div className="text-center py-12">
                        <div className="text-6xl mb-4">🎓</div>
                        <h3 className="text-xl font-semibold text-gray-700 mb-2">
                          Eğitim Asistanınıza Hoş Geldiniz!
                        </h3>
                        <p className="text-gray-500 max-w-md mx-auto">
                          Ders materyalleriniz hakkında soru sorarak öğrenme
                          sürecinizi destekleyin.
                        </p>
                      </div>
                    )}
                    {chatHistory
                      .slice()
                      .reverse()
                      .map((chat, index) => (
                        <div key={index} className="space-y-3">
                          {/* Teacher Question */}
                          <div className="w-full">
                            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-blue-500 p-4 rounded-r-lg shadow-sm">
                              <div className="flex items-start gap-3">
                                <span className="text-xl">👨‍🏫</span>
                                <div className="flex-1">
                                  <div className="flex items-center justify-between mb-1">
                                    <p className="font-medium text-sm text-blue-700">
                                      Soru
                                    </p>
                                    {chat.timestamp && (
                                      <p className="text-xs text-gray-500">
                                        {formatTimestamp(chat.timestamp)}
                                      </p>
                                    )}
                                  </div>
                                  <p className="text-gray-800">{chat.user}</p>
                                </div>
                              </div>
                            </div>
                          </div>
                          {/* AI Assistant Response */}
                          <div className="w-full">
                            <div className="bg-gradient-to-br from-white to-gray-50 border-2 border-indigo-100 p-5 rounded-xl shadow-md hover:shadow-lg transition-shadow">
                              {chat.bot === "..." ? (
                                <div className="py-6">
                                  <div className="flex items-center gap-4 mb-4">
                                    <div className="relative">
                                      <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center animate-pulse">
                                        <span className="text-2xl">🤖</span>
                                      </div>
                                      <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white animate-ping"></div>
                                    </div>
                                    <div className="flex-1">
                                      <div className="flex items-center gap-2 mb-2">
                                        <div className="h-2 w-2 bg-blue-500 rounded-full animate-pulse"></div>
                                        <span className="text-sm font-medium text-gray-700">
                                          Cevap hazırlanıyor...
                                        </span>
                                      </div>
                                      <div className="space-y-2">
                                        {/* Animated progress bars */}
                                        <div className="flex items-center gap-2">
                                          <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                            <div
                                              className="h-full bg-gradient-to-r from-blue-400 to-indigo-500 rounded-full animate-pulse"
                                              style={{ width: "70%" }}
                                            ></div>
                                          </div>
                                          <span className="text-xs text-gray-500">
                                            Kaynaklar taranıyor
                                          </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                          <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                            <div
                                              className="h-full bg-gradient-to-r from-purple-400 to-pink-500 rounded-full animate-pulse"
                                              style={{
                                                width: "45%",
                                                animationDelay: "150ms",
                                              }}
                                            ></div>
                                          </div>
                                          <span className="text-xs text-gray-500">
                                            Cevap üretiliyor
                                          </span>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                  {/* Typing indicator */}
                                  <div className="flex items-center gap-2 pl-16">
                                    <div className="flex gap-1">
                                      <div
                                        className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"
                                        style={{ animationDelay: "0ms" }}
                                      ></div>
                                      <div
                                        className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"
                                        style={{ animationDelay: "150ms" }}
                                      ></div>
                                      <div
                                        className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce"
                                        style={{ animationDelay: "300ms" }}
                                      ></div>
                                    </div>
                                    <span className="text-xs text-gray-400 italic">
                                      DYSK katmanı değerlendiriliyor...
                                    </span>
                                  </div>
                                </div>
                              ) : (
                                <div className="flex items-start gap-4">
                                  <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white text-lg shadow-md">
                                    🎓
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between mb-3 pb-2 border-b border-gray-200">
                                      <p className="font-semibold text-base text-gray-800">
                                        Cevap
                                      </p>
                                      <div className="flex items-center gap-2">
                                        {chat.timestamp && (
                                          <span className="text-xs text-gray-500">
                                            {formatTimestamp(chat.timestamp)}
                                          </span>
                                        )}
                                        {(chat.sources?.length ?? 0) > 0 && (
                                          <span className="text-xs font-medium text-indigo-700 bg-indigo-50 px-2.5 py-1 rounded-full border border-indigo-200">
                                            {chat.sources?.length} kaynak
                                          </span>
                                        )}
                                        {chat.durationMs != null && (
                                          <span className="text-xs text-gray-600 bg-gray-100 px-2.5 py-1 rounded-full font-medium">
                                            {chat.durationMs} ms
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                    <div className="prose prose-base max-w-none text-gray-800 leading-relaxed markdown-content">
                                      <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                        components={{
                                          p: ({ node, ...props }) => (
                                            <p
                                              className="mb-4 text-gray-800 leading-7"
                                              {...props}
                                            />
                                          ),
                                          h1: ({ node, ...props }) => (
                                            <h1
                                              className="text-2xl font-bold mb-3 mt-6 text-gray-900 border-b border-gray-200 pb-2"
                                              {...props}
                                            />
                                          ),
                                          h2: ({ node, ...props }) => (
                                            <h2
                                              className="text-xl font-bold mb-3 mt-5 text-gray-900"
                                              {...props}
                                            />
                                          ),
                                          h3: ({ node, ...props }) => (
                                            <h3
                                              className="text-lg font-semibold mb-2 mt-4 text-gray-900"
                                              {...props}
                                            />
                                          ),
                                          ul: ({ node, ...props }) => (
                                            <ul
                                              className="list-disc list-inside mb-4 space-y-2 text-gray-800"
                                              {...props}
                                            />
                                          ),
                                          ol: ({ node, ...props }) => (
                                            <ol
                                              className="list-decimal list-inside mb-4 space-y-2 text-gray-800"
                                              {...props}
                                            />
                                          ),
                                          li: ({ node, ...props }) => (
                                            <li className="ml-4" {...props} />
                                          ),
                                          code: ({
                                            node,
                                            inline,
                                            ...props
                                          }: any) =>
                                            inline ? (
                                              <code
                                                className="bg-gray-100 text-indigo-700 px-1.5 py-0.5 rounded text-sm font-mono"
                                                {...props}
                                              />
                                            ) : (
                                              <code
                                                className="block bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm font-mono mb-4"
                                                {...props}
                                              />
                                            ),
                                          pre: ({ node, ...props }) => (
                                            <pre
                                              className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4"
                                              {...props}
                                            />
                                          ),
                                          blockquote: ({ node, ...props }) => (
                                            <blockquote
                                              className="border-l-4 border-indigo-500 pl-4 italic text-gray-700 my-4 bg-indigo-50 py-2 rounded-r"
                                              {...props}
                                            />
                                          ),
                                          strong: ({ node, ...props }) => (
                                            <strong
                                              className="font-bold text-gray-900"
                                              {...props}
                                            />
                                          ),
                                          em: ({ node, ...props }) => (
                                            <em
                                              className="italic text-gray-700"
                                              {...props}
                                            />
                                          ),
                                          a: ({ node, ...props }) => (
                                            <a
                                              className="text-indigo-600 hover:text-indigo-800 underline"
                                              {...props}
                                            />
                                          ),
                                        }}
                                      >
                                        {chat.bot}
                                      </ReactMarkdown>
                                    </div>

                                    {/* Correction Notice */}
                                    {(chat as any).correction &&
                                      (chat as any).correction
                                        .was_corrected && (
                                        <div className="mt-4 mb-4 p-4 bg-amber-50 border-l-4 border-amber-500 rounded-r-lg shadow-sm">
                                          <div className="flex items-start gap-3">
                                            <div className="text-amber-600 mt-0.5">
                                              <svg
                                                className="w-5 h-5"
                                                fill="none"
                                                stroke="currentColor"
                                                viewBox="0 0 24 24"
                                              >
                                                <path
                                                  strokeLinecap="round"
                                                  strokeLinejoin="round"
                                                  strokeWidth="2"
                                                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                                                />
                                              </svg>
                                            </div>
                                            <div className="flex-1">
                                              <h4 className="text-sm font-bold text-amber-800 mb-1">
                                                ⚠️ Otomatik Doğrulama ve
                                                Düzeltme
                                              </h4>
                                              <p className="text-xs text-amber-700 mb-2">
                                                Yapay zeka, ilk cevabında
                                                tutarsızlıklar tespit etti ve
                                                aşağıdaki nedenlerle cevabı
                                                güncelledi:
                                              </p>
                                              <ul className="list-disc list-inside text-xs text-amber-800 space-y-1 bg-amber-100/50 p-2 rounded">
                                                {(
                                                  chat as any
                                                ).correction.issues.map(
                                                  (
                                                    issue: string,
                                                    idx: number
                                                  ) => (
                                                    <li key={idx}>{issue}</li>
                                                  )
                                                )}
                                              </ul>
                                            </div>
                                          </div>
                                        </div>
                                      )}

                                    {/* Verification Success Notice (Debug) */}
                                    {(chat as any).correction &&
                                      !(chat as any).correction.was_corrected &&
                                      (chat as any).correction.issues &&
                                      (chat as any).correction.issues.length ===
                                        0 && (
                                        <div className="mt-4 mb-4 p-3 bg-green-50 border-l-4 border-green-500 rounded-r-lg shadow-sm">
                                          <div className="flex items-start gap-2">
                                            <div className="text-green-600 mt-0.5">
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
                                                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                                                />
                                              </svg>
                                            </div>
                                            <div className="flex-1">
                                              <p className="text-xs font-medium text-green-800">
                                                ✅ Cevap doğrulandı - Tutarlılık
                                                kontrolü başarılı
                                              </p>
                                            </div>
                                          </div>
                                        </div>
                                      )}

                                    {Array.isArray(chat.suggestions) &&
                                      chat.suggestions.length > 0 && (
                                        <div className="mt-4 pt-4 border-t border-gray-100">
                                          <div className="flex items-center gap-2 mb-3">
                                            <svg
                                              className="w-4 h-4 text-indigo-600"
                                              fill="none"
                                              stroke="currentColor"
                                              viewBox="0 0 24 24"
                                            >
                                              <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                strokeWidth={2}
                                                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                                              />
                                            </svg>
                                            <span className="text-sm font-semibold text-gray-700">
                                              İlgili Sorular
                                            </span>
                                          </div>
                                          <div className="flex flex-wrap gap-2">
                                            {chat.suggestions.map(
                                              (s: string, i: number) => (
                                                <button
                                                  key={i}
                                                  onClick={() =>
                                                    handleSuggestionClick(s)
                                                  }
                                                  className="group px-3 py-2 text-sm bg-gradient-to-r from-indigo-50 to-purple-50 text-indigo-700 border border-indigo-200 rounded-lg hover:from-indigo-100 hover:to-purple-100 hover:border-indigo-300 hover:shadow-md transition-all duration-200 flex items-center gap-2"
                                                  title="Bu soruyu sor"
                                                >
                                                  <svg
                                                    className="w-4 h-4 text-indigo-500 group-hover:text-indigo-600"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    viewBox="0 0 24 24"
                                                  >
                                                    <path
                                                      strokeLinecap="round"
                                                      strokeLinejoin="round"
                                                      strokeWidth={2}
                                                      d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                                    />
                                                  </svg>
                                                  <span>{s}</span>
                                                </button>
                                              )
                                            )}
                                          </div>
                                        </div>
                                      )}
                                  </div>
                                </div>
                              )}

                              {/* Enhanced Sources Display for Teachers - Tab System */}
                              {(() => {
                                // Show all sources - no similarity filtering since scores vary by embedding model
                                const filteredSources = chat.sources || [];

                                if (filteredSources.length === 0) return null;

                                return (
                                  <div className="mt-4 pt-4 border-t border-gray-100">
                                    <div className="flex items-center justify-between gap-2 mb-3">
                                      <div className="flex items-center gap-2">
                                        <span className="text-sm">📚</span>
                                        <span className="text-sm font-medium text-gray-600">
                                          Güvenilir Kaynaklar (
                                          {filteredSources.length})
                                        </span>
                                      </div>
                                    </div>

                                    {/* Tab System for Sources */}
                                    <div className="bg-gray-50 rounded-lg border border-gray-200 overflow-hidden">
                                      <div className="flex flex-wrap border-b border-gray-200 bg-white">
                                        {filteredSources.map(
                                          (source: any, idx: number) => (
                                            <button
                                              key={idx}
                                              className="px-3 py-2 text-xs font-medium text-gray-700 hover:text-blue-600 hover:bg-blue-50 border-r border-gray-200 last:border-r-0 transition-colors flex items-center gap-2"
                                              onClick={(e) => {
                                                const panel =
                                                  e.currentTarget.parentElement
                                                    ?.nextElementSibling
                                                    ?.children[idx];
                                                const allPanels =
                                                  e.currentTarget.parentElement
                                                    ?.nextElementSibling
                                                    ?.children;
                                                const allTabs =
                                                  e.currentTarget.parentElement
                                                    ?.children;

                                                // Hide all panels and deactivate tabs
                                                if (allPanels && allTabs) {
                                                  for (
                                                    let i = 0;
                                                    i < allPanels.length;
                                                    i++
                                                  ) {
                                                    allPanels[i].classList.add(
                                                      "hidden"
                                                    );
                                                    allTabs[i].classList.remove(
                                                      "bg-blue-100",
                                                      "text-blue-700"
                                                    );
                                                  }
                                                }

                                                // Show selected panel and activate tab
                                                panel?.classList.remove(
                                                  "hidden"
                                                );
                                                e.currentTarget.classList.add(
                                                  "bg-blue-100",
                                                  "text-blue-700"
                                                );
                                              }}
                                            >
                                              <span className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full text-xs font-bold">
                                                {idx + 1}
                                              </span>
                                              <span
                                                className="truncate max-w-20 cursor-pointer hover:text-blue-700 underline"
                                                onClick={(e) => {
                                                  e.stopPropagation();
                                                  handleOpenSourceModal(source);
                                                }}
                                                title="Kaynağı detaylı görüntülemek için tıklayın"
                                              >
                                                {source.metadata?.source_file?.replace(
                                                  ".md",
                                                  ""
                                                ) ||
                                                  source.metadata?.filename?.replace(
                                                    ".md",
                                                    ""
                                                  ) ||
                                                  `Kaynak ${idx + 1}`}
                                              </span>
                                              <span className="flex items-center gap-1">
                                                {source.crag_score != null ? (
                                                  <>
                                                    <span className="text-xs text-gray-500">
                                                      DYSK:
                                                    </span>
                                                    <span className="text-indigo-700 font-bold">
                                                      {Math.round(
                                                        source.crag_score * 100
                                                      )}
                                                      %
                                                    </span>
                                                  </>
                                                ) : (
                                                  <span className="text-green-600 font-bold">
                                                    {Math.round(
                                                      source.score * 100
                                                    )}
                                                    %
                                                  </span>
                                                )}
                                              </span>
                                            </button>
                                          )
                                        )}
                                      </div>

                                      {/* Tab Panels */}
                                      <div className="relative">
                                        {filteredSources.map(
                                          (source: any, idx: number) => (
                                            <div
                                              key={idx}
                                              className={`p-3 ${
                                                idx !== 0 ? "hidden" : ""
                                              }`}
                                            >
                                              <div className="space-y-2">
                                                <div className="flex items-center justify-between">
                                                  <div
                                                    className="text-xs font-semibold text-blue-700 cursor-pointer hover:text-blue-900 hover:underline"
                                                    onClick={() =>
                                                      handleOpenSourceModal(
                                                        source
                                                      )
                                                    }
                                                    title="Kaynağı detaylı görüntülemek için tıklayın"
                                                  >
                                                    📄{" "}
                                                    {source.metadata
                                                      ?.source_file ||
                                                      source.metadata
                                                        ?.filename ||
                                                      "Belge"}
                                                  </div>
                                                  <div className="flex items-center gap-3">
                                                    {source.crag_score !=
                                                      null && (
                                                      <div className="flex items-center gap-1">
                                                        <span className="text-xs text-gray-500">
                                                          DYSK:
                                                        </span>
                                                        <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                                          <div
                                                            className="h-full bg-gradient-to-r from-indigo-400 to-purple-500 rounded-full"
                                                            style={{
                                                              width: `${
                                                                source.crag_score *
                                                                100
                                                              }%`,
                                                            }}
                                                          ></div>
                                                        </div>
                                                        <span className="text-xs text-indigo-700 font-bold">
                                                          {Math.round(
                                                            source.crag_score *
                                                              100
                                                          )}
                                                          %
                                                        </span>
                                                      </div>
                                                    )}
                                                    <div className="flex items-center gap-1">
                                                      <span className="text-xs text-gray-500">
                                                        Benzerlik:
                                                      </span>
                                                      <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                                        <div
                                                          className="h-full bg-gradient-to-r from-green-400 to-blue-500 rounded-full"
                                                          style={{
                                                            width: `${
                                                              source.score * 100
                                                            }%`,
                                                          }}
                                                        ></div>
                                                      </div>
                                                      <span className="text-xs text-green-600 font-bold">
                                                        {Math.round(
                                                          source.score * 100
                                                        )}
                                                        %
                                                      </span>
                                                    </div>
                                                  </div>
                                                </div>

                                                {(source.metadata
                                                  ?.chunk_index ||
                                                  source.metadata
                                                    ?.chunk_title) && (
                                                  <div className="text-xs text-gray-600">
                                                    📍 Bölüm{" "}
                                                    {source.metadata
                                                      ?.chunk_index || "?"}{" "}
                                                    /{" "}
                                                    {source.metadata
                                                      ?.total_chunks || "?"}
                                                    {source.metadata
                                                      ?.chunk_title && (
                                                      <div className="font-medium text-gray-700 mt-1">
                                                        🏷️{" "}
                                                        {
                                                          source.metadata
                                                            .chunk_title
                                                        }
                                                      </div>
                                                    )}
                                                  </div>
                                                )}

                                                {source.content && (
                                                  <div className="text-xs text-gray-700 bg-white p-2 rounded border-l-2 border-blue-300">
                                                    <div className="line-clamp-3">
                                                      "
                                                      {source.content.substring(
                                                        0,
                                                        150
                                                      )}
                                                      ..."
                                                    </div>
                                                    <button
                                                      onClick={() =>
                                                        handleOpenSourceModal(
                                                          source
                                                        )
                                                      }
                                                      className="mt-2 text-xs text-blue-600 hover:text-blue-800 font-medium"
                                                    >
                                                      Tam içeriği görüntüle →
                                                    </button>
                                                  </div>
                                                )}
                                              </div>
                                            </div>
                                          )
                                        )}
                                      </div>
                                    </div>

                                    <div className="mt-2 text-xs text-gray-500 text-center">
                                      💡 Not: Tüm güvenilir kaynaklar
                                      gösterilmektedir
                                    </div>
                                  </div>
                                );
                              })()}

                              {/* Feedback Button - APRAG */}
                              {chat.interactionId &&
                                chat.bot !== "..." &&
                                !chat.bot.startsWith("Hata:") && (
                                  <div className="mt-4 pt-4 border-t border-gray-200">
                                    <button
                                      onClick={() => {
                                        setSelectedInteractionForFeedback({
                                          interactionId: chat.interactionId!,
                                          query: chat.user,
                                          answer: chat.bot,
                                        });
                                        setFeedbackModalOpen(true);
                                      }}
                                      className="w-full px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg hover:from-green-600 hover:to-emerald-700 transition-all duration-200 flex items-center justify-center gap-2 font-medium shadow-sm hover:shadow-md"
                                    >
                                      <svg
                                        className="w-5 h-5"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                      >
                                        <path
                                          strokeLinecap="round"
                                          strokeLinejoin="round"
                                          strokeWidth="2"
                                          d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
                                        />
                                      </svg>
                                      <span>Geri Bildirim Ver</span>
                                    </button>
                                  </div>
                                )}
                            </div>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {!selectedSessionId && !useDirectLLM && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 text-center">
              <div className="text-4xl mb-3">⚠️</div>
              <h3 className="text-lg font-semibold text-yellow-800 mb-2">
                Ders Oturumu Seçin
              </h3>
              <p className="text-yellow-700">
                Soru sormak için önce yukarıdan bir ders oturumu seçin veya
                "LLM'den direkt cevap al" seçeneğini açın.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === "query" && (
        <div className="space-y-8">
          {/* Header Section */}
          <div className="bg-gradient-to-br from-blue-50 via-white to-indigo-50 rounded-2xl p-8 border border-blue-100">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <div className="p-4 bg-gradient-to-br from-blue-500 to-indigo-600 text-white rounded-2xl shadow-lg">
                  <span className="text-3xl">🎓</span>
                </div>
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    RAG Temelli Eğitim Asistanı
                  </h2>
                  <p className="text-gray-600 mt-1">
                    Ders materyalleri hakkında sorularınızı sorun
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Session Selection */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden p-6">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                🎯 Ders Oturumu Seçin
              </label>
              <div className="flex items-center gap-3">
                <select
                  value={selectedSessionId}
                  onChange={(e) => setSelectedSessionId(e.target.value)}
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all bg-white shadow-sm"
                >
                  <option value="">Ders Oturumu Seçin</option>
                  {sessions
                    .filter((s) => s.status === "active")
                    .map((session) => (
                      <option
                        key={session.session_id}
                        value={session.session_id}
                      >
                        📚 {session.name} ({session.document_count} doküman)
                      </option>
                    ))}
                </select>
              </div>
            </div>
          </div>

          {/* Chat Interface - Use existing chat code from dashboard */}
          {(selectedSessionId || useDirectLLM) && (
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
              <div className="p-8">
                {/* Query Input */}
                <form
                  onSubmit={handleQuery}
                  className="bg-white rounded-xl border border-gray-200 shadow-lg p-4 mb-6"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex-1 relative">
                      <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ders materyalleri hakkında sorunuzu yazın..."
                        className="w-full p-4 pr-12 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-gray-800 placeholder-gray-400"
                        disabled={
                          isQuerying ||
                          (!sessionRagSettings?.model && !selectedQueryModel)
                        }
                      />
                      <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400">
                        💭
                      </div>
                    </div>
                    <button
                      type="submit"
                      disabled={
                        isQuerying ||
                        !query.trim() ||
                        (!sessionRagSettings?.model && !selectedQueryModel)
                      }
                      className="px-6 py-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg font-medium"
                    >
                      {isQuerying ? (
                        <>
                          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin inline-block mr-2"></div>
                          <span>Düşünüyor...</span>
                        </>
                      ) : (
                        <>
                          <span className="text-lg">🚀</span>
                          <span>Sor</span>
                        </>
                      )}
                    </button>
                  </div>
                </form>

                {/* Chat History - Use existing chat history rendering */}
                <div
                  id="chat-history-container"
                  className="relative min-h-[50vh] max-h-[65vh] overflow-y-auto bg-gradient-to-b from-indigo-50/40 to-white rounded-xl border border-gray-200 p-3 sm:p-4 space-y-3 sm:space-y-4 flex flex-col"
                >
                  {chatHistory.length === 0 && (
                    <div className="text-center py-12">
                      <div className="text-6xl mb-4">🎓</div>
                      <h3 className="text-xl font-semibold text-gray-700 mb-2">
                        Eğitim Asistanınıza Hoş Geldiniz!
                      </h3>
                      <p className="text-gray-500 max-w-md mx-auto">
                        Ders materyalleriniz hakkında soru sorarak öğrenme
                        sürecinizi destekleyin.
                      </p>
                    </div>
                  )}
                  {chatHistory
                    .slice()
                    .reverse()
                    .map((chat, index) => (
                      <div key={index} className="space-y-3">
                        {/* Teacher Question */}
                        <div className="w-full">
                          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-blue-500 p-4 rounded-r-lg shadow-sm">
                            <div className="flex items-start gap-3">
                              <span className="text-xl">👨‍🏫</span>
                              <div className="flex-1">
                                <div className="flex items-center justify-between mb-1">
                                  <p className="font-medium text-sm text-blue-700">
                                    Soru
                                  </p>
                                  {chat.timestamp && (
                                    <p className="text-xs text-gray-500">
                                      {formatTimestamp(chat.timestamp)}
                                    </p>
                                  )}
                                </div>
                                <p className="text-gray-800">{chat.user}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                        {/* AI Assistant Response */}
                        <div className="w-full">
                          <div className="bg-gradient-to-br from-white to-gray-50 border-2 border-indigo-100 p-5 rounded-xl shadow-md hover:shadow-lg transition-shadow">
                            {chat.bot === "..." ? (
                              <div className="py-6">
                                <div className="flex items-center gap-4 mb-4">
                                  <div className="relative">
                                    <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center animate-pulse">
                                      <span className="text-2xl">🤖</span>
                                    </div>
                                  </div>
                                  <div className="flex-1">
                                    <span className="text-sm font-medium text-gray-700">
                                      Cevap hazırlanıyor...
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ) : (
                              <div className="flex items-start gap-4">
                                <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white text-lg shadow-md">
                                  🎓
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="prose prose-base max-w-none text-gray-800 leading-relaxed">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                      {chat.bot}
                                    </ReactMarkdown>
                                  </div>

                                  {/* Sources */}
                                  {chat.sources && chat.sources.length > 0 && (
                                    <div className="mt-6 pt-4 border-t border-gray-100">
                                      <div className="flex flex-wrap gap-2">
                                        {chat.sources.map((source: any, i: number) => (
                                          <button
                                            key={i}
                                            onClick={() => handleOpenSourceModal(source)}
                                            className="px-3 py-2 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg border border-gray-200 transition-all"
                                          >
                                            {source.source || source.metadata?.source_file || `Kaynak ${i + 1}`}
                                          </button>
                                        ))}
                                      </div>
                                    </div>
                                  )}

                                  {/* Suggestions */}
                                  {Array.isArray(chat.suggestions) && chat.suggestions.length > 0 && (
                                    <div className="mt-6 pt-4 border-t border-gray-100">
                                      <div className="flex flex-wrap gap-2">
                                        {chat.suggestions.map((suggestion: string, i: number) => (
                                          <button
                                            key={i}
                                            onClick={() => handleSuggestionClick(suggestion)}
                                            className="px-4 py-3 text-sm bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-full hover:shadow-md transition-all min-h-[44px]"
                                          >
                                            {suggestion}
                                          </button>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          )}

          {!selectedSessionId && !useDirectLLM && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 text-center">
              <div className="text-4xl mb-3">⚠️</div>
              <h3 className="text-lg font-semibold text-yellow-800 mb-2">
                Ders Oturumu Seçin
              </h3>
              <p className="text-yellow-700">
                Soru sormak için önce yukarıdan bir ders oturumu seçin.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === "analytics" && (
        <div className="space-y-6">
          {selectedSessionId ? (
            <TopicAnalyticsDashboard sessionId={selectedSessionId} />
          ) : (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📊</div>
              <h3 className="text-2xl font-bold text-gray-700 mb-4">
                Analytics Dashboard
              </h3>
              <p className="text-gray-500 max-w-md mx-auto mb-8">
                Konu seviyesi analytics verilerini görüntülemek için bir ders
                oturumu seçin.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === "modules" && (
        <div className="space-y-8">
          {/* Header Section */}
          <div className="bg-gradient-to-br from-indigo-50 via-white to-purple-50 rounded-2xl p-8 border border-indigo-100">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <div className="p-4 bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-2xl shadow-lg">
                  <span className="text-3xl">🧩</span>
                </div>
                <div>
                  <h2 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                    Modül Sistemi
                  </h2>
                  <p className="text-gray-600 mt-1">
                    Konuları modüllerle organize edin ve öğrenme yolculuğunu
                    yapılandırın
                  </p>
                </div>
              </div>
            </div>

            {/* Info Card */}
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 flex items-start gap-3">
              <span className="text-2xl">💡</span>
              <div className="flex-1">
                <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                  Nasıl Çalışır?{" "}
                </span>
                <span className="text-sm text-blue-700 dark:text-blue-300">
                  Ders oturumlarındaki konuları LLM desteğiyle Türkiye MEB
                  müfredatına uygun modüllere ayırın. Bu sistem konuları
                  pedagojik sıraya göre organize ederek öğrenme sürecini
                  optimize eder.
                </span>
              </div>
            </div>
          </div>

          {/* Module Management Tabs */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
            <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-8 py-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-800">
                  Modül Yönetimi
                </h2>
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                    <span>LLM Aktif</span>
                  </span>
                  <span className="text-gray-400">•</span>
                  <span>Türkiye MEB Müfredatı</span>
                </div>
              </div>
            </div>

            <div className="p-8">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Module Extraction Panel */}
                <div className="lg:col-span-2">
                  <ModuleExtractionPanel sessionId={selectedSessionId} />
                </div>

                {/* Module Management Dashboard */}
                <div className="lg:col-span-1">
                  <ModuleManagementDashboard
                    sessionId={selectedSessionId}
                    modules={modules}
                    onModuleUpdate={async (moduleId, updates) => {
                      // Refresh modules after update
                      await fetchModules(selectedSessionId);
                    }}
                    onModuleDelete={async (moduleId) => {
                      // Refresh modules after delete
                      await fetchModules(selectedSessionId);
                    }}
                  />
                </div>
              </div>

              {/* Progress Monitoring Dashboard */}
              <div className="mt-8">
                <ModuleProgressMonitoringDashboard
                  sessionId={selectedSessionId}
                  modules={modules}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "query" && (
        <div className="space-y-6 -mt-4">
          <EducationAssistantContent />
        </div>
      )}

      {/* Create Session Modal */}
      <Modal
        isOpen={isCreateSessionModalOpen}
        onClose={() => setIsCreateSessionModalOpen(false)}
        title="Yeni Ders Oturumu Oluştur"
      >
        <form onSubmit={handleCreateSession} className="space-y-6">
          <div>
            <label htmlFor="name" className="label">
              Ders Oturumu Adı
            </label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input"
              placeholder="Örn: Biyoloji 9. Sınıf - Hücre Bölünmesi"
              required
            />
          </div>
          <div>
            <label htmlFor="description" className="label">
              Açıklama
            </label>
            <textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="input"
              placeholder="Bu oturumda işlenecek konuların kısa bir özeti..."
              rows={3}
            />
          </div>
          <div>
            <label htmlFor="category" className="label">
              Kategori
            </label>
            <select
              id="category"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="input"
            >
              <option value="research">🔬 Araştırma</option>
              <option value="general">📚 Genel</option>
              <option value="exam_prep">📝 Sınav Hazırlık</option>
              <option value="science">🔭 Fen Bilimleri</option>
              <option value="mathematics">➗ Matematik</option>
              <option value="language">🗣️ Dil</option>
              <option value="social_studies">🌍 Sosyal Bilgiler</option>
              <option value="history">🏛️ Tarih</option>
              <option value="geography">🗺️ Coğrafya</option>
              <option value="biology">🧬 Biyoloji</option>
              <option value="chemistry">⚗️ Kimya</option>
              <option value="physics">🧲 Fizik</option>
              <option value="computer_science">💻 Bilgisayar Bil.</option>
              <option value="art">🎨 Sanat</option>
              <option value="music">🎵 Müzik</option>
              <option value="physical_education">🏃‍♂️ Beden Eğitimi</option>
            </select>
          </div>
          <div className="pt-2">
            <button type="submit" className="btn btn-primary group w-full">
              <svg
                className="w-5 h-5 mr-2 group-hover:rotate-90 transition-transform duration-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
              <span>Ders Oturumu Oluştur</span>
            </button>
          </div>
        </form>
      </Modal>

      {/* Enhanced Document Upload Modal */}
      <EnhancedDocumentUploadModal
        isOpen={isDocumentUploadModalOpen}
        onClose={() => setIsDocumentUploadModalOpen(false)}
        onSuccess={handleDocumentUploadSuccess}
        onError={handleDocumentUploadError}
        onMarkdownFilesUpdate={fetchMarkdownFiles}
      />

      {/* Markdown Viewer Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        title={
          selectedFileName
            ? `${selectedFileName.replace(".md", "")}`
            : "Markdown Dosyası"
        }
        size="2xl"
      >
        <MarkdownViewer
          content={selectedFileContent}
          isLoading={isLoadingContent}
          error={modalError}
        />
      </Modal>

      {/* Source Detail Modal */}
      <Modal
        isOpen={sourceModalOpen}
        onClose={handleCloseSourceModal}
        title={
          selectedSource
            ? `Kaynak Detayları: ${
                selectedSource.metadata?.source_file ||
                selectedSource.metadata?.filename ||
                "Belge"
              }`
            : "Kaynak Detayları"
        }
        size="xl"
      >
        {selectedSource && (
          <div className="space-y-6">
            {/* Header Info */}
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-bold text-blue-800">
                  📄{" "}
                  {selectedSource.metadata?.source_file ||
                    selectedSource.metadata?.filename ||
                    "Belge Adı"}
                </h3>
                <div className="flex items-center gap-2">
                  <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-400 to-blue-500 rounded-full"
                      style={{
                        width: `${selectedSource.score * 100}%`,
                      }}
                    ></div>
                  </div>
                  <span className="text-sm font-bold text-green-600">
                    {Math.round(selectedSource.score * 100)}% Benzerlik
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                {selectedSource.metadata?.chunk_index && (
                  <div className="bg-white p-3 rounded border">
                    <div className="text-gray-600 font-medium">Bölüm</div>
                    <div className="text-gray-800 font-bold">
                      {selectedSource.metadata.chunk_index} /{" "}
                      {selectedSource.metadata?.total_chunks || "?"}
                    </div>
                  </div>
                )}

                {selectedSource.metadata?.chunk_title && (
                  <div className="bg-white p-3 rounded border">
                    <div className="text-gray-600 font-medium">Başlık</div>
                    <div className="text-gray-800 font-bold">
                      {selectedSource.metadata.chunk_title}
                    </div>
                  </div>
                )}

                <div className="bg-white p-3 rounded border">
                  <div className="text-gray-600 font-medium">
                    İçerik Uzunluğu
                  </div>
                  <div className="text-gray-800 font-bold">
                    {selectedSource.content?.length || 0} karakter
                  </div>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="space-y-3">
              <h4 className="text-md font-semibold text-gray-700 flex items-center gap-2">
                <span>📝</span>
                Tam İçerik
              </h4>
              <div className="bg-gray-50 p-4 rounded-lg border max-h-96 overflow-y-auto">
                <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed">
                  {selectedSource.content ? (
                    <div
                      dangerouslySetInnerHTML={{
                        __html: selectedSource.content.replace(/\n/g, "<br>"),
                      }}
                    />
                  ) : (
                    <div className="text-gray-500 italic">
                      İçerik bulunamadı
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Additional Metadata */}
            {(selectedSource.metadata?.page_number ||
              selectedSource.metadata?.section) && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-md font-semibold text-gray-700 mb-2">
                  📋 Belge Bilgileri
                </h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  {selectedSource.metadata?.page_number && (
                    <div>
                      <span className="text-gray-600">Sayfa:</span>
                      <span className="ml-2 font-medium">
                        {selectedSource.metadata.page_number}
                      </span>
                    </div>
                  )}
                  {selectedSource.metadata?.section && (
                    <div>
                      <span className="text-gray-600">Bölüm:</span>
                      <span className="ml-2 font-medium">
                        {selectedSource.metadata.section}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>

      <Modal
        isOpen={isRecentModalOpen}
        onClose={() => setIsRecentModalOpen(false)}
        title="Son Öğrenci Sorguları"
      >
        <div className="mb-3 flex items-center gap-2">
          <div className="flex gap-2 mb-3">
            <input
              value={recentSearch}
              onChange={(e) => setRecentSearch(e.target.value)}
              placeholder="Soru veya cevapta ara..."
              className="flex-1 border rounded-lg px-3 py-2"
            />
            <button
              onClick={() => loadRecentInteractions(1)}
              className="px-4 py-2 rounded-lg bg-blue-600 text-white"
            >
              Ara
            </button>
          </div>
          <div className="flex items-center justify-between mb-3">
            <div className="text-lg font-semibold">Son Öğrenci Sorguları</div>
            <div className="flex items-center gap-2">
              {selectedSessionId && (
                <button
                  className="px-3 py-2 text-sm rounded-md bg-red-50 text-red-700 border border-red-200 hover:bg-red-100"
                  onClick={async () => {
                    if (!selectedSessionId) return;
                    if (
                      !confirm(
                        "Bu oturumun tüm sorgularını temizlemek istiyor musunuz?"
                      )
                    )
                      return;
                    try {
                      await clearSessionInteractions(selectedSessionId);
                      await loadRecentInteractions(1);
                      await refreshSessions();
                    } catch (e: any) {
                      setError(e.message || "Sorgular temizlenemedi");
                    }
                  }}
                >
                  Bu Oturumun Sorgularını Temizle
                </button>
              )}
              <button
                onClick={() => setIsRecentModalOpen(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
          </div>
        </div>
        {recentLoading ? (
          <div>Yükleniyor...</div>
        ) : recentInteractions.length === 0 ? (
          <div className="text-gray-600">Kayıt bulunamadı.</div>
        ) : (
          <>
            <div className="space-y-3 max-h-[50vh] sm:max-h-[60vh] overflow-y-auto">
              {recentInteractions.map((it, idx) => (
                <div
                  key={it.interaction_id || idx}
                  className="p-3 bg-gray-50 rounded-lg border"
                >
                  <div className="text-xs text-gray-500 flex items-center justify-between">
                    <span>
                      {new Date(it.timestamp).toLocaleString()} • Kullanıcı:{" "}
                      {it.user_id} • Oturum: {it.session_id || "-"}
                    </span>
                    <span
                      className={
                        it.success === false ? "text-red-600" : "text-green-700"
                      }
                    >
                      {it.chain_type ? `${it.chain_type}` : ""}
                      {it.processing_time_ms != null
                        ? ` • ${it.processing_time_ms}ms`
                        : ""}
                      {it.success === false ? " • Hata" : ""}
                    </span>
                  </div>
                  {it.error_message && it.success === false ? (
                    <div className="mt-1 text-xs text-red-600">
                      Hata: {it.error_message}
                    </div>
                  ) : null}
                  <div className="mt-1">
                    <span className="font-semibold">Soru:</span> {it.query}
                  </div>
                  <div className="mt-1 text-gray-700">
                    <span className="font-semibold">Cevap:</span>{" "}
                    {it.response?.slice(0, 500) || ""}
                    {it.response && it.response.length > 500 ? "..." : ""}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 flex items-center justify-between">
              <div className="text-sm text-gray-600">
                Toplam {recentTotal} kayıt
              </div>
              <div className="flex items-center gap-2">
                <button
                  disabled={recentPage <= 1}
                  onClick={() => loadRecentInteractions(recentPage - 1)}
                  className="px-3 py-1 text-sm border rounded disabled:opacity-50"
                >
                  Önceki
                </button>
                <div className="text-sm">Sayfa {recentPage}</div>
                <button
                  disabled={recentPage * recentLimit >= recentTotal}
                  onClick={() => loadRecentInteractions(recentPage + 1)}
                  className="px-3 py-1 text-sm border rounded disabled:opacity-50"
                >
                  Sonraki
                </button>
              </div>
            </div>
          </>
        )}
      </Modal>

      {/* Profile Settings Modal */}
      {isProfileModalOpen && (
        <Modal
          isOpen={isProfileModalOpen}
          onClose={() => {
            setIsProfileModalOpen(false);
            setProfileError(null);
            setProfileSuccess(null);
            setPasswordFormData({
              old_password: "",
              new_password: "",
              confirm_password: "",
            });
          }}
          title="Profil Ayarları"
        >
          <div className="space-y-6">
            {profileSuccess && (
              <div className="bg-green-50 border border-green-200 text-green-800 p-4 rounded-lg">
                {profileSuccess}
              </div>
            )}
            {profileError && (
              <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg">
                {profileError}
              </div>
            )}

            {/* Profile Information */}
            {userProfile && (
              <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-sm text-gray-600 mb-2">
                  Kullanıcı Bilgileri
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-500">Rol</div>
                    <div className="font-semibold">{userProfile.role_name}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Hesap Durumu</div>
                    <div className="font-semibold">
                      {userProfile.is_active ? "Aktif" : "Pasif"}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Edit Profile Form */}
            <div>
              <h3 className="text-lg font-semibold mb-4 text-gray-800">
                Profil Bilgileri
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Kullanıcı Adı
                  </label>
                  <input
                    type="text"
                    value={profileFormData.username}
                    onChange={(e) =>
                      setProfileFormData({
                        ...profileFormData,
                        username: e.target.value,
                      })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    placeholder="Kullanıcı adınız"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    3-50 karakter, sadece harf, rakam, alt çizgi ve tire
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    E-posta
                  </label>
                  <input
                    type="email"
                    value={profileFormData.email}
                    onChange={(e) =>
                      setProfileFormData({
                        ...profileFormData,
                        email: e.target.value,
                      })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Ad
                    </label>
                    <input
                      type="text"
                      value={profileFormData.first_name}
                      onChange={(e) =>
                        setProfileFormData({
                          ...profileFormData,
                          first_name: e.target.value,
                        })
                      }
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Soyad
                    </label>
                    <input
                      type="text"
                      value={profileFormData.last_name}
                      onChange={(e) =>
                        setProfileFormData({
                          ...profileFormData,
                          last_name: e.target.value,
                        })
                      }
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                  </div>
                </div>
                <button
                  onClick={async () => {
                    try {
                      setProfileLoading(true);
                      setProfileError(null);
                      setProfileSuccess(null);
                      const updated = await updateProfile(profileFormData);
                      setUserProfile(updated);
                      setProfileSuccess("Profil başarıyla güncellendi!");
                      setTimeout(() => {
                        setIsProfileModalOpen(false);
                        setProfileSuccess(null);
                      }, 2000);
                    } catch (e: any) {
                      setProfileError(e.message || "Profil güncellenemedi");
                    } finally {
                      setProfileLoading(false);
                    }
                  }}
                  disabled={profileLoading}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white px-4 py-3 rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all"
                >
                  {profileLoading ? "Güncelleniyor..." : "Profil Güncelle"}
                </button>
              </div>
            </div>

            {/* Change Password Form */}
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">
                Şifre Değiştir
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Eski Şifre
                  </label>
                  <input
                    type="password"
                    value={passwordFormData.old_password}
                    onChange={(e) =>
                      setPasswordFormData({
                        ...passwordFormData,
                        old_password: e.target.value,
                      })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Yeni Şifre
                  </label>
                  <input
                    type="password"
                    value={passwordFormData.new_password}
                    onChange={(e) =>
                      setPasswordFormData({
                        ...passwordFormData,
                        new_password: e.target.value,
                      })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Yeni Şifre (Tekrar)
                  </label>
                  <input
                    type="password"
                    value={passwordFormData.confirm_password}
                    onChange={(e) =>
                      setPasswordFormData({
                        ...passwordFormData,
                        confirm_password: e.target.value,
                      })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
                {passwordFormData.new_password &&
                  passwordFormData.confirm_password &&
                  passwordFormData.new_password !==
                    passwordFormData.confirm_password && (
                    <div className="text-red-600 text-sm">
                      Şifreler eşleşmiyor
                    </div>
                  )}
                <button
                  onClick={async () => {
                    if (
                      passwordFormData.new_password !==
                      passwordFormData.confirm_password
                    ) {
                      setProfileError("Şifreler eşleşmiyor");
                      return;
                    }
                    if (passwordFormData.new_password.length < 6) {
                      setProfileError("Yeni şifre en az 6 karakter olmalıdır");
                      return;
                    }
                    try {
                      setProfileLoading(true);
                      setProfileError(null);
                      setProfileSuccess(null);
                      await changePassword(
                        passwordFormData.old_password,
                        passwordFormData.new_password
                      );
                      setProfileSuccess("Şifre başarıyla değiştirildi!");
                      setPasswordFormData({
                        old_password: "",
                        new_password: "",
                        confirm_password: "",
                      });
                      setTimeout(() => {
                        setProfileSuccess(null);
                      }, 3000);
                    } catch (e: any) {
                      setProfileError(e.message || "Şifre değiştirilemedi");
                    } finally {
                      setProfileLoading(false);
                    }
                  }}
                  disabled={
                    profileLoading ||
                    passwordFormData.new_password !==
                      passwordFormData.confirm_password
                  }
                  className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white px-4 py-3 rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {profileLoading ? "Değiştiriliyor..." : "Şifreyi Değiştir"}
                </button>
              </div>
            </div>
          </div>
        </Modal>
      )}

      {/* Feedback Modal - APRAG (only when enabled) */}
      {apragEnabled && selectedInteractionForFeedback && (
        <FeedbackModal
          isOpen={feedbackModalOpen}
          onClose={() => {
            setFeedbackModalOpen(false);
            setSelectedInteractionForFeedback(null);
          }}
          onSubmit={async (feedback: FeedbackData) => {
            try {
              // Get user info
              const userId =
                user?.id?.toString() || user?.username || "anonymous";
              const sessionId =
                selectedSessionId ||
                localStorage.getItem("currentSessionId") ||
                "";

              if (!sessionId) {
                throw new Error("Session ID bulunamadı");
              }

              const feedbackData: FeedbackCreate = {
                interaction_id: feedback.interaction_id,
                user_id: userId,
                session_id: sessionId,
                understanding_level: feedback.understanding_level,
                answer_adequacy: feedback.answer_adequacy,
                satisfaction_level: feedback.satisfaction_level,
                difficulty_level: feedback.difficulty_level,
                topic_understood: feedback.topic_understood,
                answer_helpful: feedback.answer_helpful,
                needs_more_explanation: feedback.needs_more_explanation,
                comment: feedback.comment,
              };

              await submitFeedback(feedbackData);
              setSuccess("Geri bildiriminiz için teşekkürler! 🎉");
              setTimeout(() => setSuccess(null), 5000);
            } catch (e: any) {
              setError(e.message || "Geri bildirim gönderilemedi");
            }
          }}
          interactionId={parseInt(selectedInteractionForFeedback.interactionId)}
          query={selectedInteractionForFeedback.query}
          answer={selectedInteractionForFeedback.answer}
        />
      )}

      {/* Category Management Modal */}
      {isCategoryModalOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          onClick={() => setIsCategoryModalOpen(false)}
        >
          <div
            className="bg-white rounded-xl shadow-2xl w-full max-w-md max-h-[80vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6 pb-0">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-900">
                  Markdown Kategorileri
                </h3>
                <button
                  onClick={() => setIsCategoryModalOpen(false)}
                  className="text-gray-500 hover:text-gray-700"
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
                      strokeWidth="2"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>

              {/* Add New Category */}
              <div className="flex gap-2 mb-6">
                <input
                  type="text"
                  value={newCategoryName}
                  onChange={(e) => setNewCategoryName(e.target.value)}
                  placeholder="Yeni kategori adı"
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  onKeyDown={(e) => e.key === "Enter" && handleCreateCategory()}
                />
                <button
                  onClick={handleCreateCategory}
                  disabled={!newCategoryName.trim() || isCategoryLoading}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isCategoryLoading ? "Ekleniyor..." : "Ekle"}
                </button>
              </div>

              {/* Categories List */}
              <div className="max-h-[50vh] overflow-y-auto">
                {categories.length === 0 ? (
                  <p className="text-gray-500 text-center py-4">
                    Henüz kategori bulunmuyor
                  </p>
                ) : (
                  <ul className="divide-y divide-gray-200">
                    {categories.map((category) => (
                      <li
                        key={category.id}
                        className="py-3 px-1 flex items-center justify-between hover:bg-gray-50"
                      >
                        <span className="text-gray-800">{category.name}</span>
                        <button
                          onClick={() => handleDeleteCategory(category.id)}
                          disabled={isCategoryLoading}
                          className="text-red-500 hover:text-red-700 disabled:opacity-50"
                          title="Kategoriyi sil"
                        >
                          <svg
                            className="w-5 h-5"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 bg-gray-50 rounded-b-xl flex justify-end border-t border-gray-200">
              <button
                onClick={() => setIsCategoryModalOpen(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Kapat
              </button>
            </div>
          </div>
        </div>
      )}
    </TeacherLayout>
  );
}
