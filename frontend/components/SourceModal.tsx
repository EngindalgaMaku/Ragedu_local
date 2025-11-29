"use client";

import React, { useState, useEffect, useMemo } from "react";
import { X, FileText, Hash, Loader2 } from "lucide-react";
import { RAGSource, getApiUrl } from "@/lib/api";
import ReactMarkdown from "react-markdown";

// Component to format content (JSON, Markdown, or plain text)
function FormattedContent({ content }: { content: any }) {
  if (!content) return <p className="text-gray-500">İçerik bulunamadı</p>;
  
  // If content is an object, try to format it nicely
  if (typeof content === 'object' && content !== null) {
    // Check if it's a KB structure
    if (content.topic_summary || content.key_concepts || content.learning_objectives) {
      return (
        <div className="space-y-6">
          {content.topic_summary && (
            <div>
              <h3 className="text-lg font-bold text-gray-900 mb-3">📝 Özet</h3>
              <div className="text-gray-800 leading-relaxed break-words">
                <ReactMarkdown 
                  components={{
                    p: ({ children }) => <p className="mb-3">{children}</p>,
                  }}
                >
                  {String(content.topic_summary)}
                </ReactMarkdown>
              </div>
            </div>
          )}
          
          {content.key_concepts && Array.isArray(content.key_concepts) && content.key_concepts.length > 0 && (
            <div>
              <h3 className="text-lg font-bold text-gray-900 mb-3">🔑 Anahtar Kavramlar</h3>
              <div className="space-y-3">
                {content.key_concepts.map((concept: any, idx: number) => (
                  <div key={idx} className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded">
                    {typeof concept === 'object' && concept !== null ? (
                      <div>
                        {concept.term && (
                          <div className="font-semibold text-blue-900 mb-1">
                            {concept.term}
                            {concept.importance && (
                              <span className="ml-2 text-xs font-normal text-blue-700">
                                ({concept.importance})
                              </span>
                            )}
                          </div>
                        )}
                        {concept.definition && (
                          <div className="text-blue-800 text-sm mt-1">
                            <ReactMarkdown>{concept.definition}</ReactMarkdown>
                          </div>
                        )}
                        {concept.category && (
                          <div className="text-xs text-blue-600 mt-1">
                            Kategori: {concept.category}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-blue-800">{String(concept)}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Öğrenme Hedefleri ve Örnekler bölümleri kaldırıldı - sadece Özet ve Anahtar Kavramlar gösteriliyor */}
        </div>
      );
    }
    
    // If it's a plain object, try to format as JSON
    try {
      const jsonStr = JSON.stringify(content, null, 2);
      return (
        <pre className="bg-gray-50 border border-gray-200 rounded-lg p-4 overflow-x-auto text-sm">
          <code className="text-gray-800">{jsonStr}</code>
        </pre>
      );
    } catch {
      return <p className="text-gray-800">{String(content)}</p>;
    }
  }
  
  // If content is a string, try to detect if it's JSON or Markdown
  const contentStr = String(content).trim();
  
  // Try to parse as JSON
  if (contentStr.startsWith('{') || contentStr.startsWith('[')) {
    try {
      const parsed = JSON.parse(contentStr);
      const jsonStr = JSON.stringify(parsed, null, 2);
      return (
        <pre className="bg-gray-50 border border-gray-200 rounded-lg p-4 overflow-x-auto text-sm">
          <code className="text-gray-800">{jsonStr}</code>
        </pre>
      );
    } catch {
      // Not valid JSON, treat as markdown/text
    }
  }
  
  // Treat as Markdown or plain text - FULL CONTENT, NO TRUNCATION
  return (
    <div className="text-gray-800 leading-relaxed">
      <ReactMarkdown 
        components={{
          // Ensure all content is displayed, no truncation
          p: ({ children }) => <p className="mb-3">{children}</p>,
          h1: ({ children }) => <h1 className="text-2xl font-bold mb-4 mt-6">{children}</h1>,
          h2: ({ children }) => <h2 className="text-xl font-bold mb-3 mt-5">{children}</h2>,
          h3: ({ children }) => <h3 className="text-lg font-bold mb-2 mt-4">{children}</h3>,
          ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
          li: ({ children }) => <li className="ml-2">{children}</li>,
        }}
      >
        {contentStr}
      </ReactMarkdown>
    </div>
  );
}

interface SourceModalProps {
  source: RAGSource | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function SourceModal({ source, isOpen, onClose }: SourceModalProps) {
  const [kbData, setKbData] = useState<any>(null);
  const [loadingKB, setLoadingKB] = useState(false);
  const [kbError, setKbError] = useState<string | null>(null);

  const rawType = source
    ? ((source.metadata as any)?.source_type ||
       (source.metadata as any)?.source ||
       "unknown")
    : "unknown";

  const isKnowledgeBase = rawType === "structured_kb" || rawType === "knowledge_base";
  const topicId = source ? (source.metadata as any)?.topic_id : undefined;

  const sourceTypeLabel =
    rawType === "direct_qa" || rawType === "qa_pair" || rawType === "question_answer"
      ? "Soru Bankası (Otomatik QA)"
      : isKnowledgeBase
      ? "Bilgi Tabanı Özeti"
      : rawType === "vector_search"
      ? "Döküman Parçası (RAG)"
      : "Döküman";

  // Load knowledge base data if it's a KB source with topic_id
  useEffect(() => {
    // Only load if modal is open, it's a KB source, has topicId, and we haven't loaded yet
    if (isOpen && isKnowledgeBase && topicId && !kbData && !loadingKB) {
      setLoadingKB(true);
      setKbError(null);
      
      let cancelled = false;
      
      fetch(`${getApiUrl()}/aprag/knowledge/kb/${topicId}`)
        .then(async (res) => {
          if (cancelled) return;
          if (!res.ok) {
            throw new Error("Bilgi tabanı yüklenemedi");
          }
          const data = await res.json();
          if (data.success && data.knowledge_base) {
            if (!cancelled) {
              setKbData(data.knowledge_base);
            }
          } else {
            throw new Error("Bilgi tabanı bulunamadı");
          }
        })
        .catch((err) => {
          if (cancelled) return;
          console.error("KB load error:", err);
          setKbError(err.message || "Bilgi tabanı yüklenemedi");
        })
        .finally(() => {
          if (!cancelled) {
            setLoadingKB(false);
          }
        });
      
      return () => {
        cancelled = true;
      };
    }
  }, [isOpen, isKnowledgeBase, topicId]); // Removed kbData and loadingKB from dependencies

  // Reset KB data when modal closes or source changes
  useEffect(() => {
    if (!isOpen) {
      setKbData(null);
      setKbError(null);
      setLoadingKB(false);
    }
  }, [isOpen]);

  // Reset KB data when source changes (different source clicked)
  useEffect(() => {
    if (source) {
      setKbData(null);
      setKbError(null);
      setLoadingKB(false);
    }
  }, [source]);

  // Helper functions to format KB content
  const formatConcept = React.useCallback((concept: any): string => {
    if (typeof concept === 'string') return concept;
    if (concept.term) {
      return `${concept.term}${concept.definition ? `: ${concept.definition}` : ''}`;
    }
    if (concept.concept) {
      return `${concept.concept}${concept.definition ? `: ${concept.definition}` : ''}`;
    }
    if (concept.name) {
      return concept.name;
    }
    return JSON.stringify(concept);
  }, []);

  const formatObjective = React.useCallback((obj: any): string => {
    if (typeof obj === 'string') return obj;
    if (obj.objective) {
      const level = obj.level ? ` [${obj.level}]` : '';
      return `${obj.objective}${level}`;
    }
    if (obj.description) return obj.description;
    return JSON.stringify(obj);
  }, []);

  const formatExample = React.useCallback((ex: any): string => {
    if (typeof ex === 'string') return ex;
    if (ex.example) return ex.example;
    if (ex.description) return ex.description;
    if (ex.text) return ex.text;
    return JSON.stringify(ex);
  }, []);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen || !source) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
      onClick={handleBackdropClick}
    >
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6" />
            <div>
              <h3 className="text-lg font-semibold">Kaynak Detayları</h3>
              <p className="text-sm text-blue-100">
                {sourceTypeLabel}
                {(source.metadata as any)?.topic_title
                  ? ` • ${(source.metadata as any).topic_title}`
                  : source.metadata?.filename || source.metadata?.source_file
                  ? ` • ${source.metadata.filename || source.metadata.source_file}`
                  : ""}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors"
            title="Kapat"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Loading KB Indicator */}
          {isKnowledgeBase && topicId && loadingKB && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center gap-3 mb-4">
              <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
              <span className="text-sm text-blue-800">Bilgi tabanı yükleniyor...</span>
            </div>
          )}

          {/* KB Error */}
          {isKnowledgeBase && topicId && kbError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
              <p className="text-sm text-red-800">{kbError}</p>
            </div>
          )}

          {/* Bilgi Tabanı İçeriği - Sadece İçerik, Tam ve Formatlanmış, Scroll Edilebilir */}
          {isKnowledgeBase && (kbData || source.content) ? (
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="max-h-[calc(90vh-250px)] overflow-y-auto pr-2">
                <div className="prose prose-sm max-w-none break-words">
                  <FormattedContent content={kbData || source.content} />
                </div>
              </div>
            </div>
          ) : !isKnowledgeBase ? (
            <>
              {/* Metadata Tags - Sadece non-KB için */}
              <div className="flex flex-wrap gap-2 mb-4">
                <span className="inline-flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
                  <Hash className="w-3 h-3" />
                  Skor: {(source.score * 100).toFixed(1)}%
                </span>
                
                {source.metadata?.chunk_index !== undefined && (
                  <span className="inline-flex items-center gap-1 px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">
                    Parça {source.metadata.chunk_index + 1}
                    {source.metadata?.total_chunks && ` / ${source.metadata.total_chunks}`}
                  </span>
                )}
                
                {source.metadata?.page_number && (
                  <span className="inline-flex items-center px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                    Sayfa {source.metadata.page_number}
                  </span>
                )}
                
                {source.metadata?.section && (
                  <span className="inline-flex items-center px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-sm font-medium">
                    {source.metadata.section}
                  </span>
                )}
              </div>

              {/* Chunk Title */}
              {source.metadata?.chunk_title && (
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
                  <h4 className="text-sm font-semibold text-gray-700 mb-1">Bölüm Başlığı</h4>
                  <p className="text-gray-900 font-medium">{source.metadata.chunk_title}</p>
                </div>
              )}

              {/* Source Content */}
              <div className="bg-white border border-gray-200 rounded-lg p-6 space-y-6">
                <h4 className="text-sm font-semibold text-gray-700 mb-3">İçerik</h4>
            
                {/* Regular Content (non-KB sources) */}
                <div className="prose prose-sm max-w-none">
                <div className="text-gray-800 leading-relaxed whitespace-pre-wrap">
                  {(() => {
                    if (!source) return "İçerik bulunamadı";
                    
                    const base = (source.content || "").toString().trim();
                    if (base.length > 0) {
                      return base.split('\n').map((line, idx) => {
                        if (line.startsWith('## ')) {
                          return <h3 key={idx} className="text-lg font-bold mt-4 mb-2 text-gray-900">{line.substring(3)}</h3>;
                        } else if (line.startsWith('- ')) {
                          return <li key={idx} className="ml-4">{line.substring(2)}</li>;
                        } else if (line.match(/^\d+\.\s/)) {
                          return <p key={idx} className="mb-2">{line}</p>;
                        } else if (line.trim() === '') {
                          return <br key={idx} />;
                        } else {
                          return <p key={idx} className="mb-2">{line}</p>;
                        }
                      });
                    }

                    // QA kaynağıysa
                    const meta: any = source.metadata || {};
                    if (
                      (rawType === "direct_qa" || rawType === "qa_pair" || rawType === "question_answer") &&
                      (meta.qa_question || meta.qa_answer)
                    ) {
                      return (
                        <div className="space-y-3">
                          <div className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded">
                            <div className="font-semibold text-blue-900 mb-1">Soru:</div>
                            <div className="text-blue-800">{meta.qa_question || "-"}</div>
                          </div>
                          <div className="bg-green-50 border-l-4 border-green-500 p-3 rounded">
                            <div className="font-semibold text-green-900 mb-1">Cevap:</div>
                            <div className="text-green-800">{meta.qa_answer || "-"}</div>
                          </div>
                        </div>
                      );
                    }

                    return <p className="text-gray-500">İçerik bulunamadı</p>;
                  })()}
                </div>
              </div>
            </div>

              {/* Additional Metadata - Sadece non-KB için */}
              {!isKnowledgeBase && source.metadata && Object.keys(source.metadata).length > 0 && (
                <details className="bg-gray-50 border border-gray-200 rounded-lg p-4 mt-4">
              <summary className="text-sm font-semibold text-gray-700 cursor-pointer hover:text-gray-900">
                Ek Bilgiler
              </summary>
              <div className="mt-3 space-y-2 text-sm">
                {Object.entries(source.metadata).map(([key, value]) => {
                  // Skip fields that are already displayed or are part of KB content
                  if (
                    [
                      "filename",
                      "source_file",
                      "chunk_title",
                      "chunk_index",
                      "total_chunks",
                      "page_number",
                      "section",
                      "qa_question",
                      "qa_answer",
                      "concepts", // Already displayed in KB content section
                      "objectives", // Already displayed in KB content section
                      "examples", // Already displayed in KB content section
                      "topic_id", // Already used to load KB
                      "topic_title", // Already displayed in header
                      "quality_score", // Can show but formatted
                    ].includes(key)
                  ) {
                    return null; // Already displayed above or in KB sections
                  }
                  
                  // Türkçe anahtar ismi
                  const label =
                    key === "source_type"
                      ? "Kaynak Türü"
                      : key === "qa_similarity"
                      ? "Benzerlik"
                      : key === "quality_score"
                      ? "Kalite Skoru"
                      : key;
                  
                  // Format value nicely
                  let displayValue: React.ReactNode;
                  if (value === null || value === undefined) {
                    displayValue = "-";
                  } else if (typeof value === "boolean") {
                    displayValue = value ? "Evet" : "Hayır";
                  } else if (typeof value === "number") {
                    displayValue = key === "quality_score" 
                      ? `${(value * 100).toFixed(0)}%`
                      : String(value);
                  } else if (Array.isArray(value)) {
                    // For arrays, show count or first few items
                    if (value.length === 0) {
                      displayValue = "Yok";
                    } else if (value.length <= 3) {
                      displayValue = value.join(", ");
                    } else {
                      displayValue = `${value.length} öğe (${value.slice(0, 3).join(", ")}, ...)`;
                    }
                  } else if (typeof value === "object") {
                    // For objects, show a summary instead of full JSON
                    const keys = Object.keys(value);
                    if (keys.length === 0) {
                      displayValue = "Boş";
                    } else if (keys.length <= 3) {
                      displayValue = keys.join(", ");
                    } else {
                      displayValue = `${keys.length} alan (${keys.slice(0, 3).join(", ")}, ...)`;
                    }
                  } else {
                    displayValue = String(value);
                  }
                  
                  return (
                    <div key={key} className="flex items-start gap-2">
                      <span className="font-medium text-gray-600 min-w-[120px]">
                        {label}:
                      </span>
                      <span className="text-gray-800 flex-1 break-words">
                        {displayValue}
                      </span>
                    </div>
                  );
                })}
              </div>
            </details>
              )}
            </>
          ) : null}
        </div>

        {/* Footer */}
        <div className="bg-gray-50 border-t border-gray-200 px-6 py-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
          >
            Kapat
          </button>
        </div>
      </div>
    </div>
  );
}

