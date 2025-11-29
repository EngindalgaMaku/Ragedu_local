"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, Terminal, Copy, Download } from "lucide-react";

interface QueryDebugPanelProps {
  debugData: {
    adaptiveQueryResult?: any;
    personalizationData?: any;
    lastQuery?: string | null;
  } | null;
}

export default function QueryDebugPanel({ debugData }: QueryDebugPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!debugData || !debugData.adaptiveQueryResult) {
    return (
      <div className="border-t border-gray-200 bg-gray-50 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Terminal className="w-4 h-4" />
            <span>Query Debug Panel</span>
          </div>
          <span className="text-xs text-gray-400">
            Henüz sorgu yapılmadı. İlk sorgu sonrası detaylar burada görünecek.
          </span>
        </div>
      </div>
    );
  }

  const generateDebugReport = (): string => {
    if (!debugData || !debugData.adaptiveQueryResult) {
      return "Debug data not available.";
    }
    
    const data = debugData.adaptiveQueryResult;
    const hybridDebug = data.hybrid_rag_debug || {};
    const personalization = debugData.personalizationData || {};
    
    // Try to get comprehensive_debug from multiple sources
    let comprehensiveDebug = personalization?.comprehensive_debug;
    if (!comprehensiveDebug && hybridDebug?.comprehensive_debug) {
        comprehensiveDebug = hybridDebug.comprehensive_debug;
    }
    if (!comprehensiveDebug && data?.comprehensive_debug) {
        comprehensiveDebug = data.comprehensive_debug;
    }
    
    let report = "";
    report += "=".repeat(80) + "\n";
    report += " QUERY DEBUG REPORT - TAM SÜREÇ DÖKÜMÜ\n";
    report += "=".repeat(80) + "\n\n";
    
    // 1. İSTEK PARAMETRELERİ
    report += "[1] İSTEK PARAMETRELERİ\n";
    report += "-".repeat(80) + "\n";
    if (hybridDebug.request_params) {
      const params = hybridDebug.request_params;
      report += `Session ID    : ${params.session_id || "N/A"}\n`;
      report += `Query         : ${params.query || "N/A"}\n`;
      report += `Top K         : ${params.top_k || "N/A"}\n`;
      report += `Use KB        : ${params.use_kb ? "Yes" : "No"}\n`;
      report += `Use QA Pairs  : ${params.use_qa_pairs ? "Yes" : "No"}\n`;
      report += `Use Rerank    : ${params.use_crag ? "Yes" : "No"}\n`;
      report += `Model         : ${params.model || "N/A"}\n`;
      report += `Embedding Model: ${params.embedding_model || "N/A"}\n`;
      report += `Max Tokens    : ${params.max_tokens || "N/A"}\n`;
      report += `Temperature  : ${params.temperature || "N/A"}\n`;
      report += `Max Context   : ${params.max_context_chars || "N/A"} chars\n`;
    } else {
      report += "İstek parametreleri bulunamadı.\n";
    }
    report += "\n";
    
    // 2. SESSION SETTINGS
    report += "[2] SESSION RAG SETTINGS\n";
    report += "-".repeat(80) + "\n";
    if (hybridDebug.session_settings) {
      const settings = hybridDebug.session_settings;
      report += `Effective Model        : ${settings.effective_model || "N/A"}\n`;
      report += `Effective Embedding    : ${settings.effective_embedding_model || "N/A"}\n`;
      if (settings.session_rag_settings) {
        report += `Session Model          : ${settings.session_rag_settings.model || "N/A"}\n`;
        report += `Session Embedding      : ${settings.session_rag_settings.embedding_model || "N/A"}\n`;
      }
    } else {
      report += "Session settings bulunamadı.\n";
    }
    report += "\n";
    
    // 3. RETRIEVAL AŞAMALARI
    report += "[3] RETRIEVAL AŞAMALARI\n";
    report += "-".repeat(80) + "\n";
    if (hybridDebug.retrieval_stages) {
      const stages = hybridDebug.retrieval_stages;
      
      // Topic Classification
      if (stages.topic_classification) {
        report += "\n[3.1] Topic Classification\n";
        report += `  Matched Topics    : ${stages.topic_classification.topics_count || 0}\n`;
        const topicConfidence = Number(stages.topic_classification?.confidence ?? 0);
        report += `  Confidence        : ${(topicConfidence * 100).toFixed(1)}%\n`;
        if (stages.topic_classification.matched_topics && stages.topic_classification.matched_topics.length > 0) {
          stages.topic_classification.matched_topics.forEach((topic: any, idx: number) => {
            const confidence = Number(topic?.confidence ?? 0);
            report += `  Topic ${idx + 1}        : ${topic?.topic_title || "N/A"} (confidence: ${(confidence * 100).toFixed(1)}%)\n`;
          });
        }
      }
      
      // Chunk Retrieval
      if (stages.chunk_retrieval) {
        report += "\n[3.2] Chunk Retrieval\n";
        report += `  Chunks Retrieved  : ${stages.chunk_retrieval.chunks_retrieved || 0}\n`;
        if (stages.chunk_retrieval.chunks && stages.chunk_retrieval.chunks.length > 0) {
          stages.chunk_retrieval.chunks.forEach((chunk: any, idx: number) => {
            const score = Number(chunk?.score ?? 0);
            report += `  Chunk ${idx + 1}       : Score=${(score * 100).toFixed(1)}%, Preview="${chunk?.content_preview || ""}"\n`;
          });
        }
      }
      
      // QA Matching
      if (stages.qa_matching) {
        report += "\n[3.3] QA Pairs Matching\n";
        report += `  QA Pairs Matched  : ${stages.qa_matching.qa_pairs_matched || 0}\n`;
        if (stages.qa_matching.qa_pairs && stages.qa_matching.qa_pairs.length > 0) {
          stages.qa_matching.qa_pairs.forEach((qa: any, idx: number) => {
            const similarity = Number(qa?.similarity ?? 0);
            report += `  QA ${idx + 1}          : Similarity=${(similarity * 100).toFixed(1)}%, Q="${qa?.question || ""}"\n`;
          });
        }
      }
      
      // KB Retrieval
      if (stages.kb_retrieval) {
        report += "\n[3.4] Knowledge Base Retrieval\n";
        report += `  KB Items Retrieved: ${stages.kb_retrieval.kb_items_retrieved || 0}\n`;
        if (stages.kb_retrieval.kb_items && stages.kb_retrieval.kb_items.length > 0) {
          stages.kb_retrieval.kb_items.forEach((kb: any, idx: number) => {
            const relevanceScore = Number(kb?.relevance_score ?? 0);
            report += `  KB ${idx + 1}          : Topic="${kb?.topic_title || ""}", Relevance=${(relevanceScore * 100).toFixed(1)}%\n`;
          });
        }
      }
      
      // Merged Results
      if (stages.merged_results) {
        report += "\n[3.5] Merged Results\n";
        report += `  Total Merged      : ${stages.merged_results.total_merged || 0}\n`;
        if (stages.merged_results.by_source) {
          report += `  By Source:\n`;
          report += `    - Chunks        : ${stages.merged_results.by_source.chunks || 0}\n`;
          report += `    - KB            : ${stages.merged_results.by_source.kb || 0}\n`;
          report += `    - QA Pairs      : ${stages.merged_results.by_source.qa_pairs || 0}\n`;
        }
      }
      
      // Context Built
      if (stages.context_built) {
        report += "\n[3.6] Context Built\n";
        report += `  Context Length    : ${stages.context_built.context_length || 0} chars\n`;
        report += `  Context Preview   : ${stages.context_built.context_preview || "N/A"}\n`;
      }
    } else {
      report += "Retrieval aşamaları bulunamadı.\n";
    }
    report += "\n";
    
    // 4. RERANK SCORES
    report += "[4] RERANK SCORES\n";
    report += "-".repeat(80) + "\n";
    if (hybridDebug.rerank_scores) {
      const rerank = hybridDebug.rerank_scores;
      const maxScore = Number(rerank?.max_score ?? 0);
      report += `Max Score      : ${maxScore > 0 ? maxScore.toFixed(4) : "N/A"}\n`;
      const avgScore = Number(rerank?.avg_score ?? 0);
      report += `Avg Score      : ${avgScore > 0 ? avgScore.toFixed(4) : "N/A"}\n`;
      report += `Reranker Type  : ${rerank.reranker_type || "N/A"}\n`;
      report += `Documents Reranked: ${rerank.documents_reranked || 0}\n`;
      if (rerank.scores && rerank.scores.length > 0) {
        report += `\n[4.1] Individual Scores\n`;
        rerank.scores.slice(0, 5).forEach((score: any, idx: number) => {
          const normalizedScore = Number(score?.normalized_score ?? 0);
          const rawScore = Number(score?.score ?? 0);
          report += `  Doc ${idx + 1}        : Normalized=${(normalizedScore * 100).toFixed(1)}%, Raw=${rawScore.toFixed(4)}\n`;
        });
      }
    } else {
      report += "Rerank yapılmadı.\n";
    }
    report += "\n";
    
    // 5. LLM GENERATION
    report += "[5] LLM GENERATION\n";
    report += "-".repeat(80) + "\n";
    if (hybridDebug.llm_generation) {
      const llm = hybridDebug.llm_generation;
      report += `Model          : ${llm.model || "N/A"}\n`;
      report += `Max Tokens     : ${llm.max_tokens || "N/A"}\n`;
      report += `Temperature    : ${llm.temperature || "N/A"}\n`;
      report += `Context Length : ${llm.context_length?.toLocaleString() || "N/A"} chars\n`;
      report += `Query Length   : ${llm.query_length?.toLocaleString() || "N/A"} chars\n`;
      report += `Prompt Length  : ${llm.prompt_length?.toLocaleString() || "N/A"} chars\n`;
      report += `LLM URL        : ${llm.llm_url || "N/A"}\n`;
      const llmDuration = Number(llm?.llm_duration_ms ?? 0);
      report += `LLM Duration   : ${llmDuration > 0 ? llmDuration.toFixed(2) : "N/A"} ms\n`;
      report += `Response Status: ${llm.llm_response_status || "N/A"}\n`;
      
      if (llm.prompt) {
        report += "\n[5.1] LLM PROMPT (Full System Prompt)\n";
        report += "-".repeat(80) + "\n";
        report += llm.prompt + "\n";
      }
      
      if (llm.raw_response) {
        report += "\n[5.2] LLM RAW RESPONSE\n";
        report += "-".repeat(80) + "\n";
        report += JSON.stringify(llm.raw_response, null, 2) + "\n";
      }
    } else {
      report += "LLM generation detayları bulunamadı.\n";
    }
    report += "\n";
    
    // 6. RESPONSE
    report += "[6] RESPONSE\n";
    report += "-".repeat(80) + "\n";
    if (hybridDebug.response) {
      const resp = hybridDebug.response;
      report += `Answer Length  : ${resp.answer_length?.toLocaleString() || "N/A"} chars\n`;
      report += `Confidence     : ${resp.confidence || "N/A"}\n`;
      report += `Strategy       : ${resp.retrieval_strategy || "N/A"}\n`;
      report += `Answer Preview : ${resp.answer_preview || "N/A"}\n`;
    }
    report += "\n";
    
    // 7. TIMING
    report += "[7] TIMING\n";
    report += "-".repeat(80) + "\n";
    if (hybridDebug.timing) {
      const timing = hybridDebug.timing;
      const totalMs = Number(timing?.total_processing_ms ?? 0);
      const retrievalMs = Number(timing?.retrieval_time_ms ?? 0);
      report += `Total Processing: ${totalMs > 0 ? totalMs.toFixed(2) : "N/A"} ms (${totalMs > 0 ? (totalMs / 1000).toFixed(2) : "N/A"} s)\n`;
      report += `Retrieval Time : ${retrievalMs > 0 ? retrievalMs.toFixed(2) : "N/A"} ms\n`;
      if (totalMs > 0 && retrievalMs > 0) {
        const llmTime = totalMs - retrievalMs;
        report += `LLM Time       : ${llmTime.toFixed(2)} ms\n`;
      }
    }
    report += "\n";
    
    // 8. SOURCES SUMMARY
    report += "[8] SOURCES SUMMARY\n";
    report += "-".repeat(80) + "\n";
    if (hybridDebug.sources_summary) {
      const sources = hybridDebug.sources_summary;
      report += `Chunks Used    : ${sources.chunks || 0}\n`;
      report += `KB Items Used  : ${sources.kb || 0}\n`;
      report += `QA Pairs Used  : ${sources.qa_pairs || 0}\n`;
    }
    report += "\n";
    
    // 9. APRAG PERSONALIZATION (if available)
    // comprehensiveDebug is already extracted above
    if (data.pedagogical_context || personalization.personalization_factors || comprehensiveDebug || hybridDebug?.comprehensive_debug) {
      report += "[9] APRAG PERSONALIZATION & ADAPTIVE LEARNING\n";
      report += "-".repeat(80) + "\n";
      
      // Use comprehensiveDebug from any source
      const finalComprehensiveDebug = comprehensiveDebug || hybridDebug?.comprehensive_debug;
      
      // 9.0 Feature Flags & Session Settings
      if (finalComprehensiveDebug?.feature_flags) {
        report += "\n[9.0] Feature Flags & Session Settings\n";
        const ff = comprehensiveDebug.feature_flags;
        report += `Eğitsel-KBRAG Enabled: ${ff.egitsel_kbrag_enabled ? "Yes" : "No"}\n`;
        report += `Session Settings Loaded: ${ff.session_settings_loaded ? "Yes" : "No"}\n`;
        if (ff.components_active) {
          report += `Components Active:\n`;
          Object.entries(ff.components_active).forEach(([key, value]) => {
            report += `  - ${key}: ${value ? "✅ Enabled" : "❌ Disabled"}\n`;
          });
        }
        if (finalComprehensiveDebug?.session_settings) {
          report += `Session Settings:\n`;
          const ss = finalComprehensiveDebug.session_settings;
          report += `  - enable_cacs: ${ss.enable_cacs !== undefined ? ss.enable_cacs : "N/A"}\n`;
          report += `  - enable_zpd: ${ss.enable_zpd !== undefined ? ss.enable_zpd : "N/A"}\n`;
          report += `  - enable_bloom: ${ss.enable_bloom !== undefined ? ss.enable_bloom : "N/A"}\n`;
          report += `  - enable_cognitive_load: ${ss.enable_cognitive_load !== undefined ? ss.enable_cognitive_load : "N/A"}\n`;
          report += `  - enable_emoji_feedback: ${ss.enable_emoji_feedback !== undefined ? ss.enable_emoji_feedback : "N/A"}\n`;
        }
      }
      
      // 9.0.1 Student Profile
      if (finalComprehensiveDebug?.student_profile) {
        report += "\n[9.0.1] Student Profile\n";
        const sp = finalComprehensiveDebug.student_profile;
        report += `Profile Exists        : ${sp.exists ? "Yes" : "No"}\n`;
        if (sp.exists) {
          report += `Average Understanding : ${sp.average_understanding !== null ? sp.average_understanding.toFixed(2) : "N/A"}\n`;
          report += `Average Satisfaction  : ${sp.average_satisfaction !== null ? sp.average_satisfaction.toFixed(2) : "N/A"}\n`;
          report += `Total Interactions    : ${sp.total_interactions || 0}\n`;
          report += `Total Feedback Count  : ${sp.total_feedback_count || 0}\n`;
        }
      }
      
      // 9.0.2 Recent Interactions
      if (finalComprehensiveDebug?.recent_interactions) {
        report += "\n[9.0.2] Recent Interactions History\n";
        const ri = finalComprehensiveDebug.recent_interactions;
        report += `Total Recent Interactions: ${ri.count || 0}\n`;
        if (ri.last_5_interactions && ri.last_5_interactions.length > 0) {
          report += `Last 5 Interactions:\n`;
          ri.last_5_interactions.forEach((interaction: any, idx: number) => {
            report += `  ${idx + 1}. Query: "${interaction.query || "N/A"}"\n`;
            report += `     Emoji Feedback: ${interaction.emoji_feedback || "None"}\n`;
            if (interaction.feedback_score !== null && interaction.feedback_score !== undefined) {
              report += `     Feedback Score: ${Number(interaction.feedback_score).toFixed(2)}\n`;
            }
          });
        }
      }
      
      // 9.1 CACS Scoring Details
      if (finalComprehensiveDebug?.cacs_scoring) {
        report += "\n[9.1] CACS Document Scoring Details\n";
        const cacs = finalComprehensiveDebug.cacs_scoring;
        report += `CACS Applied          : ${cacs.applied ? "Yes" : "No"}\n`;
        report += `Documents Scored      : ${cacs.documents_scored || 0}\n`;
        if (cacs.scoring_details && cacs.scoring_details.length > 0) {
          report += `Scoring Details:\n`;
          cacs.scoring_details.forEach((doc: any, idx: number) => {
            report += `  Doc ${idx + 1} (${doc.doc_id || "N/A"}):\n`;
            report += `    Final Score    : ${(doc.final_score * 100).toFixed(2)}%\n`;
            report += `    Base Score     : ${(doc.base_score * 100).toFixed(2)}%\n`;
            report += `    Personal Score : ${(doc.personal_score * 100).toFixed(2)}%\n`;
            report += `    Global Score   : ${(doc.global_score * 100).toFixed(2)}%\n`;
            report += `    Context Score  : ${(doc.context_score * 100).toFixed(2)}%\n`;
            report += `    Rank           : ${doc.rank || "N/A"}\n`;
          });
        }
      }
      
      // 9.2 Pedagogical Context
      if (data.pedagogical_context) {
        const pc = data.pedagogical_context;
        report += "\n[9.2] Pedagogical Context\n";
        report += `ZPD Recommended    : ${pc.zpd_recommended || "N/A"}\n`;
        report += `Bloom Level         : ${pc.bloom_level || "N/A"}\n`;
        const cognitiveLoad = Number(pc?.cognitive_load ?? 0);
        report += `Cognitive Load      : ${cognitiveLoad > 0 ? cognitiveLoad.toFixed(2) : "N/A"}\n`;
      }
      
      // 9.3 Pedagogical Analysis Details
      if (finalComprehensiveDebug?.pedagogical_analysis) {
        report += "\n[9.3] Pedagogical Analysis Details\n";
        const pa = finalComprehensiveDebug.pedagogical_analysis;
        
        if (pa.zpd) {
          report += `ZPD Analysis:\n`;
          report += `  Enabled           : ${pa.zpd.enabled ? "Yes" : "No"}\n`;
          report += `  Current Level     : ${pa.zpd.current_level || "N/A"}\n`;
          report += `  Recommended Level : ${pa.zpd.recommended_level || "N/A"}\n`;
          if (pa.zpd.success_rate !== null && pa.zpd.success_rate !== undefined) {
            report += `  Success Rate      : ${(Number(pa.zpd.success_rate) * 100).toFixed(2)}%\n`;
          }
          report += `  Calculation Method: ${pa.zpd.calculation_method || "N/A"}\n`;
        }
        
        if (pa.bloom) {
          report += `Bloom Taxonomy:\n`;
          report += `  Enabled           : ${pa.bloom.enabled ? "Yes" : "No"}\n`;
          report += `  Level             : ${pa.bloom.level || "N/A"}\n`;
          report += `  Level Index        : ${pa.bloom.level_index !== null && pa.bloom.level_index !== undefined ? pa.bloom.level_index : "N/A"}\n`;
          if (pa.bloom.confidence !== null && pa.bloom.confidence !== undefined) {
            report += `  Confidence        : ${(Number(pa.bloom.confidence) * 100).toFixed(2)}%\n`;
          }
          report += `  Detection Method  : ${pa.bloom.detection_method || "N/A"}\n`;
        }
        
        if (pa.cognitive_load) {
          report += `Cognitive Load:\n`;
          report += `  Enabled           : ${pa.cognitive_load.enabled ? "Yes" : "No"}\n`;
          if (pa.cognitive_load.total_load !== null && pa.cognitive_load.total_load !== undefined) {
            report += `  Total Load        : ${Number(pa.cognitive_load.total_load).toFixed(2)}\n`;
          }
          report += `  Needs Simplification: ${pa.cognitive_load.needs_simplification ? "Yes" : "No"}\n`;
          report += `  Calculation Method: ${pa.cognitive_load.calculation_method || "N/A"}\n`;
        }
      }
      
      // 9.4 Personalization Factors
      if (personalization.personalization_factors) {
        report += "\n[9.4] Personalization Factors\n";
        Object.entries(personalization.personalization_factors).forEach(([key, value]) => {
          report += `  ${key}: ${JSON.stringify(value)}\n`;
        });
      }
      
      // 9.5 Pedagogical Instructions
      if (personalization.pedagogical_instructions) {
        report += "\n[9.5] Pedagogical Instructions (Sent to LLM)\n";
        report += "-".repeat(80) + "\n";
        report += personalization.pedagogical_instructions + "\n";
      }
      
      // 9.6 Response Comparison
      if (finalComprehensiveDebug?.response_comparison) {
        report += "\n[9.6] Response Comparison\n";
        const rc = finalComprehensiveDebug.response_comparison;
        report += `Original Length      : ${rc.original_length || 0} chars\n`;
        report += `Personalized Length  : ${rc.personalized_length || 0} chars\n`;
        report += `Length Difference    : ${rc.length_difference || 0} chars\n`;
        report += `Is Different         : ${rc.is_different ? "Yes" : "No"}\n`;
        if (rc.similarity_ratio !== null && rc.similarity_ratio !== undefined) {
          report += `Similarity Ratio     : ${(Number(rc.similarity_ratio) * 100).toFixed(2)}%\n`;
        }
      }
      
      // 9.7 Original vs Personalized Response
      if (data.original_response && data.personalized_response) {
        report += "\n[9.7] Original vs Personalized Response\n";
        report += "-".repeat(80) + "\n";
        report += "ORIGINAL:\n";
        report += data.original_response.substring(0, 500) + (data.original_response.length > 500 ? "..." : "") + "\n\n";
        report += "PERSONALIZED:\n";
        report += data.personalized_response.substring(0, 500) + (data.personalized_response.length > 500 ? "..." : "") + "\n";
      }
      report += "\n";
    }
    
    // 10. FULL RAW DATA
    report += "[10] FULL RAW DATA (JSON)\n";
    report += "=".repeat(80) + "\n";
    report += JSON.stringify({
      hybrid_rag_debug: hybridDebug,
      adaptive_query_result: data,
      personalization_data: personalization
    }, null, 2);
    
    report += "\n\n" + "=".repeat(80) + "\n";
    report += `Report Generated: ${new Date().toLocaleString('tr-TR')}\n`;
    report += "=".repeat(80) + "\n";
    
    return report;
  };

  const handleCopy = () => {
    const report = generateDebugReport();
    navigator.clipboard.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const report = generateDebugReport();
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query-debug-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadJSON = () => {
    if (!debugData || !debugData.adaptiveQueryResult) return;
    
    const data = debugData.adaptiveQueryResult;
    const hybridDebug = data.hybrid_rag_debug || {};
    const personalization = debugData.personalizationData || {};
    
    const jsonData = {
      timestamp: new Date().toISOString(),
      query: debugData.lastQuery || "N/A",
      hybrid_rag_debug: hybridDebug,
      adaptive_query_result: data,
      personalization_data: personalization,
      comprehensive_debug: personalization?.comprehensive_debug || null
    };
    
    const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query-debug-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Only generate report when needed (when panel is open)
  const report = isOpen ? generateDebugReport() : "";

  return (
    <div className="border-t border-gray-300 bg-gray-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 bg-gradient-to-r from-gray-800 to-gray-900 hover:from-gray-900 hover:to-black border-b border-gray-700 flex items-center justify-between transition-colors text-white"
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5" />
          <span className="font-semibold">📋 Query Debug Panel (Tam Süreç Dökümü)</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleCopy();
            }}
            className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs flex items-center gap-1"
            title="Copy to clipboard"
          >
            <Copy className="w-3 h-3" />
            {copied ? "Copied!" : "Copy"}
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleDownload();
            }}
            className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs flex items-center gap-1"
            title="Download as text file"
          >
            <Download className="w-3 h-3" />
            TXT
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleDownloadJSON();
            }}
            className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs flex items-center gap-1"
            title="Download as JSON for analysis"
          >
            <Download className="w-3 h-3" />
            JSON
          </button>
          {isOpen ? (
            <ChevronUp className="w-5 h-5" />
          ) : (
            <ChevronDown className="w-5 h-5" />
          )}
        </div>
      </button>

      {isOpen && (
        <div className="bg-black text-green-400 font-mono text-xs p-4 overflow-auto" style={{ maxHeight: "calc(100vh - 400px)", minHeight: "400px" }}>
          <pre className="whitespace-pre-wrap">{report}</pre>
        </div>
      )}
    </div>
  );
}

