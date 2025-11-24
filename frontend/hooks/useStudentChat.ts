/**
 * Custom hook for managing student chat history with database persistence
 * Provides auto-save functionality and session-specific chat management
 */

"use client";

import { useState, useCallback, useEffect } from "react";
import {
  StudentChatMessage,
  getStudentChatHistory,
  saveStudentChatMessage,
  clearStudentChatHistory,
  ragQuery,
  generateSuggestions,
  createAPRAGInteraction,
  apragAdaptiveQuery,
  getAPRAGSettings,
  classifyQuestion,
  hybridRAGQuery,
} from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

interface UseStudentChatOptions {
  sessionId: string;
  autoSave?: boolean;
  maxMessages?: number;
}

interface UseStudentChatReturn {
  messages: StudentChatMessage[];
  isLoading: boolean;
  isQuerying: boolean;
  error: string | null;
  sendMessage: (query: string, sessionRagSettings?: any) => Promise<void>;
  clearHistory: () => Promise<void>;
  refreshHistory: () => Promise<void>;
  handleSuggestionClick: (
    suggestion: string,
    sessionRagSettings?: any
  ) => Promise<void>;
}

export function useStudentChat({
  sessionId,
  autoSave = true,
  maxMessages = 50,
}: UseStudentChatOptions): UseStudentChatReturn {
  const [messages, setMessages] = useState<StudentChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();

  // Load chat history from database on mount or sessionId change
  const refreshHistory = useCallback(async () => {
    if (!sessionId || !user) return;

    try {
      setIsLoading(true);
      setError(null);

      const chatHistory = await getStudentChatHistory(sessionId);
      setMessages(chatHistory || []);
    } catch (err: any) {
      console.error("Failed to load chat history:", err);
      setError(err.message || "Sohbet geçmişi yüklenemedi");
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, user]);

  // Save message to database
  const saveMessage = useCallback(
    async (message: Omit<StudentChatMessage, "id" | "timestamp">) => {
      if (!autoSave || !user) return null;

      try {
        const savedMessage = await saveStudentChatMessage(message);
        return savedMessage;
      } catch (err) {
        console.error("Failed to save message:", err);
        return null;
      }
    },
    [autoSave, user]
  );

  // Send a new message (question + get AI response)
  const sendMessage = useCallback(
    async (query: string, sessionRagSettings?: any) => {
      if (!query.trim() || !sessionId || isQuerying) return;

      const userMessage: Omit<StudentChatMessage, "id" | "timestamp"> = {
        user: query,
        bot: "...",
        session_id: sessionId,
      };

      // Add user message immediately to UI
      setMessages((prev) => [
        ...prev,
        { ...userMessage, timestamp: new Date().toISOString() },
      ]);
      setIsQuerying(true);
      setError(null);

      const startTime = Date.now();

      try {
        // Build conversation history for context
        const conversationHistory = messages.slice(-4).flatMap((msg) => {
          const history: Array<{
            role: "user" | "assistant";
            content: string;
          }> = [];
          if (msg.user && msg.user.trim() && msg.user !== "...") {
            history.push({ role: "user", content: msg.user });
          }
          if (
            msg.bot &&
            msg.bot.trim() &&
            msg.bot !== "..." &&
            !msg.bot.startsWith("Hata:")
          ) {
            history.push({ role: "assistant", content: msg.bot });
          }
          return history;
        });

        // Prepare classic RAG payload (for metadata / APRAG logging)
        const payload: any = {
          session_id: sessionId,
          query,
          top_k: 5,
          use_rerank: sessionRagSettings?.use_rerank ?? false, // Use session settings
          min_score: sessionRagSettings?.min_score ?? 0.5,
          max_context_chars: 8000,
          use_direct_llm: false,
          max_tokens: 2048,
          conversation_history:
            conversationHistory.length > 0 ? conversationHistory : undefined,
        };

        // Use session settings if available
        if (sessionRagSettings?.model) {
          payload.model = sessionRagSettings.model;
        }
        if (sessionRagSettings?.chain_type) {
          payload.chain_type = sessionRagSettings.chain_type;
        }
        if (sessionRagSettings?.embedding_model) {
          payload.embedding_model = sessionRagSettings.embedding_model;
        }

        // KB-Enhanced RAG payload (Hybrid RAG)
        const hybridPayload = {
          session_id: sessionId,
          query,
          top_k: 5,
          use_kb: true,
          use_qa_pairs: true,
          use_crag: true,
          model: sessionRagSettings?.model,
          max_tokens: 2048,
          temperature: 0.7,
          max_context_chars: 8000,
          include_examples: true,
          include_sources: true,
        };

        // Get AI response from KB-Enhanced Hybrid RAG
        const result = await hybridRAGQuery(hybridPayload);
        const actualDurationMs = Date.now() - startTime;

        // Check if APRAG is enabled for adaptive learning
        let finalResponse = result.answer;
        let apragInteractionId: number | null = null;
        let pedagogicalInfo: any = null;

        if (user?.id) {
          try {
            // Check APRAG status
            const apragSettings = await getAPRAGSettings(sessionId);
            
            if (apragSettings.enabled && apragSettings.features.cacs) {
              // Use APRAG Adaptive Query for personalized learning
              console.log("🎓 Using APRAG Adaptive Query for personalized response...");
              
              const adaptiveResult = await apragAdaptiveQuery({
                user_id: user.id.toString(),
                session_id: sessionId,
                query: query,
                rag_documents: (result.sources || []).map((s: any) => ({
                  doc_id: s.metadata?.source_file || s.metadata?.filename || "unknown",
                  content: s.content || "",
                  score: s.score || 0,
                  metadata: s.metadata || {},
                })),
                rag_response: result.answer,
              });

              // Use personalized response
              finalResponse = adaptiveResult.personalized_response;
              apragInteractionId = adaptiveResult.interaction_id;
              pedagogicalInfo = {
                zpd: adaptiveResult.pedagogical_context.zpd_recommended,
                bloom: adaptiveResult.pedagogical_context.bloom_level,
                cognitive_load: adaptiveResult.pedagogical_context.cognitive_load,
                cacs_applied: adaptiveResult.cacs_applied,
              };

              console.log(`✅ APRAG Applied: ZPD=${pedagogicalInfo.zpd}, Bloom=${pedagogicalInfo.bloom}, CACS=${pedagogicalInfo.cacs_applied}`);
            } else {
              // Fallback: Manual APRAG interaction logging
              const apragResult = await createAPRAGInteraction({
                user_id: user.id.toString(),
                session_id: sessionId,
                query: query,
                response: result.answer,
                processing_time_ms: actualDurationMs,
                model_used: payload.model,
                chain_type: payload.chain_type,
                sources: result.sources?.map((s: any) => ({
                  content: s.content || "",
                  score: s.score || 0,
                  metadata: s.metadata,
                })),
              });
              apragInteractionId = apragResult.interaction_id;
            }
          } catch (apragError) {
            // Don't fail the whole request if APRAG fails
            console.error("Failed to use APRAG adaptive query:", apragError);
            
            // Fallback to manual interaction logging
            try {
              const apragResult = await createAPRAGInteraction({
                user_id: user.id.toString(),
                session_id: sessionId,
                query: query,
                response: result.answer,
                processing_time_ms: actualDurationMs,
                model_used: payload.model,
                chain_type: payload.chain_type,
                sources: result.sources?.map((s: any) => ({
                  content: s.content || "",
                  score: s.score || 0,
                  metadata: s.metadata,
                })),
              });
              apragInteractionId = apragResult.interaction_id;
            } catch (fallbackError) {
              console.error("Fallback interaction logging also failed:", fallbackError);
            }
          }
        }

        // Get topic classification first (if available)
        let topicInfo: { topic_id: number; topic_title: string; confidence_score: number } | undefined;
        if (apragInteractionId && sessionId) {
          try {
            const classificationResult = await classifyQuestion({
              question: query,
              session_id: sessionId,
              interaction_id: apragInteractionId,
            });
            
            if (classificationResult.success && classificationResult.topic_id) {
              topicInfo = {
                topic_id: classificationResult.topic_id,
                topic_title: classificationResult.topic_title,
                confidence_score: classificationResult.confidence_score,
              };
            }
          } catch (classificationError) {
            console.warn(
              "Question classification for topic progress failed:",
              classificationError
            );
          }
        }

        // Create complete message object with personalized response
        const completeMessage: Omit<StudentChatMessage, "id" | "timestamp"> = {
          user: query,
          bot: finalResponse,
          sources: result.sources || [],
          durationMs: actualDurationMs,
          session_id: sessionId,
          suggestions: [], // Will be filled asynchronously
          aprag_interaction_id: apragInteractionId || undefined,
          topic: topicInfo, // Include topic information
        };

        // Update UI with response
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...completeMessage,
            timestamp: new Date().toISOString(),
          };
          return updated;
        });

        // Save to database
        await saveMessage(completeMessage);

        // Generate suggestions asynchronously (non-blocking)
        // CRITICAL: Save suggestions to database after generation
        (async () => {
          try {
            const suggestions = await generateSuggestions({
              question: query,
              answer: finalResponse, // Use personalized response for better context
              sources: result.sources || [],
            });

            if (Array.isArray(suggestions) && suggestions.length > 0) {
              // Update UI immediately
              setMessages((prev) => {
                const updated = [...prev];
                const lastIndex = updated.length - 1;
                if (lastIndex >= 0 && updated[lastIndex]) {
                  updated[lastIndex] = {
                    ...updated[lastIndex],
                    suggestions: suggestions,
                  };
                }
                return updated;
              });
              
              // CRITICAL: Save suggestions to database by updating existing message
              try {
                // Get current message from database to preserve all fields
                const { getStudentChatHistory, saveStudentChatMessage } = await import("@/lib/api");
                const chatHistory = await getStudentChatHistory(sessionId);
                const messageToUpdate = chatHistory.find(
                  (msg) => msg.aprag_interaction_id === apragInteractionId
                );
                
                if (messageToUpdate) {
                  // Update with suggestions, preserve all other fields
                  // CRITICAL: Use topicInfo if available, otherwise keep existing topic
                  const finalTopic = topicInfo || messageToUpdate.topic;
                  await saveStudentChatMessage({
                    user: messageToUpdate.user,
                    bot: messageToUpdate.bot,
                    sources: messageToUpdate.sources || [],
                    durationMs: messageToUpdate.durationMs || 0,
                    session_id: messageToUpdate.session_id,
                    suggestions: suggestions, // Save suggestions
                    aprag_interaction_id: messageToUpdate.aprag_interaction_id,
                    topic: finalTopic, // Use topicInfo if available, otherwise keep existing
                    emoji_feedback: messageToUpdate.emoji_feedback, // Keep existing emoji
                  });
                  console.log("✅ Suggestions saved to database", { 
                    suggestionsCount: suggestions.length,
                    topic: finalTopic,
                    emoji: messageToUpdate.emoji_feedback 
                  });
                } else {
                  console.warn("⚠️ Message not found for suggestions update", { apragInteractionId });
                }
              } catch (saveErr) {
                console.error("❌ Failed to save suggestions to database:", saveErr);
              }
            }
          } catch (suggErr) {
            console.error("Failed to generate suggestions:", suggErr);
            // Don't show error to user - suggestions are optional
          }
        })();
      } catch (err: any) {
        const errorMessage = err.message || "Sorgu başarısız oldu";
        setError(errorMessage);

        // Update UI with error
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            bot: `Hata: ${errorMessage}`,
          };
          return updated;
        });

        // Save error message too
        await saveMessage({
          user: query,
          bot: `Hata: ${errorMessage}`,
          session_id: sessionId,
        });
      } finally {
        setIsQuerying(false);
      }
    },
    [sessionId, messages, isQuerying, saveMessage]
  );

  // Handle suggestion click
  const handleSuggestionClick = useCallback(
    async (suggestion: string, sessionRagSettings?: any) => {
      await sendMessage(suggestion, sessionRagSettings);
    },
    [sendMessage]
  );

  // Clear all chat history
  const clearHistory = useCallback(async () => {
    if (!sessionId || !user) return;

    try {
      await clearStudentChatHistory(sessionId);
      setMessages([]);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Sohbet geçmişi temizlenemedi");
    }
  }, [sessionId, user]);

  // Load history on mount and sessionId change
  useEffect(() => {
    if (sessionId) {
      refreshHistory();
    }
  }, [sessionId, refreshHistory]);

  // Limit messages in memory (keep most recent)
  useEffect(() => {
    if (messages.length > maxMessages) {
      setMessages((prev) => prev.slice(-maxMessages));
    }
  }, [messages.length, maxMessages]);

  return {
    messages,
    isLoading,
    isQuerying,
    error,
    sendMessage,
    clearHistory,
    refreshHistory,
    handleSuggestionClick,
  };
}
