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

        // Prepare RAG query payload
        const payload: any = {
          session_id: sessionId,
          query,
          top_k: 5,
          use_rerank: true,
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

        // Get AI response
        const result = await ragQuery(payload);
        const actualDurationMs = Date.now() - startTime;

        // Create complete message object
        const completeMessage: Omit<StudentChatMessage, "id" | "timestamp"> = {
          user: query,
          bot: result.answer,
          sources: result.sources || [],
          durationMs: actualDurationMs,
          session_id: sessionId,
          suggestions: [], // Will be filled asynchronously
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
        if (result.sources && result.sources.length > 0) {
          try {
            const suggestions = await generateSuggestions({
              question: query,
              answer: result.answer,
              sources: result.sources,
            });

            if (Array.isArray(suggestions) && suggestions.length > 0) {
              setMessages((prev) => {
                const updated = [...prev];
                if (updated[updated.length - 1]) {
                  updated[updated.length - 1].suggestions = suggestions;

                  // Update database with suggestions
                  if (autoSave) {
                    saveStudentChatMessage({
                      ...updated[updated.length - 1],
                      suggestions,
                    }).catch(console.error);
                  }
                }
                return updated;
              });
            }
          } catch (suggErr) {
            console.error("Failed to generate suggestions:", suggErr);
          }
        }
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
