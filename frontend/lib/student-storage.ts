/**
 * Student Storage Manager - Database-based persistence
 * Handles student-specific data with database storage
 * Fallback to localStorage for offline scenarios
 */

"use client";

import {
  StudentChatMessage,
  getStudentChatHistory,
  saveStudentChatMessage,
  getStudentSessions,
} from "./api";

// Types for student data management
export interface StudentSession {
  sessionId: string;
  sessionName: string;
  lastAccessed: string;
  messageCount: number;
}

export interface StudentProgress {
  sessionId: string;
  questionsAsked: number;
  topicsExplored: string[];
  lastActivity: string;
  averageResponseTime?: number;
}

/**
 * Student Storage Manager Class
 * Manages student-specific data with database persistence
 */
class StudentStorageManager {
  private readonly STORAGE_PREFIX = "student_";
  private readonly CACHE_DURATION = 5 * 60 * 1000; // 5 minutes cache

  // Memory cache for quick access
  private cache = new Map<string, { data: any; timestamp: number }>();

  /**
   * Get student's session list with activity info
   */
  async getStudentSessions(): Promise<StudentSession[]> {
    const cacheKey = "student_sessions";
    const cached = this.getCached(cacheKey);

    if (cached) {
      return cached;
    }

    try {
      const sessionsData = await getStudentSessions();
      const sessions: StudentSession[] = sessionsData.map((item) => ({
        sessionId: item.session.session_id,
        sessionName: item.session.name,
        lastAccessed: item.last_message_time || item.session.updated_at,
        messageCount: item.message_count,
      }));

      this.setCache(cacheKey, sessions);
      return sessions;
    } catch (error) {
      console.error("Failed to load student sessions from database:", error);
      // Fallback to localStorage
      return this.getFromLocalStorage("sessions", []);
    }
  }

  /**
   * Get chat history for a specific session
   */
  async getChatHistory(sessionId: string): Promise<StudentChatMessage[]> {
    const cacheKey = `chat_history_${sessionId}`;
    const cached = this.getCached(cacheKey);

    if (cached) {
      return cached;
    }

    try {
      const messages = await getStudentChatHistory(sessionId);
      this.setCache(cacheKey, messages);
      return messages;
    } catch (error) {
      console.error(
        `Failed to load chat history for session ${sessionId}:`,
        error
      );
      // Fallback to localStorage
      return this.getFromLocalStorage(`chat_${sessionId}`, []);
    }
  }

  /**
   * Save a chat message
   */
  async saveChatMessage(
    message: Omit<StudentChatMessage, "id" | "timestamp">
  ): Promise<void> {
    try {
      const savedMessage = await saveStudentChatMessage(message);

      // Update cache
      const cacheKey = `chat_history_${message.session_id}`;
      const currentHistory = this.getCached(cacheKey) || [];
      const updatedHistory = [...currentHistory, savedMessage];
      this.setCache(cacheKey, updatedHistory);

      // Clear sessions cache to force refresh
      this.clearCache("student_sessions");
    } catch (error) {
      console.error("Failed to save chat message to database:", error);
      // Fallback to localStorage
      this.saveToLocalStorage(`chat_${message.session_id}`, message);
    }
  }

  /**
   * Get student progress for a session
   */
  getStudentProgress(sessionId: string): StudentProgress {
    const progress = this.getFromLocalStorage(`progress_${sessionId}`, null);

    return (
      progress || {
        sessionId,
        questionsAsked: 0,
        topicsExplored: [],
        lastActivity: new Date().toISOString(),
      }
    );
  }

  /**
   * Update student progress
   */
  updateStudentProgress(progress: StudentProgress): void {
    this.saveToLocalStorage(`progress_${progress.sessionId}`, {
      ...progress,
      lastActivity: new Date().toISOString(),
    });

    // Clear cache to force refresh
    this.clearCache(`progress_${progress.sessionId}`);
  }

  /**
   * Clear all data for a specific session
   */
  async clearSessionData(sessionId: string): Promise<void> {
    // Clear from cache
    this.clearCache(`chat_history_${sessionId}`);
    this.clearCache(`progress_${sessionId}`);
    this.clearCache("student_sessions");

    // Clear from localStorage
    this.removeFromLocalStorage(`chat_${sessionId}`);
    this.removeFromLocalStorage(`progress_${sessionId}`);

    console.log(`Cleared all data for session ${sessionId}`);
  }

  /**
   * Get user preferences
   */
  getUserPreferences(): Record<string, any> {
    return this.getFromLocalStorage("preferences", {
      theme: "light",
      fontSize: "medium",
      notifications: true,
      autoSave: true,
    });
  }

  /**
   * Save user preferences
   */
  saveUserPreferences(preferences: Record<string, any>): void {
    this.saveToLocalStorage("preferences", preferences);
    this.clearCache("preferences");
  }

  // Private helper methods
  private getCached(key: string): any {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.CACHE_DURATION) {
      return cached.data;
    }
    return null;
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
    });
  }

  private clearCache(key: string): void {
    this.cache.delete(key);
  }

  private getFromLocalStorage(key: string, defaultValue: any): any {
    if (typeof window === "undefined") return defaultValue;

    try {
      const item = localStorage.getItem(this.STORAGE_PREFIX + key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.warn(`Failed to get ${key} from localStorage:`, error);
      return defaultValue;
    }
  }

  private saveToLocalStorage(key: string, data: any): void {
    if (typeof window === "undefined") return;

    try {
      localStorage.setItem(this.STORAGE_PREFIX + key, JSON.stringify(data));
    } catch (error) {
      console.warn(`Failed to save ${key} to localStorage:`, error);
    }
  }

  private removeFromLocalStorage(key: string): void {
    if (typeof window === "undefined") return;

    try {
      localStorage.removeItem(this.STORAGE_PREFIX + key);
    } catch (error) {
      console.warn(`Failed to remove ${key} from localStorage:`, error);
    }
  }
}

// Export singleton instance
export const studentStorage = new StudentStorageManager();

// Export utility functions
export const getStudentSessionData = (sessionId: string) =>
  studentStorage.getChatHistory(sessionId);

export const saveStudentMessage = (
  message: Omit<StudentChatMessage, "id" | "timestamp">
) => studentStorage.saveChatMessage(message);

export const getStudentPreferences = () => studentStorage.getUserPreferences();

export const saveStudentPreferences = (preferences: Record<string, any>) =>
  studentStorage.saveUserPreferences(preferences);
