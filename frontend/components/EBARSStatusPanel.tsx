"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

interface EBARSStatusPanelProps {
  userId: string;
  sessionId: string;
  onFeedbackSubmitted?: () => void;
  refreshTrigger?: number; // When this changes, refresh the state
}

interface EBARSState {
  comprehension_score: number;
  difficulty_level: string;
  prompt_parameters: any;
  adaptive_prompt?: string; // Generated prompt for display
  statistics: {
    total_feedback_count: number;
    positive_feedback_count: number;
    negative_feedback_count: number;
    consecutive_positive: number;
    consecutive_negative: number;
    last_feedback_at: string | null;
  };
}

const DIFFICULTY_LABELS: Record<string, { label: string; color: string; description: string }> = {
  'very_struggling': {
    label: 'Çok Zorlanıyor',
    color: 'bg-red-100 text-red-800 border-red-300',
    description: 'Sistem cevapları çok basit ve detaylı açıklamalı üretiyor'
  },
  'struggling': {
    label: 'Zorlanıyor',
    color: 'bg-orange-100 text-orange-800 border-orange-300',
    description: 'Sistem cevapları basitleştirilmiş ve örneklerle desteklenmiş üretiyor'
  },
  'normal': {
    label: 'Normal',
    color: 'bg-blue-100 text-blue-800 border-blue-300',
    description: 'Sistem cevapları dengeli seviyede üretiyor'
  },
  'good': {
    label: 'İyi',
    color: 'bg-green-100 text-green-800 border-green-300',
    description: 'Sistem cevapları zorlayıcı ve derinlemesine üretiyor'
  },
  'excellent': {
    label: 'Mükemmel',
    color: 'bg-purple-100 text-purple-800 border-purple-300',
    description: 'Sistem cevapları ileri seviye ve teknik üretiyor'
  },
};

export default function EBARSStatusPanel({
  userId,
  sessionId,
  onFeedbackSubmitted,
  refreshTrigger,
}: EBARSStatusPanelProps) {
  const [ebarsState, setEbarsState] = useState<EBARSState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [previousScore, setPreviousScore] = useState<number | null>(null);
  const [previousDifficulty, setPreviousDifficulty] = useState<string | null>(null);
  const [showChangeAnimation, setShowChangeAnimation] = useState(false);
  const [scoreDelta, setScoreDelta] = useState<number | null>(null);
  const [difficultyChange, setDifficultyChange] = useState<string | null>(null);
  const [isCollapsed, setIsCollapsed] = useState(true); // Start collapsed
  const [showPrompt, setShowPrompt] = useState(false); // Show/hide prompt
  const [showInfoModal, setShowInfoModal] = useState(false); // Show/hide info modal

  // Track if we've successfully loaded EBARS state at least once
  const hasLoadedOnceRef = useRef(false);
  const isInitialLoadRef = useRef(true);
  const isFetchingRef = useRef(false); // Prevent concurrent fetches

  // Fetch EBARS state
  const fetchEBARSState = useCallback(async () => {
    // Prevent concurrent fetches
    if (isFetchingRef.current) {
      return;
    }
    
    const isInitialLoad = isInitialLoadRef.current;
    isFetchingRef.current = true;
    
    try {
      // Only set loading to true on initial load
      if (isInitialLoad) {
        setLoading(true);
      }
      setError(null);

      const response = await fetch(
        `/api/aprag/ebars/state/${userId}/${sessionId}`
      );

      if (!response.ok) {
        // EBARS might be disabled, that's okay
        if (response.status === 403) {
          // Only clear state on initial load, not during refresh
          if (isInitialLoad && !hasLoadedOnceRef.current) {
            setEbarsState(null);
            setLoading(false);
          }
          isInitialLoadRef.current = false;
          return;
        }
        // For other errors, only throw on initial load
        if (isInitialLoad && !hasLoadedOnceRef.current) {
          setEbarsState(null);
          setLoading(false);
          isInitialLoadRef.current = false;
          throw new Error("EBARS durumu alınamadı");
        } else {
          // During refresh, just log and keep existing state - DO NOT CLEAR STATE
          console.warn("EBARS state fetch failed during refresh:", response.status);
          return;
        }
      }

      const data = await response.json();
      if (data.success && data.data) {
        const newState = data.data;
        
        // Mark that we've successfully loaded at least once
        hasLoadedOnceRef.current = true;
        isInitialLoadRef.current = false;
        
        // Check for changes and show animation
        const scoreChanged = previousScore !== null && 
          Math.abs(previousScore - newState.comprehension_score) > 0.1;
        const difficultyChanged = previousDifficulty !== null && 
          previousDifficulty !== newState.difficulty_level;
        
        if (scoreChanged) {
          const delta = newState.comprehension_score - (previousScore || 0);
          setScoreDelta(delta);
          setTimeout(() => setScoreDelta(null), 3000);
        }
        
        if (difficultyChanged) {
          setDifficultyChange(`${previousDifficulty} → ${newState.difficulty_level}`);
          setTimeout(() => setDifficultyChange(null), 3000);
        }
        
        if (scoreChanged || difficultyChanged) {
          setShowChangeAnimation(true);
          setTimeout(() => setShowChangeAnimation(false), 2000);
          
          // Log change for debugging
          if (scoreChanged) {
            console.log(`📊 EBARS Score changed: ${previousScore?.toFixed(1)} → ${newState.comprehension_score.toFixed(1)} (${scoreDelta ? (scoreDelta > 0 ? '+' : '') + scoreDelta.toFixed(1) : 'N/A'})`);
          }
          if (difficultyChanged) {
            console.log(`📈 EBARS Difficulty changed: ${previousDifficulty} → ${newState.difficulty_level}`);
          }
        }

        setPreviousScore(newState.comprehension_score);
        setPreviousDifficulty(newState.difficulty_level);
        setEbarsState(newState); // Always update state if we got valid data
      } else {
        // Only clear state if this is initial load and we haven't loaded before
        if (isInitialLoad && !hasLoadedOnceRef.current) {
          setEbarsState(null);
        }
        isInitialLoadRef.current = false;
      }
    } catch (e: any) {
      // Non-critical error - NEVER clear state if we've loaded before
      if (isInitialLoad && !hasLoadedOnceRef.current) {
        setEbarsState(null);
        setError(null); // Don't show error, just hide panel
      } else {
        // During refresh, just log the error but KEEP showing the panel with existing state
        console.warn("EBARS state fetch error during refresh:", e);
        // DO NOT clear ebarsState - keep showing the last known good state
      }
      isInitialLoadRef.current = false;
    } finally {
      if (isInitialLoad) {
        setLoading(false);
      }
      isFetchingRef.current = false;
    }
  }, [userId, sessionId]);

  // Initial load
  useEffect(() => {
    if (userId && sessionId) {
      hasLoadedOnceRef.current = false;
      isInitialLoadRef.current = true;
      fetchEBARSState();
    }
  }, [userId, sessionId, fetchEBARSState]);

  // Refresh when refreshTrigger changes (parent updates this when query/feedback happens)
  useEffect(() => {
    if (refreshTrigger !== undefined && refreshTrigger > 0 && userId && sessionId && hasLoadedOnceRef.current) {
      // Only refresh if we've loaded at least once
      isInitialLoadRef.current = false;
      fetchEBARSState();
    }
  }, [refreshTrigger, userId, sessionId, fetchEBARSState]);

  // Show loading state only on initial load
  if (loading && !ebarsState && isInitialLoadRef.current) {
    return (
      <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl border-2 border-indigo-200 p-4 mb-4 shadow-md">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-sm animate-pulse">
            EB
          </div>
          <div className="flex-1">
            <div className="h-4 bg-gray-200 rounded w-32 mb-2 animate-pulse"></div>
            <div className="h-3 bg-gray-200 rounded w-48 animate-pulse"></div>
          </div>
        </div>
      </div>
    );
  }

  // Don't render if EBARS state is not available and we've never loaded successfully
  if (!ebarsState && !hasLoadedOnceRef.current) {
    return null;
  }

  // If we have state, always show it (even if a refresh is in progress)
  if (!ebarsState && hasLoadedOnceRef.current) {
    // This shouldn't happen, but if it does, show a minimal placeholder
    return null;
  }

  // At this point, ebarsState is guaranteed to be non-null (we checked above)
  if (!ebarsState) {
    return null; // This should never happen, but TypeScript needs this
  }

  const difficulty = DIFFICULTY_LABELS[ebarsState.difficulty_level] || DIFFICULTY_LABELS['normal'];
  const score = ebarsState.comprehension_score;
  const scoreColor = 
    score >= 80 ? 'text-green-600' :
    score >= 60 ? 'text-blue-600' :
    score >= 40 ? 'text-yellow-600' :
    'text-red-600';
  const difficultyColorClass = difficulty.color.split(' ')[1] || 'text-gray-800';

  return (
    <div className={`bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl border-2 border-indigo-200 mb-4 shadow-md transition-all duration-300 ${showChangeAnimation ? 'ring-4 ring-indigo-300 animate-pulse' : ''}`}>
      {/* Header - Always visible */}
      <div 
        className="flex items-start justify-between p-4 cursor-pointer hover:bg-indigo-50/50 transition-colors rounded-t-xl"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div className="flex items-center gap-2 flex-1">
          <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">
            EB
          </div>
          <div className="flex-1">
            <h3 className="text-sm font-semibold text-gray-800">
              Adaptif Öğrenme Sistemi
            </h3>
            <p className="text-xs text-gray-600">
              {isCollapsed ? (
                <>
                  Puan: <span className={`font-semibold ${scoreColor}`}>{score.toFixed(0)}/100</span> • 
                  Seviye: <span className={`font-semibold ${difficultyColorClass}`}>{difficulty.label}</span>
                </>
              ) : (
                "Anlama seviyenize göre cevaplar otomatik ayarlanıyor"
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowInfoModal(true);
            }}
            className="text-xs text-indigo-600 hover:text-indigo-800 px-2 py-1 rounded hover:bg-indigo-100 transition-colors"
            title="Bilgi"
          >
            ℹ️
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              fetchEBARSState();
            }}
            className="text-xs text-indigo-600 hover:text-indigo-800 px-2 py-1 rounded hover:bg-indigo-100 transition-colors"
            title="Yenile"
          >
            🔄
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsCollapsed(!isCollapsed);
            }}
            className="text-xs text-indigo-600 hover:text-indigo-800 px-2 py-1 rounded hover:bg-indigo-100 transition-colors"
            title={isCollapsed ? "Genişlet" : "Daralt"}
          >
            {isCollapsed ? "▼" : "▲"}
          </button>
        </div>
      </div>

      {/* Content - Collapsible */}
      {!isCollapsed && (
        <div className="px-4 pb-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            {/* Comprehension Score */}
            <div className="bg-white/80 rounded-lg p-3 border border-gray-200 relative">
              <div className="text-xs text-gray-600 mb-1">Algılama Puanı</div>
              <div className="flex items-baseline gap-2">
                <div className={`text-2xl font-bold ${scoreColor} transition-colors duration-300`}>
                  {score.toFixed(0)}
                </div>
                <span className="text-sm text-gray-500">/100</span>
                {scoreDelta !== null && (
                  <div className={`text-xs font-semibold px-2 py-0.5 rounded ${
                    scoreDelta > 0 
                      ? 'bg-green-100 text-green-700 animate-pulse' 
                      : 'bg-red-100 text-red-700 animate-pulse'
                  }`}>
                    {scoreDelta > 0 ? '+' : ''}{scoreDelta.toFixed(1)}
                  </div>
                )}
              </div>
              <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    score >= 80 ? 'bg-green-500' :
                    score >= 60 ? 'bg-blue-500' :
                    score >= 40 ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`}
                  style={{ width: `${score}%` }}
                />
              </div>
            </div>

            {/* Difficulty Level */}
            <div className="bg-white/80 rounded-lg p-3 border border-gray-200">
              <div className="text-xs text-gray-600 mb-1">Zorluk Seviyesi</div>
              <div className="flex items-center gap-2 mb-1">
                <div className={`text-sm font-semibold px-2 py-1 rounded border ${difficulty.color} transition-all duration-300`}>
                  {difficulty.label}
                </div>
                {difficultyChange && (
                  <div className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded animate-pulse">
                    ⚡ Değişti
                  </div>
                )}
              </div>
              <div className="text-xs text-gray-500 mt-2">
                {difficulty.description}
              </div>
            </div>
          </div>

          {/* Statistics */}
          <div className="bg-white/60 rounded-lg p-2 border border-gray-200">
            <div className="text-xs text-gray-600 mb-1">İstatistikler</div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Toplam:</span>
                <span className="font-semibold text-gray-800 ml-1">
                  {ebarsState.statistics.total_feedback_count}
                </span>
              </div>
              <div>
                <span className="text-green-600">Olumlu:</span>
                <span className="font-semibold text-gray-800 ml-1">
                  {ebarsState.statistics.positive_feedback_count}
                </span>
              </div>
              <div>
                <span className="text-red-600">Olumsuz:</span>
                <span className="font-semibold text-gray-800 ml-1">
                  {ebarsState.statistics.negative_feedback_count}
                </span>
              </div>
            </div>
          </div>

          {/* System Explanation */}
          <div className="p-2 bg-indigo-100/50 rounded-lg border border-indigo-200">
            <div className="text-xs text-gray-700 leading-relaxed">
              <span className="font-semibold">💡 Nasıl Çalışıyor?</span>
              <br />
              Emoji geri bildirimlerinize göre sistem, cevapların zorluk seviyesini otomatik olarak ayarlar. 
              {score >= 70 && " Şu anda iyi seviyedesiniz, sistem size daha zorlayıcı cevaplar sunuyor."}
              {score < 70 && score >= 50 && " Sistem size dengeli seviyede cevaplar sunuyor."}
              {score < 50 && " Sistem size daha basit ve açıklayıcı cevaplar sunuyor."}
            </div>
          </div>

          {/* Adaptive Prompt Display */}
          {ebarsState.adaptive_prompt && (
            <div className="mt-3">
              <button
                onClick={() => setShowPrompt(!showPrompt)}
                className="w-full text-left text-xs font-semibold text-indigo-700 hover:text-indigo-900 flex items-center justify-between p-2 bg-indigo-50 hover:bg-indigo-100 rounded-lg border border-indigo-200 transition-colors"
              >
                <span>📝 Kişiselleştirilmiş Prompt {showPrompt ? "(Gizle)" : "(Göster)"}</span>
                <span>{showPrompt ? "▲" : "▼"}</span>
              </button>
              {showPrompt && (
                <div className="mt-2 p-3 bg-gray-50 rounded-lg border border-gray-300 max-h-96 overflow-y-auto">
                  <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">
                    {ebarsState.adaptive_prompt}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Info Modal */}
      {showInfoModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowInfoModal(false)}>
          <div className="bg-white rounded-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="sticky top-0 bg-gradient-to-r from-indigo-500 to-purple-600 text-white p-4 rounded-t-xl flex items-center justify-between">
              <h2 className="text-lg font-bold">EBARS Puanlama Sistemi</h2>
              <button
                onClick={() => setShowInfoModal(false)}
                className="text-white hover:text-gray-200 text-xl font-bold"
              >
                ×
              </button>
            </div>
            
            <div className="p-6 space-y-4">
              <div>
                <h3 className="text-base font-semibold text-gray-800 mb-2">📊 Nasıl Çalışıyor?</h3>
                <p className="text-sm text-gray-700 leading-relaxed">
                  EBARS (Emoji Geri Bildirimi Tabanlı Adaptif Cevap Sistemi), emoji geri bildirimlerinize göre 
                  anlama puanınızı hesaplar ve sistemin cevaplarının zorluk seviyesini otomatik olarak ayarlar.
                </p>
              </div>

              <div>
                <h3 className="text-base font-semibold text-gray-800 mb-2">😊 Emoji Puanlama Sistemi</h3>
                <div className="bg-gray-50 rounded-lg p-3 space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">👍</span>
                    <span className="flex-1"><strong>Tam Anladım:</strong> +5 puan (yüksek seviyede +3.5, düşük seviyede +6.5)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">😊</span>
                    <span className="flex-1"><strong>Genel Anladım:</strong> +2 puan (yüksek seviyede +1.4, düşük seviyede +2.6)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">😐</span>
                    <span className="flex-1"><strong>Kısmen Anladım:</strong> -3 puan (yüksek seviyede -2.1, düşük seviyede -3.9)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">❌</span>
                    <span className="flex-1"><strong>Anlamadım:</strong> -5 puan (yüksek seviyede -3.5, düşük seviyede -6.5)</span>
                  </div>
                </div>
                <p className="text-xs text-gray-600 mt-2">
                  <strong>Dinamik Delta Sistemi:</strong> Puanınız yüksekse (70+) değişimler daha yavaş, 
                  düşükse (30-) daha hızlı olur. Bu sayede sistem daha dengeli çalışır.
                </p>
              </div>

              <div>
                <h3 className="text-base font-semibold text-gray-800 mb-2">📈 Zorluk Seviyeleri</h3>
                <div className="bg-gray-50 rounded-lg p-3 space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-1 bg-red-100 text-red-800 rounded text-xs font-semibold">Çok Zorlanıyor</span>
                    <span className="text-gray-600">0-30 puan</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-1 bg-orange-100 text-orange-800 rounded text-xs font-semibold">Zorlanıyor</span>
                    <span className="text-gray-600">31-45 puan</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-semibold">Normal</span>
                    <span className="text-gray-600">46-70 puan</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-semibold">İyi</span>
                    <span className="text-gray-600">71-80 puan</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs font-semibold">Mükemmel</span>
                    <span className="text-gray-600">81-100 puan</span>
                  </div>
                </div>
                <p className="text-xs text-gray-600 mt-2">
                  <strong>Histeresis Mekanizması:</strong> Zorluk seviyesi değişimleri daha stabil olması için 
                  farklı giriş/çıkış eşikleri kullanılır. Örneğin "Normal" seviyesine 50 puanla girilir, 
                  ancak çıkmak için 75 puana çıkmak gerekir.
                </p>
              </div>

              <div>
                <h3 className="text-base font-semibold text-gray-800 mb-2">⚡ Özel Durumlar</h3>
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm space-y-2">
                  <div>
                    <strong>Hızlı Düşüş:</strong> 3 ardışık olumsuz feedback (😐 veya ❌) alırsanız, 
                    zorluk seviyesi hemen düşürülür.
                  </div>
                  <div>
                    <strong>Hızlı Yükseliş:</strong> 4 ardışık olumlu feedback (👍 veya 😊) alırsanız, 
                    zorluk seviyesi hemen yükseltilir.
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-base font-semibold text-gray-800 mb-2">💡 İpuçları</h3>
                <ul className="text-sm text-gray-700 space-y-1 list-disc list-inside">
                  <li>Dürüst geri bildirim verin - sistem size en uygun seviyeyi bulacaktır</li>
                  <li>Puanınız 50 ile başlar (Normal seviye)</li>
                  <li>Her emoji geri bildirimi puanınızı etkiler</li>
                  <li>Sistem, cevapların zorluk seviyesini otomatik olarak ayarlar</li>
                  <li>Yüksek puanlarda daha zorlayıcı, düşük puanlarda daha açıklayıcı cevaplar alırsınız</li>
                </ul>
              </div>

              <div className="pt-4 border-t border-gray-200">
                <p className="text-xs text-gray-500 text-center">
                  Bu sistem, öğrenme deneyiminizi kişiselleştirmek için tasarlanmıştır. 
                  Daha fazla bilgi için akademik makaleye bakabilirsiniz.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

