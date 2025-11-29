"use client";

import React, { useState, useEffect } from "react";
import {
  getSessionInteractions,
  getRecentInteractions,
  getTotalInteractions,
  getAPRAGSettings,
  APRAGInteraction,
  APRAGInteractionsResponse,
} from "@/lib/api";

interface DiagnosticResult {
  testName: string;
  status: "success" | "error" | "warning";
  message: string;
  data?: any;
  duration?: number;
  error?: string;
}

interface InteractionsTabDiagnosticsProps {
  sessionId: string;
}

export default function InteractionsTabDiagnostics({
  sessionId,
}: InteractionsTabDiagnosticsProps) {
  const [results, setResults] = useState<DiagnosticResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [interactionsData, setInteractionsData] =
    useState<APRAGInteractionsResponse | null>(null);
  const [apragSettings, setApragSettings] = useState<any>(null);

  const addResult = (result: DiagnosticResult) => {
    setResults((prev) => [...prev, result]);
  };

  const runDiagnostics = async () => {
    setIsRunning(true);
    setResults([]);

    // Test 1: APRAG Settings Check
    try {
      const startTime = performance.now();
      const settings = await getAPRAGSettings(sessionId);
      const duration = performance.now() - startTime;
      setApragSettings(settings);

      addResult({
        testName: "APRAG Settings Status",
        status: settings.enabled ? "success" : "warning",
        message: settings.enabled
          ? `APRAG Enabled - Global: ${settings.global_enabled}, Session: ${settings.session_enabled}`
          : "APRAG Disabled - Interactions may be limited",
        data: settings,
        duration,
      });
    } catch (error: any) {
      addResult({
        testName: "APRAG Settings Status",
        status: "error",
        message: "APRAG ayarları yüklenemedi",
        error: error.message,
      });
    }

    // Test 2: Session Interactions API Test
    try {
      const startTime = performance.now();
      const interactions = await getSessionInteractions(sessionId, 20, 0);
      const duration = performance.now() - startTime;
      setInteractionsData(interactions);

      addResult({
        testName: "Session Interactions API",
        status: "success",
        message: `Session interactions loaded: ${
          interactions.interactions?.length || 0
        } items, Total: ${interactions.total || 0}`,
        data: {
          count: interactions.interactions?.length || 0,
          total: interactions.total || 0,
          limit: interactions.limit,
          offset: interactions.offset,
        },
        duration,
      });
    } catch (error: any) {
      addResult({
        testName: "Session Interactions API",
        status: "error",
        message: `getSessionInteractions API failed: ${error.message}`,
        error: error.message,
      });
    }

    // Test 3: Recent Interactions API Test
    try {
      const startTime = performance.now();
      const recent = await getRecentInteractions({
        limit: 10,
        session_id: sessionId,
      });
      const duration = performance.now() - startTime;

      addResult({
        testName: "Recent Interactions API",
        status: "success",
        message: `Recent interactions loaded: ${
          recent.items?.length || 0
        } items (Page ${recent.page})`,
        data: {
          count: recent.items?.length || 0,
          total: recent.count || 0,
          page: recent.page,
          limit: recent.limit,
        },
        duration,
      });
    } catch (error: any) {
      addResult({
        testName: "Recent Interactions API",
        status: "error",
        message: `getRecentInteractions API failed: ${error.message}`,
        error: error.message,
      });
    }

    // Test 4: Total Interactions Count
    try {
      const startTime = performance.now();
      const totalData = await getTotalInteractions([sessionId]);
      const duration = performance.now() - startTime;

      addResult({
        testName: "Total Interactions Count",
        status: "success",
        message: `Total interactions count: ${totalData.total || 0}`,
        data: totalData,
        duration,
      });
    } catch (error: any) {
      addResult({
        testName: "Total Interactions Count",
        status: "error",
        message: `getTotalInteractions API failed: ${error.message}`,
        error: error.message,
      });
    }

    // Test 5: Interaction Data Structure Validation
    if (
      interactionsData?.interactions &&
      interactionsData.interactions.length > 0
    ) {
      const interaction = interactionsData.interactions[0];
      const hasRequiredFields = !!(
        interaction.interaction_id &&
        interaction.user_id &&
        interaction.session_id &&
        interaction.query &&
        interaction.original_response &&
        interaction.timestamp
      );

      addResult({
        testName: "Interaction Data Structure",
        status: hasRequiredFields ? "success" : "warning",
        message: hasRequiredFields
          ? "Interaction data structure is complete with all required fields"
          : "Some required fields missing in interaction data",
        data: {
          sample_interaction: interaction,
          required_fields_present: {
            interaction_id: !!interaction.interaction_id,
            user_id: !!interaction.user_id,
            session_id: !!interaction.session_id,
            query: !!interaction.query,
            original_response: !!interaction.original_response,
            timestamp: !!interaction.timestamp,
            student_name: !!interaction.student_name,
            topic_info: !!interaction.topic_info,
          },
        },
      });
    } else {
      addResult({
        testName: "Interaction Data Structure",
        status: "warning",
        message: "No interactions available to validate data structure",
        data: { interactions_count: 0 },
      });
    }

    // Test 6: Pagination Functionality
    try {
      const startTime = performance.now();
      const paginated = await getSessionInteractions(sessionId, 5, 5); // Different limit/offset
      const duration = performance.now() - startTime;

      addResult({
        testName: "Pagination Functionality",
        status: "success",
        message: `Pagination works: Retrieved ${
          paginated.interactions?.length || 0
        } items with offset 5`,
        data: {
          offset_5_count: paginated.interactions?.length || 0,
          original_count: interactionsData?.interactions?.length || 0,
          limit_used: 5,
          offset_used: 5,
        },
        duration,
      });
    } catch (error: any) {
      addResult({
        testName: "Pagination Functionality",
        status: "error",
        message: `Pagination test failed: ${error.message}`,
        error: error.message,
      });
    }

    setIsRunning(false);
  };

  useEffect(() => {
    // Auto-run diagnostics on component mount
    runDiagnostics();
  }, [sessionId]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "success":
        return "text-green-600 bg-green-50 border-green-200";
      case "error":
        return "text-red-600 bg-red-50 border-red-200";
      case "warning":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "success":
        return "✅";
      case "error":
        return "❌";
      case "warning":
        return "⚠️";
      default:
        return "ℹ️";
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg border">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-lg font-semibold">
            🪲 Interactions Tab API Diagnostics
          </h3>
          <p className="text-sm text-gray-600">Session ID: {sessionId}</p>
        </div>
        <button
          onClick={runDiagnostics}
          disabled={isRunning}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {isRunning ? "Testing..." : "Run Diagnostics"}
        </button>
      </div>

      {/* APRAG Status Summary */}
      {apragSettings && (
        <div
          className={`mb-6 p-4 rounded border ${
            apragSettings.enabled
              ? "bg-green-50 border-green-200"
              : "bg-yellow-50 border-yellow-200"
          }`}
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">
              {apragSettings.enabled ? "🟢" : "🟡"}
            </span>
            <strong>
              APRAG Status: {apragSettings.enabled ? "ENABLED" : "DISABLED"}
            </strong>
          </div>
          <div className="text-sm space-y-1">
            <div>
              Global Enabled: {apragSettings.global_enabled ? "✅" : "❌"}
            </div>
            <div>
              Session Enabled:{" "}
              {apragSettings.session_enabled !== null
                ? apragSettings.session_enabled
                  ? "✅"
                  : "❌"
                : "⚪"}
            </div>
            <div>
              Features: Analytics:{" "}
              {apragSettings.features?.analytics ? "✅" : "❌"}, Feedback:{" "}
              {apragSettings.features?.feedback_collection ? "✅" : "❌"}
            </div>
          </div>
        </div>
      )}

      {/* Interactions Data Summary */}
      {interactionsData && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">📊</span>
            <strong>Interactions Data Summary</strong>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div>
                Total Interactions:{" "}
                <strong>{interactionsData.total || 0}</strong>
              </div>
              <div>
                Current Page:{" "}
                <strong>{interactionsData.interactions?.length || 0}</strong>{" "}
                items
              </div>
            </div>
            <div>
              <div>Limit: {interactionsData.limit}</div>
              <div>Offset: {interactionsData.offset}</div>
            </div>
          </div>
        </div>
      )}

      {/* Diagnostic Results */}
      <div className="space-y-3">
        {results.map((result, index) => (
          <div
            key={index}
            className={`p-4 rounded border ${getStatusColor(result.status)}`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-3 flex-1">
                <span className="text-lg">{getStatusIcon(result.status)}</span>
                <div className="flex-1">
                  <div className="font-medium">{result.testName}</div>
                  <div className="text-sm mt-1">{result.message}</div>
                  {result.duration && (
                    <div className="text-xs text-gray-500 mt-1">
                      Response time: {result.duration.toFixed(2)}ms
                    </div>
                  )}
                  {result.error && (
                    <div className="text-xs text-red-600 mt-2 font-mono bg-red-100 p-2 rounded">
                      Error: {result.error}
                    </div>
                  )}
                </div>
              </div>
            </div>
            {result.data && (
              <details className="mt-3">
                <summary className="text-xs text-gray-600 cursor-pointer hover:text-gray-800">
                  View response data
                </summary>
                <pre className="text-xs bg-gray-100 p-2 rounded mt-2 overflow-auto max-h-40">
                  {JSON.stringify(result.data, null, 2)}
                </pre>
              </details>
            )}
          </div>
        ))}
      </div>

      {isRunning && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded flex items-center gap-2">
          <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
          <span className="text-blue-700">Running diagnostics...</span>
        </div>
      )}
    </div>
  );
}
