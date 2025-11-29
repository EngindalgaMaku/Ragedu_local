"use client";

import React, { useState, useEffect } from "react";
import { getApiUrl, getSessionTopics } from "@/lib/api";

interface TopicsTabDiagnosticsProps {
  sessionId: string;
}

interface TestResult {
  success: boolean;
  status?: number;
  data?: any;
  note?: string;
}

export const TopicsTabDiagnostics: React.FC<TopicsTabDiagnosticsProps> = ({
  sessionId,
}) => {
  const [results, setResults] = useState<any[]>([]);
  const [testing, setTesting] = useState(false);
  const [apragStatus, setApragStatus] = useState<any>(null);

  const testEndpoints = [
    // lib/api.ts functions
    {
      name: "getSessionTopics (lib/api.ts)",
      test: async (): Promise<TestResult> => {
        const response = await getSessionTopics(sessionId);
        return {
          success: response.success,
          data: response,
          status: response.success ? 200 : 500,
          note: `Found ${response.topics?.length || 0} topics`,
        };
      },
    },
    // Direct API calls used in TopicManagementPanel
    {
      name: "APRAG Settings Status",
      test: async (): Promise<TestResult> => {
        const response = await fetch(`${getApiUrl()}/aprag/settings/status`);
        const data = response.ok ? await response.json() : null;
        return {
          success: response.ok,
          data: data,
          status: response.status,
          note: response.ok
            ? `APRAG ${data?.global_enabled ? "enabled" : "disabled"}`
            : "Check status",
        };
      },
    },
    {
      name: "Topics Re-extract Endpoint",
      test: async (): Promise<TestResult> => {
        // Only test GET to see if endpoint exists (don't actually trigger extraction)
        const response = await fetch(
          `/api/aprag/topics/re-extract/${sessionId}`,
          {
            method: "GET",
          }
        );
        return {
          success: response.status !== 404,
          status: response.status,
          data: null,
          note:
            response.status === 405
              ? "Endpoint exists (Method Not Allowed)"
              : "Check status",
        };
      },
    },
    {
      name: "Knowledge Batch Extract Endpoint",
      test: async (): Promise<TestResult> => {
        const response = await fetch(
          `/api/aprag/knowledge/extract-batch/${sessionId}`,
          {
            method: "GET",
          }
        );
        return {
          success: response.status !== 404,
          status: response.status,
          data: null,
          note:
            response.status === 405
              ? "Endpoint exists (Method Not Allowed)"
              : "Check status",
        };
      },
    },
    {
      name: "Knowledge Base Endpoint (Test Topic ID 1)",
      test: async (): Promise<TestResult> => {
        const response = await fetch(`/api/aprag/knowledge/kb/1`);
        const data = response.ok ? await response.json() : null;
        return {
          success: response.ok || response.status !== 404,
          status: response.status,
          data: data,
          note: response.ok
            ? "KB data found"
            : response.status === 404
            ? "Endpoint not found"
            : "Check status",
        };
      },
    },
  ];

  const runDiagnostics = async () => {
    setTesting(true);
    setResults([]);

    // First check APRAG status
    try {
      const response = await fetch(`${getApiUrl()}/aprag/settings/status`);
      if (response.ok) {
        const data = await response.json();
        setApragStatus(data);
      }
    } catch (e) {
      console.error("APRAG status check failed:", e);
    }

    const testResults = [];

    for (const endpoint of testEndpoints) {
      try {
        console.log(`🧪 Testing: ${endpoint.name}`);
        const startTime = Date.now();
        const result = await endpoint.test();
        const duration = Date.now() - startTime;

        testResults.push({
          name: endpoint.name,
          success: result.success,
          duration: `${duration}ms`,
          status: result.status || (result.success ? "OK" : "FAIL"),
          data: result.data,
          note: result.note,
          timestamp: new Date().toLocaleTimeString(),
        });
      } catch (error: any) {
        testResults.push({
          name: endpoint.name,
          success: false,
          error: error.message,
          duration: "N/A",
          timestamp: new Date().toLocaleTimeString(),
        });
      }
    }

    setResults(testResults);
    setTesting(false);
  };

  useEffect(() => {
    runDiagnostics();
  }, [sessionId]);

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            🧠 Topics Tab Diagnostics
          </h3>
          <p className="text-sm text-gray-600">
            Session: {sessionId} • Testing APRAG Topic Management APIs
          </p>
        </div>
        <button
          onClick={runDiagnostics}
          disabled={testing}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
        >
          {testing ? "Testing..." : "Re-test"}
        </button>
      </div>

      {/* APRAG Status */}
      {apragStatus && (
        <div className="mb-4 p-3 bg-blue-50 rounded-md">
          <h4 className="text-sm font-semibold text-blue-900 mb-2">
            📊 APRAG Status
          </h4>
          <div className="text-xs text-blue-800 space-y-1">
            <div>
              Global Enabled:{" "}
              <span className="font-mono">
                {apragStatus.global_enabled ? "✅ Yes" : "❌ No"}
              </span>
            </div>
            <div>
              Session Enabled:{" "}
              <span className="font-mono">
                {apragStatus.session_enabled ? "✅ Yes" : "❌ No"}
              </span>
            </div>
            <div>
              Features:{" "}
              <span className="font-mono">
                {JSON.stringify(apragStatus.features || {})}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Test Results */}
      <div className="space-y-3">
        {results.map((result, index) => (
          <div
            key={index}
            className={`p-3 rounded-md ${
              result.success
                ? "bg-green-50 border border-green-200"
                : "bg-red-50 border border-red-200"
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={`text-sm ${
                      result.success ? "text-green-800" : "text-red-800"
                    }`}
                  >
                    {result.success ? "✅" : "❌"}
                  </span>
                  <span className="font-semibold text-sm text-gray-900">
                    {result.name}
                  </span>
                  <span className="text-xs text-gray-500">
                    {result.duration}
                  </span>
                </div>

                <div className="text-xs space-y-1">
                  <div>
                    <span className="font-semibold">Status:</span>
                    <span className="font-mono ml-1">{result.status}</span>
                  </div>

                  {result.note && (
                    <div>
                      <span className="font-semibold">Note:</span>
                      <span className="ml-1">{result.note}</span>
                    </div>
                  )}

                  {result.error && (
                    <div>
                      <span className="font-semibold text-red-700">Error:</span>
                      <span className="ml-1 text-red-600">{result.error}</span>
                    </div>
                  )}

                  {result.data && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                        View Response Data
                      </summary>
                      <pre className="mt-1 p-2 bg-gray-100 rounded text-xs overflow-auto max-h-32">
                        {JSON.stringify(result.data, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>

              <span className="text-xs text-gray-500 ml-4">
                {result.timestamp}
              </span>
            </div>
          </div>
        ))}
      </div>

      {results.length === 0 && !testing && (
        <div className="text-center py-8">
          <p className="text-gray-500">
            No test results yet. Click "Re-test" to start.
          </p>
        </div>
      )}
    </div>
  );
};
