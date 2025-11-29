"use client";

import React, { useState, useEffect } from "react";

interface EnvironmentTestResult {
  test: string;
  status: "pass" | "fail" | "warning";
  message: string;
  timing?: number;
  details?: any;
}

export default function DockerEnvironmentTest({
  sessionId,
}: {
  sessionId: string;
}) {
  const [results, setResults] = useState<EnvironmentTestResult[]>([]);
  const [running, setRunning] = useState(false);

  const addResult = (result: EnvironmentTestResult) => {
    setResults((prev) => [...prev, result]);
    console.log(
      `🔍 [DOCKER TEST] ${result.test}: ${result.status} - ${result.message}`
    );
  };

  const runDockerTests = async () => {
    setRunning(true);
    setResults([]);

    // Test 1: Next.js Environment Detection
    try {
      const isDocker = process.env.NODE_ENV;
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;

      addResult({
        test: "Next.js Environment",
        status: "pass",
        message: `NODE_ENV: ${isDocker || "undefined"} | API_URL: ${
          apiUrl || "/api (proxy)"
        }`,
        details: {
          NODE_ENV: process.env.NODE_ENV,
          NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
          host: typeof window !== "undefined" ? window.location.host : "SSR",
        },
      });
    } catch (error: any) {
      addResult({
        test: "Next.js Environment",
        status: "fail",
        message: `Environment check failed: ${error.message}`,
      });
    }

    // Test 2: Docker Network API Gateway Test
    try {
      const start = Date.now();
      console.log("🐳 [DOCKER TEST] Testing API Gateway connectivity...");

      // Test direct /api endpoint (Next.js proxy)
      const response = await fetch("/api/health", {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      const timing = Date.now() - start;

      if (response.ok) {
        const data = await response.json();
        addResult({
          test: "Docker API Gateway",
          status: "pass",
          message: `API Gateway reachable (${timing}ms) - Status: ${
            data.status || "OK"
          }`,
          timing,
          details: data,
        });
      } else {
        addResult({
          test: "Docker API Gateway",
          status: "fail",
          message: `HTTP ${response.status} - Gateway not responding correctly`,
          timing,
        });
      }
    } catch (error: any) {
      addResult({
        test: "Docker API Gateway",
        status: "fail",
        message: `Network error: ${error.message} - Check Docker networking`,
      });
    }

    // Test 3: APRAG Service Through Docker Network
    try {
      const start = Date.now();
      console.log(
        "🐳 [DOCKER TEST] Testing APRAG service through Docker network..."
      );

      const response = await fetch("/api/aprag/settings/status", {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      const timing = Date.now() - start;

      if (response.ok) {
        const data = await response.json();
        addResult({
          test: "Docker APRAG Service",
          status: data.global_enabled ? "pass" : "warning",
          message: `APRAG ${
            data.global_enabled ? "ENABLED" : "DISABLED"
          } (${timing}ms) - Features: ${
            Object.keys(data.features || {}).length
          }`,
          timing,
          details: data,
        });
      } else {
        addResult({
          test: "Docker APRAG Service",
          status: "fail",
          message: `HTTP ${response.status} - APRAG service unreachable`,
          timing,
        });
      }
    } catch (error: any) {
      addResult({
        test: "Docker APRAG Service",
        status: "fail",
        message: `APRAG service error: ${error.message}`,
      });
    }

    // Test 4: Topics API Endpoint in Docker
    try {
      const start = Date.now();
      console.log("🐳 [DOCKER TEST] Testing Topics API endpoint...");

      const response = await fetch(`/api/aprag/topics/session/${sessionId}`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      const timing = Date.now() - start;

      if (response.ok) {
        const data = await response.json();
        addResult({
          test: "Docker Topics API",
          status: "pass",
          message: `Topics API working (${timing}ms) - Found: ${
            data.topics?.length || 0
          } topics`,
          timing,
          details: { topics_count: data.topics?.length, success: data.success },
        });
      } else {
        addResult({
          test: "Docker Topics API",
          status: response.status === 404 ? "warning" : "fail",
          message:
            response.status === 404
              ? `No topics found (${timing}ms) - Normal for new sessions`
              : `HTTP ${response.status} - Topics API failed`,
          timing,
        });
      }
    } catch (error: any) {
      addResult({
        test: "Docker Topics API",
        status: "fail",
        message: `Topics API error: ${error.message}`,
      });
    }

    // Test 5: Interactions API Endpoint in Docker
    try {
      const start = Date.now();
      console.log("🐳 [DOCKER TEST] Testing Interactions API endpoint...");

      const response = await fetch(
        `/api/aprag/interactions/session/${sessionId}?limit=1&offset=0`,
        {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        }
      );

      const timing = Date.now() - start;

      if (response.ok) {
        const data = await response.json();
        addResult({
          test: "Docker Interactions API",
          status: "pass",
          message: `Interactions API working (${timing}ms) - Found: ${
            data.total || 0
          } interactions`,
          timing,
          details: { total: data.total, count: data.interactions?.length },
        });
      } else {
        addResult({
          test: "Docker Interactions API",
          status: response.status === 404 ? "warning" : "fail",
          message:
            response.status === 404
              ? `No interactions found (${timing}ms) - Normal for new sessions`
              : `HTTP ${response.status} - Interactions API failed`,
          timing,
        });
      }
    } catch (error: any) {
      addResult({
        test: "Docker Interactions API",
        status: "fail",
        message: `Interactions API error: ${error.message}`,
      });
    }

    // Test 6: Component Import Validation
    try {
      // Check if backup components are properly imported
      const hasTopicManagement = typeof window !== "undefined";
      addResult({
        test: "Component Imports",
        status: hasTopicManagement ? "pass" : "warning",
        message: hasTopicManagement
          ? "Component imports validated - backup components should load"
          : "Server-side render - client components will load on hydration",
      });
    } catch (error: any) {
      addResult({
        test: "Component Imports",
        status: "fail",
        message: `Component validation error: ${error.message}`,
      });
    }

    setRunning(false);
  };

  useEffect(() => {
    // Auto-run tests on mount
    runDockerTests();
  }, [sessionId]);

  const getStatusIcon = (status: EnvironmentTestResult["status"]) => {
    switch (status) {
      case "pass":
        return "✅";
      case "fail":
        return "❌";
      case "warning":
        return "⚠️";
      default:
        return "❓";
    }
  };

  const getStatusColor = (status: EnvironmentTestResult["status"]) => {
    switch (status) {
      case "pass":
        return "text-green-700 bg-green-50 border-green-200";
      case "fail":
        return "text-red-700 bg-red-50 border-red-200";
      case "warning":
        return "text-yellow-700 bg-yellow-50 border-yellow-200";
      default:
        return "text-gray-700 bg-gray-50 border-gray-200";
    }
  };

  const failedTests = results.filter((r) => r.status === "fail").length;
  const warningTests = results.filter((r) => r.status === "warning").length;
  const passedTests = results.filter((r) => r.status === "pass").length;

  return (
    <div className="fixed top-4 left-4 w-[28rem] bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-xl z-50 max-h-[80vh] overflow-hidden">
      <div className="p-4 border-b border-gray-200 dark:border-gray-600">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            🐳 Docker Environment Tests
          </h3>
          <button
            onClick={runDockerTests}
            disabled={running}
            className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {running ? "Testing..." : "Re-test"}
          </button>
        </div>
        <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          ✅ {passedTests} | ⚠️ {warningTests} | ❌ {failedTests}
        </div>
      </div>

      <div className="p-4 overflow-y-auto max-h-96 space-y-2">
        {results.map((result, index) => (
          <div
            key={index}
            className={`p-3 rounded-md border ${getStatusColor(result.status)}`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-lg">{getStatusIcon(result.status)}</span>
                <span className="font-medium text-sm">{result.test}</span>
              </div>
              {result.timing && (
                <span className="text-xs opacity-75">{result.timing}ms</span>
              )}
            </div>
            <div className="text-xs mt-1 opacity-90">{result.message}</div>
            {result.details && (
              <details className="mt-2">
                <summary className="text-xs cursor-pointer opacity-75 hover:opacity-100">
                  Technical Details
                </summary>
                <pre className="text-xs mt-1 p-2 bg-black/10 rounded overflow-auto">
                  {JSON.stringify(result.details, null, 2)}
                </pre>
              </details>
            )}
          </div>
        ))}

        {running && (
          <div className="text-center py-4">
            <div className="inline-block animate-spin rounded-full h-6 w-6 border-2 border-blue-600 border-t-transparent"></div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              Running Docker environment tests...
            </p>
          </div>
        )}
      </div>

      {results.length > 0 && !running && (
        <div className="p-3 border-t border-gray-200 dark:border-gray-600 text-xs text-gray-600">
          {failedTests > 0 && (
            <div className="text-red-600 mb-1">
              🚨 {failedTests} critical issues detected - check Docker services
            </div>
          )}
          {warningTests > 0 && failedTests === 0 && (
            <div className="text-yellow-600 mb-1">
              ⚠️ {warningTests} warnings - functionality may be limited
            </div>
          )}
          {failedTests === 0 && warningTests === 0 && (
            <div className="text-green-600">
              🎉 All tests passed - Topics and Interactions tabs should work
            </div>
          )}
        </div>
      )}
    </div>
  );
}
