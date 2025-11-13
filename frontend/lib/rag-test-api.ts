/**
 * RAG Testing API Client
 *
 * Functions for interacting with RAG testing endpoints
 */

import { URLS } from "@/config/ports";

const API_BASE = URLS.API_GATEWAY;

export interface TestResult {
  test_id: string;
  query: string;
  expected_relevant: boolean;
  expected_result: string;
  actual_result: string;
  passed: boolean;
  documents_retrieved: number;
  execution_time_ms: number;
  category?: string;
  results_preview?: Array<{
    text: string;
    score: number;
    evaluator_score?: number;
  }>;
}

export interface TestSummary {
  total_tests: number;
  passed: number;
  failed: number;
  success_rate: number;
  avg_execution_time_ms: number;
  false_positive_rate: number;
  false_negative_rate: number;
  false_positives: number;
  false_negatives: number;
  timestamp: string;
}

export interface SystemStatus {
  chromadb_available: boolean;
  document_count: number;
  chunk_count: number;
  crag_enabled: boolean;
  crag_evaluator_available: boolean;
  test_generation_available: boolean;
  crag_config?: {
    loaded: boolean;
    model_name: string;
    correct_threshold: number;
    incorrect_threshold: number;
    filter_threshold: number;
  };
}

export interface ManualTestRequest {
  query: string;
  expected_relevant: boolean;
  category?: string;
}

export interface AutoTestConfig {
  num_tests: number;
  include_relevant: boolean;
  include_irrelevant: boolean;
}

export interface TestQuery {
  query: string;
  expected_relevant: boolean;
  category: string;
  source_chunk?: string;
}

/**
 * Get RAG testing system status
 */
export async function getRAGTestStatus(token: string): Promise<SystemStatus> {
  const response = await fetch(`${API_BASE}/api/admin/rag-tests/status`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    throw new Error("Failed to get RAG test status");
  }

  const data = await response.json();
  return data.status;
}

/**
 * Execute a manual test
 */
export async function executeManualTest(
  token: string,
  request: ManualTestRequest
): Promise<TestResult> {
  const response = await fetch(
    `${API_BASE}/api/admin/rag-tests/execute-manual`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(request),
    }
  );

  if (!response.ok) {
    throw new Error("Failed to execute manual test");
  }

  const data = await response.json();
  return data.test_result;
}

/**
 * Generate automatic test queries
 */
export async function generateAutoTests(
  token: string,
  config: AutoTestConfig
): Promise<TestQuery[]> {
  const response = await fetch(
    `${API_BASE}/api/admin/rag-tests/generate-auto-tests`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(config),
    }
  );

  if (!response.ok) {
    throw new Error("Failed to generate auto tests");
  }

  const data = await response.json();
  return data.test_queries;
}

/**
 * Execute batch tests
 */
export async function executeBatchTests(
  token: string,
  testQueries: TestQuery[]
): Promise<{ summary: TestSummary; results: TestResult[] }> {
  const response = await fetch(
    `${API_BASE}/api/admin/rag-tests/execute-batch`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(testQueries),
    }
  );

  if (!response.ok) {
    throw new Error("Failed to execute batch tests");
  }

  const data = await response.json();
  return {
    summary: data.summary,
    results: data.results,
  };
}

/**
 * Get sample test queries
 */
export async function getSampleQueries(
  token: string,
  category: "relevant" | "irrelevant" | "all" = "all"
): Promise<TestQuery[]> {
  const response = await fetch(
    `${API_BASE}/api/admin/rag-tests/sample-queries?category=${category}`,
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error("Failed to get sample queries");
  }

  const data = await response.json();
  return data.queries;
}

/**
 * Export test results to CSV
 */
export function exportTestResultsToCSV(
  results: TestResult[],
  filename: string = "rag_test_results.csv"
) {
  const headers = [
    "Test ID",
    "Query",
    "Expected",
    "Actual",
    "Passed",
    "Documents",
    "Time (ms)",
    "Category",
  ];

  const rows = results.map((r) => [
    r.test_id,
    `"${r.query.replace(/"/g, '""')}"`,
    r.expected_result,
    r.actual_result,
    r.passed ? "PASS" : "FAIL",
    r.documents_retrieved,
    (r.execution_time_ms || 0).toFixed(2),
    r.category || "unknown",
  ]);

  const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");

  // Create download link
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);

  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  link.style.visibility = "hidden";

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Export summary to JSON
 */
export function exportSummaryToJSON(
  summary: TestSummary,
  results: TestResult[],
  filename: string = "rag_test_report.json"
) {
  const report = {
    generated_at: new Date().toISOString(),
    summary,
    results,
  };

  const json = JSON.stringify(report, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);

  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  link.style.visibility = "hidden";

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
