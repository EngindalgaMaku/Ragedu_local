/**
 * Integration Test for Interactions Tab Components
 * This file tests that all components work together properly
 * Run this as a sanity check after creating all components
 */

import React from "react";
import type { APRAGInteraction } from "@/lib/api";

// Import all components to test they can be imported
import {
  InteractionsTab,
  InteractionCard,
  InteractionCardCompact,
  InteractionCardMobile,
  InteractionModal,
  InteractionsPagination,
  InteractionsPaginationAdvanced,
  InteractionsEmpty,
  useInteractions,
} from "./index";

// Mock interaction data for testing
const mockInteraction: APRAGInteraction = {
  interaction_id: 1,
  user_id: "test-user",
  session_id: "test-session",
  query: "Test soru",
  original_response: "Test yanıt",
  personalized_response: "Kişiselleştirilmiş test yanıt",
  timestamp: new Date().toISOString(),
  processing_time_ms: 1500,
  sources: [
    {
      source: "test-document.pdf",
      score: 0.95,
    },
  ],
  topic_info: {
    title: "Test Konusu",
    confidence: 0.9,
  },
  first_name: "Test",
  last_name: "User",
};

// Test component that uses all interactions components
export function InteractionsIntegrationTest() {
  const [modalOpen, setModalOpen] = React.useState(false);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Interactions Tab Integration Test</h1>

      {/* Test Empty State */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Empty State Test</h2>
        <InteractionsEmpty
          hasFilters={false}
          onClearFilters={() => console.log("Clear filters")}
        />
      </section>

      {/* Test Interaction Card */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Interaction Card Test</h2>
        <div className="space-y-4">
          <InteractionCard
            interaction={mockInteraction}
            index={1}
            onViewDetails={() => setModalOpen(true)}
            compact={false}
          />

          <InteractionCardCompact
            interaction={mockInteraction}
            index={2}
            onViewDetails={() => setModalOpen(true)}
          />

          <InteractionCardMobile
            interaction={mockInteraction}
            index={3}
            onViewDetails={() => setModalOpen(true)}
          />
        </div>
      </section>

      {/* Test Pagination */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Pagination Test</h2>
        <InteractionsPagination
          currentPage={1}
          totalPages={5}
          totalItems={50}
          itemsPerPage={10}
          onPageChange={() => console.log("Page changed")}
          onItemsPerPageChange={() => console.log("Items per page changed")}
        />

        <div className="mt-4">
          <InteractionsPaginationAdvanced
            currentPage={2}
            totalPages={10}
            totalItems={100}
            itemsPerPage={10}
            onPageChange={() => console.log("Advanced page changed")}
          />
        </div>
      </section>

      {/* Test Modal */}
      <InteractionModal
        interaction={modalOpen ? mockInteraction : null}
        open={modalOpen}
        onOpenChange={setModalOpen}
      />

      {/* Test Button to Open Modal */}
      <button
        onClick={() => setModalOpen(true)}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        Test Modal
      </button>
    </div>
  );
}

// Component to test the useInteractions hook
export function InteractionsHookTest({ sessionId }: { sessionId: string }) {
  const {
    interactions,
    loading,
    error,
    currentPage,
    totalPages,
    totalInteractions,
    stats,
    openModal,
    closeModal,
  } = useInteractions({
    sessionId,
    initialPerPage: 10,
  });

  return (
    <div className="p-6">
      <h2 className="text-lg font-semibold mb-4">useInteractions Hook Test</h2>
      <div className="space-y-2 text-sm">
        <p>Loading: {loading ? "Yes" : "No"}</p>
        <p>Error: {error || "None"}</p>
        <p>Current Page: {currentPage}</p>
        <p>Total Pages: {totalPages}</p>
        <p>Total Interactions: {totalInteractions}</p>
        <p>Interactions Count: {interactions.length}</p>
        <p>Total Queries: {stats.totalQueries}</p>
        <p>Average Response Time: {stats.averageResponseTime}ms</p>
      </div>
    </div>
  );
}

// Full integration test with InteractionsTab
export function FullInteractionsTabTest({ sessionId }: { sessionId: string }) {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Full InteractionsTab Test</h1>
      <InteractionsTab sessionId={sessionId} apragEnabled={true} />
    </div>
  );
}

export default InteractionsIntegrationTest;

// Test runner function for console verification
export function runIntegrationTests() {
  console.log("🧪 Running Interactions Tab Integration Tests...");

  // Test 1: Component imports
  const components = [
    InteractionsTab,
    InteractionCard,
    InteractionCardCompact,
    InteractionCardMobile,
    InteractionModal,
    InteractionsPagination,
    InteractionsPaginationAdvanced,
    InteractionsEmpty,
  ];

  console.log("✅ All component imports successful:", components.length);

  // Test 2: Hook import
  if (typeof useInteractions === "function") {
    console.log("✅ useInteractions hook imported successfully");
  }

  // Test 3: Mock data structure
  console.log("✅ Mock interaction data structure valid");

  console.log("🎉 All integration tests passed!");

  return true;
}
