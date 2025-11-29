/**
 * RAG Settings Tab - Index file for all RAG settings components
 * Centralized export for all RAG settings functionality
 */

// Main component
export { default as RagSettingsTab } from "./RagSettingsTab";

// Individual components
export { ModelSelector } from "./ModelSelector";
export { EmbeddingSelector } from "./EmbeddingSelector";
export { RerankerSelector } from "./RerankerSelector";
export { SettingsSaveButton } from "./SettingsSaveButton";

// Hook
export { useRagSettings } from "../../hooks/useRagSettings";

// Re-export the main component as default
export { default } from "./RagSettingsTab";
