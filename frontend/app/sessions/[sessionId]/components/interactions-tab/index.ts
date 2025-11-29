/**
 * Interactions Tab Components - Main exports
 * Phase 6: Complete interaction management system
 */

// Main tab component
export { default as InteractionsTab } from "./InteractionsTab";

// Individual interaction components
export {
  default as InteractionCard,
  InteractionCardCompact,
  InteractionCardMobile,
} from "./InteractionCard";
export { default as InteractionModal } from "./InteractionModal";

// UI components
export {
  default as InteractionsPagination,
  InteractionsPaginationAdvanced,
} from "./InteractionsPagination";
export { default as InteractionsEmpty } from "./InteractionsEmpty";

// Re-export the hook for convenience
export { useInteractions } from "../../hooks/useInteractions";

// Type exports for external use
export type {
  UseInteractionsState,
  UseInteractionsActions,
  UseInteractionsOptions,
} from "../../hooks/useInteractions";
