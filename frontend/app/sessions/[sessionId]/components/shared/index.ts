/**
 * Shared Components Export Index
 * Centralized exports for all shared components and hooks
 */

// Main Components
export {
  default as TabNavigation,
  MobileTabNavigation,
  CompactTabNavigation,
} from "./TabNavigation";
export type { TabId, TabItem } from "./TabNavigation";

export {
  default as LoadingSpinner,
  Skeleton,
  DocumentsLoadingSkeleton,
  ChunksLoadingSkeleton,
  RagSettingsLoadingSkeleton,
  SessionSettingsLoadingSkeleton,
  TopicsLoadingSkeleton,
  InteractionsLoadingSkeleton,
  LoadingOverlay,
  PageLoadingSpinner,
} from "./LoadingSpinner";
export type { LoadingSpinnerProps, SkeletonProps } from "./LoadingSpinner";

export {
  default as ErrorBoundary,
  SessionErrorBoundary,
  TabErrorBoundary,
  APIErrorBoundary,
  withErrorBoundary,
  useErrorBoundaryHandler,
} from "./ErrorBoundary";

export {
  default as EmptyState,
  DocumentsEmptyState,
  ChunksEmptyState,
  RagSettingsEmptyState,
  SessionSettingsEmptyState,
  TopicsEmptyState,
  InteractionsEmptyState,
  SearchEmptyState,
  LoadingEmptyState,
  ErrorEmptyState,
} from "./EmptyState";
export type {
  EmptyStateProps,
  EmptyStateAction,
  EmptyStateType,
} from "./EmptyState";

// Hooks are exported from their respective locations
// Import like: import { useTabNavigation } from '../../hooks/useTabNavigation'
