/**
 * TypeScript Type Definitions for Module Management System
 * Comprehensive types for educational module extraction and management
 */

// ===== CORE MODULE TYPES =====

export interface Module {
  module_id: number;
  session_id: string;
  module_title: string;
  module_description: string | null;
  module_order: number;
  estimated_duration_minutes: number | null;
  difficulty_level: "beginner" | "intermediate" | "advanced";
  learning_objectives: string[];
  prerequisites: number[]; // Other module IDs
  created_at: string;
  updated_at: string;
  is_active: boolean;
  extraction_metadata: {
    extraction_method: string;
    confidence_score: number;
    curriculum_standard: string | null;
    topic_count: number;
    word_count: number;
  } | null;
}

export interface ModuleTemplate {
  template_id: number;
  template_name: string;
  description: string;
  grade_level: string;
  subject_area: string;
  curriculum_standard: string;
  structure_template: {
    sections: Array<{
      section_name: string;
      section_order: number;
      required: boolean;
    }>;
    assessment_types: string[];
    learning_activities: string[];
  };
  created_at: string;
  updated_at: string;
}

export interface ModuleProgress {
  progress_id: number;
  user_id: number;
  module_id: number;
  session_id: string;
  completion_status: "not_started" | "in_progress" | "completed" | "mastered";
  progress_percentage: number;
  time_spent_minutes: number;
  last_accessed: string;
  learning_objectives_completed: number[];
  assessment_scores: {
    objective_id: number;
    score: number;
    max_score: number;
    completion_date: string;
  }[];
  difficulty_adjustments: {
    original_level: string;
    current_level: string;
    reason: string;
    adjusted_at: string;
  }[];
  created_at: string;
  updated_at: string;
}

export interface ModuleTopicRelationship {
  relationship_id: number;
  module_id: number;
  topic_id: number;
  relationship_type: "core" | "supplementary" | "prerequisite" | "extension";
  weight: number;
  sequence_order: number;
  created_at: string;
}

// ===== MODULE EXTRACTION TYPES =====

export interface ModuleExtractionJob {
  job_id: number;
  session_id: string;
  user_id: number;
  extraction_type: "automatic" | "manual" | "hybrid";
  status: "pending" | "in_progress" | "completed" | "failed";
  curriculum_prompt: string;
  extraction_config: {
    max_modules: number;
    min_topics_per_module: number;
    difficulty_distribution: {
      beginner: number;
      intermediate: number;
      advanced: number;
    };
    include_assessments: boolean;
    curriculum_alignment: boolean;
  };
  result_summary: {
    modules_extracted: number;
    topics_organized: number;
    learning_objectives_generated: number;
    processing_time_ms: number;
    confidence_scores: {
      average: number;
      min: number;
      max: number;
    };
  } | null;
  error_details: {
    error_type: string;
    error_message: string;
    failed_step: string;
    recoverable: boolean;
  } | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface CurriculumStandard {
  standard_id: number;
  country: string;
  education_level: "elementary" | "middle" | "high" | "university";
  subject_area: string;
  grade_level: string;
  standard_code: string;
  standard_title: string;
  description: string;
  learning_domains: string[];
  assessment_criteria: string[];
  prerequisites: string[];
  created_at: string;
  updated_at: string;
}

// ===== API REQUEST/RESPONSE TYPES =====

export interface ModuleExtractionRequest {
  session_id: string;
  curriculum_prompt: string;
  extraction_config?: {
    max_modules?: number;
    min_topics_per_module?: number;
    difficulty_distribution?: {
      beginner?: number;
      intermediate?: number;
      advanced?: number;
    };
    include_assessments?: boolean;
    curriculum_alignment?: boolean;
    force_refresh?: boolean;
  };
  curriculum_standard_id?: number;
  template_id?: number;
}

export interface ModuleExtractionResponse {
  success: boolean;
  job_id: number;
  message: string;
  estimated_processing_time: number;
  modules_preview?: Array<{
    title: string;
    description: string;
    topic_count: number;
    difficulty_level: string;
  }>;
}

export interface ModuleListResponse {
  success: boolean;
  modules: Module[];
  total_count: number;
  session_id: string;
  extraction_job?: {
    job_id: number;
    status: string;
    created_at: string;
  };
}

export interface ModuleDetailsResponse {
  success: boolean;
  module: Module;
  topics: Array<{
    topic_id: number;
    topic_title: string;
    relationship_type: string;
    sequence_order: number;
  }>;
  progress_stats?: {
    total_students: number;
    students_started: number;
    students_completed: number;
    average_progress: number;
    average_completion_time: number;
  };
}

export interface ModuleProgressResponse {
  success: boolean;
  progress: ModuleProgress;
  module: Module;
  next_objectives: Array<{
    objective_id: number;
    objective_text: string;
    estimated_time: number;
  }>;
  recommendations: Array<{
    type: "review" | "practice" | "advance" | "help";
    message: string;
    action_url?: string;
  }>;
}

// ===== TEACHER DASHBOARD TYPES =====

export interface TeacherModuleDashboard {
  session_id: string;
  session_name: string;
  total_modules: number;
  extraction_status: "not_started" | "in_progress" | "completed" | "error";
  last_extraction: {
    job_id: number;
    completed_at: string;
    modules_count: number;
  } | null;
  curriculum_alignment: {
    standard_name: string;
    alignment_percentage: number;
    gaps: string[];
  } | null;
  student_engagement: {
    total_students: number;
    active_students: number;
    modules_in_progress: number;
    average_completion_rate: number;
  };
  quality_metrics: {
    average_confidence: number;
    modules_need_review: number;
    objectives_coverage: number;
    assessment_alignment: number;
  };
}

export interface StudentModuleOverview {
  user_id: number;
  session_id: string;
  available_modules: Array<{
    module: Module;
    progress: ModuleProgress | null;
    is_unlocked: boolean;
    prerequisites_met: boolean;
    estimated_time_remaining: number;
  }>;
  current_module: {
    module: Module;
    progress: ModuleProgress;
    current_objective: {
      objective_id: number;
      objective_text: string;
      progress_percentage: number;
    } | null;
  } | null;
  learning_path: {
    completed_modules: number;
    total_modules: number;
    next_recommended: Module | null;
    mastery_level: "beginner" | "developing" | "proficient" | "advanced";
  };
  achievements: Array<{
    achievement_type:
      | "module_completed"
      | "objective_mastered"
      | "streak"
      | "excellence";
    title: string;
    description: string;
    earned_at: string;
    icon: string;
  }>;
}

// ===== FORM AND UI TYPES =====

export interface CurriculumPromptForm {
  curriculum_prompt: string;
  subject_area: string;
  grade_level: string;
  education_system: string;
  max_modules: number;
  difficulty_distribution: {
    beginner: number;
    intermediate: number;
    advanced: number;
  };
  include_assessments: boolean;
  curriculum_alignment: boolean;
  template_id?: number;
}

export interface ModuleEditForm {
  module_title: string;
  module_description: string;
  difficulty_level: "beginner" | "intermediate" | "advanced";
  estimated_duration_minutes: number;
  learning_objectives: string[];
  prerequisites: number[];
  is_active: boolean;
}

export interface ModuleQualityReview {
  module_id: number;
  review_status: "pending" | "approved" | "needs_revision" | "rejected";
  quality_score: number;
  review_comments: string;
  reviewer_id: number;
  review_criteria: {
    objective_clarity: number;
    content_alignment: number;
    difficulty_appropriateness: number;
    progression_logic: number;
    assessment_quality: number;
  };
  suggested_improvements: Array<{
    category: string;
    suggestion: string;
    priority: "high" | "medium" | "low";
  }>;
  reviewed_at: string;
}

// ===== ANALYTICS AND REPORTING TYPES =====

export interface ModuleAnalytics {
  module_id: number;
  time_period: {
    start_date: string;
    end_date: string;
  };
  engagement_metrics: {
    total_enrollments: number;
    active_learners: number;
    completion_rate: number;
    average_time_to_complete: number;
    dropout_points: Array<{
      objective_id: number;
      dropout_rate: number;
    }>;
  };
  learning_outcomes: {
    objective_mastery_rates: Record<number, number>;
    assessment_scores: {
      average: number;
      distribution: Record<string, number>;
    };
    skill_development: Array<{
      skill: string;
      improvement_rate: number;
    }>;
  };
  feedback_summary: {
    difficulty_ratings: Record<string, number>;
    satisfaction_scores: Record<string, number>;
    common_challenges: string[];
    improvement_suggestions: string[];
  };
}

export interface LearningPathAnalytics {
  session_id: string;
  student_journey: {
    total_students: number;
    path_completion_rate: number;
    average_journey_time: number;
    common_paths: Array<{
      path: number[];
      frequency: number;
      success_rate: number;
    }>;
  };
  module_relationships: {
    prerequisite_effectiveness: Record<string, number>;
    transition_success_rates: Record<string, number>;
    bottleneck_modules: number[];
  };
  personalization_impact: {
    adaptive_adjustments: number;
    difficulty_modifications: Record<string, number>;
    learning_preference_accommodations: Record<string, number>;
  };
}

// ===== ERROR AND STATUS TYPES =====

export interface ModuleError {
  error_type:
    | "extraction_failed"
    | "validation_error"
    | "api_error"
    | "permission_error";
  error_code: string;
  message: string;
  details?: {
    field?: string;
    expected?: string;
    received?: string;
    suggestions?: string[];
  };
}

export interface ModuleOperationStatus {
  operation: "extract" | "create" | "update" | "delete" | "review";
  status: "idle" | "loading" | "success" | "error";
  progress?: number;
  message?: string;
  error?: ModuleError;
}

// ===== UTILITY TYPES =====

export type ModuleDifficultyLevel = "beginner" | "intermediate" | "advanced";
export type ModuleCompletionStatus =
  | "not_started"
  | "in_progress"
  | "completed"
  | "mastered";
export type ModuleExtractionStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed";
export type ModuleReviewStatus =
  | "pending"
  | "approved"
  | "needs_revision"
  | "rejected";

// ===== TYPE GUARDS =====

export function isModule(obj: any): obj is Module {
  return (
    obj &&
    typeof obj.module_id === "number" &&
    typeof obj.session_id === "string" &&
    typeof obj.module_title === "string" &&
    typeof obj.module_order === "number" &&
    ["beginner", "intermediate", "advanced"].includes(obj.difficulty_level)
  );
}

export function isModuleProgress(obj: any): obj is ModuleProgress {
  return (
    obj &&
    typeof obj.progress_id === "number" &&
    typeof obj.user_id === "number" &&
    typeof obj.module_id === "number" &&
    ["not_started", "in_progress", "completed", "mastered"].includes(
      obj.completion_status
    )
  );
}

export function isModuleExtractionJob(obj: any): obj is ModuleExtractionJob {
  return (
    obj &&
    typeof obj.job_id === "number" &&
    typeof obj.session_id === "string" &&
    ["pending", "in_progress", "completed", "failed"].includes(obj.status)
  );
}

// ===== CONSTANTS =====

export const MODULE_DIFFICULTY_LEVELS: ModuleDifficultyLevel[] = [
  "beginner",
  "intermediate",
  "advanced",
];

export const MODULE_COMPLETION_STATUSES: ModuleCompletionStatus[] = [
  "not_started",
  "in_progress",
  "completed",
  "mastered",
];

export const CURRICULUM_STANDARDS = {
  TURKEY: {
    code: "TR_MEB",
    name: "Türkiye Millî Eğitim Bakanlığı Müfredatı",
    levels: ["ilkokul", "ortaokul", "lise"],
  },
  COMMON_CORE: {
    code: "US_CC",
    name: "Common Core State Standards",
    levels: ["elementary", "middle", "high"],
  },
} as const;

export const DEFAULT_EXTRACTION_CONFIG = {
  max_modules: 8,
  min_topics_per_module: 3,
  difficulty_distribution: {
    beginner: 40,
    intermediate: 40,
    advanced: 20,
  },
  include_assessments: true,
  curriculum_alignment: true,
} as const;

// ===== HELPER FUNCTIONS =====

export function getDifficultyColor(level: ModuleDifficultyLevel): string {
  switch (level) {
    case "beginner":
      return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300";
    case "intermediate":
      return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300";
    case "advanced":
      return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300";
  }
}

export function getDifficultyLabel(level: ModuleDifficultyLevel): string {
  switch (level) {
    case "beginner":
      return "Başlangıç";
    case "intermediate":
      return "Orta";
    case "advanced":
      return "İleri";
  }
}

export function getCompletionStatusColor(
  status: ModuleCompletionStatus
): string {
  switch (status) {
    case "not_started":
      return "text-gray-500";
    case "in_progress":
      return "text-blue-600";
    case "completed":
      return "text-green-600";
    case "mastered":
      return "text-purple-600";
  }
}

export function getCompletionStatusLabel(
  status: ModuleCompletionStatus
): string {
  switch (status) {
    case "not_started":
      return "Başlanmadı";
    case "in_progress":
      return "Devam Ediyor";
    case "completed":
      return "Tamamlandı";
    case "mastered":
      return "Ustalaşıldı";
  }
}

export function calculateModuleProgress(progress: ModuleProgress): {
  percentage: number;
  timeSpent: string;
  objectivesCompleted: number;
  totalObjectives: number;
} {
  const hours = Math.floor(progress.time_spent_minutes / 60);
  const minutes = progress.time_spent_minutes % 60;
  const timeSpent = hours > 0 ? `${hours}s ${minutes}dk` : `${minutes}dk`;

  return {
    percentage: progress.progress_percentage,
    timeSpent,
    objectivesCompleted: progress.learning_objectives_completed.length,
    totalObjectives:
      progress.learning_objectives_completed.length +
      (progress.assessment_scores?.length || 0),
  };
}
