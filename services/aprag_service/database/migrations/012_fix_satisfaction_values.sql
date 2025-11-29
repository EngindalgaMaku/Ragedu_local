-- Migration: Fix satisfaction values that are incorrectly set to same as understanding
-- This migration ensures that satisfaction is only set when multi-dimensional feedback is provided

-- Step 1: Find profiles where satisfaction equals understanding (likely from old bug)
-- and set satisfaction to NULL if no multi-dimensional feedback exists

-- Check if detailed_feedback_entries table exists
-- If a user has only emoji feedback (no multi-dimensional feedback),
-- their satisfaction should be NULL, not equal to understanding

-- Update profiles where satisfaction = understanding and no multi-dimensional feedback exists
UPDATE student_profiles
SET average_satisfaction = NULL
WHERE average_understanding IS NOT NULL
  AND average_satisfaction IS NOT NULL
  AND ABS(average_understanding - average_satisfaction) < 0.01
  AND user_id || '|' || session_id NOT IN (
    SELECT DISTINCT user_id || '|' || session_id
    FROM detailed_feedback_entries
    WHERE feedback_type = 'multi_dimensional'
  );

-- Log the fix
-- This ensures that satisfaction is only shown when it has been explicitly set
-- via multi-dimensional feedback (relevance + clarity)



