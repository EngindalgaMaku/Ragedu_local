"""
Module Quality Validation System
Ensures extracted modules meet educational quality standards and curriculum compliance
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    ERROR = "error"      # Must be fixed before accepting module
    WARNING = "warning"  # Should be fixed but not blocking
    INFO = "info"        # Informational only


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a module"""
    severity: ValidationSeverity
    category: str
    message: str
    suggested_fix: str
    affected_items: List[str]
    auto_fixable: bool = False
    module_id: Optional[int] = None


class ModuleQualityValidator:
    """Validates educational modules for quality and compliance"""

    def __init__(self, curriculum_standards_service=None):
        self.curriculum_standards = curriculum_standards_service
        self.logger = logging.getLogger(__name__ + '.ModuleQualityValidator')
        self.logger.info("Module Quality Validator initialized")

    async def validate_module_set(
        self,
        modules: List[Dict[str, Any]],
        course_context: Dict[str, Any],
        validation_options: Dict[str, Any] = None
    ) -> Tuple[bool, List[ValidationIssue], List[Dict[str, Any]]]:
        """
        Validate complete set of modules for a course
        
        Args:
            modules: List of module candidates to validate
            course_context: Course information and context
            validation_options: Validation configuration options
            
        Returns:
            Tuple of (is_valid, issues_list, corrected_modules)
        """
        self.logger.info(f"Validating set of {len(modules)} modules")
        
        if not modules:
            return False, [ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="input",
                message="No modules provided for validation",
                suggested_fix="Provide at least one module for validation",
                affected_items=["modules"]
            )], []

        all_issues = []
        corrected_modules = []
        options = validation_options or {}

        # Course-level validation
        course_issues = await self._validate_course_level_requirements(modules, course_context)
        all_issues.extend(course_issues)

        # Individual module validation
        for i, module in enumerate(modules):
            is_valid, module_issues, corrected_module = await self._validate_individual_module(
                module, course_context, i + 1
            )

            all_issues.extend(module_issues)
            corrected_modules.append(corrected_module)

        # Inter-module relationship validation
        relationship_issues = await self._validate_module_relationships(corrected_modules, course_context)
        all_issues.extend(relationship_issues)

        # Apply auto-fixes if enabled
        if options.get('apply_auto_fixes', True):
            corrected_modules, fix_issues = await self._apply_auto_fixes(corrected_modules, all_issues, course_context)
            all_issues.extend(fix_issues)

        # Determine overall validity
        error_count = sum(1 for issue in all_issues if issue.severity == ValidationSeverity.ERROR)
        is_valid = error_count == 0

        self.logger.info(f"Validation completed: {len(all_issues)} issues ({error_count} errors), valid: {is_valid}")
        return is_valid, all_issues, corrected_modules

    async def _validate_course_level_requirements(
        self,
        modules: List[Dict[str, Any]],
        course_context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate course-level requirements"""
        issues = []

        # Module count validation
        module_count = len(modules)
        min_modules = course_context.get('min_modules', 2)
        max_modules = course_context.get('max_modules', 15)

        if module_count < min_modules:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="course_structure",
                message=f"Course has only {module_count} modules, expected at least {min_modules}",
                suggested_fix=f"Add more modules or combine with related content to reach at least {min_modules} modules",
                affected_items=["module_count"],
                auto_fixable=False
            ))
        elif module_count > max_modules:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="course_structure",
                message=f"Course has {module_count} modules, which exceeds recommended maximum of {max_modules}",
                suggested_fix="Consider combining related modules or splitting the course",
                affected_items=["module_count"],
                auto_fixable=False
            ))

        # Total duration validation
        total_duration = sum(module.get('estimated_duration_hours', 0) for module in modules)
        expected_duration = course_context.get('total_hours', 144)  # Default for Turkish high school

        if total_duration < expected_duration * 0.7:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="course_duration",
                message=f"Total module duration ({total_duration}h) is significantly less than expected ({expected_duration}h)",
                suggested_fix="Increase module durations or add more comprehensive content",
                affected_items=["total_duration"],
                auto_fixable=True
            ))
        elif total_duration > expected_duration * 1.3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="course_duration",
                message=f"Total module duration ({total_duration}h) significantly exceeds expected ({expected_duration}h)",
                suggested_fix="Reduce module durations or split into multiple courses",
                affected_items=["total_duration"],
                auto_fixable=True
            ))

        # Curriculum coverage validation
        if course_context.get('curriculum_standards'):
            coverage_issues = await self._validate_curriculum_coverage(modules, course_context)
            issues.extend(coverage_issues)

        # Module order validation
        order_issues = await self._validate_module_ordering(modules)
        issues.extend(order_issues)

        return issues

    async def _validate_curriculum_coverage(
        self,
        modules: List[Dict[str, Any]],
        course_context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate curriculum standards coverage"""
        issues = []
        
        # Collect all curriculum standards mentioned in modules
        covered_standards = set()
        for module in modules:
            standards = module.get('curriculum_standards', [])
            if isinstance(standards, list):
                covered_standards.update(standards)
        
        # Get expected standards from course context
        expected_standards = course_context.get('curriculum_standards', [])
        expected_standard_codes = {std.get('standard_code', '') for std in expected_standards if std.get('standard_code')}
        
        # Find missing standards
        missing_standards = expected_standard_codes - covered_standards
        if missing_standards and len(missing_standards) > len(expected_standard_codes) * 0.3:  # More than 30% missing
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="curriculum_coverage",
                message=f"Significant curriculum standards not covered: {len(missing_standards)} out of {len(expected_standard_codes)}",
                suggested_fix="Review module content to ensure better curriculum alignment",
                affected_items=list(missing_standards)[:5],  # Limit to first 5 for readability
                auto_fixable=False
            ))
        
        return issues

    async def _validate_module_ordering(self, modules: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate module ordering and prerequisites"""
        issues = []
        
        # Check for duplicate orders
        orders = [module.get('module_order', 0) for module in modules]
        if len(set(orders)) != len(orders):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="module_ordering",
                message="Duplicate module orders found",
                suggested_fix="Ensure each module has a unique order number",
                affected_items=["module_order"],
                auto_fixable=True
            ))
        
        # Check for gaps in ordering
        orders = sorted(orders)
        expected_orders = list(range(1, len(modules) + 1))
        if orders != expected_orders:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="module_ordering",
                message="Gaps or inconsistencies in module ordering",
                suggested_fix="Use consecutive numbering starting from 1",
                affected_items=["module_order"],
                auto_fixable=True
            ))
        
        return issues

    async def _validate_individual_module(
        self,
        module: Dict[str, Any],
        course_context: Dict[str, Any],
        module_number: int
    ) -> Tuple[bool, List[ValidationIssue], Dict[str, Any]]:
        """Validate individual module"""
        issues = []
        corrected_module = module.copy()

        # Structure validation
        structure_issues, corrected_module = await self._validate_module_structure(corrected_module, module_number)
        issues.extend(structure_issues)

        # Content quality validation
        content_issues = await self._validate_content_quality(corrected_module, course_context)
        issues.extend(content_issues)

        # Educational alignment validation
        alignment_issues = await self._validate_educational_alignment(corrected_module, course_context)
        issues.extend(alignment_issues)

        # Topic relationship validation
        topic_issues = await self._validate_topic_relationships(corrected_module)
        issues.extend(topic_issues)

        # Difficulty and progression validation
        difficulty_issues = await self._validate_difficulty_progression(corrected_module)
        issues.extend(difficulty_issues)

        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        is_valid = not has_errors

        return is_valid, issues, corrected_module

    async def _validate_module_structure(
        self,
        module: Dict[str, Any],
        module_number: int
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate basic module structure"""
        issues = []
        corrected_module = module.copy()

        # Required fields validation
        required_fields = {
            'module_title': str,
            'module_description': str,
            'topics': list,
            'estimated_duration_hours': (int, float),
        }

        for field, expected_types in required_fields.items():
            if field not in module:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="structure",
                    message=f"Module {module_number} missing required field: {field}",
                    suggested_fix=f"Add {field} to module definition",
                    affected_items=[field],
                    auto_fixable=True
                ))
                
                # Auto-fix with default values
                corrected_module[field] = self._get_default_value(field, module_number)
                
            elif not isinstance(module[field], expected_types):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="structure",
                    message=f"Module {module_number} field {field} has wrong type: expected {expected_types}",
                    suggested_fix=f"Convert {field} to appropriate type",
                    affected_items=[field],
                    auto_fixable=True
                ))

        # Title validation
        title = corrected_module.get('module_title', '')
        if len(title) < 5:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="content_quality",
                message=f"Module {module_number} title is too short ({len(title)} chars)",
                suggested_fix="Use a more descriptive title (at least 5 characters)",
                affected_items=['module_title'],
                auto_fixable=True
            ))
            if len(title) < 5:
                corrected_module['module_title'] = f"Eğitim Modülü {module_number}"
                
        elif len(title) > 100:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="content_quality",
                message=f"Module {module_number} title is too long ({len(title)} chars)",
                suggested_fix="Shorten title to under 100 characters",
                affected_items=['module_title'],
                auto_fixable=True
            ))
            corrected_module['module_title'] = title[:97] + "..."

        # Topic count validation
        topics = corrected_module.get('topics', [])
        topic_count = len(topics)

        if topic_count < 2:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message=f"Module {module_number} has too few topics ({topic_count})",
                suggested_fix="Add more topics or merge with another module (minimum 2 topics recommended)",
                affected_items=['topics'],
                auto_fixable=False
            ))
        elif topic_count > 20:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message=f"Module {module_number} has too many topics ({topic_count})",
                suggested_fix="Consider splitting into multiple modules (maximum 20 topics recommended)",
                affected_items=['topics'],
                auto_fixable=False
            ))

        # Duration validation
        duration = corrected_module.get('estimated_duration_hours', 0)
        if duration <= 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="structure",
                message=f"Module {module_number} has invalid duration ({duration})",
                suggested_fix="Set a positive duration value",
                affected_items=['estimated_duration_hours'],
                auto_fixable=True
            ))
            corrected_module['estimated_duration_hours'] = max(topic_count * 2, 8)
        elif duration > 80:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message=f"Module {module_number} has very long duration ({duration}h)",
                suggested_fix="Consider reducing duration or splitting content",
                affected_items=['estimated_duration_hours'],
                auto_fixable=True
            ))

        return issues, corrected_module

    async def _validate_content_quality(
        self,
        module: Dict[str, Any],
        course_context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate educational content quality"""
        issues = []
        module_number = module.get('module_order', 'X')

        # Learning outcomes validation
        learning_outcomes = module.get('learning_outcomes', [])
        if not learning_outcomes:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="educational_content",
                message=f"Module {module_number} has no learning outcomes",
                suggested_fix="Define clear, measurable learning outcomes",
                affected_items=['learning_outcomes'],
                auto_fixable=False
            ))
        else:
            for i, outcome in enumerate(learning_outcomes):
                if isinstance(outcome, str) and len(outcome) < 10:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="educational_content",
                        message=f"Module {module_number} learning outcome {i+1} is very brief",
                        suggested_fix="Write more detailed and specific learning outcomes",
                        affected_items=[f'learning_outcomes[{i}]'],
                        auto_fixable=False
                    ))

        # Assessment methods validation
        assessment_methods = module.get('assessment_methods', [])
        if not assessment_methods:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="educational_content",
                message=f"Module {module_number} has no assessment methods",
                suggested_fix="Define appropriate assessment methods (quiz, project, exam, etc.)",
                affected_items=['assessment_methods'],
                auto_fixable=True
            ))

        # Duration vs content validation
        duration = module.get('estimated_duration_hours', 0)
        topic_count = len(module.get('topics', []))

        if topic_count > 0 and duration > 0:
            hours_per_topic = duration / topic_count
            if hours_per_topic < 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="educational_content",
                    message=f"Module {module_number} has very low time per topic ({hours_per_topic:.1f}h)",
                    suggested_fix="Increase module duration or reduce topic count for adequate coverage",
                    affected_items=['estimated_duration_hours'],
                    auto_fixable=True
                ))
            elif hours_per_topic > 10:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="educational_content",
                    message=f"Module {module_number} has very high time per topic ({hours_per_topic:.1f}h)",
                    suggested_fix="Reduce module duration or add more topics",
                    affected_items=['estimated_duration_hours'],
                    auto_fixable=True
                ))

        return issues

    async def _validate_educational_alignment(
        self,
        module: Dict[str, Any],
        course_context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate educational alignment and pedagogy"""
        issues = []
        module_number = module.get('module_order', 'X')

        # Difficulty level validation
        difficulty = module.get('difficulty_level', '').lower()
        valid_difficulties = ['beginner', 'başlangıç', 'baslangic', 'intermediate', 'orta', 'advanced', 'ileri']
        
        if difficulty and difficulty not in valid_difficulties:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="educational_alignment",
                message=f"Module {module_number} has invalid difficulty level: {difficulty}",
                suggested_fix=f"Use one of: {', '.join(valid_difficulties)}",
                affected_items=['difficulty_level'],
                auto_fixable=True
            ))

        # Grade level appropriateness
        grade_level = course_context.get('grade_level', '')
        if grade_level:
            grade_issues = await self._validate_grade_appropriateness(module, grade_level)
            issues.extend(grade_issues)

        # Curriculum standards validation
        standards = module.get('curriculum_standards', [])
        if isinstance(standards, list) and course_context.get('curriculum_standard'):
            curriculum_type = course_context['curriculum_standard']
            invalid_standards = []
            
            for standard in standards:
                if isinstance(standard, str) and not self._is_valid_curriculum_code(standard, curriculum_type):
                    invalid_standards.append(standard)
            
            if invalid_standards:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="educational_alignment",
                    message=f"Module {module_number} may have invalid curriculum standard codes",
                    suggested_fix="Verify curriculum standard codes with official documentation",
                    affected_items=invalid_standards,
                    auto_fixable=False
                ))

        return issues

    async def _validate_grade_appropriateness(
        self,
        module: Dict[str, Any],
        grade_level: str
    ) -> List[ValidationIssue]:
        """Validate if module content is appropriate for grade level"""
        issues = []
        module_number = module.get('module_order', 'X')
        
        # Simple heuristics for grade appropriateness
        if grade_level.isdigit():
            grade_num = int(grade_level)
            difficulty = module.get('difficulty_level', 'intermediate').lower()
            
            # Lower grades should not have advanced difficulty
            if grade_num <= 10 and difficulty in ['advanced', 'ileri']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="educational_alignment",
                    message=f"Module {module_number} may be too advanced for grade {grade_level}",
                    suggested_fix="Consider adjusting difficulty level or content complexity",
                    affected_items=['difficulty_level'],
                    auto_fixable=False
                ))
            
            # Higher grades should not have beginner difficulty
            elif grade_num >= 11 and difficulty in ['beginner', 'başlangıç', 'baslangic']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="educational_alignment",
                    message=f"Module {module_number} may be too basic for grade {grade_level}",
                    suggested_fix="Consider increasing difficulty level or content depth",
                    affected_items=['difficulty_level'],
                    auto_fixable=False
                ))
        
        return issues

    async def _validate_topic_relationships(self, module: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate topic relationships within module"""
        issues = []
        module_number = module.get('module_order', 'X')
        topics = module.get('topics', [])

        if not topics:
            return issues

        # Check for duplicate topic IDs
        topic_ids = [topic.get('topic_id') for topic in topics if topic.get('topic_id')]
        if len(topic_ids) != len(set(topic_ids)):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="topic_relationships",
                message=f"Module {module_number} has duplicate topic IDs",
                suggested_fix="Ensure each topic appears only once per module",
                affected_items=['topics'],
                auto_fixable=False
            ))

        # Check topic ordering
        orders = [topic.get('topic_order_in_module', 0) for topic in topics]
        if orders and max(orders) > 0:  # Only check if orders are set
            expected_orders = list(range(1, len(topics) + 1))
            if sorted(orders) != expected_orders:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="topic_relationships",
                    message=f"Module {module_number} has inconsistent topic ordering",
                    suggested_fix="Use consecutive numbering for topic order within module",
                    affected_items=['topic_order_in_module'],
                    auto_fixable=True
                ))

        # Validate importance levels
        valid_importance = ['low', 'düşük', 'medium', 'orta', 'high', 'yüksek', 'critical', 'kritik']
        invalid_importance = []
        
        for i, topic in enumerate(topics):
            importance = topic.get('importance_level', '').lower()
            if importance and importance not in valid_importance:
                invalid_importance.append(f"topic_{i+1}")
        
        if invalid_importance:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="topic_relationships",
                message=f"Module {module_number} has invalid importance levels",
                suggested_fix=f"Use valid importance levels: {', '.join(valid_importance)}",
                affected_items=invalid_importance,
                auto_fixable=True
            ))

        return issues

    async def _validate_difficulty_progression(self, module: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate difficulty progression within module"""
        issues = []
        module_number = module.get('module_order', 'X')
        topics = module.get('topics', [])

        if len(topics) < 3:  # Need at least 3 topics to assess progression
            return issues

        # Get difficulty scores for progression analysis
        difficulty_scores = []
        for topic in topics:
            difficulty = topic.get('estimated_difficulty', 'intermediate').lower()
            score = self._get_difficulty_score(difficulty)
            difficulty_scores.append(score)

        # Check if there's some progression (not just random)
        if len(set(difficulty_scores)) == 1:  # All same difficulty
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="difficulty_progression",
                message=f"Module {module_number} has uniform difficulty level across all topics",
                suggested_fix="Consider varying difficulty levels to provide better learning progression",
                affected_items=['estimated_difficulty'],
                auto_fixable=False
            ))

        # Check for too many difficulty jumps (chaotic progression)
        difficulty_changes = 0
        for i in range(1, len(difficulty_scores)):
            if abs(difficulty_scores[i] - difficulty_scores[i-1]) > 1:
                difficulty_changes += 1

        if difficulty_changes > len(difficulty_scores) * 0.5:  # More than 50% are big jumps
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="difficulty_progression",
                message=f"Module {module_number} has erratic difficulty progression",
                suggested_fix="Consider reordering topics for smoother difficulty progression",
                affected_items=['topic_order_in_module'],
                auto_fixable=False
            ))

        return issues

    async def _validate_module_relationships(
        self,
        modules: List[Dict[str, Any]],
        course_context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate relationships between modules"""
        issues = []

        if len(modules) < 2:
            return issues

        # Check prerequisite relationships
        for module in modules:
            prerequisites = module.get('prerequisites', [])
            if prerequisites:
                module_ids = [m.get('module_id', m.get('module_order', 0)) for m in modules]
                invalid_prereqs = [p for p in prerequisites if p not in module_ids]
                
                if invalid_prereqs:
                    module_num = module.get('module_order', 'X')
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="module_relationships",
                        message=f"Module {module_num} has invalid prerequisite references",
                        suggested_fix="Ensure prerequisite modules exist in the course",
                        affected_items=['prerequisites'],
                        auto_fixable=False
                    ))

        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies(modules)
        if circular_deps:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="module_relationships",
                message="Circular dependency detected in module prerequisites",
                suggested_fix="Remove circular dependencies in prerequisite relationships",
                affected_items=circular_deps,
                auto_fixable=False
            ))

        return issues

    async def _apply_auto_fixes(
        self,
        modules: List[Dict[str, Any]],
        issues: List[ValidationIssue],
        course_context: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[ValidationIssue]]:
        """Apply automatic fixes to validation issues"""
        fixed_modules = []
        fix_issues = []

        for module in modules:
            fixed_module = module.copy()
            module_number = module.get('module_order', 0)

            # Apply structure fixes
            fixed_module = await self._apply_structure_fixes(fixed_module, issues)
            
            # Apply content fixes
            fixed_module = await self._apply_content_fixes(fixed_module, issues)
            
            # Apply ordering fixes
            fixed_module = await self._apply_ordering_fixes(fixed_module, module_number)

            fixed_modules.append(fixed_module)

        # Apply course-level fixes
        fixed_modules = await self._apply_course_level_fixes(fixed_modules, issues, course_context)

        return fixed_modules, fix_issues

    async def _apply_structure_fixes(
        self,
        module: Dict[str, Any],
        issues: List[ValidationIssue]
    ) -> Dict[str, Any]:
        """Apply structure-related fixes"""
        fixed_module = module.copy()
        
        # Fix missing required fields
        if 'module_title' not in fixed_module or not fixed_module['module_title']:
            module_num = fixed_module.get('module_order', 1)
            fixed_module['module_title'] = f"Eğitim Modülü {module_num}"
        
        if 'module_description' not in fixed_module or not fixed_module['module_description']:
            topic_count = len(fixed_module.get('topics', []))
            fixed_module['module_description'] = f"Eğitim modülü - {topic_count} konu içerir"
        
        if 'topics' not in fixed_module:
            fixed_module['topics'] = []
        
        if 'estimated_duration_hours' not in fixed_module or fixed_module.get('estimated_duration_hours', 0) <= 0:
            topic_count = len(fixed_module.get('topics', []))
            fixed_module['estimated_duration_hours'] = max(topic_count * 2, 8)
        
        # Fix topic ordering
        topics = fixed_module.get('topics', [])
        for i, topic in enumerate(topics):
            if 'topic_order_in_module' not in topic or topic.get('topic_order_in_module', 0) <= 0:
                topic['topic_order_in_module'] = i + 1
        
        return fixed_module

    async def _apply_content_fixes(
        self,
        module: Dict[str, Any],
        issues: List[ValidationIssue]
    ) -> Dict[str, Any]:
        """Apply content-related fixes"""
        fixed_module = module.copy()
        
        # Fix assessment methods
        if not fixed_module.get('assessment_methods'):
            fixed_module['assessment_methods'] = ['quiz', 'assignment']
        
        # Fix difficulty level
        difficulty = fixed_module.get('difficulty_level', '').lower()
        valid_difficulties = ['beginner', 'intermediate', 'advanced']
        if difficulty not in valid_difficulties:
            fixed_module['difficulty_level'] = 'intermediate'
        
        # Fix learning outcomes
        if not fixed_module.get('learning_outcomes'):
            module_title = fixed_module.get('module_title', 'Bu modül')
            fixed_module['learning_outcomes'] = [f"{module_title} konularını anlayabilir"]
        
        return fixed_module

    async def _apply_ordering_fixes(
        self,
        module: Dict[str, Any],
        expected_order: int
    ) -> Dict[str, Any]:
        """Apply ordering-related fixes"""
        fixed_module = module.copy()
        
        if 'module_order' not in fixed_module or fixed_module.get('module_order', 0) <= 0:
            fixed_module['module_order'] = expected_order
        
        return fixed_module

    async def _apply_course_level_fixes(
        self,
        modules: List[Dict[str, Any]],
        issues: List[ValidationIssue],
        course_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply course-level fixes"""
        fixed_modules = []
        
        # Fix module ordering
        for i, module in enumerate(modules):
            fixed_module = module.copy()
            fixed_module['module_order'] = i + 1
            fixed_modules.append(fixed_module)
        
        # Fix duration proportionally if needed
        total_duration = sum(m.get('estimated_duration_hours', 0) for m in fixed_modules)
        expected_duration = course_context.get('total_hours', 144)
        
        if total_duration < expected_duration * 0.7:
            # Increase all durations proportionally
            adjustment_factor = (expected_duration * 0.8) / total_duration if total_duration > 0 else 1.2
            for module in fixed_modules:
                current_duration = module.get('estimated_duration_hours', 20)
                module['estimated_duration_hours'] = min(int(current_duration * adjustment_factor), 60)
        
        return fixed_modules

    # Utility methods
    def _get_default_value(self, field: str, module_number: int) -> Any:
        """Get default value for a field"""
        defaults = {
            'module_title': f"Eğitim Modülü {module_number}",
            'module_description': f"Eğitim modülü {module_number} - temel konular",
            'topics': [],
            'estimated_duration_hours': 20,
            'difficulty_level': 'intermediate',
            'learning_outcomes': [f"Modül {module_number} temel hedefleri"],
            'assessment_methods': ['quiz']
        }
        return defaults.get(field, None)

    def _get_difficulty_score(self, difficulty: str) -> int:
        """Get numeric score for difficulty level"""
        scores = {
            'beginner': 1, 'başlangıç': 1, 'baslangic': 1,
            'intermediate': 2, 'orta': 2,
            'advanced': 3, 'ileri': 3, 'gelişmiş': 3
        }
        return scores.get(difficulty.lower(), 2)

    def _is_valid_curriculum_code(self, code: str, curriculum_type: str) -> bool:
        """Check if curriculum code follows expected format"""
        if not code or not isinstance(code, str):
            return False
        
        if curriculum_type == 'MEB_2018':
            # MEB codes like B.9.1.1, M.10.2.3, F.11.3.2
            pattern = r'^[A-Z]\.\d{1,2}\.\d+\.\d+$'
            return bool(re.match(pattern, code))
        
        return True  # Allow other formats for non-MEB curricula

    def _detect_circular_dependencies(self, modules: List[Dict[str, Any]]) -> List[str]:
        """Detect circular dependencies in module prerequisites"""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(module_id: int, adjacency: Dict[int, List[int]]) -> bool:
            visited.add(module_id)
            rec_stack.add(module_id)
            
            for neighbor in adjacency.get(module_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, adjacency):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(module_id)
            return False
        
        # Build adjacency list
        adjacency = {}
        for module in modules:
            module_id = module.get('module_id', module.get('module_order', 0))
            prerequisites = module.get('prerequisites', [])
            adjacency[module_id] = prerequisites
        
        # Check for cycles
        for module_id in adjacency:
            if module_id not in visited:
                if has_cycle(module_id, adjacency):
                    return [str(module_id)]  # Return first found cycle
        
        return []