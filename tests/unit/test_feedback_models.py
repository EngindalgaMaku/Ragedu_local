import pytest
from datetime import datetime
from typing import Dict, Any, List
from pydantic import ValidationError

from src.models.feedback import (
    FeedbackType, 
    FeedbackBase, 
    FeedbackCreate, 
    FeedbackResponse, 
    BulkFeedbackCreate, 
    FeedbackStats
)


@pytest.fixture
def valid_feedback_data():
    """Valid feedback data for testing."""
    return {
        "interaction_id": "interaction-123",
        "student_id": "student-456",
        "feedback_type": FeedbackType.HELPFUL,
        "comment": "This was very helpful!"
    }


@pytest.fixture
def valid_feedback_response_data():
    """Valid feedback response data for testing."""
    return {
        "interaction_id": "interaction-789",
        "student_id": "student-101",
        "feedback_type": FeedbackType.CORRECT,
        "comment": "Perfect answer",
        "id": 42,
        "created_at": datetime(2023, 10, 15, 14, 30, 0)
    }


@pytest.fixture
def sample_feedback_stats():
    """Sample feedback statistics data."""
    return {
        "total_feedbacks": 100,
        "positive_feedbacks": 75,
        "negative_feedbacks": 25,
        "by_type": {
            FeedbackType.HELPFUL: 40,
            FeedbackType.CORRECT: 35,
            FeedbackType.UNCLEAR: 15,
            FeedbackType.INCORRECT: 10
        }
    }


class TestFeedbackType:
    """Tests for FeedbackType enum."""
    
    @pytest.mark.unit
    def test_feedback_type_values(self):
        """Tests that all feedback type values are defined correctly."""
        assert FeedbackType.CORRECT == "correct"
        assert FeedbackType.INCORRECT == "incorrect"
        assert FeedbackType.HELPFUL == "helpful"
        assert FeedbackType.NOT_HELPFUL == "not_helpful"
        assert FeedbackType.UNCLEAR == "unclear"
        assert FeedbackType.OFF_TOPIC == "off_topic"
    
    @pytest.mark.unit
    def test_feedback_type_string_conversion(self):
        """Tests string representation of feedback types."""
        # Enum string representation shows the full enum name by default
        assert str(FeedbackType.HELPFUL) == "FeedbackType.HELPFUL"
        assert str(FeedbackType.CORRECT) == "FeedbackType.CORRECT"
        assert str(FeedbackType.UNCLEAR) == "FeedbackType.UNCLEAR"
        
        # Value access gives the actual string value
        assert FeedbackType.HELPFUL.value == "helpful"
        assert FeedbackType.CORRECT.value == "correct"
        assert FeedbackType.UNCLEAR.value == "unclear"
    
    @pytest.mark.unit
    def test_feedback_type_enum_members(self):
        """Tests that all enum members are accessible."""
        feedback_types = list(FeedbackType)
        expected_types = [
            FeedbackType.CORRECT,
            FeedbackType.INCORRECT,
            FeedbackType.HELPFUL,
            FeedbackType.NOT_HELPFUL,
            FeedbackType.UNCLEAR,
            FeedbackType.OFF_TOPIC
        ]
        
        assert len(feedback_types) == 6
        for expected_type in expected_types:
            assert expected_type in feedback_types
    
    @pytest.mark.unit
    def test_feedback_type_value_access(self):
        """Tests accessing feedback type values."""
        assert FeedbackType.HELPFUL.value == "helpful"
        assert FeedbackType.NOT_HELPFUL.value == "not_helpful"
        assert FeedbackType.OFF_TOPIC.value == "off_topic"
    
    @pytest.mark.unit
    def test_feedback_type_from_string(self):
        """Tests creating FeedbackType from string values."""
        assert FeedbackType("correct") == FeedbackType.CORRECT
        assert FeedbackType("helpful") == FeedbackType.HELPFUL
        assert FeedbackType("off_topic") == FeedbackType.OFF_TOPIC
    
    @pytest.mark.unit
    def test_feedback_type_invalid_string(self):
        """Tests creating FeedbackType with invalid string raises ValueError."""
        with pytest.raises(ValueError, match="'invalid_type' is not a valid FeedbackType"):
            FeedbackType("invalid_type")


class TestFeedbackBase:
    """Tests for FeedbackBase model."""
    
    @pytest.mark.unit
    def test_feedback_base_creation_success(self, valid_feedback_data):
        """Tests successful creation of FeedbackBase model."""
        feedback = FeedbackBase(**valid_feedback_data)
        
        assert feedback.interaction_id == "interaction-123"
        assert feedback.student_id == "student-456"
        assert feedback.feedback_type == FeedbackType.HELPFUL
        assert feedback.comment == "This was very helpful!"
    
    @pytest.mark.unit
    def test_feedback_base_without_comment(self):
        """Tests FeedbackBase creation without optional comment."""
        data = {
            "interaction_id": "interaction-999",
            "student_id": "student-888",
            "feedback_type": FeedbackType.CORRECT
        }
        
        feedback = FeedbackBase(**data)
        
        assert feedback.interaction_id == "interaction-999"
        assert feedback.student_id == "student-888"
        assert feedback.feedback_type == FeedbackType.CORRECT
        assert feedback.comment is None
    
    @pytest.mark.unit
    def test_feedback_base_field_descriptions(self):
        """Tests that field descriptions are properly defined."""
        # Access the model fields to verify descriptions (Pydantic v2 syntax)
        fields = FeedbackBase.model_fields
        
        assert fields['interaction_id'].description == "İlişkili etkileşim ID'si"
        assert fields['student_id'].description == "Geri bildirimi veren öğrenci ID'si"
        assert fields['feedback_type'].description == "Geri bildirim türü"
        assert fields['comment'].description == "Ek yorumlar"
    
    @pytest.mark.unit
    def test_feedback_base_required_fields(self):
        """Tests that required fields are properly validated."""
        # Missing interaction_id
        with pytest.raises(ValidationError) as exc_info:
            FeedbackBase(
                student_id="student-123",
                feedback_type=FeedbackType.HELPFUL
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('interaction_id',)
        assert errors[0]['type'] == 'missing'
        
        # Missing student_id
        with pytest.raises(ValidationError) as exc_info:
            FeedbackBase(
                interaction_id="interaction-123",
                feedback_type=FeedbackType.HELPFUL
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('student_id',)
        assert errors[0]['type'] == 'missing'
        
        # Missing feedback_type
        with pytest.raises(ValidationError) as exc_info:
            FeedbackBase(
                interaction_id="interaction-123",
                student_id="student-456"
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('feedback_type',)
        assert errors[0]['type'] == 'missing'
    
    @pytest.mark.unit
    def test_feedback_base_invalid_feedback_type(self):
        """Tests validation error for invalid feedback type."""
        with pytest.raises(ValidationError) as exc_info:
            FeedbackBase(
                interaction_id="interaction-123",
                student_id="student-456",
                feedback_type="invalid_type"
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('feedback_type',)
        assert 'value_error' in errors[0]['type'] or 'enum' in errors[0]['type']
    
    @pytest.mark.unit
    def test_feedback_base_empty_string_fields(self):
        """Tests behavior with empty string values."""
        # Empty strings should be accepted (not converted to None)
        feedback = FeedbackBase(
            interaction_id="",
            student_id="",
            feedback_type=FeedbackType.HELPFUL,
            comment=""
        )
        
        assert feedback.interaction_id == ""
        assert feedback.student_id == ""
        assert feedback.comment == ""
    
    @pytest.mark.unit
    def test_feedback_base_with_all_feedback_types(self):
        """Tests FeedbackBase with all possible feedback types."""
        feedback_types = [
            FeedbackType.CORRECT,
            FeedbackType.INCORRECT,
            FeedbackType.HELPFUL,
            FeedbackType.NOT_HELPFUL,
            FeedbackType.UNCLEAR,
            FeedbackType.OFF_TOPIC
        ]
        
        for feedback_type in feedback_types:
            feedback = FeedbackBase(
                interaction_id="test-interaction",
                student_id="test-student",
                feedback_type=feedback_type,
                comment=f"Comment for {feedback_type}"
            )
            
            assert feedback.feedback_type == feedback_type
            assert feedback.comment == f"Comment for {feedback_type}"


class TestFeedbackCreate:
    """Tests for FeedbackCreate model."""
    
    @pytest.mark.unit
    def test_feedback_create_inheritance(self, valid_feedback_data):
        """Tests that FeedbackCreate properly inherits from FeedbackBase."""
        feedback = FeedbackCreate(**valid_feedback_data)
        
        # Should have all FeedbackBase properties
        assert feedback.interaction_id == "interaction-123"
        assert feedback.student_id == "student-456"
        assert feedback.feedback_type == FeedbackType.HELPFUL
        assert feedback.comment == "This was very helpful!"
        
        # Should be instance of both classes
        assert isinstance(feedback, FeedbackCreate)
        assert isinstance(feedback, FeedbackBase)
    
    @pytest.mark.unit
    def test_feedback_create_validation(self):
        """Tests FeedbackCreate validation rules."""
        # Valid creation
        feedback = FeedbackCreate(
            interaction_id="create-test",
            student_id="create-student",
            feedback_type=FeedbackType.CORRECT
        )
        
        assert feedback.interaction_id == "create-test"
        assert feedback.student_id == "create-student"
        assert feedback.feedback_type == FeedbackType.CORRECT
        assert feedback.comment is None
    
    @pytest.mark.unit
    def test_feedback_create_with_long_comment(self):
        """Tests FeedbackCreate with very long comment."""
        long_comment = "A" * 10000
        
        feedback = FeedbackCreate(
            interaction_id="long-comment-test",
            student_id="test-student",
            feedback_type=FeedbackType.HELPFUL,
            comment=long_comment
        )
        
        assert feedback.comment == long_comment
        assert len(feedback.comment) == 10000
    
    @pytest.mark.unit
    def test_feedback_create_with_special_characters(self):
        """Tests FeedbackCreate with special characters in fields."""
        special_data = {
            "interaction_id": "interaction-ñüéß-中文-🤖",
            "student_id": "student-ñüéß-العربية-🎓",
            "feedback_type": FeedbackType.HELPFUL,
            "comment": "Comment with special chars: àáâãäåæçèéêë ñóôõö üýÿ 中文 العربية 🎉🚀"
        }
        
        feedback = FeedbackCreate(**special_data)
        
        assert feedback.interaction_id == "interaction-ñüéß-中文-🤖"
        assert feedback.student_id == "student-ñüéß-العربية-🎓"
        assert feedback.comment == "Comment with special chars: àáâãäåæçèéêë ñóôõö üýÿ 中文 العربية 🎉🚀"
    
    @pytest.mark.unit
    def test_feedback_create_serialization(self, valid_feedback_data):
        """Tests FeedbackCreate serialization to dictionary."""
        feedback = FeedbackCreate(**valid_feedback_data)
        serialized = feedback.dict()
        
        expected_keys = {"interaction_id", "student_id", "feedback_type", "comment"}
        assert set(serialized.keys()) == expected_keys
        
        assert serialized["interaction_id"] == "interaction-123"
        assert serialized["student_id"] == "student-456"
        assert serialized["feedback_type"] == FeedbackType.HELPFUL
        assert serialized["comment"] == "This was very helpful!"


class TestFeedbackResponse:
    """Tests for FeedbackResponse model."""
    
    @pytest.mark.unit
    def test_feedback_response_creation_success(self, valid_feedback_response_data):
        """Tests successful creation of FeedbackResponse model."""
        feedback = FeedbackResponse(**valid_feedback_response_data)
        
        assert feedback.interaction_id == "interaction-789"
        assert feedback.student_id == "student-101"
        assert feedback.feedback_type == FeedbackType.CORRECT
        assert feedback.comment == "Perfect answer"
        assert feedback.id == 42
        assert feedback.created_at == datetime(2023, 10, 15, 14, 30, 0)
    
    @pytest.mark.unit
    def test_feedback_response_inheritance(self, valid_feedback_response_data):
        """Tests that FeedbackResponse inherits from FeedbackBase."""
        feedback = FeedbackResponse(**valid_feedback_response_data)
        
        # Should be instance of both classes
        assert isinstance(feedback, FeedbackResponse)
        assert isinstance(feedback, FeedbackBase)
    
    @pytest.mark.unit
    def test_feedback_response_required_additional_fields(self):
        """Tests that FeedbackResponse requires id and created_at fields."""
        base_data = {
            "interaction_id": "test-interaction",
            "student_id": "test-student",
            "feedback_type": FeedbackType.HELPFUL
        }
        
        # Missing id
        with pytest.raises(ValidationError) as exc_info:
            FeedbackResponse(**base_data, created_at=datetime.now())
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('id',)
        assert errors[0]['type'] == 'missing'
        
        # Missing created_at
        with pytest.raises(ValidationError) as exc_info:
            FeedbackResponse(**base_data, id=123)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('created_at',)
        assert errors[0]['type'] == 'missing'
    
    @pytest.mark.unit
    def test_feedback_response_orm_mode_config(self):
        """Tests that FeedbackResponse has orm_mode enabled."""
        # The model still uses the old orm_mode configuration
        config = FeedbackResponse.model_config
        assert 'orm_mode' in config
        assert config['orm_mode'] is True
    
    @pytest.mark.unit
    def test_feedback_response_with_current_datetime(self):
        """Tests FeedbackResponse with current datetime."""
        now = datetime.now()
        
        feedback = FeedbackResponse(
            interaction_id="datetime-test",
            student_id="datetime-student",
            feedback_type=FeedbackType.UNCLEAR,
            comment="Datetime test",
            id=999,
            created_at=now
        )
        
        assert feedback.created_at == now
        assert feedback.id == 999
    
    @pytest.mark.unit
    def test_feedback_response_id_validation(self):
        """Tests FeedbackResponse id field validation."""
        base_data = {
            "interaction_id": "id-test",
            "student_id": "id-student",
            "feedback_type": FeedbackType.CORRECT,
            "created_at": datetime.now()
        }
        
        # Valid positive integer
        feedback = FeedbackResponse(**base_data, id=12345)
        assert feedback.id == 12345
        
        # Zero should be valid
        feedback = FeedbackResponse(**base_data, id=0)
        assert feedback.id == 0
        
        # Negative integers should be valid (database might use them)
        feedback = FeedbackResponse(**base_data, id=-1)
        assert feedback.id == -1
    
    @pytest.mark.unit
    def test_feedback_response_serialization(self, valid_feedback_response_data):
        """Tests FeedbackResponse serialization to dictionary."""
        feedback = FeedbackResponse(**valid_feedback_response_data)
        serialized = feedback.dict()
        
        expected_keys = {"interaction_id", "student_id", "feedback_type", "comment", "id", "created_at"}
        assert set(serialized.keys()) == expected_keys
        
        assert serialized["id"] == 42
        assert serialized["created_at"] == datetime(2023, 10, 15, 14, 30, 0)


class TestBulkFeedbackCreate:
    """Tests for BulkFeedbackCreate model."""
    
    @pytest.mark.unit
    def test_bulk_feedback_create_empty_list(self):
        """Tests BulkFeedbackCreate with empty feedback list."""
        bulk_feedback = BulkFeedbackCreate(feedbacks=[])
        
        assert bulk_feedback.feedbacks == []
        assert len(bulk_feedback.feedbacks) == 0
    
    @pytest.mark.unit
    def test_bulk_feedback_create_single_feedback(self, valid_feedback_data):
        """Tests BulkFeedbackCreate with single feedback item."""
        feedback_item = FeedbackCreate(**valid_feedback_data)
        bulk_feedback = BulkFeedbackCreate(feedbacks=[feedback_item])
        
        assert len(bulk_feedback.feedbacks) == 1
        assert bulk_feedback.feedbacks[0] == feedback_item
        assert bulk_feedback.feedbacks[0].interaction_id == "interaction-123"
    
    @pytest.mark.unit
    def test_bulk_feedback_create_multiple_feedbacks(self):
        """Tests BulkFeedbackCreate with multiple feedback items."""
        feedbacks = [
            FeedbackCreate(
                interaction_id=f"interaction-{i}",
                student_id=f"student-{i}",
                feedback_type=FeedbackType.HELPFUL,
                comment=f"Comment {i}"
            )
            for i in range(5)
        ]
        
        bulk_feedback = BulkFeedbackCreate(feedbacks=feedbacks)
        
        assert len(bulk_feedback.feedbacks) == 5
        for i, feedback in enumerate(bulk_feedback.feedbacks):
            assert feedback.interaction_id == f"interaction-{i}"
            assert feedback.student_id == f"student-{i}"
            assert feedback.comment == f"Comment {i}"
    
    @pytest.mark.unit
    def test_bulk_feedback_create_mixed_feedback_types(self):
        """Tests BulkFeedbackCreate with different feedback types."""
        feedbacks = [
            FeedbackCreate(
                interaction_id="int-1",
                student_id="stu-1",
                feedback_type=FeedbackType.CORRECT
            ),
            FeedbackCreate(
                interaction_id="int-2",
                student_id="stu-2",
                feedback_type=FeedbackType.HELPFUL,
                comment="Very helpful"
            ),
            FeedbackCreate(
                interaction_id="int-3",
                student_id="stu-3",
                feedback_type=FeedbackType.UNCLEAR,
                comment="Needs clarification"
            )
        ]
        
        bulk_feedback = BulkFeedbackCreate(feedbacks=feedbacks)
        
        assert len(bulk_feedback.feedbacks) == 3
        assert bulk_feedback.feedbacks[0].feedback_type == FeedbackType.CORRECT
        assert bulk_feedback.feedbacks[1].feedback_type == FeedbackType.HELPFUL
        assert bulk_feedback.feedbacks[2].feedback_type == FeedbackType.UNCLEAR
    
    @pytest.mark.unit
    def test_bulk_feedback_create_validation_error(self):
        """Tests BulkFeedbackCreate validation with invalid feedback items."""
        # Invalid feedback item (missing required field)
        with pytest.raises(ValidationError) as exc_info:
            BulkFeedbackCreate(feedbacks=[
                {
                    "interaction_id": "valid-interaction",
                    "feedback_type": FeedbackType.HELPFUL
                    # Missing required student_id
                }
            ])
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('feedbacks', 0, 'student_id')
    
    @pytest.mark.unit
    def test_bulk_feedback_create_large_batch(self):
        """Tests BulkFeedbackCreate with large number of feedback items."""
        feedbacks = [
            FeedbackCreate(
                interaction_id=f"bulk-interaction-{i}",
                student_id=f"bulk-student-{i}",
                feedback_type=FeedbackType.HELPFUL if i % 2 == 0 else FeedbackType.CORRECT
            )
            for i in range(100)
        ]
        
        bulk_feedback = BulkFeedbackCreate(feedbacks=feedbacks)
        
        assert len(bulk_feedback.feedbacks) == 100
        
        # Verify alternating feedback types
        for i, feedback in enumerate(bulk_feedback.feedbacks):
            expected_type = FeedbackType.HELPFUL if i % 2 == 0 else FeedbackType.CORRECT
            assert feedback.feedback_type == expected_type
    
    @pytest.mark.unit
    def test_bulk_feedback_create_serialization(self):
        """Tests BulkFeedbackCreate serialization to dictionary."""
        feedbacks = [
            FeedbackCreate(
                interaction_id="ser-int-1",
                student_id="ser-stu-1",
                feedback_type=FeedbackType.HELPFUL
            ),
            FeedbackCreate(
                interaction_id="ser-int-2",
                student_id="ser-stu-2",
                feedback_type=FeedbackType.CORRECT,
                comment="Good answer"
            )
        ]
        
        bulk_feedback = BulkFeedbackCreate(feedbacks=feedbacks)
        serialized = bulk_feedback.dict()
        
        assert "feedbacks" in serialized
        assert len(serialized["feedbacks"]) == 2
        assert serialized["feedbacks"][0]["interaction_id"] == "ser-int-1"
        assert serialized["feedbacks"][1]["comment"] == "Good answer"


class TestFeedbackStats:
    """Tests for FeedbackStats model."""
    
    @pytest.mark.unit
    def test_feedback_stats_creation_success(self, sample_feedback_stats):
        """Tests successful creation of FeedbackStats model."""
        stats = FeedbackStats(**sample_feedback_stats)
        
        assert stats.total_feedbacks == 100
        assert stats.positive_feedbacks == 75
        assert stats.negative_feedbacks == 25
        assert len(stats.by_type) == 4
        assert stats.by_type[FeedbackType.HELPFUL] == 40
        assert stats.by_type[FeedbackType.CORRECT] == 35
    
    @pytest.mark.unit
    def test_feedback_stats_required_fields(self):
        """Tests that all fields are required in FeedbackStats."""
        # Missing total_feedbacks
        with pytest.raises(ValidationError) as exc_info:
            FeedbackStats(
                positive_feedbacks=10,
                negative_feedbacks=5,
                by_type={}
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('total_feedbacks',)
        
        # Missing positive_feedbacks
        with pytest.raises(ValidationError) as exc_info:
            FeedbackStats(
                total_feedbacks=15,
                negative_feedbacks=5,
                by_type={}
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('positive_feedbacks',)
    
    @pytest.mark.unit
    def test_feedback_stats_zero_values(self):
        """Tests FeedbackStats with zero values."""
        stats = FeedbackStats(
            total_feedbacks=0,
            positive_feedbacks=0,
            negative_feedbacks=0,
            by_type={}
        )
        
        assert stats.total_feedbacks == 0
        assert stats.positive_feedbacks == 0
        assert stats.negative_feedbacks == 0
        assert stats.by_type == {}
    
    @pytest.mark.unit
    def test_feedback_stats_by_type_with_all_types(self):
        """Tests FeedbackStats by_type dictionary with all feedback types."""
        by_type_data = {
            FeedbackType.CORRECT: 20,
            FeedbackType.INCORRECT: 5,
            FeedbackType.HELPFUL: 30,
            FeedbackType.NOT_HELPFUL: 8,
            FeedbackType.UNCLEAR: 12,
            FeedbackType.OFF_TOPIC: 3
        }
        
        stats = FeedbackStats(
            total_feedbacks=78,
            positive_feedbacks=50,
            negative_feedbacks=28,
            by_type=by_type_data
        )
        
        assert len(stats.by_type) == 6
        for feedback_type, count in by_type_data.items():
            assert stats.by_type[feedback_type] == count
    
    @pytest.mark.unit
    def test_feedback_stats_by_type_partial(self):
        """Tests FeedbackStats with partial by_type data."""
        partial_by_type = {
            FeedbackType.HELPFUL: 25,
            FeedbackType.UNCLEAR: 10
        }
        
        stats = FeedbackStats(
            total_feedbacks=35,
            positive_feedbacks=25,
            negative_feedbacks=10,
            by_type=partial_by_type
        )
        
        assert len(stats.by_type) == 2
        assert stats.by_type[FeedbackType.HELPFUL] == 25
        assert stats.by_type[FeedbackType.UNCLEAR] == 10
        
        # Verify other types are not present
        assert FeedbackType.CORRECT not in stats.by_type
        assert FeedbackType.INCORRECT not in stats.by_type
    
    @pytest.mark.unit
    def test_feedback_stats_negative_numbers(self):
        """Tests FeedbackStats validation with negative numbers."""
        # Negative numbers should be accepted (might represent adjustments)
        stats = FeedbackStats(
            total_feedbacks=-5,
            positive_feedbacks=-2,
            negative_feedbacks=-3,
            by_type={FeedbackType.HELPFUL: -1}
        )
        
        assert stats.total_feedbacks == -5
        assert stats.positive_feedbacks == -2
        assert stats.negative_feedbacks == -3
        assert stats.by_type[FeedbackType.HELPFUL] == -1
    
    @pytest.mark.unit
    def test_feedback_stats_large_numbers(self):
        """Tests FeedbackStats with very large numbers."""
        large_stats = FeedbackStats(
            total_feedbacks=999999999,
            positive_feedbacks=800000000,
            negative_feedbacks=199999999,
            by_type={
                FeedbackType.HELPFUL: 500000000,
                FeedbackType.CORRECT: 300000000
            }
        )
        
        assert large_stats.total_feedbacks == 999999999
        assert large_stats.positive_feedbacks == 800000000
        assert large_stats.by_type[FeedbackType.HELPFUL] == 500000000
    
    @pytest.mark.unit
    def test_feedback_stats_serialization(self, sample_feedback_stats):
        """Tests FeedbackStats serialization to dictionary."""
        stats = FeedbackStats(**sample_feedback_stats)
        serialized = stats.dict()
        
        expected_keys = {"total_feedbacks", "positive_feedbacks", "negative_feedbacks", "by_type"}
        assert set(serialized.keys()) == expected_keys
        
        assert serialized["total_feedbacks"] == 100
        assert serialized["positive_feedbacks"] == 75
        assert isinstance(serialized["by_type"], dict)
        
        # Verify by_type serialization preserves enum keys
        for feedback_type, count in serialized["by_type"].items():
            assert isinstance(feedback_type, FeedbackType)
            assert isinstance(count, int)


class TestFeedbackModelsIntegration:
    """Integration tests across feedback models."""
    
    @pytest.mark.unit
    def test_feedback_workflow_create_to_response(self, valid_feedback_data):
        """Tests workflow from FeedbackCreate to FeedbackResponse."""
        # Create feedback
        create_feedback = FeedbackCreate(**valid_feedback_data)
        
        # Simulate database processing (add id and timestamp)
        response_data = create_feedback.dict()
        response_data.update({
            "id": 789,
            "created_at": datetime.now()
        })
        
        # Create response model
        response_feedback = FeedbackResponse(**response_data)
        
        # Verify data consistency
        assert response_feedback.interaction_id == create_feedback.interaction_id
        assert response_feedback.student_id == create_feedback.student_id
        assert response_feedback.feedback_type == create_feedback.feedback_type
        assert response_feedback.comment == create_feedback.comment
        assert response_feedback.id == 789
        assert response_feedback.created_at is not None
    
    @pytest.mark.unit
    def test_bulk_feedback_to_stats_conversion(self):
        """Tests conceptual conversion from bulk feedback to stats."""
        # Create bulk feedback with various types
        feedbacks = [
            FeedbackCreate(
                interaction_id=f"bulk-{i}",
                student_id=f"student-{i}",
                feedback_type=FeedbackType.HELPFUL
            )
            for i in range(30)
        ] + [
            FeedbackCreate(
                interaction_id=f"bulk-{i}",
                student_id=f"student-{i}",
                feedback_type=FeedbackType.CORRECT
            )
            for i in range(30, 50)
        ] + [
            FeedbackCreate(
                interaction_id=f"bulk-{i}",
                student_id=f"student-{i}",
                feedback_type=FeedbackType.UNCLEAR
            )
            for i in range(50, 60)
        ]
        
        bulk_feedback = BulkFeedbackCreate(feedbacks=feedbacks)
        
        # Simulate stats calculation
        by_type_counts = {}
        for feedback in bulk_feedback.feedbacks:
            by_type_counts[feedback.feedback_type] = by_type_counts.get(feedback.feedback_type, 0) + 1
        
        stats = FeedbackStats(
            total_feedbacks=len(bulk_feedback.feedbacks),
            positive_feedbacks=by_type_counts.get(FeedbackType.HELPFUL, 0) + by_type_counts.get(FeedbackType.CORRECT, 0),
            negative_feedbacks=by_type_counts.get(FeedbackType.UNCLEAR, 0),
            by_type=by_type_counts
        )
        
        # Verify calculated stats
        assert stats.total_feedbacks == 60
        assert stats.positive_feedbacks == 50  # 30 helpful + 20 correct
        assert stats.negative_feedbacks == 10  # 10 unclear
        assert stats.by_type[FeedbackType.HELPFUL] == 30
        assert stats.by_type[FeedbackType.CORRECT] == 20
        assert stats.by_type[FeedbackType.UNCLEAR] == 10


class TestFeedbackModelsErrorHandling:
    """Tests for error handling and edge cases."""
    
    @pytest.mark.unit
    def test_invalid_data_types(self):
        """Tests models with invalid data types."""
        # Non-string interaction_id
        with pytest.raises(ValidationError):
            FeedbackCreate(
                interaction_id=12345,  # Should be string
                student_id="student-123",
                feedback_type=FeedbackType.HELPFUL
            )
        
        # Non-integer count in stats
        with pytest.raises(ValidationError):
            FeedbackStats(
                total_feedbacks="not a number",  # Should be int
                positive_feedbacks=10,
                negative_feedbacks=5,
                by_type={}
            )
    
    @pytest.mark.unit
    def test_none_values_for_required_fields(self):
        """Tests behavior with None values for required fields."""
        with pytest.raises(ValidationError):
            FeedbackCreate(
                interaction_id=None,
                student_id="student-123",
                feedback_type=FeedbackType.HELPFUL
            )
    
    @pytest.mark.unit
    def test_extremely_long_field_values(self):
        """Tests models with extremely long field values."""
        very_long_string = "A" * 1000000  # 1MB string
        
        # Should handle very long strings without error
        feedback = FeedbackCreate(
            interaction_id=very_long_string,
            student_id=very_long_string,
            feedback_type=FeedbackType.HELPFUL,
            comment=very_long_string
        )
        
        assert len(feedback.interaction_id) == 1000000
        assert len(feedback.student_id) == 1000000
        assert len(feedback.comment) == 1000000
    
    @pytest.mark.unit
    def test_unicode_edge_cases(self):
        """Tests models with various Unicode edge cases."""
        unicode_strings = [
            "🤖🔬📚",  # Emojis
            "नमस्ते",    # Devanagari
            "مرحبا",     # Arabic
            "こんにちは",   # Japanese
            "🇹🇷🇺🇸🇯🇵",  # Flag emojis
            "test\x00null",  # Null character
            "test\uffefBOM", # BOM character
        ]
        
        for unicode_string in unicode_strings:
            feedback = FeedbackCreate(
                interaction_id=f"unicode-test-{unicode_string}",
                student_id=f"student-{unicode_string}",
                feedback_type=FeedbackType.HELPFUL,
                comment=f"Comment with {unicode_string}"
            )
            
            assert unicode_string in feedback.interaction_id
            assert unicode_string in feedback.student_id
            assert unicode_string in feedback.comment
    
    @pytest.mark.unit
    def test_model_equality_and_comparison(self, valid_feedback_data):
        """Tests model equality and comparison behavior."""
        feedback1 = FeedbackCreate(**valid_feedback_data)
        feedback2 = FeedbackCreate(**valid_feedback_data)
        
        # Same data should create equal models
        assert feedback1.dict() == feedback2.dict()
        
        # Different data should create different models
        different_data = valid_feedback_data.copy()
        different_data["comment"] = "Different comment"
        feedback3 = FeedbackCreate(**different_data)
        
        assert feedback1.dict() != feedback3.dict()
    
    @pytest.mark.unit
    def test_model_json_serialization(self, valid_feedback_data, sample_feedback_stats):
        """Tests JSON serialization and deserialization of models."""
        # Test FeedbackCreate JSON
        feedback = FeedbackCreate(**valid_feedback_data)
        json_str = feedback.json()
        assert isinstance(json_str, str)
        assert "helpful" in json_str  # feedback_type value
        
        # Test FeedbackStats JSON
        stats = FeedbackStats(**sample_feedback_stats)
        json_str = stats.json()
        assert isinstance(json_str, str)
        assert "100" in json_str  # total_feedbacks value