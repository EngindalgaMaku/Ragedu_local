import pytest
from unittest.mock import MagicMock, patch, call
import logging

from src.services.feedback_processor import FeedbackProcessor
from src.models.feedback import FeedbackCreate, BulkFeedbackCreate, FeedbackType


@pytest.fixture
def mock_db():
    """Veritabanı sınıfı için bir MagicMock nesnesi oluşturur."""
    return MagicMock()


@pytest.fixture
def feedback_processor(mock_db):
    """
    FeedbackProcessor'ı mock veritabanı ile başlatır.
    'get_experiment_db' fonksiyonunu patch'leyerek, her çağrıldığında
    bizim mock_db'mizi döndürmesini sağlarız.
    """
    with patch('src.services.feedback_processor.get_experiment_db') as mock_get_db:
        mock_get_db.return_value = mock_db
        processor = FeedbackProcessor()
        yield processor


@pytest.fixture
def sample_feedback():
    """Test için örnek feedback verisi."""
    return FeedbackCreate(
        interaction_id="interaction-1",
        student_id="student-1",
        feedback_type=FeedbackType.HELPFUL,
        comment="Good explanation."
    )


class TestFeedbackProcessorInit:
    """Constructor initialization tests."""
    
    @patch('src.services.feedback_processor.get_experiment_db')
    @patch('src.services.feedback_processor.logger')
    def test_init_success(self, mock_logger, mock_get_db):
        """Tests successful initialization of FeedbackProcessor."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        processor = FeedbackProcessor()
        
        mock_logger.info.assert_called_once_with("FeedbackProcessor başlatıldı.")
        mock_get_db.assert_called_once()
        assert processor.db == mock_db
    
    @patch('src.services.feedback_processor.get_experiment_db')
    def test_init_database_connection_failure(self, mock_get_db):
        """Tests initialization when database connection fails."""
        mock_get_db.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception, match="Database connection failed"):
            FeedbackProcessor()


class TestProcessFeedback:
    """Tests for process_feedback method."""
    
    @pytest.mark.unit
    def test_process_feedback_success(self, feedback_processor: FeedbackProcessor, mock_db, sample_feedback):
        """Tek bir geri bildirimin başarılı bir şekilde işlenmesini test eder."""
        # Arrange
        mock_db.add_feedback.return_value = 123
        
        # Act
        result = feedback_processor.process_feedback(sample_feedback)
        
        # Assert
        mock_db.add_feedback.assert_called_once_with(sample_feedback)
        assert result["status"] == "success"
        assert result["message"] == "Feedback processed successfully"
        assert result["feedback_id"] == 123
    
    @pytest.mark.unit
    def test_process_feedback_with_all_feedback_types(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests processing feedback with all different FeedbackType values."""
        feedback_types = [
            FeedbackType.CORRECT,
            FeedbackType.INCORRECT,
            FeedbackType.HELPFUL,
            FeedbackType.NOT_HELPFUL,
            FeedbackType.UNCLEAR,
            FeedbackType.OFF_TOPIC
        ]
        
        mock_db.add_feedback.side_effect = [100, 101, 102, 103, 104, 105]
        
        for i, feedback_type in enumerate(feedback_types):
            feedback_data = FeedbackCreate(
                interaction_id=f"interaction-{i}",
                student_id=f"student-{i}",
                feedback_type=feedback_type,
                comment=f"Test comment for {feedback_type}"
            )
            
            result = feedback_processor.process_feedback(feedback_data)
            
            assert result["status"] == "success"
            assert result["feedback_id"] == 100 + i
    
    @pytest.mark.unit
    def test_process_feedback_without_comment(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests processing feedback without optional comment."""
        mock_db.add_feedback.return_value = 124
        
        feedback_data = FeedbackCreate(
            interaction_id="interaction-2",
            student_id="student-2",
            feedback_type=FeedbackType.CORRECT
        )
        
        result = feedback_processor.process_feedback(feedback_data)
        
        mock_db.add_feedback.assert_called_once_with(feedback_data)
        assert result["status"] == "success"
        assert result["feedback_id"] == 124
    
    @pytest.mark.unit
    def test_process_feedback_with_long_comment(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests processing feedback with very long comment."""
        mock_db.add_feedback.return_value = 125
        long_comment = "A" * 1000  # Very long comment
        
        feedback_data = FeedbackCreate(
            interaction_id="interaction-long",
            student_id="student-long",
            feedback_type=FeedbackType.HELPFUL,
            comment=long_comment
        )
        
        result = feedback_processor.process_feedback(feedback_data)
        
        assert result["status"] == "success"
        assert result["feedback_id"] == 125
    
    @pytest.mark.unit
    @patch('src.services.feedback_processor.logger')
    def test_process_feedback_logs_correctly(self, mock_logger, feedback_processor: FeedbackProcessor, 
                                           mock_db, sample_feedback):
        """Tests that process_feedback logs correctly."""
        mock_db.add_feedback.return_value = 126
        
        feedback_processor.process_feedback(sample_feedback)
        
        mock_logger.info.assert_called_with(f"Geri bildirim işleniyor: {sample_feedback.interaction_id}")
    
    @pytest.mark.unit
    @patch('src.services.feedback_processor.logger')
    def test_process_feedback_failure(self, mock_logger, feedback_processor: FeedbackProcessor, 
                                    mock_db, sample_feedback):
        """Veritabanı hatası durumunda geri bildirim işlemenin başarısız olmasını test eder."""
        # Arrange
        error_message = "Database connection failed"
        mock_db.add_feedback.side_effect = Exception(error_message)
        
        # Act & Assert
        with pytest.raises(Exception, match=error_message):
            feedback_processor.process_feedback(sample_feedback)
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        args = mock_logger.error.call_args
        assert error_message in str(args[0][0])


class TestProcessBulkFeedback:
    """Tests for process_bulk_feedback method."""
    
    @pytest.mark.unit
    def test_process_bulk_feedback_success(self, feedback_processor: FeedbackProcessor, mock_db):
        """Toplu geri bildirimlerin başarılı işlenmesini test eder."""
        # Arrange
        feedbacks = [
            FeedbackCreate(interaction_id="int-1", student_id="stu-1", feedback_type=FeedbackType.CORRECT),
            FeedbackCreate(interaction_id="int-2", student_id="stu-2", feedback_type=FeedbackType.UNCLEAR)
        ]
        bulk_data = BulkFeedbackCreate(feedbacks=feedbacks)
        
        # Her add_feedback çağrısı için farklı bir ID döndür
        mock_db.add_feedback.side_effect = [10, 20]
        
        # Act
        result = feedback_processor.process_bulk_feedback(bulk_data)
        
        # Assert
        assert mock_db.add_feedback.call_count == 2
        assert result["status"] == "success"
        assert result["processed_ids"] == [10, 20]
        assert "2 of 2 feedbacks processed" in result["message"]
    
    @pytest.mark.unit
    def test_process_bulk_feedback_empty_list(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests processing empty bulk feedback list."""
        bulk_data = BulkFeedbackCreate(feedbacks=[])
        
        result = feedback_processor.process_bulk_feedback(bulk_data)
        
        assert mock_db.add_feedback.call_count == 0
        assert result["status"] == "success"
        assert result["processed_ids"] == []
        assert "0 of 0 feedbacks processed" in result["message"]
    
    @pytest.mark.unit
    def test_process_bulk_feedback_with_partial_failure(self, feedback_processor: FeedbackProcessor, mock_db):
        """Toplu geri bildirimde bazı işlemlerin başarısız olmasını test eder."""
        # Arrange
        feedbacks = [
            FeedbackCreate(interaction_id="int-1", student_id="stu-1", feedback_type=FeedbackType.CORRECT),
            FeedbackCreate(interaction_id="int-2", student_id="stu-2", feedback_type=FeedbackType.INCORRECT),
            FeedbackCreate(interaction_id="int-3", student_id="stu-3", feedback_type=FeedbackType.HELPFUL)
        ]
        bulk_data = BulkFeedbackCreate(feedbacks=feedbacks)
        
        # İkinci çağrıda hata fırlat
        mock_db.add_feedback.side_effect = [10, Exception("Failed to process"), 30]
        
        # Act
        result = feedback_processor.process_bulk_feedback(bulk_data)
        
        # Assert
        assert mock_db.add_feedback.call_count == 3
        assert result["status"] == "success"
        assert result["processed_ids"] == [10, 30]  # Only successful ones
        assert "2 of 3 feedbacks processed" in result["message"]
    
    @pytest.mark.unit
    def test_process_bulk_feedback_all_failures(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests bulk feedback processing when all feedbacks fail."""
        feedbacks = [
            FeedbackCreate(interaction_id="int-1", student_id="stu-1", feedback_type=FeedbackType.CORRECT),
            FeedbackCreate(interaction_id="int-2", student_id="stu-2", feedback_type=FeedbackType.INCORRECT)
        ]
        bulk_data = BulkFeedbackCreate(feedbacks=feedbacks)
        
        mock_db.add_feedback.side_effect = [Exception("Error 1"), Exception("Error 2")]
        
        result = feedback_processor.process_bulk_feedback(bulk_data)
        
        assert mock_db.add_feedback.call_count == 2
        assert result["status"] == "success"
        assert result["processed_ids"] == []
        assert "0 of 2 feedbacks processed" in result["message"]
    
    @pytest.mark.unit
    @patch('src.services.feedback_processor.logger')
    def test_process_bulk_feedback_logs_correctly(self, mock_logger, feedback_processor: FeedbackProcessor, mock_db):
        """Tests that bulk feedback processing logs correctly."""
        feedbacks = [
            FeedbackCreate(interaction_id="int-1", student_id="stu-1", feedback_type=FeedbackType.CORRECT)
        ]
        bulk_data = BulkFeedbackCreate(feedbacks=feedbacks)
        mock_db.add_feedback.return_value = 10
        
        feedback_processor.process_bulk_feedback(bulk_data)
        
        # Check initial logging
        mock_logger.info.assert_any_call("1 adet geri bildirim toplu olarak işleniyor.")
    
    @pytest.mark.unit
    @patch('src.services.feedback_processor.logger')
    def test_process_bulk_feedback_logs_warnings_on_failures(self, mock_logger, feedback_processor: FeedbackProcessor, mock_db):
        """Tests that bulk feedback processing logs warnings on individual failures."""
        feedbacks = [
            FeedbackCreate(interaction_id="int-1", student_id="stu-1", feedback_type=FeedbackType.CORRECT)
        ]
        bulk_data = BulkFeedbackCreate(feedbacks=feedbacks)
        error_message = "Individual processing failed"
        mock_db.add_feedback.side_effect = Exception(error_message)
        
        feedback_processor.process_bulk_feedback(bulk_data)
        
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Toplu geri bildirim işlenirken bir geri bildirimde hata oluştu" in warning_call


class TestGetFeedbacksByInteraction:
    """Tests for get_feedbacks_by_interaction method."""
    
    @pytest.mark.unit
    def test_get_feedbacks_by_interaction_success(self, feedback_processor: FeedbackProcessor, mock_db):
        """Etkileşim ID'sine göre geri bildirimlerin başarılı getirilmesini test eder."""
        # Arrange
        interaction_id = "interaction-abc"
        mock_return_value = [
            {"id": 1, "comment": "Test comment 1", "feedback_type": "helpful"},
            {"id": 2, "comment": "Test comment 2", "feedback_type": "correct"}
        ]
        mock_db.get_feedbacks_by_interaction_id.return_value = mock_return_value
        
        # Act
        result = feedback_processor.get_feedbacks_by_interaction(interaction_id)
        
        # Assert
        mock_db.get_feedbacks_by_interaction_id.assert_called_once_with(interaction_id)
        assert result == mock_return_value
    
    @pytest.mark.unit
    def test_get_feedbacks_by_interaction_empty_result(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests getting feedbacks when no feedbacks exist for interaction."""
        interaction_id = "nonexistent-interaction"
        mock_db.get_feedbacks_by_interaction_id.return_value = []
        
        result = feedback_processor.get_feedbacks_by_interaction(interaction_id)
        
        assert result == []
        mock_db.get_feedbacks_by_interaction_id.assert_called_once_with(interaction_id)
    
    @pytest.mark.unit
    @patch('src.services.feedback_processor.logger')
    def test_get_feedbacks_by_interaction_database_error(self, mock_logger, feedback_processor: FeedbackProcessor, mock_db):
        """Tests get_feedbacks_by_interaction when database error occurs."""
        interaction_id = "error-interaction"
        error_message = "Database query failed"
        mock_db.get_feedbacks_by_interaction_id.side_effect = Exception(error_message)
        
        with pytest.raises(Exception, match=error_message):
            feedback_processor.get_feedbacks_by_interaction(interaction_id)
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert interaction_id in error_call
        assert "geri bildirimler getirilirken hata oluştu" in error_call
    
    @pytest.mark.unit
    @patch('src.services.feedback_processor.logger')
    def test_get_feedbacks_by_interaction_logs_correctly(self, mock_logger, feedback_processor: FeedbackProcessor, mock_db):
        """Tests that get_feedbacks_by_interaction logs correctly."""
        interaction_id = "test-interaction"
        mock_db.get_feedbacks_by_interaction_id.return_value = []
        
        feedback_processor.get_feedbacks_by_interaction(interaction_id)
        
        mock_logger.info.assert_called_with(f"'{interaction_id}' için geri bildirimler getiriliyor.")


class TestGetFeedbackStats:
    """Tests for get_feedback_stats method."""
    
    @pytest.mark.unit
    def test_get_feedback_stats_success(self, feedback_processor: FeedbackProcessor, mock_db):
        """Geri bildirim istatistiklerinin başarılı getirilmesini test eder."""
        # Arrange
        mock_stats = {
            "total": 100,
            "positive": 80,
            "negative": 20,
            "by_type": {
                "helpful": 30,
                "correct": 50,
                "incorrect": 15,
                "unclear": 5
            }
        }
        mock_db.get_feedback_statistics.return_value = mock_stats
        
        # Act
        result = feedback_processor.get_feedback_stats()
        
        # Assert
        mock_db.get_feedback_statistics.assert_called_once()
        assert result == mock_stats
    
    @pytest.mark.unit
    def test_get_feedback_stats_empty_stats(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests getting stats when no feedback data exists."""
        mock_empty_stats = {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "by_type": {}
        }
        mock_db.get_feedback_statistics.return_value = mock_empty_stats
        
        result = feedback_processor.get_feedback_stats()
        
        assert result == mock_empty_stats
    
    @pytest.mark.unit
    @patch('src.services.feedback_processor.logger')
    def test_get_feedback_stats_database_error(self, mock_logger, feedback_processor: FeedbackProcessor, mock_db):
        """Tests get_feedback_stats when database error occurs."""
        error_message = "Statistics calculation failed"
        mock_db.get_feedback_statistics.side_effect = Exception(error_message)
        
        with pytest.raises(Exception, match=error_message):
            feedback_processor.get_feedback_stats()
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "Geri bildirim istatistikleri hesaplanırken hata oluştu" in error_call
    
    @pytest.mark.unit
    @patch('src.services.feedback_processor.logger')
    def test_get_feedback_stats_logs_correctly(self, mock_logger, feedback_processor: FeedbackProcessor, mock_db):
        """Tests that get_feedback_stats logs correctly."""
        mock_db.get_feedback_statistics.return_value = {"total": 10}
        
        feedback_processor.get_feedback_stats()
        
        mock_logger.info.assert_called_with("Geri bildirim istatistikleri hesaplanıyor.")


class TestFeedbackProcessorIntegration:
    """Integration-style tests testing multiple methods together."""
    
    @pytest.mark.unit
    def test_feedback_processor_workflow(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests a typical workflow of processing feedback and retrieving statistics."""
        # Process some feedback
        feedback_data = FeedbackCreate(
            interaction_id="workflow-test",
            student_id="student-workflow",
            feedback_type=FeedbackType.HELPFUL,
            comment="Workflow test comment"
        )
        
        mock_db.add_feedback.return_value = 200
        mock_db.get_feedbacks_by_interaction_id.return_value = [{"id": 200, "comment": "Workflow test comment"}]
        mock_db.get_feedback_statistics.return_value = {"total": 1, "positive": 1, "negative": 0}
        
        # Act
        process_result = feedback_processor.process_feedback(feedback_data)
        feedbacks = feedback_processor.get_feedbacks_by_interaction("workflow-test")
        stats = feedback_processor.get_feedback_stats()
        
        # Assert
        assert process_result["status"] == "success"
        assert process_result["feedback_id"] == 200
        assert len(feedbacks) == 1
        assert feedbacks[0]["id"] == 200
        assert stats["total"] == 1
        assert stats["positive"] == 1


class TestFeedbackProcessorEdgeCases:
    """Edge case tests for FeedbackProcessor."""
    
    @pytest.mark.unit
    def test_feedback_with_special_characters_in_comment(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests processing feedback with special characters in comment."""
        mock_db.add_feedback.return_value = 300
        
        feedback_data = FeedbackCreate(
            interaction_id="special-chars",
            student_id="student-special",
            feedback_type=FeedbackType.HELPFUL,
            comment="Special chars: àáâãäåæçèéêë ñóôõö üýÿ 中文 العربية 🎉🚀"
        )
        
        result = feedback_processor.process_feedback(feedback_data)
        
        assert result["status"] == "success"
        assert result["feedback_id"] == 300
    
    @pytest.mark.unit
    def test_feedback_with_very_long_interaction_id(self, feedback_processor: FeedbackProcessor, mock_db):
        """Tests processing feedback with very long interaction ID."""
        mock_db.add_feedback.return_value = 301
        long_interaction_id = "interaction-" + "x" * 1000
        
        feedback_data = FeedbackCreate(
            interaction_id=long_interaction_id,
            student_id="student-long-id",
            feedback_type=FeedbackType.CORRECT,
            comment="Test with long interaction ID"
        )
        
        result = feedback_processor.process_feedback(feedback_data)
        
        assert result["status"] == "success"
        assert result["feedback_id"] == 301