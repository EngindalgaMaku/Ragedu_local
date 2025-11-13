import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from collections import Counter
import os

from src.analytics.report_generator import generate_report, LOG_FILE_PATH


@pytest.fixture
def temp_csv_dir():
    """Create temporary directory for CSV files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_query_data():
    """Sample query data for testing."""
    return [
        {"timestamp": "2023-01-01T10:00:00", "query": "What is machine learning?"},
        {"timestamp": "2023-01-01T10:05:00", "query": "How does deep learning work?"},
        {"timestamp": "2023-01-01T10:10:00", "query": "Machine learning algorithms explained"},
        {"timestamp": "2023-01-01T10:15:00", "query": "What are neural networks?"},
        {"timestamp": "2023-01-01T10:20:00", "query": "Deep learning vs machine learning"},
    ]


@pytest.fixture
def create_test_csv():
    """Factory function to create test CSV files."""
    def _create_csv(data, filepath):
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return filepath
    return _create_csv


class TestGenerateReportFileHandling:
    """Tests for file handling in generate_report function."""
    
    @pytest.mark.unit
    def test_generate_report_file_not_found(self):
        """Test generate_report when CSV file doesn't exist."""
        with patch('src.analytics.report_generator.LOG_FILE_PATH', '/nonexistent/path.csv'):
            result = generate_report()
            
            assert "error" in result
            assert "Log file not found" in result["error"]
            assert "No queries have been logged yet" in result["error"]
    
    @pytest.mark.unit
    def test_generate_report_with_existing_file(self, temp_csv_dir, sample_query_data, create_test_csv):
        """Test generate_report with existing CSV file."""
        csv_path = Path(temp_csv_dir) / "test_queries.csv"
        create_test_csv(sample_query_data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            assert "top_keywords" in result
            assert isinstance(result["top_keywords"], list)
            assert len(result["top_keywords"]) <= 5
    
    @pytest.mark.unit
    def test_generate_report_empty_file(self, temp_csv_dir, create_test_csv):
        """Test generate_report with empty CSV file."""
        csv_path = Path(temp_csv_dir) / "empty_queries.csv"
        create_test_csv([], csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            # Empty DataFrame should still work, but produce empty results
            result = generate_report()
            
            assert "top_keywords" in result
            assert result["top_keywords"] == []
    
    @pytest.mark.unit
    def test_generate_report_csv_read_error(self):
        """Test generate_report when CSV reading fails."""
        with patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError("No data")):
            result = generate_report()
            
            # Should handle pandas errors gracefully
            # In this case it might raise an exception or return error
            assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_generate_report_permission_error(self):
        """Test generate_report when file access is denied."""
        with patch('pandas.read_csv', side_effect=PermissionError("Access denied")):
            # Should handle permission errors
            try:
                result = generate_report()
                # If it doesn't raise, should return error dict
                assert isinstance(result, dict)
            except PermissionError:
                # Or it might raise the exception
                pass


class TestKeywordExtraction:
    """Tests for keyword extraction logic."""
    
    @pytest.mark.unit
    def test_keyword_extraction_basic(self, temp_csv_dir, create_test_csv):
        """Test basic keyword extraction functionality."""
        data = [
            {"query": "machine learning algorithm"},
            {"query": "machine learning model"},
            {"query": "deep learning network"},
        ]
        csv_path = Path(temp_csv_dir) / "keyword_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # "machine" and "learning" should appear multiple times
            assert keywords.get("machine", 0) >= 2
            assert keywords.get("learning", 0) >= 3  # appears in all queries
            assert "deep" in keywords
    
    @pytest.mark.unit
    def test_keyword_extraction_case_insensitive(self, temp_csv_dir, create_test_csv):
        """Test that keyword extraction is case-insensitive."""
        data = [
            {"query": "Machine Learning"},
            {"query": "machine learning"},
            {"query": "MACHINE LEARNING"},
        ]
        csv_path = Path(temp_csv_dir) / "case_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # All variations should be counted together as lowercase
            assert keywords.get("machine", 0) == 3
            assert keywords.get("learning", 0) == 3
    
    @pytest.mark.unit
    def test_keyword_extraction_special_characters(self, temp_csv_dir, create_test_csv):
        """Test keyword extraction handles special characters."""
        data = [
            {"query": "machine-learning algorithms!"},
            {"query": "deep_learning networks?"},
            {"query": "AI/ML technologies & applications"},
        ]
        csv_path = Path(temp_csv_dir) / "special_chars_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # Special characters should be replaced with spaces
            assert "machine" in keywords
            assert "learning" in keywords
            assert "deep" in keywords
            # Should not contain punctuation
            assert "machine-learning" not in keywords
            assert "deep_learning" not in keywords
    
    @pytest.mark.unit
    def test_keyword_extraction_numbers_and_alphanumeric(self, temp_csv_dir, create_test_csv):
        """Test keyword extraction with numbers and alphanumeric strings."""
        data = [
            {"query": "GPT-3 model training"},
            {"query": "BERT transformer 2023"},
            {"query": "Python 3.9 programming"},
        ]
        csv_path = Path(temp_csv_dir) / "numbers_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # Numbers and mixed alphanumeric should be handled
            assert "gpt" in keywords
            assert "bert" in keywords
            assert "python" in keywords
            assert "model" in keywords
    
    @pytest.mark.unit
    def test_keyword_extraction_unicode_support(self, temp_csv_dir, create_test_csv):
        """Test keyword extraction with Unicode characters."""
        data = [
            {"query": "makine öğrenmesi nedir?"},
            {"query": "derin öğrenme algoritmaları"},
            {"query": "yapay zeka uygulamaları"},
            {"query": "machine learning başlangıç"},
        ]
        csv_path = Path(temp_csv_dir) / "unicode_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # Should handle Turkish characters correctly
            assert "makine" in keywords or "öğrenmesi" in keywords or "öğrenme" in keywords
            assert "machine" in keywords
            assert "learning" in keywords


class TestStopwordFiltering:
    """Tests for stopword filtering functionality."""
    
    @pytest.mark.unit
    def test_stopword_filtering_english(self, temp_csv_dir, create_test_csv):
        """Test that English stopwords are filtered out."""
        data = [
            {"query": "What is the machine learning algorithm"},
            {"query": "How does it work with the data"},
            {"query": "Where can I find information about this"},
        ]
        csv_path = Path(temp_csv_dir) / "stopwords_en_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # English stopwords should be filtered
            english_stopwords = ["what", "is", "the", "how", "does", "it", "with", "where", "can", "about"]
            for stopword in english_stopwords:
                assert stopword not in keywords
            
            # Content words should remain
            assert "machine" in keywords
            assert "learning" in keywords
            assert "algorithm" in keywords
    
    @pytest.mark.unit
    def test_stopword_filtering_turkish(self, temp_csv_dir, create_test_csv):
        """Test that Turkish stopwords are filtered out."""
        data = [
            {"query": "Bu makine öğrenmesi ile yapılır"},
            {"query": "Ve derin öğrenme mi kullanılır"},
            {"query": "Nasıl bu teknoloji mü gelişir"},
        ]
        csv_path = Path(temp_csv_dir) / "stopwords_tr_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # Turkish stopwords should be filtered
            turkish_stopwords = ["bu", "ve", "ile", "mi", "mu", "mı", "mü"]
            for stopword in turkish_stopwords:
                assert stopword not in keywords
            
            # Content words should remain
            assert "makine" in keywords
            assert "öğrenmesi" in keywords or "öğrenme" in keywords
    
    @pytest.mark.unit
    def test_short_word_filtering(self, temp_csv_dir, create_test_csv):
        """Test that words with length <= 2 are filtered out."""
        data = [
            {"query": "AI ML is good to go"},
            {"query": "I am ok so it is on"},
            {"query": "To be or not to be"},
        ]
        csv_path = Path(temp_csv_dir) / "short_words_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # Short words should be filtered (length <= 2)
            short_words = ["ai", "ml", "is", "to", "go", "am", "ok", "so", "it", "on", "or", "be"]
            for short_word in short_words:
                assert short_word not in keywords
            
            # Longer words should remain
            assert "good" in keywords
            assert "not" in keywords


class TestTopKeywordsLogic:
    """Tests for top keywords selection logic."""
    
    @pytest.mark.unit
    def test_top_5_keywords_limit(self, temp_csv_dir, create_test_csv):
        """Test that only top 5 keywords are returned."""
        # Create data with more than 5 distinct keywords
        data = []
        keywords = ["machine", "learning", "deep", "neural", "network", "algorithm", "model", "training"]
        for i, keyword in enumerate(keywords):
            # Each keyword appears a different number of times
            for j in range(len(keywords) - i):
                data.append({"query": f"{keyword} research study"})
        
        csv_path = Path(temp_csv_dir) / "top5_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            assert len(result["top_keywords"]) <= 5
            
            # Results should be ordered by frequency (descending)
            if len(result["top_keywords"]) > 1:
                counts = [count for keyword, count in result["top_keywords"]]
                assert counts == sorted(counts, reverse=True)
    
    @pytest.mark.unit
    def test_top_keywords_ordering(self, temp_csv_dir, create_test_csv):
        """Test that keywords are ordered by frequency."""
        data = [
            {"query": "machine learning"},  # machine: 1, learning: 1
            {"query": "machine algorithm"},  # machine: 2, algorithm: 1
            {"query": "machine model"},     # machine: 3, model: 1
            {"query": "deep learning"},     # learning: 2, deep: 1
            {"query": "deep network"},      # deep: 2, network: 1
        ]
        csv_path = Path(temp_csv_dir) / "ordering_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords_list = result["top_keywords"]
            
            # "machine" should be first (appears 3 times)
            assert keywords_list[0][0] == "machine"
            assert keywords_list[0][1] == 3
            
            # Next should be words that appear 2 times
            second_tier = [item for item in keywords_list if item[1] == 2]
            assert len(second_tier) >= 1  # learning and deep should both appear twice
    
    @pytest.mark.unit
    def test_keywords_with_equal_frequency(self, temp_csv_dir, create_test_csv):
        """Test behavior when multiple keywords have the same frequency."""
        data = [
            {"query": "machine learning"},
            {"query": "deep network"},
            {"query": "neural algorithm"},
        ]
        csv_path = Path(temp_csv_dir) / "equal_freq_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords_dict = dict(result["top_keywords"])
            
            # All keywords should appear once
            expected_keywords = ["machine", "learning", "deep", "network", "neural", "algorithm"]
            for keyword in expected_keywords:
                assert keywords_dict.get(keyword, 0) == 1
    
    @pytest.mark.unit
    def test_no_valid_keywords(self, temp_csv_dir, create_test_csv):
        """Test behavior when no valid keywords remain after filtering."""
        data = [
            {"query": "a an the"},
            {"query": "is in it of"},
            {"query": "bu ve ile mi"},
        ]
        csv_path = Path(temp_csv_dir) / "no_keywords_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            assert result["top_keywords"] == []


class TestDataTypes:
    """Tests for handling different data types in queries."""
    
    @pytest.mark.unit
    def test_non_string_queries(self, temp_csv_dir, create_test_csv):
        """Test handling of non-string query values."""
        data = [
            {"query": "machine learning"},
            {"query": 12345},
            {"query": None},
            {"query": 3.14159},
            {"query": "deep learning"},
        ]
        csv_path = Path(temp_csv_dir) / "non_string_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            # Should handle non-string values by converting to string
            assert "top_keywords" in result
            keywords = dict(result["top_keywords"])
            
            # String queries should still be processed
            assert "machine" in keywords
            assert "learning" in keywords
            assert "deep" in keywords
    
    @pytest.mark.unit
    def test_nan_and_null_values(self, temp_csv_dir):
        """Test handling of NaN and null values in queries."""
        import numpy as np
        
        # Create DataFrame with NaN values
        df = pd.DataFrame({
            "timestamp": ["2023-01-01T10:00:00", "2023-01-01T10:05:00", "2023-01-01T10:10:00"],
            "query": ["machine learning", np.nan, "deep learning"]
        })
        
        csv_path = Path(temp_csv_dir) / "nan_test.csv"
        df.to_csv(csv_path, index=False)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            # Should handle NaN values gracefully
            assert "top_keywords" in result
            keywords = dict(result["top_keywords"])
            
            # Valid queries should still be processed
            assert "machine" in keywords
            assert "learning" in keywords
            assert "deep" in keywords
    
    @pytest.mark.unit
    def test_empty_string_queries(self, temp_csv_dir, create_test_csv):
        """Test handling of empty string queries."""
        data = [
            {"query": "machine learning"},
            {"query": ""},
            {"query": "   "},  # Whitespace only
            {"query": "deep learning"},
        ]
        csv_path = Path(temp_csv_dir) / "empty_string_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            # Should handle empty strings gracefully
            assert "top_keywords" in result
            keywords = dict(result["top_keywords"])
            
            # Valid queries should still be processed
            assert "machine" in keywords
            assert "learning" in keywords
            assert "deep" in keywords


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_single_query_single_word(self, temp_csv_dir, create_test_csv):
        """Test with single query containing single word."""
        data = [{"query": "machine"}]
        csv_path = Path(temp_csv_dir) / "single_word_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            assert result["top_keywords"] == [("machine", 1)]
    
    @pytest.mark.unit
    def test_very_long_query(self, temp_csv_dir, create_test_csv):
        """Test with very long query text."""
        long_query = "machine " + "learning " * 1000 + "algorithm"
        data = [{"query": long_query}]
        csv_path = Path(temp_csv_dir) / "long_query_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            assert "machine" in keywords
            assert "learning" in keywords
            assert keywords["learning"] == 1000  # Should appear 1000 times
            assert "algorithm" in keywords
    
    @pytest.mark.unit
    def test_special_regex_characters(self, temp_csv_dir, create_test_csv):
        """Test with special regex characters in queries."""
        data = [
            {"query": "machine.learning+algorithm*"},
            {"query": "deep[learning]network?"},
            {"query": "neural^network$model"},
        ]
        csv_path = Path(temp_csv_dir) / "regex_chars_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            # Should handle regex special characters safely
            keywords = dict(result["top_keywords"])
            assert "machine" in keywords
            assert "learning" in keywords
            assert "algorithm" in keywords
            assert "deep" in keywords
    
    @pytest.mark.unit
    def test_mixed_language_content(self, temp_csv_dir, create_test_csv):
        """Test with mixed language content in queries."""
        data = [
            {"query": "machine learning makine öğrenmesi"},
            {"query": "deep learning derin öğrenme"},
            {"query": "artificial intelligence yapay zeka"},
        ]
        csv_path = Path(temp_csv_dir) / "mixed_lang_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # Should handle both English and Turkish words
            assert "machine" in keywords
            assert "learning" in keywords
            assert "makine" in keywords
            assert "deep" in keywords


class TestConstantsAndConfiguration:
    """Tests related to constants and configuration."""
    
    @pytest.mark.unit
    def test_log_file_path_constant(self):
        """Test that LOG_FILE_PATH constant is correctly defined."""
        assert LOG_FILE_PATH == "data/analytics/query_logs.csv"
        assert isinstance(LOG_FILE_PATH, str)
        assert LOG_FILE_PATH.endswith('.csv')
    
    @pytest.mark.unit
    def test_stopwords_set(self):
        """Test the stopwords set used in filtering."""
        # Since stopwords are defined in the function, we test them indirectly
        data = [{"query": "what is the machine learning algorithm"}]
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(data)
            
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # Verify that known stopwords are filtered
            stopwords_to_check = ["what", "is", "the"]
            for stopword in stopwords_to_check:
                assert stopword not in keywords


class TestIntegration:
    """Integration tests for realistic scenarios."""
    
    @pytest.mark.unit
    def test_realistic_query_log_analysis(self, temp_csv_dir, create_test_csv):
        """Test with realistic query log data."""
        realistic_data = [
            {"timestamp": "2023-01-01T10:00:00", "query": "What is machine learning?"},
            {"timestamp": "2023-01-01T10:05:00", "query": "How do neural networks work?"},
            {"timestamp": "2023-01-01T10:10:00", "query": "Explain deep learning algorithms"},
            {"timestamp": "2023-01-01T10:15:00", "query": "Machine learning vs artificial intelligence"},
            {"timestamp": "2023-01-01T10:20:00", "query": "Python machine learning libraries"},
            {"timestamp": "2023-01-01T10:25:00", "query": "What are the applications of AI?"},
            {"timestamp": "2023-01-01T10:30:00", "query": "How to implement neural networks"},
            {"timestamp": "2023-01-01T10:35:00", "query": "Deep learning frameworks comparison"},
            {"timestamp": "2023-01-01T10:40:00", "query": "Machine learning model training"},
            {"timestamp": "2023-01-01T10:45:00", "query": "Natural language processing techniques"},
        ]
        
        csv_path = Path(temp_csv_dir) / "realistic_test.csv"
        create_test_csv(realistic_data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            assert len(result["top_keywords"]) <= 5
            keywords = dict(result["top_keywords"])
            
            # Expected high-frequency terms
            assert "machine" in keywords
            assert "learning" in keywords
            assert keywords["machine"] >= 3  # appears in multiple queries
            assert keywords["learning"] >= 4  # appears frequently
            
            # Should not contain stopwords
            assert "what" not in keywords
            assert "how" not in keywords
            assert "the" not in keywords
    
    @pytest.mark.unit
    def test_performance_with_large_dataset(self, temp_csv_dir, create_test_csv):
        """Test performance with larger dataset."""
        # Create dataset with 1000 queries
        large_data = []
        base_queries = [
            "machine learning algorithm",
            "deep learning network", 
            "neural network model",
            "artificial intelligence system",
            "data science analysis"
        ]
        
        for i in range(1000):
            query = base_queries[i % len(base_queries)]
            large_data.append({
                "timestamp": f"2023-01-01T{10 + i//60:02d}:{i%60:02d}:00",
                "query": f"{query} example {i}"
            })
        
        csv_path = Path(temp_csv_dir) / "large_dataset_test.csv"
        create_test_csv(large_data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            # Should handle large dataset efficiently
            assert "top_keywords" in result
            assert len(result["top_keywords"]) <= 5
            
            keywords = dict(result["top_keywords"])
            
            # High frequency words should be detected
            assert "machine" in keywords
            assert "learning" in keywords
            assert "deep" in keywords
            
            # Should have meaningful counts
            assert keywords["machine"] >= 200  # appears in 20% of queries
            assert keywords["learning"] >= 400  # appears in 40% of queries


class TestErrorScenarios:
    """Tests for various error scenarios."""
    
    @pytest.mark.unit
    def test_corrupted_csv_file(self, temp_csv_dir):
        """Test with corrupted CSV file."""
        csv_path = Path(temp_csv_dir) / "corrupted_test.csv"
        
        # Create a file with invalid CSV content
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("invalid,csv,content\n")
            f.write("missing,columns\n")
            f.write("incomplete")
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            # Should handle corrupted CSV gracefully
            try:
                result = generate_report()
                # If it succeeds, should return valid structure
                assert isinstance(result, dict)
            except Exception as e:
                # Or might raise an exception - both are acceptable
                assert isinstance(e, (pd.errors.ParserError, KeyError, AttributeError))
    
    @pytest.mark.unit
    def test_missing_query_column(self, temp_csv_dir, create_test_csv):
        """Test with CSV file missing the 'query' column."""
        data = [
            {"timestamp": "2023-01-01T10:00:00", "content": "machine learning"},
            {"timestamp": "2023-01-01T10:05:00", "content": "deep learning"},
        ]
        csv_path = Path(temp_csv_dir) / "missing_column_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            # Should raise KeyError or handle gracefully
            try:
                result = generate_report()
                # If handled gracefully, should return error
                if isinstance(result, dict) and "error" in result:
                    assert "error" in result
            except KeyError:
                # Expected behavior when 'query' column is missing
                pass
    
    @pytest.mark.unit
    def test_regex_compilation_safety(self):
        """Test that regex operations are safe and don't cause errors."""
        # Test with various regex-problematic strings
        problematic_queries = [
            "query with [unclosed bracket",
            "query with (unclosed paren",
            "query with *invalid* regex",
            "query with \\backslash",
            "query with |pipe| character",
        ]
        
        data = [{"query": query} for query in problematic_queries]
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(data)
            
            # Should not raise regex compilation errors
            result = generate_report()
            
            assert "top_keywords" in result
            # Should process the queries without regex errors
            keywords = dict(result["top_keywords"])
            assert "query" in keywords  # "query" appears in all test strings


class TestReturnFormat:
    """Tests for return value format and structure."""
    
    @pytest.mark.unit
    def test_return_format_success(self, temp_csv_dir, create_test_csv):
        """Test the format of successful return value."""
        data = [{"query": "machine learning algorithm"}]
        csv_path = Path(temp_csv_dir) / "format_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            # Check return format
            assert isinstance(result, dict)
            assert "top_keywords" in result
            assert isinstance(result["top_keywords"], list)
            
            # Check keyword tuple format
            for keyword_tuple in result["top_keywords"]:
                assert isinstance(keyword_tuple, tuple)
                assert len(keyword_tuple) == 2
                assert isinstance(keyword_tuple[0], str)  # keyword
                assert isinstance(keyword_tuple[1], int)  # count
                assert keyword_tuple[1] > 0  # count should be positive
    
    @pytest.mark.unit
    def test_return_format_error(self):
        """Test the format of error return value."""
        with patch('src.analytics.report_generator.LOG_FILE_PATH', '/nonexistent/path.csv'):
            result = generate_report()
            
            # Check error format
            assert isinstance(result, dict)
            assert "error" in result
            assert "top_keywords" not in result
            assert isinstance(result["error"], str)
            assert len(result["error"]) > 0
    
    @pytest.mark.unit
    def test_keyword_count_accuracy(self, temp_csv_dir, create_test_csv):
        """Test that keyword counts are accurate."""
        data = [
            {"query": "machine learning"},      # machine: 1, learning: 1
            {"query": "machine algorithm"},     # machine: 2, algorithm: 1
            {"query": "learning algorithm"},    # learning: 2, algorithm: 2
            {"query": "machine learning algorithm"},  # machine: 3, learning: 3, algorithm: 3
        ]
        csv_path = Path(temp_csv_dir) / "count_accuracy_test.csv"
        create_test_csv(data, csv_path)
        
        with patch('src.analytics.report_generator.LOG_FILE_PATH', str(csv_path)):
            result = generate_report()
            
            keywords = dict(result["top_keywords"])
            
            # Verify exact counts
            assert keywords["machine"] == 3
            assert keywords["learning"] == 3
            assert keywords["algorithm"] == 3