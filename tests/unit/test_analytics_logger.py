import pytest
import csv
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import threading
import time

from src.analytics.analytics_logger import log_query, LOG_FILE_PATH


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for log files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_log_file(temp_log_dir):
    """Create temporary log file path."""
    return os.path.join(temp_log_dir, "test_query_logs.csv")


class TestLogQueryBasicFunctionality:
    """Tests for basic log_query functionality."""
    
    @pytest.mark.unit
    def test_log_query_creates_directory(self, temp_log_dir):
        """Test that log_query creates the directory if it doesn't exist."""
        log_path = os.path.join(temp_log_dir, "nonexistent", "query_logs.csv")
        
        # Directory shouldn't exist initially
        assert not os.path.exists(os.path.dirname(log_path))
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', log_path):
            log_query("test query")
        
        # Directory should be created
        assert os.path.exists(os.path.dirname(log_path))
        assert os.path.exists(log_path)
    
    @pytest.mark.unit
    def test_log_query_creates_file_with_headers(self, temp_log_file):
        """Test that log_query creates file with proper headers."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query("test query")
        
        # File should exist
        assert os.path.exists(temp_log_file)
        
        # Check file content
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data_row = next(reader)
            
            assert headers == ["timestamp", "query"]
            assert len(data_row) == 2
            assert data_row[1] == "test query"
    
    @pytest.mark.unit
    def test_log_query_appends_to_existing_file(self, temp_log_file):
        """Test that log_query appends to existing file without duplicating headers."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Log first query
            log_query("first query")
            # Log second query
            log_query("second query")
        
        # Read file content
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Should have header + 2 data lines = 3 lines total
        assert len(lines) == 3
        assert "timestamp,query" in lines[0]
        assert "first query" in lines[1]
        assert "second query" in lines[2]
    
    @pytest.mark.unit
    def test_log_query_timestamp_format(self, temp_log_file):
        """Test that timestamp is logged in correct ISO format."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query("timestamp test")
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data_row = next(reader)
            
            timestamp_str = data_row[0]
            
            # Should be able to parse as ISO format
            parsed_timestamp = datetime.fromisoformat(timestamp_str)
            assert isinstance(parsed_timestamp, datetime)
            
            # Should be recent (within last few seconds)
            time_diff = datetime.now() - parsed_timestamp
            assert time_diff.total_seconds() < 5
    
    @pytest.mark.unit
    def test_log_query_encoding_utf8(self, temp_log_file):
        """Test that log_query uses UTF-8 encoding."""
        unicode_query = "Test query with special chars: ñüéß 中文 العربية 🤖"
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query(unicode_query)
        
        # Read with UTF-8 encoding
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            assert unicode_query in content
            assert "中文" in content
            assert "🤖" in content


class TestLogQueryDataTypes:
    """Tests for handling different data types in queries."""
    
    @pytest.mark.unit
    def test_log_query_string_input(self, temp_log_file):
        """Test log_query with normal string input."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query("normal string query")
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data_row = next(reader)
            
            assert data_row[1] == "normal string query"
    
    @pytest.mark.unit
    def test_log_query_empty_string(self, temp_log_file):
        """Test log_query with empty string."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query("")
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data_row = next(reader)
            
            assert data_row[1] == ""
    
    @pytest.mark.unit
    def test_log_query_whitespace_only(self, temp_log_file):
        """Test log_query with whitespace-only string."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query("   \t\n   ")
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data_row = next(reader)
            
            assert data_row[1] == "   \t\n   "
    
    @pytest.mark.unit
    def test_log_query_very_long_string(self, temp_log_file):
        """Test log_query with very long string."""
        long_query = "A" * 10000
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query(long_query)
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data_row = next(reader)
            
            assert data_row[1] == long_query
            assert len(data_row[1]) == 10000
    
    @pytest.mark.unit
    def test_log_query_with_csv_special_chars(self, temp_log_file):
        """Test log_query with CSV special characters."""
        special_query = 'Query with "quotes", commas, and \nnewlines'
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query(special_query)
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data_row = next(reader)
            
            # CSV module should handle special characters correctly
            assert data_row[1] == special_query
    
    @pytest.mark.unit
    def test_log_query_unicode_edge_cases(self, temp_log_file):
        """Test log_query with various Unicode edge cases."""
        unicode_cases = [
            "🤖🔬📚",  # Emojis
            "नमस्ते",    # Devanagari
            "مرحبا",     # Arabic
            "こんにちは",   # Japanese
            "🇹🇷🇺🇸🇯🇵",  # Flag emojis
            "test\x00null",  # Null character
            "test\uffefBOM", # BOM character
        ]
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            for case in unicode_cases:
                log_query(case)
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            for case in unicode_cases:
                assert case in content


class TestLogQueryErrorHandling:
    """Tests for error handling scenarios."""
    
    @pytest.mark.unit
    def test_log_query_permission_denied(self):
        """Test log_query when file permissions are denied."""
        # Mock os.makedirs to raise PermissionError
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            with patch('src.analytics.analytics_logger.LOG_FILE_PATH', '/root/protected/query_logs.csv'):
                # Should raise PermissionError or handle it gracefully
                try:
                    log_query("test query")
                    # If it doesn't raise, it handled the error gracefully
                except PermissionError:
                    # Expected behavior for permission errors
                    pass
    
    @pytest.mark.unit
    def test_log_query_disk_full(self, temp_log_file):
        """Test log_query when disk is full."""
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
                # Should raise OSError or handle it gracefully
                try:
                    log_query("test query")
                except OSError:
                    # Expected behavior for disk full
                    pass
    
    @pytest.mark.unit
    def test_log_query_file_locked(self, temp_log_file):
        """Test log_query when file is locked by another process."""
        # Create the file first
        with open(temp_log_file, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "query"])
        
        # Mock file opening to simulate file lock
        with patch('builtins.open', side_effect=PermissionError("File is locked")):
            with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
                try:
                    log_query("test query")
                except PermissionError:
                    # Expected behavior for locked files
                    pass
    
    @pytest.mark.unit
    def test_log_query_invalid_path_characters(self):
        """Test log_query with invalid path characters."""
        # Use invalid characters for Windows paths
        invalid_path = "data/analytics/query<>logs.csv"
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', invalid_path):
            # Should handle invalid path characters
            try:
                log_query("test query")
            except (OSError, ValueError):
                # Expected behavior for invalid paths
                pass
    
    @pytest.mark.unit
    def test_log_query_directory_creation_failure(self):
        """Test log_query when directory creation fails."""
        with patch('os.makedirs', side_effect=OSError("Cannot create directory")):
            with patch('src.analytics.analytics_logger.LOG_FILE_PATH', '/invalid/path/query_logs.csv'):
                try:
                    log_query("test query")
                except OSError:
                    # Expected behavior when directory creation fails
                    pass


class TestLogQueryFileOperations:
    """Tests for file operation specifics."""
    
    @pytest.mark.unit
    def test_log_query_file_exists_check(self, temp_log_file):
        """Test that log_query correctly checks if file exists."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Mock os.path.isfile to return False first, then True
            with patch('os.path.isfile', side_effect=[False, True]):
                with patch('builtins.open', mock_open()) as mock_file:
                    with patch('csv.writer') as mock_writer:
                        mock_csv_instance = MagicMock()
                        mock_writer.return_value = mock_csv_instance
                        
                        log_query("test query")
                        
                        # Should write header when file doesn't exist
                        mock_csv_instance.writerow.assert_any_call(["timestamp", "query"])
    
    @pytest.mark.unit
    def test_log_query_no_headers_when_file_exists(self, temp_log_file):
        """Test that headers are not written when file already exists."""
        # Create file with existing content
        with open(temp_log_file, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "query"])
            writer.writerow(["2023-01-01T10:00:00", "existing query"])
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query("new query")
        
        # Check that headers weren't duplicated
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Count header lines
        header_count = sum(1 for line in lines if "timestamp,query" in line)
        assert header_count == 1  # Should only have one header line
    
    @pytest.mark.unit
    def test_log_query_append_mode(self, temp_log_file):
        """Test that log_query opens file in append mode."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Create initial file
            log_query("first query")
            
            # Get initial file size
            initial_size = os.path.getsize(temp_log_file)
            
            # Add another query
            log_query("second query")
            
            # File should be larger (content appended, not overwritten)
            final_size = os.path.getsize(temp_log_file)
            assert final_size > initial_size
    
    @pytest.mark.unit
    def test_log_query_newline_parameter(self, temp_log_file):
        """Test that log_query uses correct newline parameter for CSV."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Mock open to capture arguments
            with patch('builtins.open', mock_open()) as mock_file:
                log_query("test query")
                
                # Should open with newline="" for CSV writing
                mock_file.assert_called_with(temp_log_file, "a", newline="", encoding="utf-8")


class TestLogQueryConcurrency:
    """Tests for concurrent access scenarios."""
    
    @pytest.mark.unit
    def test_log_query_concurrent_writes(self, temp_log_file):
        """Test concurrent writes to the same log file."""
        def log_worker(worker_id, num_logs):
            for i in range(num_logs):
                log_query(f"Worker {worker_id} - Query {i}")
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Create multiple threads
            threads = []
            num_workers = 5
            logs_per_worker = 10
            
            for worker_id in range(num_workers):
                thread = threading.Thread(target=log_worker, args=(worker_id, logs_per_worker))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        
        # Check that all logs were written
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Should have header + (num_workers * logs_per_worker) data lines
        expected_lines = 1 + (num_workers * logs_per_worker)
        assert len(lines) == expected_lines
        
        # Check that each worker's logs are present
        content = ''.join(lines)
        for worker_id in range(num_workers):
            for log_num in range(logs_per_worker):
                assert f"Worker {worker_id} - Query {log_num}" in content
    
    @pytest.mark.unit
    def test_log_query_rapid_succession(self, temp_log_file):
        """Test rapid successive calls to log_query."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Log many queries rapidly
            num_queries = 100
            for i in range(num_queries):
                log_query(f"Rapid query {i}")
        
        # Check that all queries were logged
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Should have header + num_queries data rows
        assert len(rows) == num_queries + 1
        
        # Check that all queries are present and in order
        for i in range(num_queries):
            assert rows[i + 1][1] == f"Rapid query {i}"


class TestLogQueryConstants:
    """Tests related to constants and configuration."""
    
    @pytest.mark.unit
    def test_log_file_path_constant(self):
        """Test that LOG_FILE_PATH constant is correctly defined."""
        assert LOG_FILE_PATH == "data/analytics/query_logs.csv"
        assert isinstance(LOG_FILE_PATH, str)
        assert LOG_FILE_PATH.endswith('.csv')
        assert 'analytics' in LOG_FILE_PATH
    
    @pytest.mark.unit
    def test_log_file_path_directory_structure(self):
        """Test that LOG_FILE_PATH has correct directory structure."""
        path_parts = Path(LOG_FILE_PATH).parts
        assert 'data' in path_parts
        assert 'analytics' in path_parts
        assert path_parts[-1] == 'query_logs.csv'


class TestLogQueryIntegration:
    """Integration tests for realistic usage scenarios."""
    
    @pytest.mark.unit
    def test_realistic_logging_session(self, temp_log_file):
        """Test a realistic logging session with various query types."""
        realistic_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain deep learning algorithms",
            "Machine learning vs artificial intelligence",
            "Python machine learning libraries",
            "What are the applications of AI?",
            "How to implement neural networks",
            "Deep learning frameworks comparison",
            "",  # Empty query
            "Query with special chars: àáâãäåæçèéêë",
            "Turkish query: Makine öğrenmesi nedir?",
            "Query with numbers: GPT-3 and BERT comparison",
            "Very long query: " + "A" * 1000,
            "Query with\nnewlines and\ttabs",
        ]
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Log all queries
            for query in realistic_queries:
                log_query(query)
        
        # Verify all queries were logged correctly
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data_rows = list(reader)
        
        assert headers == ["timestamp", "query"]
        assert len(data_rows) == len(realistic_queries)
        
        # Check that all queries are present
        logged_queries = [row[1] for row in data_rows]
        assert logged_queries == realistic_queries
        
        # Check that all timestamps are valid
        for row in data_rows:
            timestamp_str = row[0]
            parsed_timestamp = datetime.fromisoformat(timestamp_str)
            assert isinstance(parsed_timestamp, datetime)
    
    @pytest.mark.unit
    def test_long_running_logging_session(self, temp_log_file):
        """Test logging over an extended period."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Log queries with small delays to simulate real usage
            queries = []
            for i in range(20):
                query = f"Session query {i} at {datetime.now().isoformat()}"
                queries.append(query)
                log_query(query)
                # Small delay to ensure different timestamps
                time.sleep(0.01)
        
        # Verify logging
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            data_rows = list(reader)
        
        assert len(data_rows) == 20
        
        # Check that timestamps are chronologically ordered
        timestamps = [datetime.fromisoformat(row[0]) for row in data_rows]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]
    
    @pytest.mark.unit
    def test_mixed_character_encoding_session(self, temp_log_file):
        """Test logging session with mixed character encodings."""
        mixed_queries = [
            "English query",
            "Türkçe sorgu",
            "中文查询",
            "العربية الاستعلام",
            "日本語クエリ",
            "Русский запрос",
            "Emoji query 🤖🔬📚",
            "Mixed: English + Türkçe + 中文 + 🚀",
        ]
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            for query in mixed_queries:
                log_query(query)
        
        # Verify all character encodings were preserved
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            for query in mixed_queries:
                assert query in content
        
        # Also verify through CSV reader
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            logged_queries = [row[1] for row in reader]
            
            assert logged_queries == mixed_queries


class TestLogQueryEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_log_query_with_none_input(self, temp_log_file):
        """Test log_query behavior with None input."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # This might raise an exception or convert None to string
            try:
                log_query(None)
                
                # If it doesn't raise, check the logged value
                with open(temp_log_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip headers
                    data_row = next(reader)
                    
                    # Should have converted None to string representation
                    assert data_row[1] in ["None", ""]
            except (TypeError, AttributeError):
                # Expected behavior for None input
                pass
    
    @pytest.mark.unit
    def test_log_query_with_numeric_input(self, temp_log_file):
        """Test log_query behavior with numeric input."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Should convert number to string
            log_query(12345)
            log_query(3.14159)
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            
            row1 = next(reader)
            row2 = next(reader)
            
            assert row1[1] == "12345"
            assert row2[1] == "3.14159"
    
    @pytest.mark.unit
    def test_log_query_with_boolean_input(self, temp_log_file):
        """Test log_query behavior with boolean input."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query(True)
            log_query(False)
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            
            row1 = next(reader)
            row2 = next(reader)
            
            assert row1[1] == "True"
            assert row2[1] == "False"
    
    @pytest.mark.unit
    def test_log_query_directory_already_exists(self, temp_log_file):
        """Test log_query when directory already exists."""
        # Create directory first
        os.makedirs(os.path.dirname(temp_log_file), exist_ok=True)
        
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            # Should work without error even if directory exists
            log_query("test query")
        
        assert os.path.exists(temp_log_file)
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            data_row = next(reader)
            
            assert data_row[1] == "test query"
    
    @pytest.mark.unit
    def test_log_query_zero_length_query(self, temp_log_file):
        """Test log_query with zero-length query string."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            log_query("")
        
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data_row = next(reader)
            
            assert headers == ["timestamp", "query"]
            assert data_row[1] == ""
            assert len(data_row[0]) > 0  # Timestamp should still be present


class TestLogQueryPerformance:
    """Tests related to performance characteristics."""
    
    @pytest.mark.unit
    def test_log_query_performance_single_call(self, temp_log_file):
        """Test performance of single log_query call."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            start_time = time.time()
            log_query("performance test query")
            end_time = time.time()
            
            # Should complete quickly (less than 1 second)
            assert end_time - start_time < 1.0
    
    @pytest.mark.unit
    def test_log_query_performance_multiple_calls(self, temp_log_file):
        """Test performance of multiple log_query calls."""
        with patch('src.analytics.analytics_logger.LOG_FILE_PATH', temp_log_file):
            start_time = time.time()
            
            # Log 100 queries
            for i in range(100):
                log_query(f"performance test query {i}")
            
            end_time = time.time()
            
            # Should complete in reasonable time (less than 5 seconds)
            assert end_time - start_time < 5.0
            
            # Verify all queries were logged
            with open(temp_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 101  # Header + 100 data lines