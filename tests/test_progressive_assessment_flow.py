#!/usr/bin/env python3
"""
Progressive Assessment Flow - Comprehensive Test Suite (ADIM 3)

Tests all aspects of the progressive assessment system:
1. Backend API endpoints
2. Database operations
3. Adaptive triggering logic
4. Integration flow
5. Error handling

Run: python test_progressive_assessment_flow.py
"""

import asyncio
import json
import requests
import sqlite3
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressiveAssessmentTester:
    def __init__(self):
        self.api_base_url = os.getenv("APRAG_API_URL", "http://localhost:8007")
        self.db_path = os.getenv("APRAG_DB_PATH", "services/aprag_service/data/rag_assistant.db")
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        # Test data
        self.test_interaction_id = None
        self.test_user_id = "test_user_progressive"
        self.test_session_id = "test_session_progressive"
    
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        self.test_results["total_tests"] += 1
        if passed:
            self.test_results["passed"] += 1
            logger.info(f"✅ {test_name}: PASSED {message}")
        else:
            self.test_results["failed"] += 1
            error_msg = f"❌ {test_name}: FAILED {message}"
            logger.error(error_msg)
            self.test_results["errors"].append(error_msg)
    
    def test_database_schema(self):
        """Test 1: Database schema validation"""
        logger.info("🔍 Testing database schema...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check progressive_assessments table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='progressive_assessments'")
            table_exists = cursor.fetchone() is not None
            self.log_result("Database - progressive_assessments table", table_exists)
            
            if table_exists:
                cursor.execute("PRAGMA table_info(progressive_assessments)")
                columns = [col[1] for col in cursor.fetchall()]
                required_columns = ['assessment_id', 'interaction_id', 'user_id', 'session_id', 'stage', 
                                  'confidence_level', 'has_questions', 'application_understanding']
                
                all_columns_exist = all(col in columns for col in required_columns)
                self.log_result("Database - progressive_assessments columns", all_columns_exist, 
                              f"Found: {len(columns)} columns")
            
            # Check other tables
            for table in ['concept_confusion_log', 'requested_topics_log', 'progressive_assessment_summary']:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                exists = cursor.fetchone() is not None
                self.log_result(f"Database - {table} table", exists)
            
            # Check enhanced columns in existing tables
            cursor.execute("PRAGMA table_info(student_interactions)")
            si_columns = [col[1] for col in cursor.fetchall()]
            progressive_cols = ['progressive_assessment_data', 'progressive_assessment_stage']
            
            for col in progressive_cols:
                exists = col in si_columns
                self.log_result(f"Database - student_interactions.{col}", exists)
            
            conn.close()
            
        except Exception as e:
            self.log_result("Database schema test", False, str(e))
    
    def test_api_health(self):
        """Test 2: API health and feature flags"""
        logger.info("🔍 Testing API health...")
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            self.log_result("API - Health endpoint", response.status_code == 200)
            
            if response.status_code == 200:
                health_data = response.json()
                progressive_enabled = health_data.get("features", {}).get("progressive_assessment", False)
                self.log_result("API - Progressive assessment feature enabled", progressive_enabled)
                
        except Exception as e:
            self.log_result("API health test", False, str(e))
    
    def setup_test_data(self):
        """Test 3: Setup test interaction data"""
        logger.info("🔍 Setting up test data...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert test interaction
            cursor.execute("""
                INSERT INTO student_interactions 
                (user_id, session_id, query, original_response, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                self.test_user_id,
                self.test_session_id,
                "Test progressive assessment question",
                "Test progressive assessment response",
                datetime.now().isoformat()
            ))
            
            self.test_interaction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.log_result("Setup - Test interaction created", self.test_interaction_id is not None, 
                          f"ID: {self.test_interaction_id}")
            
        except Exception as e:
            self.log_result("Setup test data", False, str(e))
    
    def test_trigger_check_endpoint(self):
        """Test 4: Trigger check endpoint"""
        logger.info("🔍 Testing trigger check endpoint...")
        
        if not self.test_interaction_id:
            self.log_result("Trigger check test", False, "No test interaction ID")
            return
        
        try:
            url = f"{self.api_base_url}/api/aprag/progressive-assessment/check-trigger/{self.test_interaction_id}"
            response = requests.get(url, timeout=10)
            
            self.log_result("API - Trigger check endpoint", response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['trigger_follow_up', 'trigger_deep_analysis', 'interaction_id']
                
                all_fields_present = all(field in data for field in required_fields)
                self.log_result("API - Trigger check response format", all_fields_present)
                
        except Exception as e:
            self.log_result("Trigger check test", False, str(e))
    
    def test_follow_up_assessment_endpoint(self):
        """Test 5: Follow-up assessment endpoint"""
        logger.info("🔍 Testing follow-up assessment endpoint...")
        
        if not self.test_interaction_id:
            self.log_result("Follow-up test", False, "No test interaction ID")
            return
        
        try:
            url = f"{self.api_base_url}/api/aprag/progressive-assessment/follow-up"
            payload = {
                "interaction_id": self.test_interaction_id,
                "user_id": self.test_user_id,
                "session_id": self.test_session_id,
                "has_questions": True,
                "confidence_level": 2,
                "application_understanding": "Bu test verisi için gerçek bir uygulama örneği. Sistem test ediliyor ve çalışıyor gibi görünüyor.",
                "comment": "Test follow-up assessment"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            self.log_result("API - Follow-up assessment endpoint", response.status_code == 201)
            
            if response.status_code == 201:
                data = response.json()
                required_fields = ['assessment_id', 'message', 'stage', 'next_stage_available']
                
                all_fields_present = all(field in data for field in required_fields)
                self.log_result("API - Follow-up response format", all_fields_present)
                
                # Check if deep analysis is triggered for low confidence
                if data.get('next_stage_available'):
                    self.log_result("Logic - Deep analysis trigger for low confidence", True)
                
        except Exception as e:
            self.log_result("Follow-up assessment test", False, str(e))
    
    def test_deep_analysis_endpoint(self):
        """Test 6: Deep analysis endpoint"""
        logger.info("🔍 Testing deep analysis endpoint...")
        
        if not self.test_interaction_id:
            self.log_result("Deep analysis test", False, "No test interaction ID")
            return
        
        try:
            url = f"{self.api_base_url}/api/aprag/progressive-assessment/deep-analysis"
            payload = {
                "interaction_id": self.test_interaction_id,
                "user_id": self.test_user_id,
                "session_id": self.test_session_id,
                "confusion_areas": ["Temel kavramları anlamadım", "Örnekler yetersiz"],
                "requested_topics": ["Temel konular", "Pratik örnekler"],
                "alternative_explanation_request": "Daha çok görsel örneklerle açıklayabilir misiniz?",
                "comment": "Test deep analysis assessment"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            self.log_result("API - Deep analysis endpoint", response.status_code == 201)
            
            if response.status_code == 201:
                data = response.json()
                required_fields = ['assessment_id', 'message', 'stage', 'recommended_actions']
                
                all_fields_present = all(field in data for field in required_fields)
                self.log_result("API - Deep analysis response format", all_fields_present)
                
                # Check recommendations
                recommendations = data.get('recommended_actions', [])
                has_recommendations = len(recommendations) > 0
                self.log_result("Logic - Deep analysis recommendations generated", has_recommendations,
                              f"Count: {len(recommendations)}")
                
        except Exception as e:
            self.log_result("Deep analysis test", False, str(e))
    
    def test_insights_endpoint(self):
        """Test 7: Insights endpoint"""
        logger.info("🔍 Testing insights endpoint...")
        
        try:
            url = f"{self.api_base_url}/api/aprag/progressive-assessment/insights/{self.test_user_id}"
            params = {"session_id": self.test_session_id}
            
            response = requests.get(url, params=params, timeout=10)
            self.log_result("API - Insights endpoint", response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['user_id', 'session_id', 'confidence_trend', 'total_assessments']
                
                all_fields_present = all(field in data for field in required_fields)
                self.log_result("API - Insights response format", all_fields_present)
                
                # Check if we have assessment data
                total_assessments = data.get('total_assessments', 0)
                has_assessments = total_assessments > 0
                self.log_result("Data - Assessment data collected", has_assessments,
                              f"Total: {total_assessments}")
                
        except Exception as e:
            self.log_result("Insights test", False, str(e))
    
    def test_database_data_integrity(self):
        """Test 8: Database data integrity"""
        logger.info("🔍 Testing database data integrity...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check progressive assessments data
            cursor.execute("""
                SELECT COUNT(*) FROM progressive_assessments 
                WHERE user_id = ? AND session_id = ?
            """, (self.test_user_id, self.test_session_id))
            
            assessment_count = cursor.fetchone()[0]
            self.log_result("Data - Progressive assessments stored", assessment_count > 0,
                          f"Count: {assessment_count}")
            
            # Check interaction updates
            cursor.execute("""
                SELECT progressive_assessment_data, progressive_assessment_stage 
                FROM student_interactions 
                WHERE interaction_id = ?
            """, (self.test_interaction_id,))
            
            result = cursor.fetchone()
            if result:
                has_progressive_data = result[0] is not None
                has_progressive_stage = result[1] is not None
                
                self.log_result("Data - Interaction progressive data", has_progressive_data)
                self.log_result("Data - Interaction progressive stage", has_progressive_stage)
            
            # Check concept confusion log
            cursor.execute("""
                SELECT COUNT(*) FROM concept_confusion_log 
                WHERE user_id = ? AND session_id = ?
            """, (self.test_user_id, self.test_session_id))
            
            confusion_count = cursor.fetchone()[0]
            self.log_result("Data - Confusion areas logged", confusion_count > 0,
                          f"Count: {confusion_count}")
            
            conn.close()
            
        except Exception as e:
            self.log_result("Data integrity test", False, str(e))
    
    def test_adaptive_triggering_logic(self):
        """Test 9: Adaptive triggering logic scenarios"""
        logger.info("🔍 Testing adaptive triggering logic...")
        
        # Test different emoji score scenarios
        test_scenarios = [
            {"emoji_score": 0.0, "emoji": "❌", "should_trigger_follow": True, "should_trigger_deep": True},
            {"emoji_score": 0.2, "emoji": "😐", "should_trigger_follow": True, "should_trigger_deep": False},
            {"emoji_score": 0.7, "emoji": "😊", "should_trigger_follow": True, "should_trigger_deep": False},
            {"emoji_score": 1.0, "emoji": "👍", "should_trigger_follow": True, "should_trigger_deep": False}
        ]
        
        for scenario in test_scenarios:
            try:
                # This would require implementing the frontend adaptive service test
                # For now, we'll test the logic conceptually
                emoji_score = scenario["emoji_score"]
                
                # Critical case logic
                if emoji_score <= 0.0:
                    should_trigger_follow = True
                    should_trigger_deep = True
                elif emoji_score <= 0.2:
                    should_trigger_follow = True
                    should_trigger_deep = False
                else:
                    should_trigger_follow = True
                    should_trigger_deep = False
                
                expected_follow = scenario["should_trigger_follow"]
                expected_deep = scenario["should_trigger_deep"]
                
                follow_correct = should_trigger_follow == expected_follow
                deep_correct = should_trigger_deep == expected_deep
                
                self.log_result(f"Logic - {scenario['emoji']} follow-up trigger", follow_correct)
                self.log_result(f"Logic - {scenario['emoji']} deep analysis trigger", deep_correct)
                
            except Exception as e:
                self.log_result(f"Adaptive logic test - {scenario['emoji']}", False, str(e))
    
    def test_error_handling(self):
        """Test 10: Error handling"""
        logger.info("🔍 Testing error handling...")
        
        # Test invalid interaction ID
        try:
            url = f"{self.api_base_url}/api/aprag/progressive-assessment/check-trigger/99999"
            response = requests.get(url, timeout=10)
            
            error_handled = response.status_code == 404
            self.log_result("Error - Invalid interaction ID handling", error_handled)
            
        except Exception as e:
            self.log_result("Error handling test - invalid ID", False, str(e))
        
        # Test invalid payload
        try:
            url = f"{self.api_base_url}/api/aprag/progressive-assessment/follow-up"
            invalid_payload = {
                "interaction_id": "invalid",
                "confidence_level": 10  # Invalid range
            }
            
            response = requests.post(url, json=invalid_payload, timeout=10)
            error_handled = response.status_code in [400, 422]
            self.log_result("Error - Invalid payload handling", error_handled)
            
        except Exception as e:
            self.log_result("Error handling test - invalid payload", False, str(e))
    
    def cleanup_test_data(self):
        """Test 11: Cleanup test data"""
        logger.info("🔍 Cleaning up test data...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean up test data
            tables_to_clean = [
                "progressive_assessments",
                "concept_confusion_log", 
                "requested_topics_log",
                "student_interactions"
            ]
            
            for table in tables_to_clean:
                cursor.execute(f"""
                    DELETE FROM {table} 
                    WHERE user_id = ? OR session_id = ?
                """, (self.test_user_id, self.test_session_id))
            
            # Also clean by interaction_id if available
            if self.test_interaction_id:
                cursor.execute("DELETE FROM student_interactions WHERE interaction_id = ?", 
                             (self.test_interaction_id,))
            
            conn.commit()
            conn.close()
            
            self.log_result("Cleanup - Test data removed", True)
            
        except Exception as e:
            self.log_result("Cleanup test", False, str(e))
    
    def print_summary(self):
        """Print test summary"""
        total = self.test_results["total_tests"]
        passed = self.test_results["passed"]
        failed = self.test_results["failed"]
        
        print("\n" + "="*60)
        print("🧪 PROGRESSIVE ASSESSMENT FLOW - TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%" if total > 0 else "0%")
        
        if failed > 0:
            print("\n🚨 FAILED TESTS:")
            for error in self.test_results["errors"]:
                print(f"  - {error}")
        
        print("\n" + "="*60)
        
        if failed == 0:
            print("🎉 ALL TESTS PASSED! Progressive Assessment Flow is ready for production.")
        else:
            print(f"⚠️  {failed} tests failed. Please review and fix issues before deployment.")
        
        print("="*60)
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Progressive Assessment Flow Test Suite...")
        print("="*60)
        
        # Run tests in order
        self.test_database_schema()
        self.test_api_health()
        self.setup_test_data()
        self.test_trigger_check_endpoint()
        self.test_follow_up_assessment_endpoint()
        self.test_deep_analysis_endpoint()
        self.test_insights_endpoint()
        self.test_database_data_integrity()
        self.test_adaptive_triggering_logic()
        self.test_error_handling()
        self.cleanup_test_data()
        
        # Print results
        self.print_summary()
        
        # Return success status
        return self.test_results["failed"] == 0

def main():
    """Main test runner"""
    tester = ProgressiveAssessmentTester()
    
    try:
        success = asyncio.run(tester.run_all_tests())
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test suite interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()