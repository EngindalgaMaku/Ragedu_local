#!/usr/bin/env python3
"""
Comprehensive Health Check Script for RAG3 Microservices
Tests all 7 Docker services for availability and basic functionality
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import sys

class ServiceHealthChecker:
    def __init__(self):
        self.services = {
            "api-gateway": {
                "port": 8000,
                "endpoints": ["/", "/health", "/docs", "/openapi.json"],
                "expected_status": [200, 404]
            },
            "model-inference-service": {
                "port": 8002,
                "endpoints": ["/", "/health", "/models", "/docs"],
                "expected_status": [200, 404]
            },
            "document-processing-service": {
                "port": 8003,
                "endpoints": ["/", "/health", "/docs", "/process"],
                "expected_status": [200, 404, 405]
            },
            "chromadb-service": {
                "port": 8004,
                "endpoints": ["/", "/api/v1", "/api/v1/heartbeat", "/docs"],
                "expected_status": [200, 404]
            },
            "docstrange-service": {
                "port": 8005,
                "endpoints": ["/", "/health", "/convert", "/docs"],
                "expected_status": [200, 404, 405]
            },
            "auth-service": {
                "port": 8006,
                "endpoints": ["/", "/health", "/docs", "/users"],
                "expected_status": [200, 404, 401]
            },
            "rag3-frontend": {
                "port": 3000,
                "endpoints": ["/", "/_next/static", "/api"],
                "expected_status": [200, 404]
            }
        }
        
        self.results = {}
        self.base_url = "http://localhost"
        self.timeout = 10
        
    def test_service_endpoint(self, service_name: str, port: int, endpoint: str) -> Dict:
        """Test a specific endpoint of a service"""
        url = f"{self.base_url}:{port}{endpoint}"
        
        try:
            print(f"  Testing: {url}")
            start_time = time.time()
            
            response = requests.get(url, timeout=self.timeout)
            response_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                "url": url,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "success": True,
                "error": None,
                "content_type": response.headers.get('content-type', ''),
                "content_length": len(response.content)
            }
            
        except requests.exceptions.ConnectionError:
            return {
                "url": url,
                "status_code": None,
                "response_time_ms": None,
                "success": False,
                "error": "Connection refused - Service may be down",
                "content_type": None,
                "content_length": 0
            }
        except requests.exceptions.Timeout:
            return {
                "url": url,
                "status_code": None,
                "response_time_ms": None,
                "success": False,
                "error": f"Timeout after {self.timeout}s",
                "content_type": None,
                "content_length": 0
            }
        except Exception as e:
            return {
                "url": url,
                "status_code": None,
                "response_time_ms": None,
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "content_type": None,
                "content_length": 0
            }
    
    def test_service(self, service_name: str) -> Dict:
        """Test all endpoints of a service"""
        print(f"\n🔍 Testing {service_name}...")
        
        service_config = self.services[service_name]
        port = service_config["port"]
        endpoints = service_config["endpoints"]
        expected_status = service_config["expected_status"]
        
        service_results = {
            "service_name": service_name,
            "port": port,
            "status": "unknown",
            "endpoints_tested": 0,
            "endpoints_responding": 0,
            "endpoints_results": [],
            "overall_health": "unknown",
            "response_times": [],
            "errors": []
        }
        
        for endpoint in endpoints:
            result = self.test_service_endpoint(service_name, port, endpoint)
            service_results["endpoints_results"].append(result)
            service_results["endpoints_tested"] += 1
            
            if result["success"]:
                service_results["endpoints_responding"] += 1
                if result["response_time_ms"]:
                    service_results["response_times"].append(result["response_time_ms"])
            else:
                service_results["errors"].append(result["error"])
        
        # Determine overall health
        if service_results["endpoints_responding"] == 0:
            service_results["overall_health"] = "DOWN"
            service_results["status"] = "❌ Service appears to be down"
        elif service_results["endpoints_responding"] == service_results["endpoints_tested"]:
            service_results["overall_health"] = "UP"
            service_results["status"] = "✅ All endpoints responding"
        else:
            service_results["overall_health"] = "PARTIAL"
            service_results["status"] = "⚠️ Some endpoints responding"
        
        # Calculate average response time
        if service_results["response_times"]:
            avg_response_time = sum(service_results["response_times"]) / len(service_results["response_times"])
            service_results["avg_response_time_ms"] = round(avg_response_time, 2)
        else:
            service_results["avg_response_time_ms"] = None
            
        return service_results
    
    def run_all_health_checks(self) -> Dict:
        """Run health checks for all services"""
        print("🏥 Starting comprehensive health checks for all services...")
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "total_services": len(self.services),
            "services_up": 0,
            "services_down": 0,
            "services_partial": 0,
            "service_results": {}
        }
        
        for service_name in self.services.keys():
            result = self.test_service(service_name)
            all_results["service_results"][service_name] = result
            
            # Update counters
            if result["overall_health"] == "UP":
                all_results["services_up"] += 1
            elif result["overall_health"] == "DOWN":
                all_results["services_down"] += 1
            else:
                all_results["services_partial"] += 1
        
        return all_results
    
    def print_summary_report(self, results: Dict):
        """Print a summary report of all health checks"""
        print("\n" + "=" * 80)
        print("📊 HEALTH CHECK SUMMARY REPORT")
        print("=" * 80)
        
        print(f"⏰ Timestamp: {results['timestamp']}")
        print(f"📋 Total Services Tested: {results['total_services']}")
        print(f"✅ Services UP: {results['services_up']}")
        print(f"⚠️ Services PARTIAL: {results['services_partial']}")
        print(f"❌ Services DOWN: {results['services_down']}")
        
        print("\n🔍 SERVICE DETAILS:")
        print("-" * 80)
        
        for service_name, result in results["service_results"].items():
            print(f"\n🏷️ {service_name.upper()} (Port {result['port']})")
            print(f"   Status: {result['status']}")
            print(f"   Health: {result['overall_health']}")
            print(f"   Endpoints: {result['endpoints_responding']}/{result['endpoints_tested']} responding")
            
            if result["avg_response_time_ms"]:
                print(f"   Avg Response Time: {result['avg_response_time_ms']}ms")
            
            if result["errors"]:
                print(f"   Errors: {', '.join(set(result['errors']))}")
        
        print("\n" + "=" * 80)
        
        # Overall system health
        if results["services_down"] == 0:
            if results["services_partial"] == 0:
                print("🎉 OVERALL SYSTEM STATUS: ALL SYSTEMS OPERATIONAL")
            else:
                print("⚠️ OVERALL SYSTEM STATUS: SOME SERVICES HAVE ISSUES")
        else:
            print("🚨 OVERALL SYSTEM STATUS: CRITICAL - SOME SERVICES ARE DOWN")
        
        print("=" * 80)
    
    def save_detailed_report(self, results: Dict, filename: str = "health_check_report.json"):
        """Save detailed results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Detailed report saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save report: {e}")

def main():
    """Main function to run health checks"""
    checker = ServiceHealthChecker()
    
    try:
        # Run all health checks
        results = checker.run_all_health_checks()
        
        # Print summary
        checker.print_summary_report(results)
        
        # Save detailed report
        checker.save_detailed_report(results)
        
        # Exit with appropriate code
        if results["services_down"] > 0:
            print(f"\n⚠️ Exiting with code 1 - {results['services_down']} service(s) are down")
            sys.exit(1)
        elif results["services_partial"] > 0:
            print(f"\n⚠️ Exiting with code 2 - {results['services_partial']} service(s) have issues")
            sys.exit(2)
        else:
            print("\n✅ All services are healthy!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⛔ Health check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error during health check: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()