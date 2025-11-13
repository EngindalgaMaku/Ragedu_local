import requests
import json

BASE_URL = "http://127.0.0.1:8001"
ENDPOINTS = ["/health", "/health/ready", "/health/live", "/status"]

def test_endpoints():
    """Tests all health check endpoints of the local PDF processing service."""
    print("--- Starting Local PDF Service Health Check ---")
    all_tests_passed = True

    for endpoint in ENDPOINTS:
        try:
            url = BASE_URL + endpoint
            print(f"\nTesting endpoint: {url}")
            response = requests.get(url, timeout=10)

            if response.status_code == 200 or response.status_code == 503: # 503 is acceptable for readiness check initially
                print(f"✅ Status Code: {response.status_code}")
                try:
                    data = response.json()
                    print("Response JSON:")
                    print(json.dumps(data, indent=2))
                    if response.status_code != 200:
                        print("⚠️  Received non-200 status, but this might be expected if models are still loading.")
                except json.JSONDecodeError:
                    print("❌ Failed to decode JSON response.")
                    all_tests_passed = False
            else:
                print(f"❌ FAILED: Endpoint {endpoint} returned status code {response.status_code}")
                print(f"Response text: {response.text}")
                all_tests_passed = False

        except requests.exceptions.RequestException as e:
            print(f"❌ FAILED: Could not connect to {url}. Error: {e}")
            all_tests_passed = False

    print("\n--- Health Check Complete ---")
    if all_tests_passed:
        print("✅ All tested endpoints responded.")
    else:
        print("❌ Some health checks failed.")

if __name__ == "__main__":
    test_endpoints()