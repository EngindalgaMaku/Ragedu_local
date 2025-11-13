import pytest
import requests

# Define the API Gateway URL
API_GATEWAY_URL = "https://api-gateway-1051060211087.europe-west1.run.app"

def test_health_check():
    """
    Tests the /health endpoint to ensure the API Gateway is running.
    """
    response = requests.get(f"{API_GATEWAY_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_session():
    """
    Tests the /sessions endpoint for creating a new session.
    """
    response = requests.post(f"{API_GATEWAY_URL}/sessions")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Session creation placeholder"

def test_get_sessions():
    """
    Tests the /sessions endpoint for retrieving all sessions.
    """
    response = requests.get(f"{API_GATEWAY_URL}/sessions")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Get sessions placeholder"

def test_rag_query_routing():
    """
    Tests if the API Gateway attempts to route requests to the RAG service.
    Since the downstream service is not deployed, we expect a service unavailable error (e.g., 503),
    but not a 404 or 422, which would indicate a gateway-level routing issue.
    """
    payload = {"session_id": "test-session", "query": "What is the capital of Turkey?"}
    response = requests.post(f"{API_GATEWAY_URL}/rag/query", json=payload)
    
    # Assert that the gateway did not return a 404 or 422 error.
    assert response.status_code not in [404, 422]

def test_document_upload_routing():
    """
    Tests if the API Gateway attempts to route file uploads to the document processing service.
    Similar to the RAG test, we expect a 5xx error, not a 404.
    """
    # Use an in-memory file
    files = {"file": ("test_document.txt", "this is a test document", "text/plain")}
    response = requests.post(f"{API_GATEWAY_URL}/documents/upload", files=files)

    # Assert that the gateway returns a successful response
    assert response.status_code == 200