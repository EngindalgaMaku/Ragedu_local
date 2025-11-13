#!/usr/bin/env python3
"""
Test script to check communication between the API Gateway and all underlying microservices.
- PDF Processing Service
- Document Processing Service
- Model Inference Service
"""

import requests
import json
import os
import uuid
from pathlib import Path

# --- Configuration ---
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8080")

# It's better to get the service URLs from the gateway's health endpoint
# but we can define them here as a fallback or for direct testing.
PDF_PROCESSOR_URL = os.getenv('PDF_PROCESSOR_URL', 'https://pdf-processor-awe3elsvra-ew.a.run.app')
DOCUMENT_PROCESSOR_URL = os.getenv('DOCUMENT_PROCESSOR_URL', 'https://doc-proc-service-awe3elsvra-ew.a.run.app')
MODEL_INFERENCE_URL = os.getenv('MODEL_INFERENCE_URL', 'https://model-inferencer-awe3elsvra-ew.a.run.app')


def print_test_header(title):
    """Prints a formatted header for a test section."""
    print("\n" + "=" * 80)
    print(f"=== {title.upper()} ===")
    print("=" * 80)

def test_gateway_health():
    """Tests the basic health of the API Gateway."""
    print_test_header("API Gateway Health Check")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/health", timeout=10)
        if response.status_code == 200 and response.json().get("status") == "ok":
            print("✅ API Gateway is healthy and responsive.")
            return True
        else:
            print(f"❌ API Gateway health check failed. Status: {response.status_code}, Response: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"❌ Failed to connect to API Gateway at {API_GATEWAY_URL}: {e}")
        return False

def test_all_services_health_via_gateway():
    """Tests the '/health/services' endpoint to check all microservices."""
    print_test_header("Microservice Health Check via Gateway")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/health/services", timeout=60) # Increased timeout
        if response.status_code != 200:
            print(f"❌ Gateway's service health check endpoint failed. Status: {response.status_code}, Response: {response.text}")
            return False

        data = response.json()
        all_ok = True
        services = data.get("services", {})
        
        if not services:
            print("❌ No services reported in the health check response.")
            return False

        print("Service Statuses:")
        for service_name, details in services.items():
            status = details.get("status")
            url = details.get("url")
            if status == "ok":
                print(f"  ✅ {service_name:<20} | Status: OK | URL: {url}")
            else:
                all_ok = False
                error = details.get("error", "No error details provided.")
                print(f"  ❌ {service_name:<20} | Status: ERROR | URL: {url} | Error: {error}")
        
        if all_ok:
            print("\n✅ All microservices are reported as healthy by the gateway.")
        else:
            print("\n❌ One or more microservices are unhealthy.")
        
        return all_ok

    except requests.RequestException as e:
        print(f"❌ Failed to request service health from gateway: {e}")
        return False

def test_pdf_processing_flow():
    """Tests the PDF processing flow via the gateway."""
    print_test_header("PDF Processing Service Communication Test")
    # Create a dummy PDF file for testing
    from fpdf import FPDF
    pdf_path = Path("test_dummy_document.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="This is a test PDF for the gateway communication test.", ln=True, align='C')
    pdf.output(str(pdf_path))
    print(f"Created dummy PDF: {pdf_path}")

    try:
        with open(pdf_path, "rb") as f:
            files = {'file': (pdf_path.name, f, 'application/pdf')}
            response = requests.post(
                f"{API_GATEWAY_URL}/documents/convert-document-to-markdown",
                files=files,
                timeout=300 # Increased timeout for cold starts
            )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ PDF processing request successful.")
                print(f"  Message: {result.get('message')}")
                print(f"  Markdown Filename: {result.get('markdown_filename')}")
                return True
            else:
                print(f"❌ PDF processing failed. Message: {result.get('message')}")
                return False
        else:
            print(f"❌ PDF processing request failed. Status: {response.status_code}, Response: {response.text}")
            return False

    except requests.RequestException as e:
        print(f"❌ Failed to send request to PDF processing endpoint: {e}")
        return False
    finally:
        # Clean up the dummy file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"Cleaned up dummy PDF: {pdf_path}")


def test_model_inference_flow():
    """Tests the model inference service communication via the gateway."""
    print_test_header("Model Inference Service Communication Test")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/models", timeout=60) # Increased timeout
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print("✅ Successfully retrieved models from the model inference service.")
                print(f"  Found {len(models)} models. First few: {models[:5]}")
                return True
            else:
                print("❌ Model inference service returned an empty list of models.")
                return False
        else:
            print(f"❌ Failed to get models. Status: {response.status_code}, Response: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"❌ Failed to send request to model inference endpoint: {e}")
        return False

def main():
    """Runs all communication tests."""
    print_test_header("API Gateway Full Communication Test Suite")
    
    results = {
        "gateway_health": test_gateway_health(),
        "services_health": test_all_services_health_via_gateway(),
        "pdf_processing": test_pdf_processing_flow(),
        "model_inference": test_model_inference_flow(),
    }

    print_test_header("Test Summary")
    
    final_status = True
    for test_name, success in results.items():
        if success:
            print(f"  ✅ {test_name:<20} | PASSED")
        else:
            final_status = False
            print(f"  ❌ {test_name:<20} | FAILED")
            
    print("\n" + "=" * 80)
    if final_status:
        print("✅✅✅ All communication tests passed! ✅✅✅")
    else:
        print("❌❌❌ Some communication tests failed. Please review the logs. ❌❌❌")
    print("=" * 80)

if __name__ == "__main__":
    # We need fpdf to create a dummy pdf for the test
    try:
        import fpdf
    except ImportError:
        print("PyFPDF not found. Installing...")
        os.system("pip install fpdf")
    main()