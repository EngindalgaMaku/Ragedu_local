import pytest
import requests
import os
from fastapi.testclient import TestClient
from services.docstrange_service.main import app

client = TestClient(app)

# Set a dummy API key for testing
os.environ['DOCSTRANGE_API_KEY'] = 'dummy_api_key'

@pytest.fixture
def test_pdf():
    # Create a dummy PDF file for testing
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="This is a test PDF.", ln=1, align="C")
    pdf_content = pdf.output(dest='S').encode('latin-1')
    return ("test.pdf", pdf_content, "application/pdf")

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_convert_pdf_to_markdown_success(test_pdf, monkeypatch):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data
        
        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.exceptions.HTTPError()

    def mock_post(*args, **kwargs):
        return MockResponse({"result": [{"markdown": "# Test PDF"}]}, 200)

    monkeypatch.setattr(requests, "post", mock_post)
    
    files = {'file': test_pdf}
    response = client.post("/convert/pdf-to-markdown", files=files)
    
    assert response.status_code == 200
    assert response.json() == {"result": [{"markdown": "# Test PDF"}]}

def test_convert_pdf_to_markdown_invalid_file_type():
    files = {'file': ('test.txt', b'this is not a pdf', 'text/plain')}
    response = client.post("/convert/pdf-to-markdown", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_convert_pdf_to_markdown_api_error(test_pdf, monkeypatch):
    def mock_post_error(*args, **kwargs):
        raise requests.exceptions.RequestException("API is down")

    monkeypatch.setattr(requests, "post", mock_post_error)
    
    files = {'file': test_pdf}
    response = client.post("/convert/pdf-to-markdown", files=files)
    
    assert response.status_code == 500
    assert "Error communicating with DocStrange API" in response.json()["detail"]
