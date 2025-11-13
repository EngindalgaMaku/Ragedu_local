#!/usr/bin/env python3
"""
Create a simple test PDF file for testing PDF processing microservice integration.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path
import os

def create_test_pdf():
    """Create a simple test PDF with some content"""
    
    # Ensure test_documents directory exists
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    # Create the PDF file
    pdf_path = test_dir / "test_document.pdf"
    
    # Create document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Test Document for PDF Processing", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Content paragraphs
    content = [
        "This is a test PDF document created for testing the PDF Processing Service microservice integration.",
        "The document contains multiple paragraphs to test text extraction and markdown conversion.",
        "PDF Processing Service should be able to extract this text and convert it to markdown format.",
        "This test helps verify that the main API correctly delegates PDF processing to the microservice.",
        "The integration should work seamlessly between the main application and the PDF processor."
    ]
    
    for paragraph in content:
        p = Paragraph(paragraph, styles['Normal'])
        story.append(p)
        story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    
    print(f"✅ Test PDF created successfully: {pdf_path}")
    print(f"📄 File size: {pdf_path.stat().st_size} bytes")
    
    return pdf_path

if __name__ == "__main__":
    create_test_pdf()