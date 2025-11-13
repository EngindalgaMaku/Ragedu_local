"""
PDF document processing module.

This module provides functions to extract text from PDF files using PyPDF2.
"""

import PyPDF2
from ..utils.helpers import setup_logging

logger = setup_logging()

def process_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_path: The path to the PDF file.

    Returns:
        The extracted text as a single string, or an empty string if an error occurs.
    """
    logger.info(f"Processing PDF file: {file_path}")
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Inject a page marker to preserve structure in downstream chunking/UI
                    text += f"\n\n=== Page {i+1} ===\n\n" + page_text + "\n"
        logger.info(f"Successfully extracted text from {file_path}")
        return text.strip()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"An error occurred while processing {file_path}: {e}")
        return ""

if __name__ == '__main__':
    # This is for testing purposes.
    # Create a dummy PDF file to test the processor.
    # Note: This requires a sample PDF file to be present at the specified path.
    # You can create a dummy file or use an existing one.
    
    # Create a dummy data directory and a dummy file for testing
    import os
    if not os.path.exists('data/raw/sample_documents'):
        os.makedirs('data/raw/sample_documents')
    
    # The following lines are commented out because we can't create a PDF file with text directly.
    # To test this, please place a sample PDF file named 'sample.pdf' in 'data/raw/sample_documents/'
    
    # from fpdf import FPDF
    # pdf = FPDF()
    # pdf.add_page()
    # pdf.set_font("Arial", size = 12)
    # pdf.cell(200, 10, txt = "This is a test PDF document for the RAG system.", ln = True, align = 'C')
    # pdf.output("data/raw/sample_documents/sample.pdf")

    test_pdf_path = "data/raw/sample_documents/sample.pdf"
    if os.path.exists(test_pdf_path):
        extracted_text = process_pdf(test_pdf_path)
        if extracted_text:
            print("--- Extracted Text from PDF ---")
            print(extracted_text)
            print("-----------------------------")
        else:
            print("Could not extract text from the PDF.")
    else:
        print(f"Test file not found: {test_pdf_path}. Please create it to run the test.")
