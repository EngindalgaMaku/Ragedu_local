"""
Unified document processing module.

This module provides a single function to process different document types
by dispatching to the appropriate specialized processor.
"""

import os
from .pdf_processor import process_pdf
from .docx_processor import process_docx
from .pptx_processor import process_pptx
from ..utils.helpers import setup_logging

# Enhanced PDF processing with Marker
try:
    from .enhanced_pdf_processor import extract_text_from_pdf_enhanced, MARKER_AVAILABLE
    PDF_ENHANCED_AVAILABLE = True
except ImportError:
    PDF_ENHANCED_AVAILABLE = False
    extract_text_from_pdf_enhanced = None

logger = setup_logging()

def process_document(file_path: str) -> str:
    """
    Processes a document file and extracts its text content.

    The function determines the file type based on its extension and calls
    the corresponding processor.

    Args:
        file_path: The path to the document file.

    Returns:
        The extracted text as a single string, or an empty string if the
        file type is not supported or an error occurs.
    """
    logger.info(f"Starting document processing for: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return ""

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".pdf":
        # Enhanced PDF processing with Marker (fallback to basic if not available)
        if PDF_ENHANCED_AVAILABLE:
            try:
                logger.info("Using enhanced PDF processing with Marker")
                return extract_text_from_pdf_enhanced(file_path, prefer_marker=True)
            except Exception as e:
                logger.warning(f"Enhanced PDF processing failed, using fallback: {e}")
                return process_pdf(file_path)
        else:
            logger.info("Using basic PDF processing")
            return process_pdf(file_path)
    elif file_extension == ".docx":
        return process_docx(file_path)
    elif file_extension == ".pptx":
        return process_pptx(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_extension}. Cannot process {file_path}")
        return ""

if __name__ == '__main__':
    # This is for testing purposes.
    # It uses the dummy files created by the individual processors.
    
    # Test with a PDF
    pdf_path = "data/raw/sample_documents/sample.pdf"
    if os.path.exists(pdf_path):
        print(f"\n--- Processing PDF: {pdf_path} ---")
        pdf_text = process_document(pdf_path)
        if pdf_text:
            print("Extraction successful.")
            # print(pdf_text)
        else:
            print("Extraction failed.")
    else:
        print(f"\nPDF test file not found at {pdf_path}. Skipping test.")

    # Test with a DOCX
    docx_path = "data/raw/sample_documents/sample.docx"
    if os.path.exists(docx_path):
        print(f"\n--- Processing DOCX: {docx_path} ---")
        docx_text = process_document(docx_path)
        if docx_text:
            print("Extraction successful.")
            # print(docx_text)
        else:
            print("Extraction failed.")
    else:
        print(f"\nDOCX test file not found at {docx_path}. Skipping test.")

    # Test with a PPTX
    pptx_path = "data/raw/sample_documents/sample.pptx"
    if os.path.exists(pptx_path):
        print(f"\n--- Processing PPTX: {pptx_path} ---")
        pptx_text = process_document(pptx_path)
        if pptx_text:
            print("Extraction successful.")
            # print(pptx_text)
        else:
            print("Extraction failed.")
    else:
        print(f"\nPPTX test file not found at {pptx_path}. Skipping test.")

    # Test with an unsupported file type
    unsupported_path = "data/raw/sample_documents/unsupported.txt"
    print(f"\n--- Processing Unsupported File: {unsupported_path} ---")
    # Create a dummy unsupported file
    with open(unsupported_path, "w") as f:
        f.write("This is a test.")
    unsupported_text = process_document(unsupported_path)
    if not unsupported_text:
        print("Correctly handled unsupported file type.")
    else:
        print("Incorrectly processed an unsupported file type.")
    os.remove(unsupported_path)
