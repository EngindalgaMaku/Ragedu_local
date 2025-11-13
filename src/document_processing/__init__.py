# This file makes the 'document_processing' directory a Python package.

from .pdf_processor import process_pdf
from .docx_processor import process_docx
from .pptx_processor import process_pptx
from .document_processor import process_document

__all__ = [
    "process_pdf",
    "process_docx",
    "process_pptx",
    "process_document"
]