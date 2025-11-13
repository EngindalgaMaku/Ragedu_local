#!/usr/bin/env python3
"""
Test script to verify PDF marker and sentence transformers functionality
"""

import sys
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_marker_availability():
    """Test if marker-pdf is available and can be imported"""
    logger.info("🔍 Testing marker-pdf availability...")
    
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        from marker.config.parser import ConfigParser
        logger.info("✅ marker-pdf imports successful!")
        return True
    except ImportError as e:
        logger.error(f"❌ marker-pdf import failed: {e}")
        return False

def test_sentence_transformers_availability():
    """Test if sentence-transformers is available and can be imported"""
    logger.info("🔍 Testing sentence-transformers availability...")
    
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        logger.info("✅ sentence-transformers imports successful!")
        return True
    except ImportError as e:
        logger.error(f"❌ sentence-transformers import failed: {e}")
        return False

def test_enhanced_pdf_processor():
    """Test if enhanced PDF processor can be imported and initialized"""
    logger.info("🔍 Testing enhanced PDF processor...")
    
    try:
        from src.document_processing.enhanced_pdf_processor import MarkerPDFProcessor, enhanced_pdf_processor
        logger.info("✅ Enhanced PDF processor imports successful!")
        
        # Test initialization
        stats = enhanced_pdf_processor.get_processing_stats()
        logger.info(f"📊 Processing stats: {stats}")
        return True
    except ImportError as e:
        logger.error(f"❌ Enhanced PDF processor import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Enhanced PDF processor initialization failed: {e}")
        return False

def test_embedding_generator():
    """Test if embedding generator with sentence transformers works"""
    logger.info("🔍 Testing embedding generator with sentence transformers...")
    
    try:
        from src.embedding.embedding_generator import generate_embeddings
        
        # Test simple embedding generation
        test_texts = ["This is a test sentence.", "Another test sentence for embedding."]
        embeddings = generate_embeddings(test_texts, provider='sentence_transformers')
        
        if embeddings and len(embeddings) == 2:
            logger.info(f"✅ Sentence transformers embeddings working! Generated {len(embeddings)} embeddings")
            logger.info(f"📊 Embedding dimension: {len(embeddings[0]) if embeddings[0] else 'N/A'}")
            return True
        else:
            logger.error("❌ Embedding generation failed or returned empty results")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Embedding generator import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        return False

def test_pdf_file_processing():
    """Test actual PDF file processing if a test PDF is available"""
    logger.info("🔍 Testing PDF file processing...")
    
    # Look for test PDFs in data/uploads
    test_pdf_paths = []
    if os.path.exists("data/uploads"):
        for filename in os.listdir("data/uploads"):
            if filename.endswith(".pdf"):
                test_pdf_paths.append(os.path.join("data/uploads", filename))
                break  # Just use the first PDF found
    
    if not test_pdf_paths:
        logger.warning("⚠️ No test PDF files found in data/uploads - skipping file processing test")
        return True
    
    test_pdf = test_pdf_paths[0]
    logger.info(f"📄 Testing with PDF: {os.path.basename(test_pdf)}")
    
    try:
        from src.document_processing.enhanced_pdf_processor import extract_text_from_pdf_enhanced
        
        # Test extraction with minimal timeout for quick test
        text = extract_text_from_pdf_enhanced(test_pdf, prefer_marker=True)
        
        if text and len(text) > 0:
            logger.info(f"✅ PDF processing successful! Extracted {len(text)} characters")
            logger.info(f"📄 Preview (first 100 chars): {text[:100]}...")
            return True
        else:
            logger.error("❌ PDF processing returned empty text")
            return False
            
    except Exception as e:
        logger.error(f"❌ PDF processing failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting PDF marker and sentence transformers diagnostic tests...\n")
    
    tests = [
        ("Marker PDF Import", test_marker_availability),
        ("Sentence Transformers Import", test_sentence_transformers_availability),
        ("Enhanced PDF Processor", test_enhanced_pdf_processor),
        ("Embedding Generator", test_embedding_generator),
        ("PDF File Processing", test_pdf_file_processing)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST RESULTS SUMMARY:")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! PDF marker and sentence transformers are working correctly.")
        return 0
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Dependencies may need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())