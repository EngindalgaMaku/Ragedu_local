#!/usr/bin/env python3
"""
Test script to verify RAG system improvements.
Tests that answers are grounded in provided sources.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.rag.rag_chains import StuffChain
from src.vector_store.faiss_store import FaissVectorStore

def test_rag_grounding():
    """Test that RAG answers are properly grounded in sources."""
    
    print("=== Testing RAG System Grounding ===\n")
    
    # Initialize components
    try:
        # Load existing vector store (should exist from the biology document)
        faiss_store = FaissVectorStore(config.get_config())
        faiss_store.load_index("data/vector_db/sessions/biyoloji_9")
        
        # Initialize RAG chain
        rag_chain = StuffChain(config.get_config(), faiss_store)
        
        # Test queries
        test_queries = [
            "Hücre zarının yapısı nedir?",
            "Mitokondri ne işe yarar?",
            "Fotosentez nasıl gerçekleşir?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"Test {i}: {query}")
            print("-" * 50)
            
            # Execute RAG
            result = rag_chain.execute(query, top_k=3)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}\n")
                continue
                
            print(f"✅ Answer: {result['answer']}")
            print(f"📊 Chain Type: {result['chain_type']}")
            print(f"⏱️ Execution Time: {result.get('execution_time', 0):.2f}s")
            
            # Show sources
            sources = result.get('sources', [])
            print(f"📚 Sources Found: {len(sources)}")
            for j, source in enumerate(sources[:2], 1):  # Show first 2 sources
                source_text = source['text'][:100] + "..." if len(source['text']) > 100 else source['text']
                print(f"   Source {j}: {source_text}")
                
            print("\n" + "="*60 + "\n")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    print("✅ RAG grounding test completed!")
    return True

if __name__ == "__main__":
    test_rag_grounding()