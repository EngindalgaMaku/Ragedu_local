"""
FAISS vector store module.

This module provides a class to manage a FAISS index for efficient
similarity search on high-dimensional vectors.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
from .. import config
from ..utils.helpers import setup_logging

# Optional import for faiss - handle gracefully when not available (e.g., in tests)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

logger = setup_logging()

class FaissVectorStore:
    """
    A class to manage a FAISS vector store.
    
    This class handles the creation, loading, saving, and searching of a FAISS index.
    It also stores the original text chunks associated with the vectors.
    """
    def __init__(self, index_path: str = None, dimension: int = 1536):
        """
        Initializes the FaissVectorStore.

        Args:
            index_path: The path to save/load the FAISS index and chunks.
                        If None, uses the path from the config file.
            dimension: The dimensionality of the vectors (e.g., 1536 for OpenAI's ada-002).
        """
        self.index_path = index_path or config.VECTOR_STORE_PATH
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.metadata: List[Dict] = []
        self._load_store()

    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict]] = None):
        """
        Adds documents (texts and their embeddings) to the vector store.

        Args:
            texts: A list of original text chunks.
            embeddings: A list of corresponding vector embeddings.
        """
        if not texts or not embeddings or len(texts) != len(embeddings):
            logger.error("Invalid input: texts and embeddings must be non-empty and of the same length.")
            return

        # Deduplicate: skip texts already in store and duplicates within the batch
        existing = set(self.chunks) if self.chunks else set()
        seen_batch = set()
        filtered_texts: List[str] = []
        filtered_embeddings: List[List[float]] = []
        filtered_meta: List[Dict] = []
        skipped_existing = 0
        skipped_duplicates = 0
        for idx, (t, e) in enumerate(zip(texts, embeddings)):
            if t in existing:
                skipped_existing += 1
                continue
            if t in seen_batch:
                skipped_duplicates += 1
                continue
            seen_batch.add(t)
            filtered_texts.append(t)
            filtered_embeddings.append(e)
            if metadatas and idx < len(metadatas) and isinstance(metadatas[idx], dict):
                filtered_meta.append(metadatas[idx])
            else:
                filtered_meta.append({})

        if not filtered_texts:
            logger.info("No new unique documents to add (duplicates were skipped).")
            return

        embeddings_np = np.array(filtered_embeddings, dtype='float32')
        # L2-normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        embeddings_np = embeddings_np / norms
        
        if self.index is None:
            if not FAISS_AVAILABLE:
                logger.warning("FAISS not available - vector store operations disabled")
                return
            # Infer dimension from the first embedding batch if needed
            inferred_dim = embeddings_np.shape[1] if embeddings_np.ndim == 2 else len(embeddings[0])
            if inferred_dim != self.dimension:
                logger.info(f"Setting FAISS index dimension from embeddings: {inferred_dim} (was {self.dimension}).")
                self.dimension = inferred_dim
            logger.info(f"Creating a new FAISS IndexFlatIP (cosine similarity) with dimension {self.dimension}.")
            self.index = faiss.IndexFlatIP(self.dimension)
        
        self.index.add(embeddings_np)
        self.chunks.extend(filtered_texts)
        self.metadata.extend(filtered_meta)
        logger.info(
            f"Added {len(filtered_texts)} new documents to the vector store. "
            f"Skipped existing: {skipped_existing}, skipped duplicates in batch: {skipped_duplicates}. "
            f"Total documents: {self.index.ntotal}"
        )

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Performs a similarity search in the vector store.

        Args:
            query_embedding: The embedding of the query text.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A list of tuples, where each tuple contains a retrieved chunk and its
            similarity score (distance) and metadata dict. Returns an empty list if the index is empty.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search attempted on an empty or non-existent index.")
            return []

        # Accept either a single vector [d] or a list containing one vector [[d]]
        if isinstance(query_embedding, list) and query_embedding and isinstance(query_embedding[0], list):
            query_embedding_np = np.array(query_embedding, dtype='float32')
        else:
            query_embedding_np = np.array([query_embedding], dtype='float32')
        # L2-normalize query for cosine similarity
        qn = np.linalg.norm(query_embedding_np, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        query_embedding_np = query_embedding_np / qn
        if query_embedding_np.shape[1] != self.dimension:
            logger.error(f"Query embedding dimension {query_embedding_np.shape[1]} does not match index dimension {self.dimension}.")
            return []
        # With IP + normalized vectors, higher is more similar. We'll still call them 'scores'.
        distances, indices = self.index.search(query_embedding_np, k)
        
        results: List[Tuple[str, float, Dict]] = []
        # indices shape is (nq, k). We assume nq == 1 for our usage.
        for i in range(indices.shape[1]):
            idx = indices[0, i]
            if 0 <= idx < len(self.chunks):
                meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else {}
                results.append((self.chunks[idx], float(distances[0, i]), meta))
        
        logger.info(f"Search found {len(results)} results for the query.")
        return results

    def save_store(self):
        """Saves the FAISS index, text chunks and metadata to disk."""
        if self.index is None:
            logger.warning("Attempted to save an empty index. Nothing will be saved.")
            return

        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - cannot save index")
            return

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        index_file = f"{self.index_path}.index"
        chunks_file = f"{self.index_path}.chunks"
        try:
            faiss.write_index(self.index, index_file)
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for chunk in self.chunks:
                    f.write(chunk.replace('\n', '\\n') + '\n')
            # Save metadata JSONL aligned with chunks
            meta_file = f"{self.index_path}.meta.jsonl"
            with open(meta_file, 'w', encoding='utf-8') as mf:
                for i in range(len(self.chunks)):
                    md = self.metadata[i] if i < len(self.metadata) else {}
                    mf.write(json.dumps(md, ensure_ascii=False) + '\n')
            logger.info(f"Successfully saved FAISS index to {index_file} and chunks to {chunks_file} and metadata to {meta_file}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def _load_store(self):
        """Loads the FAISS index and text chunks from disk if they exist."""
        index_file = f"{self.index_path}.index"
        chunks_file = f"{self.index_path}.chunks"
        meta_file = f"{self.index_path}.meta.jsonl"

        if os.path.exists(index_file) and os.path.exists(chunks_file):
            try:
                if not FAISS_AVAILABLE:
                    logger.warning("FAISS not available - cannot load index, only loading chunks")
                    self.index = None
                else:
                    self.index = faiss.read_index(index_file)
                    # Sync dimension with the loaded index
                    try:
                        loaded_dim = int(self.index.d)
                        if loaded_dim != self.dimension:
                            logger.info(f"Loaded FAISS index dimension {loaded_dim} (was configured {self.dimension}). Syncing dimension.")
                            self.dimension = loaded_dim
                    except Exception:
                        # Best effort: if .d not available, keep current dimension
                        pass
                    # Warn if the loaded index is not IP (cosine). Users may need to reset index.
                    try:
                        if not isinstance(self.index, faiss.IndexFlatIP):
                            logger.warning("Loaded index is not IndexFlatIP (cosine). Consider resetting the index to use cosine similarity.")
                    except Exception:
                        pass
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self.chunks = [line.strip().replace('\\n', '\n') for line in f]
                # Load metadata if present
                self.metadata = []
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as mf:
                            for line in mf:
                                line = line.strip()
                                if not line:
                                    self.metadata.append({})
                                else:
                                    try:
                                        self.metadata.append(json.loads(line))
                                    except Exception:
                                        self.metadata.append({})
                    except Exception as me:
                        logger.warning(f"Failed to load metadata: {me}. Initializing empty metadata.")
                        self.metadata = [{} for _ in self.chunks]
                else:
                    # Initialize empty metadata aligned with chunks
                    self.metadata = [{} for _ in self.chunks]
                logger.info(f"Successfully loaded FAISS index and {len(self.chunks)} chunks from {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to load vector store from {self.index_path}: {e}")
                self.index = None
                self.chunks = []
                self.metadata = []
        else:
            logger.info("No existing vector store found. A new one will be created upon adding documents.")

if __name__ == '__main__':
    # This is for testing purposes.
    print("--- Testing FaissVectorStore ---")
    
    # 1. Initialize the store
    vector_store = FaissVectorStore(index_path="data/test_db/faiss_test")

    # 2. Prepare dummy data
    sample_texts = ["The sky is blue.", "The sun is bright.", "The moon is white."]
    # Dummy embeddings (in a real scenario, these would come from OpenAI)
    sample_embeddings = np.random.rand(3, 1536).astype('float32').tolist()

    # 3. Add documents
    print("\nAdding documents to the store...")
    vector_store.add_documents(sample_texts, sample_embeddings)
    
    # 4. Save the store
    print("\nSaving the store...")
    vector_store.save_store()

    # 5. Create a new instance and load the store
    print("\nLoading the store into a new instance...")
    loaded_vector_store = FaissVectorStore(index_path="data/test_db/faiss_test")
    if loaded_vector_store.index:
        print(f"Store loaded successfully with {loaded_vector_store.index.ntotal} documents.")
    else:
        print("Failed to load the store.")

    # 6. Perform a search
    if loaded_vector_store.index:
        print("\nPerforming a similarity search...")
        query_embedding = np.random.rand(1536).astype('float32').tolist()
        search_results = loaded_vector_store.search(query_embedding, k=2)
        
        if search_results:
            print("Search results:")
            for text, score in search_results:
                print(f"  - Text: '{text}', Score: {score:.4f}")
        else:
            print("Search returned no results.")
            
    # Clean up test files
    if os.path.exists("data/test_db"):
        import shutil
        shutil.rmtree("data/test_db")
        print("\nCleaned up test directory.")
