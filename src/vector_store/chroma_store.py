"""
ChromaDB vector store module.

This module provides a class to manage a ChromaDB collection for efficient
similarity search on high-dimensional vectors via HTTP API.
"""

import os
import requests
import uuid
from typing import List, Tuple, Optional, Dict, Any
import json
from .. import config
from ..utils.helpers import setup_logging

logger = setup_logging()

class ChromaVectorStore:
    """
    A class to manage a ChromaDB vector store via HTTP API.
    
    This class handles the creation, loading, and searching of a ChromaDB collection.
    It provides the same interface as FaissVectorStore for seamless replacement.
    """
    
    def __init__(self, collection_name: str = None, chroma_host: str = None, chroma_port: int = None):
        """
        Initializes the ChromaVectorStore.

        Args:
            collection_name: The name of the ChromaDB collection.
                           If None, uses 'rag_documents' as default.
            chroma_host: ChromaDB server host. If None, reads from CHROMADB_URL env var
                        or defaults to http://localhost
            chroma_port: ChromaDB server port. If None, uses 8004 for localhost
                        or reads port from CHROMADB_URL
        """
        import os
        
        self.collection_name = collection_name or "rag_documents"
        
        # Get ChromaDB URL from environment variable or use defaults
        # Google Cloud Run compatible
        chromadb_url = os.getenv('CHROMADB_URL') or os.getenv('CHROMA_SERVICE_URL')
        
        if chromadb_url:
            # Use environment variable (for Cloud Run or Docker)
            # Remove trailing slash and ensure proper format
            self.chroma_url = chromadb_url.rstrip('/')
            # Ensure it has http:// or https:// prefix
            if not self.chroma_url.startswith('http://') and not self.chroma_url.startswith('https://'):
                self.chroma_url = f"http://{self.chroma_url}"
            self.api_url = f"{self.chroma_url}/api/v1"
        else:
            # Use provided parameters or defaults (for local development)
            host = chroma_host or "http://localhost"
            port = chroma_port or 8004
            # Ensure host has protocol
            if not host.startswith('http://') and not host.startswith('https://'):
                host = f"http://{host}"
            self.chroma_url = f"{host}:{port}"
            self.api_url = f"{self.chroma_url}/api/v1"
        
        # Store document counts and metadata for compatibility
        self.chunks = []
        self.metadata = []
        self._document_count = 0
        
        # Test connection and initialize collection
        self._test_connection()
        self._ensure_collection_exists()
        self._load_existing_data()

    def _test_connection(self):
        """Test connection to ChromaDB server."""
        try:
            response = requests.get(f"{self.api_url}/heartbeat", timeout=10)
            if response.status_code == 200:
                logger.info(f"Successfully connected to ChromaDB at {self.chroma_url}")
            else:
                logger.error(f"ChromaDB connection failed: {response.status_code}")
                raise ConnectionError(f"ChromaDB server responded with {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect to ChromaDB at {self.chroma_url}: {e}")
            raise ConnectionError(f"ChromaDB connection failed: {e}")

    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't."""
        try:
            # Check if collection exists
            response = requests.get(f"{self.api_url}/collections/{self.collection_name}")
            
            if response.status_code == 200:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            elif response.status_code == 404:
                # Create collection with cosine distance function
                create_data = {
                    "name": self.collection_name,
                    "metadata": {"description": "RAG system document embeddings", "hnsw:space": "cosine"}
                }
                
                response = requests.post(
                    f"{self.api_url}/collections",
                    json=create_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"Created new collection '{self.collection_name}'")
                else:
                    logger.error(f"Failed to create collection: {response.status_code} - {response.text}")
                    raise Exception(f"Could not create ChromaDB collection: {response.text}")
            else:
                logger.error(f"Unexpected response checking collection: {response.status_code}")
                raise Exception(f"ChromaDB collection check failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise Exception(f"ChromaDB collection setup failed: {e}")

    def _load_existing_data(self):
        """Load existing documents from ChromaDB collection to sync local state."""
        try:
            response = requests.post(
                f"{self.api_url}/collections/{self.collection_name}/get",
                json={"limit": 10000},  # Get all documents (adjust limit if needed)
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])
                metadatas = data.get("metadatas", [])
                
                self.chunks = documents
                self.metadata = metadatas
                self._document_count = len(documents)
                
                logger.info(f"Loaded {self._document_count} existing documents from ChromaDB collection")
            else:
                logger.warning(f"Could not load existing data: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to load existing ChromaDB data: {e}")
            self.chunks = []
            self.metadata = []
            self._document_count = 0

    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict]] = None):
        """
        Adds documents (texts and their embeddings) to the ChromaDB collection.

        Args:
            texts: A list of original text chunks.
            embeddings: A list of corresponding vector embeddings.
            metadatas: Optional list of metadata dictionaries.
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
        filtered_ids: List[str] = []
        
        skipped_existing = 0
        skipped_duplicates = 0
        
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            if text in existing:
                skipped_existing += 1
                continue
            if text in seen_batch:
                skipped_duplicates += 1
                continue
                
            seen_batch.add(text)
            filtered_texts.append(text)
            filtered_embeddings.append(embedding)
            filtered_ids.append(str(uuid.uuid4()))
            
            if metadatas and idx < len(metadatas) and isinstance(metadatas[idx], dict):
                filtered_meta.append(metadatas[idx])
            else:
                filtered_meta.append({})

        if not filtered_texts:
            logger.info("No new unique documents to add (duplicates were skipped).")
            return

        try:
            # Add documents to ChromaDB
            add_data = {
                "ids": filtered_ids,
                "embeddings": filtered_embeddings,
                "documents": filtered_texts,
                "metadatas": filtered_meta
            }
            
            response = requests.post(
                f"{self.api_url}/collections/{self.collection_name}/add",
                json=add_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                # Update local state
                self.chunks.extend(filtered_texts)
                self.metadata.extend(filtered_meta)
                self._document_count += len(filtered_texts)
                
                logger.info(
                    f"Added {len(filtered_texts)} new documents to ChromaDB. "
                    f"Skipped existing: {skipped_existing}, skipped duplicates in batch: {skipped_duplicates}. "
                    f"Total documents: {self._document_count}"
                )
            else:
                logger.error(f"Failed to add documents to ChromaDB: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Performs a similarity search in the ChromaDB collection.

        Args:
            query_embedding: The embedding of the query text.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A list of tuples, where each tuple contains a retrieved chunk, 
            its similarity score, and metadata dict. Returns an empty list if no results.
        """
        if self._document_count == 0:
            logger.warning("Search attempted on an empty ChromaDB collection.")
            return []

        try:
            # Handle both single vector and list of vectors
            if isinstance(query_embedding[0], list):
                query_embeddings = query_embedding
            else:
                query_embeddings = [query_embedding]

            search_data = {
                "query_embeddings": query_embeddings,
                "n_results": k
            }
            
            response = requests.post(
                f"{self.api_url}/collections/{self.collection_name}/query",
                json=search_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # ChromaDB returns nested lists for batch queries
                documents = data.get("documents", [[]])[0]  # Take first query results
                distances = data.get("distances", [[]])[0]
                metadatas = data.get("metadatas", [[]])[0]
                
                results: List[Tuple[str, float, Dict]] = []
                
                for doc, distance, metadata in zip(documents, distances, metadatas):
                    # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
                    # Convert to similarity score (1 = identical, 0 = orthogonal)
                    # Use formula: 1 - distance for cosine distance
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    # Ensure metadata is a dict
                    meta_dict = metadata if isinstance(metadata, dict) else {}
                    
                    results.append((doc, float(similarity_score), meta_dict))
                
                logger.info(f"ChromaDB search found {len(results)} results for the query.")
                return results
                
            else:
                logger.error(f"ChromaDB search failed: {response.status_code} - {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during ChromaDB search: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during ChromaDB search: {e}")
            return []

    def save_store(self):
        """
        ChromaDB automatically persists data, so this is a no-op.
        Kept for compatibility with FaissVectorStore interface.
        """
        logger.info("ChromaDB automatically persists data. No manual save needed.")

    def _load_store(self):
        """
        ChromaDB loads data automatically on connection.
        Kept for compatibility with FaissVectorStore interface.
        """
        logger.info("ChromaDB loads data automatically. No manual load needed.")

    @property
    def ntotal(self) -> int:
        """
        Return total number of documents in the collection.
        For compatibility with FaissVectorStore interface.
        """
        return self._document_count

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the ChromaDB collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            response = requests.get(f"{self.api_url}/collections/{self.collection_name}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Could not get collection info: {response.status_code}")
                return {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_collection(self):
        """Delete the entire collection. Use with caution!"""
        try:
            response = requests.delete(f"{self.api_url}/collections/{self.collection_name}")
            
            if response.status_code == 200:
                logger.info(f"Collection '{self.collection_name}' deleted successfully")
                self.chunks = []
                self.metadata = []
                self._document_count = 0
            else:
                logger.error(f"Could not delete collection: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error deleting collection: {e}")

    def reset_collection(self):
        """Reset collection by deleting and recreating it."""
        logger.info(f"Resetting collection '{self.collection_name}'")
        self.delete_collection()
        self._ensure_collection_exists()


if __name__ == '__main__':
    # This is for testing purposes.
    import numpy as np
    
    print("--- Testing ChromaVectorStore ---")
    
    try:
        # 1. Initialize the store
        vector_store = ChromaVectorStore(collection_name="test_collection")
        
        # 2. Prepare dummy data
        sample_texts = ["The sky is blue.", "The sun is bright.", "The moon is white."]
        # Dummy embeddings (in a real scenario, these would come from embedding model)
        sample_embeddings = np.random.rand(3, 384).astype('float32').tolist()  # ChromaDB default dim
        sample_metadata = [
            {"source": "test1.txt", "page": 1},
            {"source": "test2.txt", "page": 1}, 
            {"source": "test3.txt", "page": 1}
        ]
        
        # 3. Add documents
        print("\nAdding documents to ChromaDB...")
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadata)
        
        # 4. Perform a search
        print(f"\nTotal documents in collection: {vector_store.ntotal}")
        
        if vector_store.ntotal > 0:
            print("\nPerforming a similarity search...")
            query_embedding = np.random.rand(384).astype('float32').tolist()
            search_results = vector_store.search(query_embedding, k=2)
            
            if search_results:
                print("Search results:")
                for text, score, metadata in search_results:
                    print(f"  - Text: '{text[:50]}...', Score: {score:.4f}, Metadata: {metadata}")
            else:
                print("Search returned no results.")
        
        # 5. Get collection info
        print(f"\nCollection info: {vector_store.get_collection_info()}")
        
        # 6. Clean up (optional)
        # vector_store.delete_collection()
        # print("\nTest collection cleaned up.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")