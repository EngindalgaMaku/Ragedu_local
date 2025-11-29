import json
from typing import Dict, Any, List
import hashlib
import time
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Optional import for ollama - handle gracefully when not available (e.g., in tests)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False
from src.vector_store.chroma_store import ChromaVectorStore
from src.embedding.embedding_generator import generate_embeddings
from src.utils.logger import get_logger
from src.utils.cache import get_cache
from src.config import is_cloud_environment, get_model_inference_url
from src.utils.memory_manager import get_memory_manager
# from src.rag.re_ranker import ReRanker  # DISABLED: Using reranker-service instead
from src.rag.retrieval_evaluator import RetrievalEvaluator

class RAGPipeline:
    """
    Implements the Retrieval-Augmented Generation (RAG) pipeline using Ollama.
    """

    def __init__(self, config: Dict[str, Any], chroma_store: ChromaVectorStore):
        """
        Initializes the RAGPipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            chroma_store (ChromaVectorStore): The ChromaDB vector store instance.
        """
        self.config = config
        self.chroma_store = chroma_store
        self.logger = get_logger(__name__, self.config)
        
        # Initialize cache
        self.cache = get_cache(ttl=config.get("cache_ttl", 3600)) if config.get("enable_cache", True) else None
        
        # Initialize memory manager
        self.memory_manager = get_memory_manager(config)
        
        # Initialize Model Inference Service URL
        self.model_inference_url = get_model_inference_url()
        
        # Initialize ReRanker - DISABLED: Use reranker-service instead
        self.reranker = None
        # Local reranker disabled - using reranker-service via CRAG evaluator
        self.logger.info("Local re-ranker disabled - using reranker-service instead")
        
        # Initialize Retrieval Evaluator (CRAG)
        self.retrieval_evaluator = None
        if self.config.get("enable_retrieval_evaluation", True):
            evaluator_config = {
                'model_name': self.config.get('retrieval_evaluator_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                'correct_threshold': self.config.get('retrieval_correct_threshold', 0.7),
                'incorrect_threshold': self.config.get('retrieval_incorrect_threshold', 0.3),
                'filter_threshold': self.config.get('retrieval_filter_threshold', 0.5)
            }
            
            self.retrieval_evaluator = RetrievalEvaluator(**evaluator_config)
            
            if not self.retrieval_evaluator.loaded:
                self.logger.warning("Retrieval Evaluator could not be loaded. Quality checks will be disabled.")
                self.retrieval_evaluator = None
            else:
                self.logger.info("✅ Retrieval Evaluator (CRAG) initialized successfully")
        
    def _test_model_inference_service(self):
        """Test connection to Model Inference Service."""
        try:
            health_response = requests.get(f"{self.model_inference_url}/health", timeout=5)
            if health_response.status_code == 200:
                self.logger.info(f"Successfully connected to Model Inference Service at {self.model_inference_url}")
                return True
            else:
                self.logger.warning(f"Model Inference Service health check failed: {health_response.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"Could not connect to Model Inference Service at {self.model_inference_url}: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant document chunks from the vector store.

        Args:
            query (str): The user query.
            top_k (int): The number of documents to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of retrieved document chunks with metadata.
        """
        self.logger.info(f"Retrieving documents for query: '{query}' (top_k={top_k})")
        
        # Try to get from cache first
        cache_key = None
        if self.cache:
            try:
                cache_key = f"retrieve:{hashlib.md5(f'{query}:{top_k}'.encode()).hexdigest()}"
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info("Found cached retrieval result for query")
                    return cached_result
            except Exception as e:
                self.logger.debug(f"Cache retrieval failed: {e}")
                cache_key = None
        
        try:
            query_embedding = generate_embeddings(
                [query],
                model=self.config.get("ollama_embedding_model"),
                use_cache=self.config.get("enable_cache", True)
            )
            
            if not query_embedding:
                self.logger.error("Failed to generate query embedding.")
                return []
            
            # Retrieve more documents initially if evaluator is enabled (will filter later)
            initial_k = top_k * 2 if self.retrieval_evaluator else top_k
            search_results = self.chroma_store.search(query_embedding, initial_k)
            
            # Convert search results to expected format
            retrieved_chunks = []
            for text, score, metadata in search_results:
                retrieved_chunks.append({
                    "text": text,
                    "score": score,
                    "metadata": metadata
                })
            
            self.logger.info(f"Initial retrieval: {len(retrieved_chunks)} chunks from vector store")
            
            # CRAG Evaluation
            if self.retrieval_evaluator and retrieved_chunks:
                # Evaluate quality
                evaluation = self.retrieval_evaluator.evaluate_retrieval(
                    query,
                    search_results
                )
                
                self.logger.info(
                    f"📊 CRAG Evaluation: {evaluation['confidence'].upper()} "
                    f"(avg={evaluation['avg_score']:.3f}, max={evaluation['max_score']:.3f}) "
                    f"→ {evaluation['action'].upper()}"
                )
                
                if evaluation['action'] == 'reject':
                    # Query is irrelevant to document corpus
                    self.logger.warning(
                        f"❌ Query rejected by CRAG evaluator: {evaluation['message']}"
                    )
                    return []
                
                elif evaluation['action'] == 'refine':
                    # Mixed results - filter by threshold
                    filtered_results = self.retrieval_evaluator.filter_by_threshold(
                        query,
                        search_results,
                        threshold=self.config.get('retrieval_filter_threshold', 0.5)
                    )
                    
                    # Update chunks with filtered results
                    retrieved_chunks = []
                    for text, score, metadata in filtered_results:
                        retrieved_chunks.append({
                            "text": text,
                            "score": score,
                            "metadata": metadata
                        })
                    
                    self.logger.info(
                        f"🔍 Refined: {len(retrieved_chunks)}/{len(search_results)} chunks "
                        f"(filtered by CRAG)"
                    )
                
                # 'accept' case - use results as-is
            
            # Take top_k after evaluation/filtering
            retrieved_chunks = retrieved_chunks[:top_k]
            
            self.logger.info(f"✅ Final: {len(retrieved_chunks)} chunks returned")
            
            # Cache the result
            if self.cache and retrieved_chunks and cache_key:
                try:
                    self.cache.set(cache_key, retrieved_chunks)
                except Exception as e:
                    self.logger.debug(f"Cache storage failed: {e}")
            
            return retrieved_chunks
            
        except Exception as e:
            self.logger.error(f"Error during document retrieval: {e}")
            return []

    def _generate_cache_key(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate cache key for generation request."""
        try:
            # Safely extract text and metadata from context for comprehensive caching
            context_data = []
            for chunk in context:
                chunk_data = {}
                
                # Extract text
                if isinstance(chunk, dict) and 'text' in chunk:
                    chunk_data['text'] = chunk['text']
                elif hasattr(chunk, 'get'):
                    chunk_data['text'] = chunk.get('text', '')
                elif hasattr(chunk, '__getitem__'):
                    try:
                        chunk_data['text'] = chunk['text'] if 'text' in chunk else ''
                    except (KeyError, TypeError):
                        chunk_data['text'] = str(chunk)
                else:
                    chunk_data['text'] = str(chunk) if chunk else ''
                
                # Extract metadata for more precise cache keys
                if isinstance(chunk, dict) and 'metadata' in chunk:
                    metadata = chunk['metadata']
                    if isinstance(metadata, dict):
                        # Include key metadata fields that affect uniqueness
                        chunk_data['source'] = metadata.get('source', '')
                        chunk_data['page'] = metadata.get('page', '')
                elif hasattr(chunk, 'get'):
                    metadata = chunk.get('metadata', {})
                    if hasattr(metadata, 'get'):
                        chunk_data['source'] = metadata.get('source', '')
                        chunk_data['page'] = metadata.get('page', '')
                
                context_data.append(chunk_data)
            
            context_hash = hashlib.md5(str(context_data).encode()).hexdigest()
            return f"generate:{hashlib.md5(f'{query}:{context_hash}'.encode()).hexdigest()}"
        except (TypeError, AttributeError, KeyError) as e:
            # Fallback for problematic contexts or mock objects
            fallback_hash = hashlib.md5(f"{query}:fallback:{time.time()}".encode()).hexdigest()
            return f"generate:{fallback_hash}"

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the Model Inference Service with the retrieved context.
        Includes caching and timeout handling.

        Args:
            query (str): The user query.
            context (List[Dict[str, Any]]): The retrieved context.

        Returns:
            str: The generated answer.
        """
        # Check cache first
        cache_key = None
        if self.cache:
            try:
                cache_key = self._generate_cache_key(query, context)
                cached_answer = self.cache.get(cache_key)
                if cached_answer:
                    self.logger.info("Found cached answer for query")
                    return cached_answer
            except Exception as e:
                self.logger.debug(f"Cache retrieval failed during generation: {e}")
                cache_key = None

        self.logger.info("Generating answer using Model Inference Service.")
        
        # Safely extract context text, handling malformed context
        try:
            context_texts = []
            for chunk in context:
                if isinstance(chunk, dict) and 'text' in chunk:
                    text = chunk['text']
                    if isinstance(text, str):
                        context_texts.append(text)
                    else:
                        context_texts.append(str(text))
                elif hasattr(chunk, '__getitem__'):
                    # Handle mock objects or other dict-like objects
                    try:
                        text = chunk.get('text', '') if hasattr(chunk, 'get') else chunk['text']
                        context_texts.append(str(text) if text else '')
                    except (KeyError, TypeError):
                        # Skip malformed chunks
                        continue
                else:
                    # Handle other object types
                    continue
            
            context_str = "\n".join(context_texts)
        except Exception as e:
            self.logger.warning(f"Error extracting context text, using fallback: {e}")
            context_str = "Context extraction failed."
        
        # Truncate context if too long to avoid token limits
        max_context_length = 4000  # Conservative limit
        if len(context_str) > max_context_length:
            context_str = context_str[:max_context_length] + "..."
            self.logger.info("Context truncated due to length limit")
        
        system_prompt = (
            "Sen, yalnızca ve yalnızca sana sağlanan BAĞLAM metnini kullanarak soruları yanıtlayan bir yapay zeka asistanısın. "
            "Görevin, kullanıcının sorusunun cevabını bu bağlam içinde bulmaktır. "
            "Eğer sorunun cevabı bağlamda açıkça yer almıyorsa, kesinlikle kendi bilgini kullanma. "
            "Bunun yerine, 'Bilgi sağlanan bağlamda bulunamadı.' şeklinde net bir cevap ver. "
            "Cevaplarını kısa ve doğrudan tut. Sadece Türkçe cevap ver."
        )
        
        full_prompt = f"""System: {system_prompt}

User: Bağlam:
{context_str}

Soru: {query}
Cevap:"""

        try:
            # Use ThreadPoolExecutor for timeout control
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._call_model_inference_service, full_prompt)
                try:
                    response = future.result(timeout=self.config.get("ollama_request_timeout", 120))
                    
                    # Cache the result
                    if self.cache and response and cache_key:
                        try:
                            self.cache.set(cache_key, response)
                        except Exception as e:
                            self.logger.debug(f"Cache storage failed during generation: {e}")
                    
                    return response
                except TimeoutError:
                    self.logger.error("Model Inference Service request timed out")
                    return "Üzgünüm, istek zaman aşımına uğradı. Lütfen daha kısa bir soru deneyin."
                except Exception as e:
                    self.logger.error(f"Error during Model Inference Service API call: {e}")
                    return "Üzgünüm, cevap oluşturulurken bir hata oluştu."
        except Exception as e:
            self.logger.error(f"Error setting up Model Inference Service request: {e}")
            return "Üzgünüm, cevap oluşturulurken bir hata oluştu."

    def _call_model_inference_service(self, prompt: str) -> str:
        """Make the actual Model Inference Service API call."""
        try:
            # Prepare request data for Model Inference Service
            request_data = {
                "prompt": prompt,
                "model": self.config.get("ollama_generation_model", "llama-3.1-8b-instant"),
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 512)
            }
            
            # Make HTTP request to Model Inference Service
            response = requests.post(
                f"{self.model_inference_url}/models/generate",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                self.logger.error(f"Model Inference Service API error: {response.status_code} - {response.text}")
                return "Üzgünüm, cevap oluşturulurken bir hata oluştu."
            
        except requests.exceptions.ConnectionError:
            self.logger.error("Could not connect to Model Inference Service")
            return "Üzgünüm, model servisi ile bağlantı kurulamadı."
        except requests.exceptions.Timeout:
            self.logger.error("Model Inference Service request timed out")
            return "Üzgünüm, istek zaman aşımına uğradı."
        except Exception as e:
            self.logger.error(f"Error in Model Inference Service API call: {e}")
            return "Üzgünüm, cevap oluşturulurken bir hata oluştu."

    def _generate_multiple_queries(self, query: str) -> List[str]:
        """
        Generates multiple search queries from a single user query using the Model Inference Service.
        """
        prompt = f"""
        You are an AI language model assistant. Your task is to generate 3 different versions of the given user
        question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user
        question, your goal is to help the user overcome some of the limitations of distance-based similarity search.
        Provide these alternative questions as a JSON array of strings.

        Original question: {query}

        JSON Array of alternative questions:
        """
        
        try:
            request_data = {
                "prompt": prompt,
                "model": self.config.get("ollama_generation_model", "llama-3.1-8b-instant"),
                "temperature": 0.3,
                "max_tokens": 256
            }
            
            response = requests.post(
                f"{self.model_inference_url}/models/generate",
                json=request_data,
                timeout=30
            )
            
            if response.status_code != 200:
                self.logger.warning(f"Model Inference Service error in multi-query generation: {response.status_code}")
                return [query]
            
            result = response.json()
            response_content = result.get("response", "[]")
            
            # Try to parse JSON response
            generated_queries = json.loads(response_content)
            
            if isinstance(generated_queries, list) and all(isinstance(q, str) for q in generated_queries):
                self.logger.info(f"Generated {len(generated_queries)} alternative queries.")
                queries = [query] + generated_queries
                # Remove duplicates
                return list(dict.fromkeys(queries))
            else:
                self.logger.warning("Model did not return a valid list of strings for multi-query. Falling back to original query.")
                return [query]

        except (json.JSONDecodeError, requests.exceptions.RequestException, Exception) as e:
            self.logger.error(f"Failed to generate or parse multiple queries: {e}")
            return [query] # Fallback to the original query

    def format_response(self, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Formats the final response with the answer and the original source objects.
        """
        return {
            "answer": answer,
            "sources": sources
        }

    def execute(self, query: str, strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline with support for multi-query retrieval and re-ranking.
        """
        if strategy_params is None:
            strategy_params = {}

        start_time = time.time()
        self.logger.info(f"Executing RAG pipeline for query: '{query}' with params: {strategy_params}")

        # Determine strategy for this execution
        use_multi_query = strategy_params.get("use_multi_query", self.config.get("use_multi_query", False))
        use_reranking = strategy_params.get("use_reranking", self.config.get("enable_reranking", False))
        
        # Adjust top_k for retrieval if re-ranking is active to provide more candidates
        if use_reranking and self.reranker:
            retrieval_top_k = strategy_params.get("retrieval_top_k", self.config.get("retrieval_top_k", 25))
        else:
            retrieval_top_k = strategy_params.get("top_k", self.config.get("top_k", 5))
        
        final_top_k = strategy_params.get("top_k", self.config.get("top_k", 5))

        try:
            # Step 1: Query Generation (Optional: Multi-Query)
            if use_multi_query:
                queries = self._generate_multiple_queries(query)
            else:
                queries = [query]
            self.logger.info(f"Using queries for retrieval: {queries}")

            # Step 2: Retrieval from all queries
            all_retrieved_chunks = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.retrieve, q, top_k=retrieval_top_k) for q in queries]
                for future in futures:
                    all_retrieved_chunks.extend(future.result())

            # Step 3: De-duplication of retrieved chunks
            unique_chunks = {}
            for chunk in all_retrieved_chunks:
                chunk_hash = hashlib.md5(chunk.get('text', '').encode()).hexdigest()
                if chunk_hash not in unique_chunks:
                    unique_chunks[chunk_hash] = chunk
            
            deduplicated_context = list(unique_chunks.values())
            self.logger.info(f"Retrieved {len(all_retrieved_chunks)} chunks, "
                             f"de-duplicated to {len(deduplicated_context)} unique chunks.")
            
            # DEBUG LOG: Log top initial results before re-ranking
            self.logger.debug(f"Top 3 initial chunks before re-ranking: "
                              f"{[chunk.get('text', '')[:150] + '...' for chunk in deduplicated_context[:3]]}")

            if not deduplicated_context:
                self.logger.warning("No context retrieved for query after all steps.")
                return {"answer": "Üzgünüm, bu konuda bilgiye sahip değilim.", "sources": []}

            # Step 4: Re-ranking (Optional)
            if use_reranking and self.reranker:
                self.logger.info(f"Applying re-ranking to {len(deduplicated_context)} documents.")
                final_context = self.reranker.rerank(query, deduplicated_context, top_k=final_top_k)
                # DEBUG LOG: Log top results after re-ranking
                self.logger.debug(f"Top 3 re-ranked chunks: "
                                  f"{[chunk.get('text', '')[:150] + '...' for chunk in final_context[:3]]}")
            else:
                # If not re-ranking, sort by original score and take top_k
                deduplicated_context.sort(key=lambda x: x.get('score', 0.0), reverse=True)
                final_context = deduplicated_context[:final_top_k]

            self.logger.info(f"Final context size after retrieval/re-ranking: {len(final_context)}")

            # Step 5: Generate answer
            generated_answer = self.generate(query, final_context)
            if not generated_answer or generated_answer.strip() == "":
                generated_answer = "Üzgünüm, bu soruya cevap oluşturamadım."

            # Step 6: Format response
            formatted_response = self.format_response(generated_answer, final_context)

            execution_time = time.time() - start_time
            self.logger.info(f"RAG pipeline execution finished in {execution_time:.2f}s")
            
            return formatted_response

        except Exception as e:
            self.logger.error(f"Error in RAG pipeline execution: {e}", exc_info=True)
            return {"answer": "Üzgünüm, sorunuzu işlerken bir hata oluştu.", "sources": []}

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Returns the current parameters of the RAG pipeline.
        """
        return {
            "embedding_model": self.config.get("ollama_embedding_model"),
            "generation_model": self.config.get("ollama_generation_model"),
            "top_k": self.config.get("top_k", 5),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 512),
            "top_p": self.config.get("top_p", 0.9),
            "repeat_penalty": self.config.get("repeat_penalty", 1.1),
            "cache_enabled": self.config.get("enable_cache", True),
            "cache_ttl": self.config.get("cache_ttl", 3600),
        }