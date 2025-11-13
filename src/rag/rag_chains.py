"""
RAG Chain Implementations for Edu-ModRAG System.

This module implements different RAG chain strategies:
- Stuff Chain: Fast, single-pass retrieval and generation
- Refine Chain: Iterative refinement for complex queries
- Map-Reduce Chain: Multi-document analysis and summarization

Features Re-ranking for improved document relevance scoring.

Based on research findings from "Şakar & Emekci (2024)" and "Gao et al. (2024)"
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import ollama
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from src.utils.logger import get_logger
from src.config import is_cloud_environment
from src.vector_store.chroma_store import ChromaVectorStore
from src.embedding.embedding_generator import generate_embeddings
from src.utils.prompt_templates import BilingualPromptManager
from src.utils.language_detector import detect_language
from src.rag.re_ranker import ReRanker

class BaseRAGChain(ABC):
    """
    Abstract base class for all RAG chain implementations.
    """
    
    def __init__(self, config: Dict[str, Any], chroma_store: ChromaVectorStore):
        """
        Initialize base RAG chain.
        
        Args:
            config: System configuration
            chroma_store: ChromaDB vector store instance
        """
        self.config = config
        self.chroma_store = chroma_store
        self.logger = get_logger(__name__, config)
        
        # Initialize Ollama client
        self.ollama_client = self._init_ollama_client()
        
        # Initialize prompt manager for consistent, strict prompts
        self.prompt_manager = BilingualPromptManager()
        
        # Initialize re-ranker for improved document relevance
        self.use_reranking = config.get("enable_reranking", True)
        self.reranker = None
        if self.use_reranking:
            try:
                self.reranker = ReRanker()
                self.logger.info("Re-ranker initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize re-ranker: {e}")
                self.use_reranking = False
        
        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "total_time": 0.0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
            "success_rate": 0.0,
            "failures": 0,
            "reranking_used": 0
        }
    
    def _init_ollama_client(self):
        """Initialize Ollama client, skipping in cloud environments."""
        if is_cloud_environment():
            self.logger.info("Cloud environment detected, skipping Ollama client initialization.")
            self.ollama_client = None
            return None
    
        try:
            client = ollama.Client(host=self.config.get("ollama_base_url"))
            return client
        except Exception as e:
            self.logger.warning(f"Could not connect to Ollama. Running without local models. Error: {e}")
            return None
    
    @abstractmethod
    def execute(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the RAG chain for the given query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer and metadata
        """
        pass
    
    def _retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector store with optional re-ranking.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata, potentially re-ranked
        """
        try:
            query_embedding = generate_embeddings(
                [query],
                model=self.config.get("ollama_embedding_model"),
                use_cache=self.config.get("enable_cache", True)
            )
            
            if not query_embedding:
                self.logger.error("Failed to generate query embedding.")
                return []
            
            # Step 1: Initial retrieval - get more documents if re-ranking is enabled
            if self.use_reranking and self.reranker:
                # Get 3-5x more documents for re-ranking
                initial_k = max(top_k * 4, 25)
                initial_k = min(initial_k, 50)  # Cap at reasonable limit
                self.logger.info(f"Using re-ranking: retrieving {initial_k} documents initially")
            else:
                initial_k = top_k
            
            search_results = self.chroma_store.search(query_embedding, initial_k)
            
            # Convert to the format expected by re-ranker
            retrieved_chunks = []
            reranker_input = []
            
            for text, score, metadata in search_results:
                retrieved_chunks.append({
                    "text": text,
                    "score": score,
                    "metadata": metadata
                })
                # Prepare for re-ranking: (text, original_score, metadata)
                reranker_input.append((text, score, metadata))
            
            # Step 2: Apply re-ranking if enabled and available
            if self.use_reranking and self.reranker and len(reranker_input) > top_k:
                try:
                    self.logger.info(f"Re-ranking {len(reranker_input)} documents")
                    reranked_results = self.reranker.rerank(query, reranker_input, top_k)
                    
                    # Convert back to our format
                    retrieved_chunks = []
                    for text, new_score, metadata in reranked_results:
                        retrieved_chunks.append({
                            "text": text,
                            "score": new_score,
                            "metadata": metadata
                        })
                    
                    self.performance_stats["reranking_used"] += 1
                    self.logger.info(f"Re-ranking completed. Final documents: {len(retrieved_chunks)}")
                    
                except Exception as e:
                    self.logger.warning(f"Re-ranking failed, using original results: {e}")
                    # Fallback to original results, truncated to top_k
                    retrieved_chunks = retrieved_chunks[:top_k]
            else:
                # Use original results
                retrieved_chunks = retrieved_chunks[:top_k]
                
            self.logger.info(f"Final retrieved documents: {len(retrieved_chunks)}")
            return retrieved_chunks
            
        except Exception as e:
            self.logger.error(f"Error during document retrieval: {e}")
            return []
    
    def _generate_with_ollama(self, system_prompt: str, user_prompt: str, 
                            temperature: float = None) -> str:
        """
        Generate response using Ollama.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Generation temperature
            
        Returns:
            Generated response
        """
        if not self.ollama_client:
            return "Ollama istemcisi mevcut değil."
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._call_ollama, 
                    system_prompt, 
                    user_prompt, 
                    temperature or self.config.get("temperature", 0.7)
                )
                
                timeout = self.config.get("ollama_request_timeout", 120)
                response = future.result(timeout=timeout)
                return response
                
        except TimeoutError:
            self.logger.error("Ollama request timed out")
            return "İstek zaman aşımına uğradı."
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            return "Cevap oluşturulurken hata oluştu."
    
    def _call_ollama(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Make actual Ollama API call."""
        response = self.ollama_client.chat(
            model=self.config.get("ollama_generation_model"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": self.config.get("max_tokens", 512),
                "top_k": self.config.get("top_k", 40),
                "top_p": self.config.get("top_p", 0.9),
                "repeat_penalty": self.config.get("repeat_penalty", 1.1),
            }
        )
        return response['message']['content'].strip()
    
    def _update_performance_stats(self, execution_time: float, success: bool, 
                                tokens_used: int = 0):
        """Update performance statistics."""
        self.performance_stats["total_queries"] += 1
        self.performance_stats["total_time"] += execution_time
        self.performance_stats["total_tokens"] += tokens_used
        
        if success:
            self.performance_stats["avg_response_time"] = (
                self.performance_stats["total_time"] / 
                self.performance_stats["total_queries"]
            )
        else:
            self.performance_stats["failures"] += 1
        
        self.performance_stats["success_rate"] = (
            (self.performance_stats["total_queries"] - 
             self.performance_stats["failures"]) / 
            self.performance_stats["total_queries"]
        )

class StuffChain(BaseRAGChain):
    """
    Stuff Chain: Fast and efficient for simple queries.
    
    Retrieves relevant documents and stuffs them into a single context
    for one-pass generation. Optimal for simple factual questions.
    """
    
    def execute(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute Stuff chain strategy.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Response dictionary with answer and sources
        """
        start_time = time.time()
        self.logger.info(f"Executing Stuff chain for query: '{query}'")
        
        try:
            # Step 1: Retrieve documents
            documents = self._retrieve_documents(query, top_k)
            if not documents:
                return {
                    "answer": "Bu konuda bilgi bulunamadı.",
                    "sources": [],
                    "chain_type": "stuff",
                    "execution_time": time.time() - start_time
                }
            
            # Step 2: Detect query language and format context properly
            query_language = detect_language(query)
            if query_language not in ['tr', 'en']:
                query_language = 'tr'  # Default to Turkish
            
            # Format context using centralized function with proper source attribution
            retrieved_texts = [doc['text'] for doc in documents]
            metas = [doc['metadata'] for doc in documents]
            
            context = self.prompt_manager.format_context_with_sources(
                query_language, retrieved_texts, metas
            )
            
            # Truncate if too long
            max_context_length = self.config.get("max_context_length", 4000)
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
                self.logger.info("Context truncated due to length limit")
            
            # Step 3: Generate response using strict centralized prompts
            system_prompt = self.prompt_manager.get_system_prompt(query_language, 'rag')
            user_prompt = self.prompt_manager.get_user_prompt(query_language, query, context)
            
            answer = self._generate_with_ollama(system_prompt, user_prompt, temperature=0.3)
            
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time, True, len(context.split()))
            
            self.logger.info(f"Stuff chain completed in {execution_time:.2f}s")
            
            return {
                "answer": answer,
                "sources": documents,
                "chain_type": "stuff",
                "execution_time": execution_time,
                "context_length": len(context)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time, False)
            self.logger.error(f"Error in Stuff chain: {e}")
            
            return {
                "answer": "Üzgünüm, cevap oluşturulurken bir hata oluştu.",
                "sources": [],
                "chain_type": "stuff",
                "execution_time": execution_time,
                "error": str(e)
            }

class RefineChain(BaseRAGChain):
    """
    Refine Chain: Iterative refinement for complex queries.
    
    Processes documents sequentially, refining the answer with each iteration.
    Optimal for complex questions requiring deep analysis.
    """
    
    def execute(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute Refine chain strategy.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Response dictionary with refined answer and sources
        """
        start_time = time.time()
        self.logger.info(f"Executing Refine chain for query: '{query}'")
        
        try:
            # Step 1: Retrieve documents
            documents = self._retrieve_documents(query, top_k)
            if not documents:
                return {
                    "answer": "Bu konuda bilgi bulunamadı.",
                    "sources": [],
                    "chain_type": "refine",
                    "execution_time": time.time() - start_time
                }
            
            # Step 2: Initial answer from first document using strict prompts
            query_language = detect_language(query)
            if query_language not in ['tr', 'en']:
                query_language = 'tr'  # Default to Turkish
                
            initial_doc = documents[0]
            system_prompt = self.prompt_manager.get_system_prompt(query_language, 'rag')
            
            # Format initial context with proper source attribution
            initial_context = self.prompt_manager.format_context_with_sources(
                query_language, [initial_doc['text']], [initial_doc['metadata']]
            )
            
            initial_prompt = self.prompt_manager.get_user_prompt(query_language, query, initial_context)
            
            current_answer = self._generate_with_ollama(
                system_prompt, initial_prompt, temperature=0.5
            )
            
            refinement_steps = []
            refinement_steps.append({
                "step": 1,
                "document_used": initial_doc,
                "answer": current_answer
            })
            
            # Step 3: Refine answer with remaining documents
            for i, doc in enumerate(documents[1:], 2):
                if i > self.config.get("max_refinement_steps", 4):
                    break  # Limit refinement steps
                
                # Format additional context with proper source attribution
                additional_context = self.prompt_manager.format_context_with_sources(
                    query_language, [doc['text']], [doc['metadata']]
                )
                
                refine_prompt_template = (
                    "Mevcut cevap:\n{current_answer}\n\n"
                    "Ek bağlam:\n{additional_context}\n\n"
                    "Soru: {query}\n\n"
                    "Yeni bağlamı kullanarak mevcut cevabı geliştirebilir misin? "
                    "SADECE yeni bağlamda geçerli bilgi varsa cevabı zenginleştir. "
                    "Bağlamda olmayan bilgi ekleme. Eğer yeni bilgi yoksa mevcut cevabı koru."
                ) if query_language == 'tr' else (
                    "Current answer:\n{current_answer}\n\n"
                    "Additional context:\n{additional_context}\n\n"
                    "Question: {query}\n\n"
                    "Can you improve the current answer using the new context? "
                    "ONLY enrich the answer if there is valid information in the new context. "
                    "Do not add information not in the context. If no new information, keep current answer."
                )
                
                refine_prompt = refine_prompt_template.format(
                    current_answer=current_answer,
                    additional_context=additional_context,
                    query=query
                )
                
                refined_answer = self._generate_with_ollama(
                    system_prompt, refine_prompt, temperature=0.4
                )
                
                # Only update if refinement adds value
                if len(refined_answer) > len(current_answer) * 0.8:  # Avoid truncated responses
                    current_answer = refined_answer
                    refinement_steps.append({
                        "step": i,
                        "document_used": doc,
                        "answer": current_answer
                    })
            
            execution_time = time.time() - start_time
            self._update_performance_stats(
                execution_time, True, 
                sum(len(doc['text'].split()) for doc in documents)
            )
            
            self.logger.info(
                f"Refine chain completed in {execution_time:.2f}s "
                f"with {len(refinement_steps)} refinement steps"
            )
            
            return {
                "answer": current_answer,
                "sources": documents,
                "chain_type": "refine",
                "execution_time": execution_time,
                "refinement_steps": len(refinement_steps),
                "refinement_details": refinement_steps
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time, False)
            self.logger.error(f"Error in Refine chain: {e}")
            
            return {
                "answer": "Üzgünüm, cevap oluşturulurken bir hata oluştu.",
                "sources": [],
                "chain_type": "refine",
                "execution_time": execution_time,
                "error": str(e)
            }

class MapReduceChain(BaseRAGChain):
    """
    Map-Reduce Chain: Multi-document summarization and analysis.
    
    Maps each document to individual summaries, then reduces them
    into a comprehensive final answer. Optimal for multi-document queries.
    """
    
    def execute(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        """
        Execute Map-Reduce chain strategy.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (higher for map-reduce)
            
        Returns:
            Response dictionary with comprehensive analysis
        """
        start_time = time.time()
        self.logger.info(f"Executing Map-Reduce chain for query: '{query}'")
        
        try:
            # Step 1: Retrieve documents (more documents for comprehensive analysis)
            documents = self._retrieve_documents(query, top_k)
            if not documents:
                return {
                    "answer": "Bu konuda bilgi bulunamadı.",
                    "sources": [],
                    "chain_type": "map_reduce",
                    "execution_time": time.time() - start_time
                }
            
            # Step 2: Map phase - Generate individual summaries using strict prompts
            query_language = detect_language(query)
            if query_language not in ['tr', 'en']:
                query_language = 'tr'  # Default to Turkish
                
            document_summaries = []
            system_prompt = self.prompt_manager.get_system_prompt(query_language, 'rag')
            
            map_template = (
                "Belge içeriği:\n{doc_content}\n\n"
                "Soru: {query}\n\n"
                "Bu belgede soruyla ilgili hangi bilgiler var? SADECE belgede yazılan bilgileri kullan. "
                "Kısa ve öz bir özet çıkar. Eğer belge soruyla ilgili değilse, 'İlgili bilgi yok' de."
            ) if query_language == 'tr' else (
                "Document content:\n{doc_content}\n\n"
                "Question: {query}\n\n"
                "What information in this document relates to the question? Use ONLY information written in the document. "
                "Create a brief summary. If the document is not relevant, say 'No relevant information'."
            )
            
            for i, doc in enumerate(documents):
                doc_context = self.prompt_manager.format_context_with_sources(
                    query_language, [doc['text']], [doc['metadata']]
                )
                
                map_prompt = map_template.format(
                    doc_content=doc_context,
                    query=query
                )
                
                summary = self._generate_with_ollama(
                    system_prompt, map_prompt, temperature=0.3
                )
                
                if "ilgili bilgi yok" not in summary.lower() and len(summary.strip()) > 10:
                    document_summaries.append({
                        "document_index": i,
                        "summary": summary,
                        "source": doc
                    })
            
            if not document_summaries:
                return {
                    "answer": "Bulunan belgelerde soruyla ilgili bilgi bulunmadı.",
                    "sources": documents,
                    "chain_type": "map_reduce",
                    "execution_time": time.time() - start_time
                }
            
            # Step 3: Reduce phase - Combine summaries into final answer
            all_summaries = "\n\n".join([
                f"Özet {i+1}: {summary['summary']}" 
                for i, summary in enumerate(document_summaries)
            ])
            
            reduce_template = (
                "Farklı kaynaklardan elde edilen özetler:\n{all_summaries}\n\n"
                "Soru: {query}\n\n"
                "Bu özetleri birleştirerek kapsamlı ve tutarlı bir cevap oluştur. "
                "SADECE özetlerde verilen bilgileri kullan. Farklı kaynaklardan gelen bilgileri entegre et "
                "ve çelişkiler varsa belirt. Özetlerde olmayan bilgi ekleme."
            ) if query_language == 'tr' else (
                "Summaries from different sources:\n{all_summaries}\n\n"
                "Question: {query}\n\n"
                "Create a comprehensive and consistent answer by combining these summaries. "
                "Use ONLY information provided in the summaries. Integrate information from different sources "
                "and note any conflicts. Do not add information not in the summaries."
            )
            
            reduce_prompt = reduce_template.format(
                all_summaries=all_summaries,
                query=query
            )
            
            final_answer = self._generate_with_ollama(
                system_prompt,
                reduce_prompt,
                temperature=0.4
            )
            
            execution_time = time.time() - start_time
            self._update_performance_stats(
                execution_time, True,
                sum(len(summary['summary'].split()) for summary in document_summaries)
            )
            
            self.logger.info(
                f"Map-Reduce chain completed in {execution_time:.2f}s "
                f"with {len(document_summaries)} document summaries"
            )
            
            return {
                "answer": final_answer,
                "sources": documents,
                "chain_type": "map_reduce",
                "execution_time": execution_time,
                "documents_processed": len(document_summaries),
                "map_summaries": document_summaries
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time, False)
            self.logger.error(f"Error in Map-Reduce chain: {e}")
            
            return {
                "answer": "Üzgünüm, cevap oluşturulurken bir hata oluştu.",
                "sources": [],
                "chain_type": "map_reduce",
                "execution_time": execution_time,
                "error": str(e)
            }

class RAGChainFactory:
    """
    Factory class for creating RAG chain instances.
    """
    
    @staticmethod
    def create_chain(chain_type: str, config: Dict[str, Any],
                    chroma_store: ChromaVectorStore) -> BaseRAGChain:
        """
        Create a RAG chain instance.
        
        Args:
            chain_type: Type of chain to create
            config: System configuration
            chroma_store: ChromaDB vector store instance
            
        Returns:
            RAG chain instance
        """
        chain_map = {
            "stuff": StuffChain,
            "refine": RefineChain,
            "map_reduce": MapReduceChain
        }
        
        if chain_type not in chain_map:
            raise ValueError(f"Unknown chain type: {chain_type}")
        
        return chain_map[chain_type](config, chroma_store)