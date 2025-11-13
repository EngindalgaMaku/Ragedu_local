from typing import Dict, Any
import time
import uuid
from src.query_processing.query_processor import QueryProcessor
from src.rag.rag_pipeline import RAGPipeline
from src.rag.adaptive_query_router import AdaptiveQueryRouter
from src.services.active_learning import ActiveLearningEngine
from src.recommendation.recommender import generate_recommendations
from src.utils.logger import get_logger
from src.analytics.database import get_experiment_db, ExperimentDatabase

class QAService:
    """
    Orchestrates the entire question-answering process.
    """

    def __init__(self, config: Dict[str, Any], query_processor: QueryProcessor, rag_pipeline: RAGPipeline, db_connection: ExperimentDatabase):
        """
        Initializes the QAService with active learning components.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            query_processor (QueryProcessor): The query processor instance.
            rag_pipeline (RAGPipeline): The RAG pipeline instance.
            db_connection (DatabaseAnalytics): The database connection for analytics.
        """
        self.config = config
        self.query_processor = query_processor
        self.rag_pipeline = rag_pipeline
        self.db = db_connection
        self.logger = get_logger(__name__, self.config)
        
        # Initialize active learning and routing components
        self.active_learning_engine = ActiveLearningEngine(self.db)
        self.adaptive_router = AdaptiveQueryRouter(self.db, self.active_learning_engine)

    def answer_question(self, query: str, user_id: str = "default_user", session_id: str = None) -> Dict[str, Any]:
        """
        Answers a user's question by coordinating the query processor and RAG pipeline.

        Args:
            query (str): The user's question.
            user_id (str): The ID of the user asking the question.
            session_id (str): The ID of the current session.

        Returns:
            Dict[str, Any]: The final answer, source citations, and interaction_id.
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())
        self.logger.info(f"Answering question for user '{user_id}' in session '{session_id}': '{query}'")
        try:
            # 1. Process the query
            self.logger.info("Processing query...")
            processed_query_data = self.query_processor.process(query)
            processed_query = processed_query_data["processed_query"]
            self.logger.info("Query processed successfully.")

            # 2. Determine the best RAG strategy using the adaptive router
            self.logger.info("Determining best RAG strategy with Adaptive Query Router...")
            # We need some metadata for the router, for now we pass an empty dict
            # In a real scenario, this could contain initial confidence scores
            strategy_name = self.adaptive_router.route_query(
                query=processed_query,
                user_id=user_id,
                query_metadata={}
            )
            self.logger.info(f"Strategy selected: {strategy_name}")

            # Define strategy parameters based on the router's decision
            # This is a simplified mapping. A more robust solution would use a config file.
            strategy_params = {}
            if strategy_name == "precise_retrieval":
                strategy_params = {"top_k": 3}
            elif strategy_name == "broad_context_retrieval":
                strategy_params = {"top_k": 10}
            elif strategy_name == "hybrid_search":
                strategy_params = {"top_k": 5} # Default

            # 3. Execute the RAG pipeline with the selected strategy
            self.logger.info(f"Executing RAG pipeline with strategy '{strategy_name}'...")
            rag_response = self.rag_pipeline.execute(processed_query, strategy_params=strategy_params)
            self.logger.info("RAG pipeline executed successfully.")

            # 4. Generate recommendations
            self.logger.info("Generating recommendations...")
            recommendations = generate_recommendations(query, rag_response["sources"])
            self.logger.info("Recommendations generated successfully.")

            # 5. (Future) Implement response quality validation and caching here
            
            # 6. Combine results and return
            final_response = {
                "query_details": processed_query_data,
                "answer": rag_response["answer"],
                "sources": rag_response["sources"],
                "recommendations": recommendations,
            }

            # Log interaction to database and get interaction_id
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Placeholder for RAG config hash. This should be dynamically determined.
            rag_params = self.rag_pipeline.get_current_parameters()
            rag_config_hash = self.db.add_or_get_rag_configuration(rag_params)

            interaction_id = self.db.add_interaction(
                user_id=user_id,
                session_id=session_id,
                query=query,
                response=rag_response["answer"],
                retrieved_context=rag_response["sources"],
                rag_config_hash=rag_config_hash,
                processing_time_ms=processing_time_ms
            )
            
            final_response["interaction_id"] = interaction_id
            
            self.logger.info(f"Successfully answered query for interaction {interaction_id}: '{query}'")
            return final_response

        except ValueError as ve:
            self.logger.error(f"Invalid query: {ve}")
            return {"error": str(ve)}
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in QAService: {e}", exc_info=True)
            return {"error": "Üzgünüm, sorunuzu yanıtlarken beklenmedik bir hata oluştu."}