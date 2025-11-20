import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain & RAGAS
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation-service")

load_dotenv()

app = FastAPI(
    title="RAG Evaluation Service",
    description="Microservice for evaluating RAG performance using RAGAS metrics",
    version="1.0.0"
)

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    overall_score: float

def get_llm():
    """Configure LLM based on environment variables"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        logger.info("Using Groq LLM for evaluation")
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
    else:
        # Fallback to OpenAI if available
        logger.info("Using OpenAI LLM for evaluation")
        return ChatOpenAI(model="gpt-3.5-turbo-0125")

def get_embeddings():
    """Configure Embeddings"""
    # RAGAS needs embeddings for some metrics. 
    # We can use OpenAI or a local HuggingFace model wrapper.
    # For simplicity in this setup, we assume OpenAI or similar is available.
    # If not, we'll need to add HuggingFaceEmbeddings to requirements.
    return OpenAIEmbeddings()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "evaluation-service"}

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(request: EvaluationRequest):
    try:
        logger.info(f"Evaluating request for question: {request.question}")
        
        # Prepare data for RAGAS
        data = {
            "question": [request.question],
            "answer": [request.answer],
            "contexts": [request.contexts],
            "ground_truth": [request.ground_truth] if request.ground_truth else [""]
        }
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics based on available data
        metrics = [faithfulness, answer_relevancy]
        if request.ground_truth:
            metrics.extend([context_precision, context_recall])
            
        # Configure RAGAS with our LLM
        llm = get_llm()
        # Note: RAGAS typically handles LLM config automatically or via passing llm argument in newer versions.
        # For v0.1.x, passing llm to evaluate is standard.
        
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=get_embeddings()
        )
        
        # Extract scores (RAGAS returns a dict-like object)
        scores = results.to_pandas().iloc[0].to_dict()
        
        # Calculate overall score (simple average of available metrics)
        valid_scores = [v for k, v in scores.items() if isinstance(v, (int, float)) and k != "question"]
        overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        return EvaluationResponse(
            faithfulness=scores.get("faithfulness", 0.0),
            answer_relevancy=scores.get("answer_relevancy", 0.0),
            context_precision=scores.get("context_precision"),
            context_recall=scores.get("context_recall"),
            overall_score=overall
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)

