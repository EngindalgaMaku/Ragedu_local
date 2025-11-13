from fastapi import FastAPI

app = FastAPI(title="Minimal RAG3 API Test", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models/list")
def list_available_models():
    """
    Get a list of available Groq cloud models.
    Returns a simple list of model name strings.
    """
    return {
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)