import os
import re
import time
import requests
from datetime import datetime
from src import config as app_config
from src.config import get_model_inference_url
from src.document_processing.document_processor import process_document
from src.document_processing.enhanced_pdf_processor import extract_text_from_pdf_enhanced, process_pdf_with_analysis, MARKER_AVAILABLE
from src.text_processing.text_chunker import chunk_text
from src.text_processing.semantic_chunker import create_semantic_chunks
from src.embedding.embedding_generator import generate_embeddings
from src.vector_store.faiss_store import FaissVectorStore
from src.utils.performance_monitor import get_performance_monitor, measure_performance
from src.utils.prompt_templates import prompt_manager
from src.utils.language_detector import detect_query_language
from src.services.prompt_manager import teacher_prompt_manager, PromptPerformance


def get_selected_provider() -> str:
    """Get the currently selected provider from environment (default: 'groq')."""
    return os.environ.get("RAG_PROVIDER", "groq")


def call_model_inference_service(prompt: str, model: str = None, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Helper function to call the Model Inference Service."""
    try:
        model_inference_url = get_model_inference_url()
        request_data = {
            "prompt": prompt,
            "model": model or "llama-3.1-8b-instant",
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{model_inference_url}/models/generate",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            return ""
            
    except Exception as e:
        return ""


def is_groq_model(model_name: str) -> bool:
    """Check if the model name is a Groq model - only tested working models."""
    groq_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b"
        # All models confirmed working via direct Groq API testing 2025-11-02
    ]
    return any(groq_model in model_name for groq_model in groq_models)


def get_session_index_path(session_name: str) -> str:
    os.makedirs("data/vector_db/sessions", exist_ok=True)
    safe = session_name.strip().replace(" ", "_") or "default"
    return os.path.join("data/vector_db/sessions", safe)

def get_store(index_path: str) -> FaissVectorStore:
    return FaissVectorStore(index_path=index_path)


def add_document_to_store(
    file_bytes: bytes,
    filename: str,
    vector_store: FaissVectorStore,
    *,
    strategy: str = "char",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
) -> dict:
    os.makedirs("data/uploads", exist_ok=True)
    save_path = os.path.join("data/uploads", filename)
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    # PDF için özel markdown tabanlı işleme - Büyük PDF'ler için optimize edildi
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension == ".pdf":
        # PDF boyutunu kontrol et
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"📊 PDF boyutu: {file_size_mb:.1f}MB - {filename}")
        
        if MARKER_AVAILABLE:
            # Büyük PDF'ler için özelleştirilmiş işleme
            try:
                print(f"🚀 Marker ile büyük PDF işleniyor: {filename}")
                text, metadata = process_pdf_with_analysis(save_path)
                
                # PDF'ler için otomatik olarak markdown stratejisini kullan
                if strategy in ["char", "paragraph", "sentence"]:
                    strategy = "markdown"
                
                # Markdown dosyasını kaydet
                os.makedirs("data/markdown", exist_ok=True)
                markdown_filename = os.path.splitext(filename)[0] + ".md"
                markdown_path = os.path.join("data/markdown", markdown_filename)
                with open(markdown_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Metadata'yı kaydet
                import json
                metadata_path = os.path.join("data/markdown", os.path.splitext(filename)[0] + "_metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                print(f"✅ PDF başarıyla işlendi: {len(text)} karakter çıkarıldı")
                    
            except Exception as e:
                print(f"❌ Marker PDF işleme hatası: {e}")
                # Büyük PDF'ler için fallback yerine hata bildir
                if "timeout" in str(e).lower():
                    print("💡 Çözüm: MARKER_TIMEOUT_SECONDS değişkenini artırın (örn: 1800)")
                    print("💡 Environment variable: MARKER_TIMEOUT_SECONDS=1800")
                elif "memory" in str(e).lower():
                    print("💡 Çözüm: MARKER_MAX_MEMORY_MB değişkenini artırın (örn: 8192)")
                    print("💡 Environment variable: MARKER_MAX_MEMORY_MB=8192")
                
                # Kritik hatalar için fallback, diğerleri için tekrar dene
                if any(error_type in str(e) for error_type in ["FileNotFoundError", "PermissionError", "corrupted"]):
                    print("🔄 Kritik hata - fallback kullanılıyor...")
                    text = process_document(save_path)
                else:
                    print("⚠️ Marker işleme başarısız - büyük PDF için ayarları optimize edin")
                    # Büyük PDF'ler için hata fırlat, fallback kullanma
                    raise Exception(f"PDF processing failed for large file: {filename}. Please optimize Marker settings.")
        else:
            print("⚠️ Marker kütüphanesi mevcut değil!")
            print("💡 Büyük PDF'ler için Marker kurulumu önerilir: pip install marker-pdf")
            # Marker yoksa fallback kullan ama uyarı ver
            text = process_document(save_path)
            print(f"📄 Fallback ile işlendi: {len(text) if text else 0} karakter")
    else:
        # PDF değilse normal işleme
        text = process_document(save_path)
    
    if not text:
        return {"added": 0, "chunks": 0, "embedding_dim": None}
    
    # Use GROQ-powered semantic chunking if available
    try:
        print(f"🧠 Using GROQ-powered semantic chunking...")
        print(f"🔧 Text length: {len(text)}, target_size: {chunk_size or 800}, overlap_ratio: {(chunk_overlap or 100) / (chunk_size or 800)}")
        print(f"🔧 Fallback strategy: {strategy or 'markdown'}")
        
        chunks = create_semantic_chunks(
            text,
            target_size=chunk_size or 800,
            overlap_ratio=(chunk_overlap or 100) / (chunk_size or 800),
            language="auto",
            fallback_strategy=strategy or "markdown"
        )
        
        # Verify if chunks are actually from semantic chunking or fallback
        if chunks:
            print(f"✅ Semantic chunking returned {len(chunks)} chunks")
            
            # Check first chunk characteristics to see if it's semantic or fallback
            first_chunk = chunks[0][:100] + "..." if len(chunks[0]) > 100 else chunks[0]
            print(f"🔍 First chunk preview: {first_chunk}")
            
            # Log chunk sizes for analysis
            chunk_sizes = [len(c) for c in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            print(f"📊 Chunk statistics: avg={avg_size:.0f}, min={min(chunk_sizes)}, max={max(chunk_sizes)}")
        else:
            print("⚠️ Semantic chunking returned empty chunks")
            
    except Exception as e:
        print(f"⚠️ Semantic chunking failed with exception: {e}")
        print(f"🔄 Falling back to traditional chunking with strategy: {strategy}")
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,  # type: ignore[arg-type]
        )
        print(f"📄 Fallback chunking created {len(chunks)} chunks")
    if not chunks:
        return {"added": 0, "chunks": 0, "embedding_dim": None}
    # Use provider-aware embedding generation
    selected_provider = get_selected_provider()
    if selected_provider == 'ollama':
        embeddings = generate_embeddings(chunks, model=embedding_model, provider='ollama')
    else:
        # For cloud providers, use local sentence transformers for embeddings
        embeddings = generate_embeddings(chunks, model=embedding_model, provider='sentence_transformers')
    if not embeddings:
        return {"added": 0, "chunks": len(chunks), "embedding_dim": None}
    embedding_dim = len(embeddings) if embeddings and isinstance(embeddings, list) else None
    before = vector_store.index.ntotal if vector_store.index is not None else 0
    # Build per-chunk metadata: source_file, page/slide markers, simple title
    def parse_marker(txt: str) -> dict:
        for line in txt.splitlines():
            ls = line.strip()
            if ls.startswith("=== Page ") and ls.endswith("==="):
                # format: === Page N ===
                try:
                    n = int(ls.replace("=", "").strip().split()[-1])
                    return {"page_number": n}
                except Exception:
                    return {}
            if ls.startswith("=== Slide ") and ls.endswith("==="):
                try:
                    n = int(ls.replace("=", "").strip().split()[-1])
                    return {"slide_number": n}
                except Exception:
                    return {}
        return {}

    def guess_title(txt: str) -> str | None:
        seen_marker = False
        for line in txt.splitlines():
            ls = line.strip()
            if not ls:
                continue
            if (ls.startswith("=== Page ") or ls.startswith("=== Slide ")) and ls.endswith("==="):
                seen_marker = True
                continue
            if seen_marker:
                return ls[:120]
        # fallback: first non-empty line
        for line in txt.splitlines():
            ls = line.strip()
            if ls:
                return ls[:120]
        return None

    metadatas = []
    for ch in chunks:
        md = {"source_file": filename}
        md.update(parse_marker(ch))
        t = guess_title(ch)
        if t:
            md["title"] = t
        metadatas.append(md)

    vector_store.add_documents(chunks, embeddings, metadatas=metadatas)
    vector_store.save_store()
    after = vector_store.index.ntotal if vector_store.index is not None else 0
    return {
        "added": max(0, after - before),
        "chunks": len(chunks),
        "embedding_dim": embedding_dim,
    }


def extract_source_label(text: str) -> str:
    """Return a short source label from chunk text (Page/Slide markers if present)."""
    for line in text.splitlines():
        ls = line.strip()
        if ls.startswith("=== Page ") or ls.startswith("=== Slide "):
            return ls.strip("=").strip()
        if ls:
            # first non-empty line fallback
            return (ls[:60] + ("..." if len(ls) > 60 else ""))
    return "(kaynak)"

def label_from_meta(meta: dict, text_fallback: str) -> str:
    fn = meta.get("source_file")
    if "page_number" in meta:
        tag = f"Page {meta['page_number']}"
    elif "slide_number" in meta:
        tag = f"Slide {meta['slide_number']}"
    else:
        tag = None
    if fn and tag:
        return f"{fn} | {tag}"
    if fn:
        return fn
    if tag:
        return tag
    return extract_source_label(text_fallback)


def retrieve_and_answer(
    vector_store: FaissVectorStore,
    query: str,
    top_k: int = 5,
    *,
    use_rerank: bool = False,
    rerank_top_n: int = 5,
    min_score: float = 0.0,
    max_context_chars: int = 8000,
    abstain_message: str | None = None,
    generation_model: str | None = None,
    embedding_model: str | None = None,
    track_performance: bool = False,
) -> tuple[str, list[str], list[float], list[dict]]:
    """
    Enhanced retrieve_and_answer with optional performance tracking for academic experiments.
    """
    # Detect query language
    detected_language = detect_query_language(query)
    
    # Use language-appropriate abstain message if not provided
    if abstain_message is None:
        abstain_message = prompt_manager.get_abstain_message(detected_language)
    
    monitor = get_performance_monitor() if track_performance else None
    
    if track_performance and monitor:
        monitor.reset_metrics()
        
    # Start timing for total response
    start_time = time.perf_counter()
    
    try:
        # Retrieval phase with performance tracking
        with measure_performance('retrieval') if track_performance else nullcontext():
            selected_provider = get_selected_provider()
            if selected_provider == 'ollama':
                query_emb = generate_embeddings([query], model=embedding_model, provider='ollama')
            else:
                # For cloud providers, use local sentence transformers for embeddings
                query_emb = generate_embeddings([query], model=embedding_model, provider='sentence_transformers')
            if not query_emb:
                error_msg = prompt_manager.get_error_message(detected_language, 'embedding_failed')
                return error_msg, [], [], []
            results = vector_store.search(query_emb, k=top_k)
            if not results:
                error_msg = prompt_manager.get_error_message(detected_language, 'no_results')
                return error_msg, [], [], []

            retrieved_texts = [text for (text, _score, _meta) in results]
            scores = [float(_score) for (_text, _score, _meta) in results]
            metas = [(_meta or {}) for (_text, _score, _meta) in results]

            # Apply minimum score threshold (cosine, higher is better)
            if min_score > 0:
                filtered = [(t, s, m) for t, s, m in zip(retrieved_texts, scores, metas) if s >= min_score]
                if filtered:
                    retrieved_texts, scores, metas = zip(*filtered)
                    retrieved_texts, scores, metas = list(retrieved_texts), list(scores), list(metas)
                else:
                    return abstain_message, [], [], []

        # Set retrieval information for performance tracking
        if track_performance and monitor:
            monitor.set_retrieval_info(len(retrieved_texts), sum(len(t) for t in retrieved_texts))

        # Optional LLM-based rerank
        if use_rerank and retrieved_texts:
            # Build a concise ranking prompt
            items = "\n\n".join([f"[#{i+1}]\n{t[:1000]}" for i, t in enumerate(retrieved_texts)])
            rerank_prompt = prompt_manager.get_rerank_prompt(
                detected_language, query, items, min(rerank_top_n, len(retrieved_texts))
            )
            rerank_system_prompt = prompt_manager.get_rerank_system_prompt(detected_language)
            
            try:
                model_to_use = generation_model or app_config.OLLAMA_GENERATION_MODEL
                
                # Check provider from session state or model name
                selected_provider = get_selected_provider()
                
                if selected_provider == "groq" or is_groq_model(model_to_use):
                    # Use Model Inference Service
                    full_prompt = f"System: {rerank_system_prompt}\n\nUser: {rerank_prompt}"
                    order_raw = call_model_inference_service(
                        prompt=full_prompt,
                        model=model_to_use,
                        temperature=0.0,
                        max_tokens=64
                    )
                else:
                    # Fallback
                    order_raw = ""
                
                idxs = [int(x)-1 for x in re.findall(r"\d+", order_raw)]
                idxs = [i for i in idxs if 0 <= i < len(retrieved_texts)]
                if idxs:
                    # Reorder according to rerank list (truncate to rerank_top_n)
                    idxs = idxs[: min(rerank_top_n, len(idxs))]
                    retrieved_texts = [retrieved_texts[i] for i in idxs]
                    scores = [scores[i] for i in idxs]
                    metas = [metas[i] for i in idxs]
            except Exception:
                # If rerank fails, continue with original order
                pass
        
        # Truncate context to max_context_chars while keeping item boundaries
        context_accum = []
        total = 0
        for t in retrieved_texts:
            add_len = len(t) + (2 if context_accum else 0)  # for \n\n join
            if total + add_len > max_context_chars:
                # try to add partial of current chunk if empty context
                remain = max_context_chars - total
                if remain > 0 and not context_accum:
                    context_accum.append(t[:remain])
                    total += remain
                break
            context_accum.append(t)
            total += add_len
        if not context_accum:
            return abstain_message, [], [], []
        context_str = "\n\n".join(context_accum)

        # Generation phase with performance tracking
        with measure_performance('generation') if track_performance else nullcontext():
            system_prompt = prompt_manager.get_system_prompt(detected_language, 'rag')
            user_prompt = prompt_manager.get_user_prompt(detected_language, query, context_str)
            model_to_use = generation_model or app_config.OLLAMA_GENERATION_MODEL
            
            # Check provider from session state or model name
            selected_provider = get_selected_provider()
            
            if selected_provider == "groq" or is_groq_model(model_to_use):
                # Use Model Inference Service
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
                answer = call_model_inference_service(
                    prompt=full_prompt,
                    model=model_to_use,
                    temperature=0.3,
                    max_tokens=1024
                )
            else:
                # Fallback
                answer = "Model not configured for cloud provider."

        # Calculate total response time and update monitor
        if track_performance and monitor:
            total_time_ms = (time.perf_counter() - start_time) * 1000
            monitor.current_metrics.total_response_time_ms = total_time_ms
            monitor.current_metrics.context_length = len(context_str)

        return answer, retrieved_texts, scores, metas
        
    except Exception as e:
        error_msg = prompt_manager.get_error_message(detected_language, 'generation_error', str(e))
        if track_performance and monitor:
            total_time_ms = (time.perf_counter() - start_time) * 1000
            monitor.current_metrics.total_response_time_ms = total_time_ms
        return error_msg, [], [], []


# Context manager for null operations
class nullcontext:
    """Context manager that does nothing - for optional performance tracking."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def generate_answer_from_context(retrieved_texts: list[str], query: str, generation_model: str | None = None) -> str:
    # Detect query language
    detected_language = detect_query_language(query)
    
    context_str = "\n\n".join(retrieved_texts)
    system_prompt = prompt_manager.get_system_prompt(detected_language, 'rag')
    user_prompt = prompt_manager.get_user_prompt(detected_language, query, context_str)
    model_to_use = generation_model or app_config.OLLAMA_GENERATION_MODEL
    
    try:
        # Check provider from session state or model name
        selected_provider = get_selected_provider()
        
        if selected_provider == "groq" or is_groq_model(model_to_use):
            # Use Model Inference Service
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            return call_model_inference_service(
                prompt=full_prompt,
                model=model_to_use,
                temperature=0.3,
                max_tokens=1024
            )
        else:
            # Fallback
            return "Model not configured for cloud provider."
    except Exception as e:
        error_msg = prompt_manager.get_error_message(detected_language, 'generation_error', str(e))
        return error_msg


def direct_answer(query: str, generation_model: str | None = None) -> str:
    """
    Generate a direct LLM answer WITHOUT providing retrieved context.
    Useful to compare with RAG-constrained answer.
    """
    # Detect query language
    detected_language = detect_query_language(query)
    
    system_prompt = prompt_manager.get_system_prompt(detected_language, 'direct')
    model_to_use = generation_model or app_config.OLLAMA_GENERATION_MODEL
    
    try:
        # Check provider from session state or model name
        selected_provider = get_selected_provider()
        
        if selected_provider == "groq" or is_groq_model(model_to_use):
            # Use Model Inference Service
            full_prompt = f"System: {system_prompt}\n\nUser: {query}"
            return call_model_inference_service(
                prompt=full_prompt,
                model=model_to_use,
                temperature=0.4,
                max_tokens=512
            )
        else:
            # Fallback
            return "Model not configured for cloud provider."
    except Exception as e:
        error_msg = prompt_manager.get_error_message(detected_language, 'direct_answer_error', str(e))
        return error_msg


def generate_multiple_answers(
    vector_store: FaissVectorStore,
    query: str,
    num_answers: int = 3,
    *,
    top_k_per_answer: int = 5,
    use_rerank: bool = False,
    generation_model: str | None = None,
    embedding_model: str | None = None,
) -> list[tuple[str, list[str], list[float], list[dict]]]:
    """
    Aynı soru için farklı kaynak kombinasyonları kullanarak birden fazla cevap üretir.
    
    Returns:
        List of (answer, sources, scores, metas) tuples
    """
    results = []
    
    # Detect query language
    detected_language = detect_query_language(query)
    
    # Daha fazla kaynak çek
    total_sources_needed = num_answers * top_k_per_answer + 5
    
    try:
        # Query embedding
        selected_provider = get_selected_provider()
        if selected_provider == 'ollama':
            query_emb = generate_embeddings([query], model=embedding_model, provider='ollama')
        else:
            # For cloud providers, use local sentence transformers for embeddings
            query_emb = generate_embeddings([query], model=embedding_model, provider='sentence_transformers')
        if not query_emb:
            return []
        
        # Retrieve more sources than needed
        all_results = vector_store.search(query_emb, k=total_sources_needed)
        if not all_results:
            return []

        all_texts = [text for (text, _score, _meta) in all_results]
        all_scores = [float(_score) for (_text, _score, _meta) in all_results]
        all_metas = [(_meta or {}) for (_text, _score, _meta) in all_results]
        
        # Her cevap için farklı kaynak setleri oluştur
        for i in range(num_answers):
            # Farklı başlangıç noktalarından kaynak seç
            start_idx = i * (len(all_texts) // num_answers)
            end_idx = start_idx + top_k_per_answer
            
            if start_idx >= len(all_texts):
                break
                
            # Bu cevap için kaynakları seç
            answer_texts = all_texts[start_idx:end_idx]
            answer_scores = all_scores[start_idx:end_idx]
            answer_metas = all_metas[start_idx:end_idx]
            
            if not answer_texts:
                continue
            
            # Context oluştur
            context_str = "\n\n".join(answer_texts)
            
            # Cevap üret
            system_prompt = prompt_manager.get_system_prompt(detected_language, 'rag')
            user_prompt = prompt_manager.get_user_prompt(detected_language, query, context_str)
            model_to_use = generation_model or app_config.OLLAMA_GENERATION_MODEL
            
            # Variety için farklı temperature değerleri kullan
            temperatures = [0.3, 0.5, 0.7]
            temperature = temperatures[i % len(temperatures)]
            
            # Check provider from session state or model name
            selected_provider = get_selected_provider()
            
            if selected_provider == "groq" or is_groq_model(model_to_use):
                # Use Model Inference Service
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
                answer = call_model_inference_service(
                    prompt=full_prompt,
                    model=model_to_use,
                    temperature=temperature,
                    max_tokens=1024
                )
            else:
                # Fallback
                answer = "Model not configured for cloud provider."
            
            if answer:
                results.append((answer, answer_texts, answer_scores, answer_metas))
        
        return results
        
    except Exception as e:
        # Hata durumunda boş liste döndür
        return []


def execute_prompt_command_with_rag(
    vector_store: FaissVectorStore,
    command: str,
    param_values: dict,
    *,
    top_k: int = 5,
    use_rerank: bool = True,
    generation_model: str | None = None,
    embedding_model: str | None = None,
    session_id: str | None = None,
    record_performance: bool = True
) -> tuple[str, list[str], list[float], list[dict], dict]:
    """
    Prompt komutunu RAG sistemi ile çalıştırır ve performansını izler.
    
    Returns:
        Tuple of (answer, sources, scores, metas, performance_data)
    """
    start_time = datetime.now()
    performance_data = {
        "execution_time": 0.0,
        "command": command,
        "success": False,
        "error": None
    }
    
    try:
        # Komutu çalıştır ve prompt oluştur
        filled_prompt, error = teacher_prompt_manager.execute_prompt_command(
            command, **param_values
        )
        
        if error:
            performance_data["error"] = error
            return "", [], [], [], performance_data
        
        # Test query oluştur (parametre değerlerinden)
        test_query = " ".join([f"{k}: {v}" for k, v in param_values.items()]) if param_values else "Test sorgusu"
        
        # RAG ile cevap al
        answer, sources, scores, metas = retrieve_and_answer(
            vector_store,
            test_query,
            top_k=top_k,
            use_rerank=use_rerank,
            generation_model=generation_model,
            embedding_model=embedding_model
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        performance_data["execution_time"] = execution_time
        performance_data["success"] = True
        
        # Performansı kaydet (isteğe bağlı)
        if record_performance:
            # Komut ID'sini al
            commands = teacher_prompt_manager.get_prompt_commands()
            command_obj = next((cmd for cmd in commands if cmd.command == command), None)
            
            if command_obj:
                performance = PromptPerformance(
                    prompt_id=command_obj.id,
                    execution_time=execution_time,
                    user_rating=None,  # Kullanıcı tarafından doldurulacak
                    response_quality=None,
                    educational_effectiveness=None,
                    engagement_score=None,
                    timestamp=datetime.now().isoformat(),
                    session_id=session_id or "unknown",
                    user_feedback=None
                )
                
                # Performans kaydını veritabanına kaydetme işlemi otomatik değerlendirme sonucu eklenebilir
                performance_data["performance_record"] = performance
        
        return answer, sources, scores, metas, performance_data
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        performance_data["execution_time"] = execution_time
        performance_data["error"] = str(e)
        return "", [], [], [], performance_data


def analyze_prompt_effectiveness(
    answer: str,
    sources: list[str],
    scores: list[float],
    query_params: dict,
    execution_time: float
) -> dict:
    """
    Prompt'un etkinliğini analiz eder ve otomatik değerlendirme yapar.
    
    Returns:
        Dictionary with effectiveness metrics
    """
    analysis = {
        "response_length": len(answer),
        "source_count": len(sources),
        "avg_relevance_score": sum(scores) / len(scores) if scores else 0.0,
        "max_relevance_score": max(scores) if scores else 0.0,
        "execution_time": execution_time,
        "estimated_quality": 0.0,
        "educational_indicators": {}
    }
    
    # Cevap kalitesi tahmini (basit heuristik)
    quality_score = 0.0
    
    # 1. Cevap uzunluğu kontrolü (çok kısa veya çok uzun olmamalı)
    if 50 <= len(answer) <= 2000:
        quality_score += 0.3
    elif 20 <= len(answer) < 50 or 2000 < len(answer) <= 3000:
        quality_score += 0.1
    
    # 2. Kaynak kullanımı
    if len(sources) >= 2:
        quality_score += 0.2
    elif len(sources) == 1:
        quality_score += 0.1
    
    # 3. Alaka düzeyi
    if analysis["avg_relevance_score"] > 0.7:
        quality_score += 0.3
    elif analysis["avg_relevance_score"] > 0.5:
        quality_score += 0.2
    elif analysis["avg_relevance_score"] > 0.3:
        quality_score += 0.1
    
    # 4. Çalışma süresi (çok hızlı veya çok yavaş olmamalı)
    if 1.0 <= execution_time <= 10.0:
        quality_score += 0.2
    elif 0.5 <= execution_time < 1.0 or 10.0 < execution_time <= 20.0:
        quality_score += 0.1
    
    analysis["estimated_quality"] = min(quality_score, 1.0)
    
    # Eğitsel göstergeler
    educational_indicators = {}
    
    # Açıklayıcı kelimeler
    explanatory_words = ["çünkü", "nedeni", "sebep", "örneğin", "yani", "dolayısıyla",
                        "because", "reason", "example", "therefore", "thus", "hence"]
    explanatory_count = sum(1 for word in explanatory_words if word in answer.lower())
    educational_indicators["explanatory_language"] = explanatory_count > 0
    
    # Yapılandırılmış içerik (listeler, adımlar)
    structured_indicators = ["1.", "2.", "3.", "•", "-", ":", "birinci", "ikinci", "üçüncü",
                           "first", "second", "third", "step", "adım"]
    structured_count = sum(1 for indicator in structured_indicators if indicator in answer)
    educational_indicators["structured_content"] = structured_count > 2
    
    # Soru sorma (öğrenci katılımı)
    question_indicators = ["?", "düşünün", "nedir", "nasıl", "neden", "what", "how", "why", "think"]
    question_count = sum(1 for indicator in question_indicators if indicator in answer.lower())
    educational_indicators["interactive_elements"] = question_count > 0
    
    analysis["educational_indicators"] = educational_indicators
    
    return analysis


def get_prompt_performance_summary(days: int = 7) -> dict:
    """
    Son X gün için prompt performans özetini getir.
    """
    try:
        analytics = teacher_prompt_manager.get_prompt_analytics(days=days)
        
        summary = {
            "total_executions": analytics.get("total_executions", 0),
            "unique_prompts": analytics.get("unique_prompts_used", 0),
            "avg_execution_time": analytics.get("avg_execution_time", 0.0),
            "avg_user_rating": analytics.get("avg_user_rating", 0.0),
            "avg_response_quality": analytics.get("avg_response_quality", 0.0),
            "avg_educational_effectiveness": analytics.get("avg_educational_effectiveness", 0.0),
            "popular_commands": analytics.get("popular_commands", [])
        }
        
        return summary
        
    except Exception as e:
        return {
            "error": str(e),
            "total_executions": 0
        }


def batch_test_prompts(
    vector_store: FaissVectorStore,
    test_cases: list[dict],
    *,
    generation_model: str | None = None
) -> list[dict]:
    """
    Birden fazla prompt komutunu toplu olarak test eder.
    
    test_cases format:
    [
        {
            "command": "/basit-anlat",
            "params": {"topic": "fotosentez", "grade_level": "5. sınıf"},
            "expected_keywords": ["güneş", "yaprak", "oksijen"]  # optional
        }
    ]
    """
    results = []
    
    for i, test_case in enumerate(test_cases):
        try:
            command = test_case["command"]
            params = test_case["params"]
            expected_keywords = test_case.get("expected_keywords", [])
            
            # Test çalıştır
            answer, sources, scores, metas, performance = execute_prompt_command_with_rag(
                vector_store=vector_store,
                command=command,
                param_values=params,
                generation_model=generation_model,
                record_performance=False  # Batch testlerde kaydetme
            )
            
            # Analiz yap
            effectiveness = analyze_prompt_effectiveness(
                answer, sources, scores, params, performance["execution_time"]
            )
            
            # Anahtar kelime kontrolü
            keyword_matches = []
            if expected_keywords:
                for keyword in expected_keywords:
                    if keyword.lower() in answer.lower():
                        keyword_matches.append(keyword)
            
            test_result = {
                "test_id": i + 1,
                "command": command,
                "params": params,
                "success": performance["success"],
                "answer_length": len(answer),
                "source_count": len(sources),
                "execution_time": performance["execution_time"],
                "estimated_quality": effectiveness["estimated_quality"],
                "keyword_matches": keyword_matches,
                "keyword_match_rate": len(keyword_matches) / len(expected_keywords) if expected_keywords else 0.0,
                "error": performance.get("error")
            }
            
            results.append(test_result)
            
        except Exception as e:
            results.append({
                "test_id": i + 1,
                "command": test_case.get("command", "unknown"),
                "success": False,
                "error": str(e)
            })
    
    return results