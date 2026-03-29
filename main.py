"""
main.py — FastAPI RAG application
Endpoints:
  POST /ingest        — upload + embed documents
  POST /query         — ask a question
  GET  /documents     — list indexed documents
  DELETE /documents/{filename} — remove a document
  GET  /health        — health check
  GET  /              — serve frontend
"""

import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from embeddings import VectorStore, get_embeddings, get_embed_text
from generation import generate_answer, route_query, execute_analytical_query
from ingestion import ingest_file

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── App + Store ───────────────────────────────────────────────────────────────

app   = FastAPI(title="RAG Agent", version="1.0.0")
store = VectorStore()

# ── Rate Limiting (simple in-memory) ─────────────────────────────────────────

_request_log: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 30   # max requests
RATE_LIMIT_WINDOW   = 60   # per 60 seconds


def check_rate_limit(client_ip: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    _request_log[client_ip] = [t for t in _request_log[client_ip] if t > window_start]
    if len(_request_log[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    _request_log[client_ip].append(now)


# ── Auth ──────────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(None)):
    """Simple API key auth. Key set via RAG_API_KEY env var."""
    expected = os.environ.get("RAG_API_KEY", "dev-secret-key")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set.")
    return key


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set.")
    return token


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 15
    min_score: float = 0.35
    filter_document: str | None = None
    groq_api_key: str | None = None
    hf_token: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return Path("index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "documents":   store.list_documents(),
        "total_chunks": store.chunk_count(),
    }


@app.post("/ingest")
async def ingest_documents(
    request: Request,
    files: list[UploadFile] = File(...),
    hf_token: str = Form(...),
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)
    check_rate_limit(request.client.host)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per upload.")

    results = []
    all_new_chunks = []

    for upload in files:
        filename = upload.filename or "unknown"
        logger.info(f"Ingesting: {filename}")

        try:
            content = await upload.read()
            chunks, error = ingest_file(filename, content)

            if error:
                results.append({"file": filename, "status": "error", "message": error})
                logger.warning(f"Ingest failed for '{filename}': {error}")
                continue

            if filename.lower().endswith(".csv"):
                csv_path = Path("data/tables") / filename
                with open(csv_path, "wb") as f:
                    f.write(content)

            all_new_chunks.extend(chunks)
            results.append({
                "file":   filename,
                "status": "queued",
                "chunks": len(chunks),
            })

        except Exception as e:
            logger.error(f"Unexpected error ingesting '{filename}': {e}")
            results.append({"file": filename, "status": "error", "message": str(e)})

    # Embed all new chunks in one batched call
    if all_new_chunks:
        try:
            texts      = [get_embed_text(c) for c in all_new_chunks]  # uses lowercased text_normalized
            embeddings = get_embeddings(texts, hf_token)
            store.add_chunks(all_new_chunks, embeddings)

            # Update results with embedded status
            embedded_docs = set(c["document"] for c in all_new_chunks)
            for r in results:
                if r["file"] in embedded_docs and r["status"] == "queued":
                    r["status"] = "success"

            logger.info(f"Embedded {len(all_new_chunks)} chunks across {len(embedded_docs)} files")

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            for r in results:
                if r["status"] == "queued":
                    r["status"] = "error"
                    r["message"] = f"Embedding failed: {e}"

    return {
        "results":    results,
        "total_chunks_in_store": store.chunk_count(),
        "documents":  store.list_documents(),
    }


@app.post("/query")
async def query_documents(req: QueryRequest, request: Request, x_api_key: str = Header(None)):
    verify_api_key(x_api_key)
    check_rate_limit(request.client.host)

    # Validate query
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 chars).")

    # Resolve credentials
    groq_key = req.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    hf_token = req.hf_token     or os.environ.get("HF_TOKEN", "")

    if not groq_key:
        raise HTTPException(status_code=400, detail="Groq API key required.")
    if not hf_token:
        raise HTTPException(status_code=400, detail="HuggingFace token required.")

    if store.chunk_count() == 0:
        return JSONResponse({
            "answer":     "No documents have been ingested yet. Please upload documents first.",
            "confidence": 0.0,
            "citations":  [],
            "status":     "no_answer",
            "reasoning":  "Vector store is empty.",
        })

    logger.info(f"Query: '{query[:80]}' | top_k={req.top_k} | filter={req.filter_document}")

    # ── ROUTING ───────────────────────────────────────────────────────────────
    intent = route_query(query, groq_key)
    logger.info(f"Intent classified as: {intent}")

    if intent == "ANALYTICAL":
        analytical_res = execute_analytical_query(query, "data/tables", groq_key)
        if analytical_res["status"] == "success":
            logger.info("Successfully executed analytical query via Pandas.")
            return JSONResponse(analytical_res)
        logger.warning(f"Analytical execution failed, falling back to Vector Search... ({analytical_res.get('reasoning')})")

    # ── RETRIEVAL ─────────────────────────────────────────────────────────────
    try:
        query_normalized = query.lower().strip()  # normalize to match lowercased chunks
        query_emb = get_embeddings([query_normalized], hf_token)[0]
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    chunks = store.search(
        query_embedding=query_emb,
        top_k=req.top_k,
        min_score=req.min_score,
        filter_document=req.filter_document,
    )

    logger.info(
        f"Retrieved {len(chunks)} chunks | "
        f"top score: {chunks[0]['similarity_score'] if chunks else 'N/A'}"
    )

    # ── GENERATION ────────────────────────────────────────────────────────────
    result = generate_answer(query, chunks, groq_key)

    logger.info(f"Answer status: {result['status']} | confidence: {result['confidence']}")

    return JSONResponse(result)


@app.get("/documents")
async def list_documents(x_api_key: str = Header(None)):
    verify_api_key(x_api_key)
    docs = store.list_documents()
    return {
        "documents":   docs,
        "total_chunks": store.chunk_count(),
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str, x_api_key: str = Header(None)):
    verify_api_key(x_api_key)
    
    if filename.lower() == "all":
        store.clear()
        tables_dir = Path("data/tables")
        if tables_dir.exists():
            for f in tables_dir.glob("*.csv"):
                f.unlink()
        logger.info("Deleted all documents")
        return {"status": "deleted", "document": "all"}

    if filename not in store.list_documents():
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in store.")
    store.remove_document(filename)
    csv_path = Path("data/tables") / filename
    if csv_path.exists():
        csv_path.unlink()
    logger.info(f"Deleted document: {filename}")
    return {"status": "deleted", "document": filename}