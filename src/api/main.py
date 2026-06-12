"""
FastAPI backend — REST endpoints + WebSocket streaming.
Sanitization errors return HTTP 422 with clear messages.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, field_validator

from config.settings import get_settings
from src.orchestrator import EnterpriseRAG
from src.utils.sanitizer import (
    SanitizationError,
    sanitize_query,
    sanitize_url,
    sanitize_text,
    sanitize_file,
    MAX_QUERY_LENGTH,
)

settings = get_settings()
rag: EnterpriseRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    logger.info("Starting EnterpriseRAG API...")
    rag = EnterpriseRAG()
    yield
    logger.info("Shutting down EnterpriseRAG API...")


app = FastAPI(
    title=settings.app_name,
    description="Production-grade multi-source RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global error handler for SanitizationError ───────────────────────────────

@app.exception_handler(SanitizationError)
async def sanitization_error_handler(request: Request, exc: SanitizationError):
    return JSONResponse(status_code=422, content={"detail": str(exc), "type": "sanitization_error"})


# ── Pydantic Models with validation ──────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    use_hyde: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        return sanitize_query(v)   # raises SanitizationError → caught by handler above


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    query_type: str
    is_grounded: bool
    latencies: dict[str, float]
    cached: bool = False


class IngestURLRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        return sanitize_url(v)


class IngestTextRequest(BaseModel):
    text: str
    source_name: str = "manual"

    @field_validator("text", "source_name")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class FeedbackRequest(BaseModel):
    run_id: str
    score: float   # 1.0 = positive, 0.0 = negative
    comment: str = ""

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.get("/stats")
async def stats():
    if not rag:
        raise HTTPException(503, "RAG not initialised")
    return rag.get_stats()


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not rag:
        raise HTTPException(503, "RAG not initialised")
    try:
        result = rag.query(req.query, use_hyde=req.use_hyde)
        return QueryResponse(**result)
    except SanitizationError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(500, "Internal server error")


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        raw_query = data.get("query", "")
        use_hyde = data.get("use_hyde", True)

        try:
            clean_query = sanitize_query(raw_query)
        except SanitizationError as e:
            await websocket.send_json({"type": "error", "data": str(e)})
            return

        async for chunk in rag.astream(clean_query, use_hyde=use_hyde):
            await websocket.send_json(chunk)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "data": "Internal server error"})


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    if not rag:
        raise HTTPException(503, "RAG not initialised")

    content = await file.read()

    try:
        clean_filename = sanitize_file(file.filename or "", len(content))
    except SanitizationError as e:
        raise HTTPException(422, str(e))

    # tmp_path = Path(f"/tmp/{clean_filename}")
    tmp_path = Path(tempfile.gettempdir()) / clean_filename
    try:
        tmp_path.write_bytes(content)
        result = rag.ingest_file(tmp_path, original_filename=clean_filename)
        return {"status": "success", "file": clean_filename, **result}
    except SanitizationError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"File ingestion error: {e}")
        raise HTTPException(500, "File ingestion failed")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ingest/url")
async def ingest_url(req: IngestURLRequest):
    if not rag:
        raise HTTPException(503, "RAG not initialised")
    try:
        result = rag.ingest_url(req.url)
        return {"status": "success", "url": req.url, **result}
    except SanitizationError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"URL ingestion error: {e}")
        raise HTTPException(500, "URL ingestion failed")


@app.post("/ingest/text")
async def ingest_text(req: IngestTextRequest):
    if not rag:
        raise HTTPException(503, "RAG not initialised")
    try:
        result = rag.ingest_text(req.text, req.source_name)
        return {"status": "success", **result}
    except SanitizationError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"Text ingestion error: {e}")
        raise HTTPException(500, "Text ingestion failed")


# ── Feedback ──────────────────────────────────────────────────────────────────

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if not rag:
        raise HTTPException(503, "RAG not initialised")
    try:
        rag.log_feedback(req.run_id, req.score, req.comment)
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"Feedback logging failed: {e}")
        return {"status": "ok"}   # non-critical, don't fail user


# ── Evaluation ────────────────────────────────────────────────────────────────

@app.post("/evaluate")
async def evaluate(payload: dict[str, Any]):
    if not rag:
        raise HTTPException(503, "RAG not initialised")
    from src.observability.evaluator import RAGASEvaluator
    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate_single(
        query=payload.get("query", ""),
        answer=payload.get("answer", ""),
        contexts=payload.get("contexts", []),
        ground_truth=payload.get("ground_truth"),
    )
    return scores


# ── Cache management ──────────────────────────────────────────────────────────

@app.post("/cache/clear")
async def clear_cache():
    if not rag:
        raise HTTPException(503, "RAG not initialised")
    rag.query_cache.invalidate_all()
    return {"status": "ok", "message": "Query cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
