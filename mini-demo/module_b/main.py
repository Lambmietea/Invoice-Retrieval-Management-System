"""FastAPI application entry point and route definitions.

Endpoints (per REQUIREMENT.md):
  POST /api/v1/invoices           → index single or batch invoices
  GET  /api/v1/invoices           → get all invoices
  POST /api/v1/invoices/search    → natural-language hybrid search
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .models import InvoiceRecord, RetrievalResult, SearchRequest, IndexResponse, BatchIndexResponse
from .service import InvoiceService

# ---------------------------------------------------------------------------
# App & service singleton
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Module B — Invoice Hybrid-RAG Retrieval",
    version="0.1.0",
    description="COMP6708 Assignment 3 — intelligent retrieval micro-service",
)

_service: InvoiceService | None = None


def get_service() -> InvoiceService:
    global _service
    if _service is None:
        _service = InvoiceService()
    return _service


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/v1/invoices", response_model=BatchIndexResponse)
async def index_invoices(invoices: list[InvoiceRecord]) -> BatchIndexResponse:
    """Index one or more invoices (both SQLite + ChromaDB)."""
    svc = get_service()
    result = svc.index_invoices(invoices)
    return BatchIndexResponse(**result)


@app.get("/api/v1/invoices", response_model=list[InvoiceRecord])
async def get_all_invoices() -> list[InvoiceRecord]:
    """Return all indexed invoices."""
    svc = get_service()
    return svc.get_all_invoices()


@app.post("/api/v1/invoices/search", response_model=RetrievalResult)
async def search_invoices(req: SearchRequest) -> RetrievalResult:
    """Natural-language hybrid search (SQL + vector)."""
    svc = get_service()
    return svc.query_invoices(req.query)


@app.get("/health")
async def health() -> dict[str, str]:
    svc = get_service()
    return {"status": "ok", "invoice_count": str(svc.invoice_count)}


# ---------------------------------------------------------------------------
# Startup: auto-load data/*.json from repo if DB is empty
# ---------------------------------------------------------------------------


def _find_data_dir() -> Path | None:
    """Look for data/*.json in several likely locations."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "data",       # repo root /data
        Path(__file__).resolve().parent.parent / "data",              # module_b/data
        Path.cwd() / "data",
    ]
    for d in candidates:
        if d.is_dir() and list(d.glob("*.json")):
            return d
    return None


@app.on_event("startup")
async def _auto_load_data() -> None:
    """Auto-index data/*.json on first startup when the DB is empty."""
    svc = get_service()
    if svc.invoice_count > 0:
        return

    data_dir = _find_data_dir()
    if not data_dir:
        print("[module_b] No data/ directory found — skipping auto-load.")
        return

    json_files = sorted(data_dir.glob("*.json"))
    records: list[InvoiceRecord] = []
    for fp in json_files:
        try:
            raw = json.loads(fp.read_bytes().decode("utf-8"))
            records.append(InvoiceRecord.from_chinese_dict(raw))
        except Exception as e:
            print(f"[module_b] Skipping {fp.name}: {e}")

    if records:
        result = svc.index_invoices(records)
        print(
            f"[module_b] Auto-loaded {result['indexed_count']} invoices from {data_dir}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli() -> None:
    """Run the FastAPI server via CLI: `module-b` or `python -m module_b.main`."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Module B — Invoice Retrieval Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "module_b.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    cli()
