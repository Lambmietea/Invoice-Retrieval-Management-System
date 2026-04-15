"""Service layer — public API consumed by FastAPI routes and Module C.

Public interface (per REQUIREMENT.md):
  - index_invoice(invoice)        → dict
  - index_invoices(invoices)      → dict
  - query_invoices(user_query)    → RetrievalResult
  - get_all_invoices()            → list[InvoiceRecord]
"""

from __future__ import annotations

from .models import InvoiceRecord, RetrievalResult
from .query_parser import parse_query
from .retriever import HybridRetriever
from .storage import SQLiteStorage
from .vector_store import VectorStore


class InvoiceService:
    """Unified service that wires storage, vector store, parser, and retriever."""

    def __init__(
        self,
        db_url: str | None = None,
        chroma_dir: str | None = None,
    ) -> None:
        self._storage = SQLiteStorage(db_url=db_url)
        self._vector = VectorStore(persist_dir=chroma_dir)
        self._retriever = HybridRetriever(self._storage, self._vector)

    # ==================================================================
    # 入库接口 (Indexing)
    # ==================================================================

    def index_invoice(self, invoice: InvoiceRecord) -> dict:
        """Index a single invoice into both SQLite and ChromaDB."""
        try:
            self._storage.insert(invoice)
            self._vector.add(invoice)
            self._invalidate_vendor_cache()
            return {
                "status": "success",
                "invoice_id": invoice.invoice_id,
                "message": f"Invoice {invoice.invoice_id} indexed.",
            }
        except Exception as e:
            return {
                "status": "error",
                "invoice_id": invoice.invoice_id,
                "message": str(e),
            }

    def index_invoices(self, invoices: list[InvoiceRecord]) -> dict:
        """Batch index invoices."""
        errors: list[str] = []
        success_count = 0
        for inv in invoices:
            res = self.index_invoice(inv)
            if res["status"] == "success":
                success_count += 1
            else:
                errors.append(f"{inv.invoice_id}: {res['message']}")
        return {
            "status": "success" if not errors else "partial",
            "indexed_count": success_count,
            "errors": errors,
        }

    # ==================================================================
    # 检索接口 (Retrieval)
    # ==================================================================

    def query_invoices(self, user_query: str) -> RetrievalResult:
        """
        Natural-language query → hybrid retrieval.

        Internally: parse_query → retriever.retrieve
        Passes known vendor names to the parser for dynamic vendor matching.
        """
        if not user_query or not user_query.strip():
            return RetrievalResult(
                query=user_query or "",
                matched_invoices=[],
                summary_context="Empty query.",
                query_type="empty",
                status="empty",
            )
        # Collect known vendors so the rule-based parser can match them
        known_vendors = self._get_known_vendors()
        parsed = parse_query(user_query, known_vendors=known_vendors)
        return self._retriever.retrieve(parsed)

    def get_all_invoices(self) -> list[InvoiceRecord]:
        """Return all invoices from SQLite (for Module C bulk tasks like dedup)."""
        return self._storage.get_all()

    # ==================================================================
    # Utilities
    # ==================================================================

    @property
    def invoice_count(self) -> int:
        return self._storage.count()

    def _get_known_vendors(self) -> list[str]:
        """Return deduplicated list of vendor names currently in the database.

        Used by the query parser for dynamic vendor name matching.
        Results are cached for the lifetime of the service instance to avoid
        repeated full-table scans on every query.
        """
        if not hasattr(self, "_vendor_cache") or self._vendor_cache is None:
            all_recs = self._storage.get_all()
            seen: set[str] = set()
            vendors: list[str] = []
            for r in all_recs:
                v = r.vendor
                if v and v not in seen:
                    seen.add(v)
                    vendors.append(v)
            self._vendor_cache: list[str] | None = vendors
        return self._vendor_cache

    def _invalidate_vendor_cache(self) -> None:
        """Clear cached vendor list — call after indexing new invoices."""
        self._vendor_cache = None

    def clear_all(self) -> None:
        """Wipe both stores — for tests."""
        self._storage.clear()
        self._vector.clear()
        self._invalidate_vendor_cache()
