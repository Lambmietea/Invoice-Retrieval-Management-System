"""ChromaDB vector store — embedding, upsert, and similarity search.

Uses ``shibing624/text2vec-base-chinese`` for Chinese-optimised embeddings.
"""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .models import InvoiceRecord

_DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_data")
_COLLECTION_NAME = "invoices"

# ---------------------------------------------------------------------------
# Chinese embedding function (text2vec-base-chinese)
# ---------------------------------------------------------------------------

_CHINESE_MODEL_NAME = "shibing624/text2vec-base-chinese"


def _get_chinese_embedding_fn():
    """Create a ChromaDB-compatible embedding function using text2vec-base-chinese."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=_CHINESE_MODEL_NAME)


class VectorStore:
    """Wraps a persistent ChromaDB collection for invoice semantic search."""

    def __init__(self, persist_dir: str | None = None) -> None:
        path = persist_dir or os.getenv("MODULE_B_CHROMA_DIR", _DEFAULT_PERSIST_DIR)
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._ef = _get_chinese_embedding_fn()
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _build_document(rec: InvoiceRecord) -> str:
        """Create a retrieval document string from an InvoiceRecord."""
        return rec.raw_text or (
            f"发票号码 {rec.invoice_id} 销售方 {rec.vendor or '未知'}, "
            f"金额 {rec.amount}, 日期 {rec.date or '未知'}, "
            f"类型 {rec.category or '未知'}"
        )

    @staticmethod
    def _build_metadata(rec: InvoiceRecord) -> dict[str, Any]:
        return {
            "invoice_id": rec.invoice_id,
            "date": rec.date or "",
            "vendor": rec.vendor or "",
            "amount": rec.amount,
            "category": rec.category or "",
            "source_file": rec.source_file or "",
        }

    # -- public API --------------------------------------------------------

    def add(self, rec: InvoiceRecord) -> None:
        """Upsert a single invoice into the vector collection."""
        self._collection.upsert(
            ids=[rec.invoice_id],
            documents=[self._build_document(rec)],
            metadatas=[self._build_metadata(rec)],
        )

    def add_many(self, records: list[InvoiceRecord]) -> int:
        """Batch upsert. Returns count processed."""
        if not records:
            return 0
        # ChromaDB batch limit is ~5461; chunk if needed
        batch_size = 5000
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self._collection.upsert(
                ids=[r.invoice_id for r in batch],
                documents=[self._build_document(r) for r in batch],
                metadatas=[self._build_metadata(r) for r in batch],
            )
            total += len(batch)
        return total

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search. Returns list of dicts with keys:
          id, distance, document, metadata

        Args:
            query_text: The search query string.
            top_k: Maximum number of results to return.
            max_distance: If set, discard results whose cosine distance exceeds
                this threshold (cosine space: 0 = identical, 2 = opposite).
                Recommended range: 0.4–0.8.
        """
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for idx in range(len(ids)):
            dist = dists[idx] if idx < len(dists) else None
            # Skip results that exceed the relevance threshold
            if max_distance is not None and dist is not None and dist > max_distance:
                continue
            out.append({
                "id": ids[idx],
                "distance": dist,
                "document": docs[idx] if idx < len(docs) else "",
                "metadata": metas[idx] if idx < len(metas) else {},
            })
        return out

    def search_with_ids(
        self,
        query_text: str,
        candidate_ids: list[str],
        top_k: int = 10,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search constrained to a set of candidate IDs (for hybrid retrieval).
        Uses ChromaDB 'where' filter on invoice_id.
        """
        if not candidate_ids:
            return []

        where_filter: dict[str, Any]
        if len(candidate_ids) == 1:
            where_filter = {"invoice_id": candidate_ids[0]}
        else:
            where_filter = {"invoice_id": {"$in": candidate_ids}}

        n = min(top_k, len(candidate_ids))
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for idx in range(len(ids)):
            dist = dists[idx] if idx < len(dists) else None
            if max_distance is not None and dist is not None and dist > max_distance:
                continue
            out.append({
                "id": ids[idx],
                "distance": dist,
                "document": docs[idx] if idx < len(docs) else "",
                "metadata": metas[idx] if idx < len(metas) else {},
            })
        return out

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """Delete all documents — useful for tests."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
