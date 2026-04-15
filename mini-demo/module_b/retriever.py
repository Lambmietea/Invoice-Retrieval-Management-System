"""Hybrid retriever — combines SQL filtering with vector semantic search.

Three retrieval modes (auto-selected based on parsed query):
  1. Pure filter  — query has structured filters but no semantic intent
  2. Pure semantic — query is fuzzy / semantic only
  3. Hybrid        — SQL narrows candidates, then vector reranks (most common)

The semantic query from the parser is a single high-density phrase (not a
keyword list).  The retriever applies two layers of precision control:
  - **Absolute distance threshold** — hard ceiling on cosine distance.
  - **Score-gap cut-off** — detects a sudden distance jump between adjacent
    results and truncates everything after the gap.  This adapts dynamically
    to each query's score distribution.
"""

from __future__ import annotations

from .models import InvoiceRecord, RetrievalResult
from .query_parser import ParsedQuery
from .storage import SQLiteStorage
from .vector_store import VectorStore


class HybridRetriever:
    """Core retrieval engine combining structured + vector search."""

    # Hard ceiling: any result with cosine distance > this is always discarded.
    DEFAULT_MAX_DISTANCE: float = 0.50

    # Score-gap cut-off: if the distance jump between two adjacent results
    # exceeds this value, all subsequent results are discarded.  This catches
    # the "relevance cliff" where the top few results are meaningful and the
    # rest are noise that merely passed the absolute threshold.
    SCORE_GAP_THRESHOLD: float = 0.04

    def __init__(self, storage: SQLiteStorage, vector_store: VectorStore) -> None:
        self._storage = storage
        self._vector = vector_store

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def retrieve(self, parsed: ParsedQuery, top_k: int = 10) -> RetrievalResult:
        """
        Execute hybrid retrieval based on parsed query.

        Decision logic:
          - has filters + has semantic → hybrid (SQL filter → vector rerank)
          - has filters only           → pure SQL filter
          - has semantic only          → pure vector search
          - neither                    → return all (capped at top_k)
        """
        has_f = parsed.has_filters
        has_s = parsed.has_semantic

        if has_f and has_s:
            return self._hybrid(parsed, top_k)
        elif has_f:
            return self._filter_only(parsed, top_k)
        elif has_s:
            return self._semantic_only(parsed, top_k)
        else:
            all_recs = self._storage.get_all()[:top_k]
            return self._build_result(
                query=parsed.original,
                invoices=all_recs,
                query_type="all",
            )

    # ------------------------------------------------------------------
    # Private strategies
    # ------------------------------------------------------------------

    def _filter_only(self, parsed: ParsedQuery, top_k: int) -> RetrievalResult:
        """Pure structured SQL filter."""
        records = self._storage.filter(parsed.filters)
        return self._build_result(
            query=parsed.original,
            invoices=records[:top_k],
            query_type="filter",
        )

    def _semantic_only(self, parsed: ParsedQuery, top_k: int) -> RetrievalResult:
        """Pure vector similarity search with score-gap cut-off."""
        hits = self._vector.search(
            parsed.semantic_query,
            top_k=top_k,
            max_distance=self.DEFAULT_MAX_DISTANCE,
        )
        hits = self._apply_score_gap_cutoff(hits)
        hit_ids = [h["id"] for h in hits]
        all_map = {r.invoice_id: r for r in self._storage.get_all()}
        records = [all_map[iid] for iid in hit_ids if iid in all_map]
        return self._build_result(
            query=parsed.original,
            invoices=records,
            query_type="semantic",
        )

    def _hybrid(self, parsed: ParsedQuery, top_k: int) -> RetrievalResult:
        """SQL filter first, then vector rerank within candidates.

        Only results that pass BOTH the SQL filter AND the vector relevance
        threshold (+ score-gap cut-off) are returned.
        """
        candidates = self._storage.filter(parsed.filters)
        if not candidates:
            return self._build_result(
                query=parsed.original,
                invoices=[],
                query_type="filter+semantic",
            )

        candidate_ids = [c.invoice_id for c in candidates]
        hits = self._vector.search_with_ids(
            parsed.semantic_query,
            candidate_ids=candidate_ids,
            top_k=top_k,
            max_distance=self.DEFAULT_MAX_DISTANCE,
        )
        hits = self._apply_score_gap_cutoff(hits)
        id_to_rec = {c.invoice_id: c for c in candidates}
        reranked = [id_to_rec[h["id"]] for h in hits if h["id"] in id_to_rec]

        return self._build_result(
            query=parsed.original,
            invoices=reranked[:top_k],
            query_type="filter+semantic",
        )

    # ------------------------------------------------------------------
    # Score-gap cut-off
    # ------------------------------------------------------------------

    def _apply_score_gap_cutoff(
        self, hits: list[dict],
    ) -> list[dict]:
        """Truncate results at the first significant distance gap.

        Walks through the distance-sorted hit list.  When the gap between
        two consecutive results exceeds ``SCORE_GAP_THRESHOLD``, everything
        from that point onward is discarded — those documents are on the
        far side of a relevance cliff and are very likely noise.

        Always keeps at least the first result (if any).
        """
        if len(hits) <= 1:
            return hits

        kept: list[dict] = [hits[0]]
        for i in range(1, len(hits)):
            prev_dist = hits[i - 1].get("distance") or 0.0
            curr_dist = hits[i].get("distance") or 0.0
            gap = curr_dist - prev_dist
            if gap >= self.SCORE_GAP_THRESHOLD:
                break  # relevance cliff — discard the rest
            kept.append(hits[i])
        return kept

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        query: str,
        invoices: list[InvoiceRecord],
        query_type: str,
    ) -> RetrievalResult:
        if not invoices:
            return RetrievalResult(
                query=query,
                matched_invoices=[],
                summary_context=f"No invoices matched: \"{query}\"",
                query_type=query_type,
                status="empty",
            )
        total = sum(r.amount for r in invoices)
        summary = (
            f"{len(invoices)} invoice(s) found for \"{query}\", "
            f"total amount {total:,.2f}"
        )
        return RetrievalResult(
            query=query,
            matched_invoices=invoices,
            summary_context=summary,
            query_type=query_type,
            status="success",
        )
