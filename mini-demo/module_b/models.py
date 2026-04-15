"""Pydantic models: InvoiceRecord, RetrievalResult, and API request/response schemas."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Core domain model
# ---------------------------------------------------------------------------

class InvoiceRecord(BaseModel):
    """Standard invoice record — maps to both the SQLite row and the API contract."""

    invoice_id: str = Field(..., description="发票号码")
    date: str | None = Field(None, description="开票日期 (YYYY-MM-DD)")
    vendor: str | None = Field(None, description="销售方名称")
    amount: float = Field(0.0, description="价税合计小写 (numeric)")
    tax: float | None = Field(None, description="税额合计")
    currency: str = Field("CNY")
    category: str | None = Field(None, description="发票类型")
    raw_text: str | None = Field(None, description="拼接后的全文，用于向量检索")
    source_file: str | None = Field(None, description="文件路径")
    status: str = Field("active")

    # --- Chinese-key helpers -------------------------------------------

    @classmethod
    def from_chinese_dict(cls, d: dict[str, Any]) -> InvoiceRecord:
        """Convert a Chinese-keyed invoice JSON (data/*.json) into an InvoiceRecord."""
        amount = _parse_amount(d.get("价税合计小写", "0"))
        tax = _sum_item_tax(d.get("项目明细"))
        raw_text = _build_raw_text(d)
        date_str = _normalize_cn_date(d.get("开票日期", ""))
        return cls(
            invoice_id=d.get("发票号码", ""),
            date=date_str,
            vendor=d.get("销售方名称"),
            amount=round(amount, 2),
            tax=round(tax, 2) if tax else None,
            currency="CNY",
            category=d.get("发票类型"),
            raw_text=raw_text,
            source_file=d.get("文件路径"),
            status="active",
        )


# ---------------------------------------------------------------------------
# Retrieval result — returned by service.query_invoices
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    query: str
    matched_invoices: list[InvoiceRecord] = Field(default_factory=list)
    summary_context: str = ""
    query_type: str = "hybrid"  # "filter" | "semantic" | "filter+semantic" | "hybrid"
    status: Literal["success", "empty", "error"] = "success"


# ---------------------------------------------------------------------------
# API request / response helpers
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query")


class BatchIndexResponse(BaseModel):
    status: str = "success"
    indexed_count: int = 0
    errors: list[str] = Field(default_factory=list)


class IndexResponse(BaseModel):
    status: str = "success"
    invoice_id: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_amount(s: Any) -> float:
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s or "0")) or "0")
    except (ValueError, TypeError):
        return 0.0


def _sum_item_tax(items: Any) -> float:
    if not isinstance(items, list):
        return 0.0
    total = 0.0
    for it in items:
        if not isinstance(it, dict):
            continue
        raw = str(it.get("税额", "0"))
        raw = re.sub(r"[^\d.\-]", "", raw)
        try:
            total += float(raw or "0")
        except ValueError:
            pass
    return total


def _normalize_cn_date(s: str) -> str | None:
    """'2024年07月20日' → '2024-07-20'."""
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", (s or "").strip())
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    stripped = (s or "").strip()
    return stripped if stripped else None


def _build_raw_text(d: dict[str, Any]) -> str:
    """Build a human-readable text block from the Chinese-keyed dict for vector embedding.

    Uses Chinese labels so that Chinese user queries produce better cosine
    similarity with the indexed documents.
    """
    parts: list[str] = []
    parts.append(f"发票号码: {d.get('发票号码', '')}")
    parts.append(f"开票日期: {d.get('开票日期', '')}")
    parts.append(f"销售方名称: {d.get('销售方名称', '')}")
    parts.append(f"购买方名称: {d.get('购买方名称', '')}")
    parts.append(f"价税合计: {d.get('价税合计小写', '')}")
    parts.append(f"发票类型: {d.get('发票类型', '')}")
    items = d.get("项目明细") or []
    if isinstance(items, list):
        for i, it in enumerate(items):
            if isinstance(it, dict):
                parts.append(
                    f"项目{i + 1}: {it.get('项目名称', '')} "
                    f"数量={it.get('数量', '')} 金额={it.get('金额', '')}"
                )
    if d.get("备注"):
        parts.append(f"备注: {d['备注']}")
    return "\n".join(parts)
