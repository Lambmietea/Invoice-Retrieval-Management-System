"""Integration tests — simulates Module C calling Module B's service layer.

Run:  cd module_b && python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from module_b.models import InvoiceRecord
from module_b.service import InvoiceService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def svc(tmp_path: Path) -> InvoiceService:
    """Create a clean InvoiceService with temp DB + ChromaDB dir."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    chroma_dir = str(tmp_path / "chroma")
    return InvoiceService(db_url=db_url, chroma_dir=chroma_dir)


@pytest.fixture()
def sample_records() -> list[InvoiceRecord]:
    """Load sample records from the repo's data/ directory."""
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    if not data_dir.is_dir():
        pytest.skip("data/ directory not found")

    records: list[InvoiceRecord] = []
    for fp in sorted(data_dir.glob("*.json"))[:10]:
        raw = json.loads(fp.read_bytes().decode("utf-8"))
        records.append(InvoiceRecord.from_chinese_dict(raw))
    if not records:
        pytest.skip("No JSON files in data/")
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIndexing:
    """Test index_invoice / index_invoices."""

    def test_index_single(self, svc: InvoiceService) -> None:
        rec = InvoiceRecord(
            invoice_id="TEST-001",
            date="2024-07-20",
            vendor="测试供应商",
            amount=199.00,
            category="电子发票(普通发票)",
            raw_text="Test invoice content",
        )
        result = svc.index_invoice(rec)
        assert result["status"] == "success"
        assert svc.invoice_count == 1

    def test_index_batch(self, svc: InvoiceService, sample_records: list[InvoiceRecord]) -> None:
        result = svc.index_invoices(sample_records)
        assert result["status"] == "success"
        assert result["indexed_count"] == len(sample_records)
        assert svc.invoice_count == len(sample_records)

    def test_upsert_no_duplicate(self, svc: InvoiceService) -> None:
        rec = InvoiceRecord(
            invoice_id="DUP-001",
            date="2024-01-01",
            vendor="Vendor A",
            amount=100.0,
        )
        svc.index_invoice(rec)
        svc.index_invoice(rec)
        assert svc.invoice_count == 1


class TestGetAll:
    """Test get_all_invoices."""

    def test_get_all_empty(self, svc: InvoiceService) -> None:
        result = svc.get_all_invoices()
        assert result == []

    def test_get_all_after_index(self, svc: InvoiceService, sample_records: list[InvoiceRecord]) -> None:
        svc.index_invoices(sample_records)
        result = svc.get_all_invoices()
        assert len(result) == len(sample_records)
        assert all(isinstance(r, InvoiceRecord) for r in result)


class TestQuery:
    """Test query_invoices (hybrid retrieval)."""

    def test_empty_query(self, svc: InvoiceService) -> None:
        result = svc.query_invoices("")
        assert result.status == "empty"

    def test_query_no_data(self, svc: InvoiceService) -> None:
        result = svc.query_invoices("找一下3月份的发票")
        assert result.status == "empty"
        assert len(result.matched_invoices) == 0

    def test_semantic_search(self, svc: InvoiceService, sample_records: list[InvoiceRecord]) -> None:
        svc.index_invoices(sample_records)
        result = svc.query_invoices("和差旅相关的票据")
        # Semantic search should return results (even if not exact match)
        assert result.status in ("success", "empty")
        assert result.query_type in ("semantic", "filter+semantic", "hybrid")

    def test_filter_by_amount(self, svc: InvoiceService, sample_records: list[InvoiceRecord]) -> None:
        svc.index_invoices(sample_records)
        result = svc.query_invoices("金额超过500的发票")
        assert result.status in ("success", "empty")
        # If matches found, all should be > 500
        for inv in result.matched_invoices:
            assert inv.amount > 500

    def test_hybrid_month_and_semantic(
        self, svc: InvoiceService, sample_records: list[InvoiceRecord]
    ) -> None:
        svc.index_invoices(sample_records)
        result = svc.query_invoices("2024年7月的发票")
        assert result.query_type in ("filter", "filter+semantic", "hybrid")
        for inv in result.matched_invoices:
            assert inv.date is not None
            assert inv.date.startswith("2024-07")


class TestModels:
    """Test data model conversion."""

    def test_from_chinese_dict(self) -> None:
        d = {
            "发票类型": "电子发票(普通发票)",
            "发票号码": "24417000000034170288",
            "开票日期": "2024年07月20日",
            "购买方名称": "同济大学",
            "销售方名称": "郑州京东优凯贸易有限公司",
            "项目明细": [
                {"项目名称": "*照明装置*台灯", "金额": "176.11", "税额": "22.89"}
            ],
            "价税合计小写": "¥199.00",
            "文件路径": "invoice_file_01.pdf",
        }
        rec = InvoiceRecord.from_chinese_dict(d)
        assert rec.invoice_id == "24417000000034170288"
        assert rec.date == "2024-07-20"
        assert rec.vendor == "郑州京东优凯贸易有限公司"
        assert rec.amount == 199.00
        assert rec.tax == 22.89
        assert rec.category == "电子发票(普通发票)"
        assert "照明装置" in (rec.raw_text or "")
