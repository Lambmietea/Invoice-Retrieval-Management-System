"""
API adapter: UI-facing API layer.

- **Parse** delegates to Module A (``parser.parse_invoice_pdf`` — Qwen-VL multimodal LLM).
- **Search** delegates to Module B (``InvoiceService`` — SQLite + ChromaDB hybrid RAG).
- **Tasks** delegates to Module C (``module_c_agent.run_task`` — Gemini LLM agent).
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Literal, TypedDict

import module_c_agent
from parser import parse_invoice_pdf as _parse_pdf

# ---------------------------------------------------------------------------
# Import Module B
# ---------------------------------------------------------------------------

# Add module_b to sys.path so it can be imported directly
_MODULE_B_ROOT = Path(__file__).resolve().parent / "module_b"
if str(_MODULE_B_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_B_ROOT))

from module_b.models import InvoiceRecord as _BInvoiceRecord
from module_b.models import RetrievalResult as _BRetrievalResult
from module_b.service import InvoiceService

# ---------------------------------------------------------------------------
# Contract schemas (unchanged for UI compatibility)
# ---------------------------------------------------------------------------


class InvoiceRecord(TypedDict):
    invoice_id: str | None
    date: str | None
    vendor: str | None
    amount: float | None
    tax: float | None
    currency: str
    category: str | None
    items: list
    raw_text: str | None
    source_file: str
    status: str
    warnings: list


class IndexResult(TypedDict):
    status: Literal["success", "error"]
    indexed_id: str | None
    message: str


class RetrievalResult(TypedDict):
    query: str
    matched_invoices: list[dict]
    summary_context: str
    query_type: str
    status: Literal["success", "empty", "error"]


class TaskResult(TypedDict):
    task_type: str
    status: Literal["success", "warning", "error"]
    message: str
    result: dict | list  # Module C 的异常检测等可能返回 list


# ---------------------------------------------------------------------------
# Data: data/*.json (中文结构) + 本会话内上传（英文 → 中文）
# ---------------------------------------------------------------------------

_DISK_CN: list[dict] | None = None
_SESSION_EN: list[InvoiceRecord] = []
_SESSION_CN: list[dict] = []

# Module B service singleton
_B_SERVICE: InvoiceService | None = None
_B_INITIALIZED: bool = False


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _load_disk_cn() -> list[dict]:
    global _DISK_CN
    if _DISK_CN is not None:
        return _DISK_CN
    out: list[dict] = []
    d = _data_dir()
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            try:
                raw_bytes = p.read_bytes()
                for enc in ("utf-8", "utf-8-sig", "gbk"):
                    try:
                        out.append(json.loads(raw_bytes.decode(enc)))
                        break
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
            except OSError:
                pass
    _DISK_CN = out
    return out


def _get_b_service() -> InvoiceService:
    """Get or create the Module B InvoiceService singleton, auto-index disk data."""
    global _B_SERVICE, _B_INITIALIZED
    if _B_SERVICE is None:
        # Store DB and ChromaDB under mini-demo/.module_b_data/
        data_root = Path(__file__).resolve().parent.parent / ".module_b_data"
        data_root.mkdir(exist_ok=True)
        db_url = f"sqlite:///{data_root / 'invoices.db'}"
        chroma_dir = str(data_root / "chroma")
        _B_SERVICE = InvoiceService(db_url=db_url, chroma_dir=chroma_dir)

    if not _B_INITIALIZED:
        _B_INITIALIZED = True
        # Auto-index disk data if DB is empty
        if _B_SERVICE.invoice_count == 0:
            disk_cn = _load_disk_cn()
            if disk_cn:
                records = []
                for d in disk_cn:
                    try:
                        records.append(_BInvoiceRecord.from_chinese_dict(d))
                    except Exception:
                        pass
                if records:
                    _B_SERVICE.index_invoices(records)
                    print(f"[adapter] Auto-indexed {len(records)} invoices into Module B")
        # Rebuild ChromaDB if vector store is empty but SQLite has data
        # (happens when embedding model changes and chroma data is cleared)
        elif _B_SERVICE._vector.count() == 0:
            all_recs = _B_SERVICE.get_all_invoices()
            if all_recs:
                _B_SERVICE._vector.add_many(all_recs)
                print(f"[adapter] Rebuilt ChromaDB vector index for {len(all_recs)} invoices")

    return _B_SERVICE


def _normalize_cn_date(s: str) -> str | None:
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", (s or "").strip())
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    stripped = (s or "").strip()
    return stripped if stripped else None


def _amount_from_xiaoxie(s: str) -> float:
    try:
        return float(re.sub(r"[^\d.-]", "", str(s or "0")) or 0)
    except ValueError:
        return 0.0


def _cn_invoice_to_record(inv: dict) -> InvoiceRecord:
    amt = _amount_from_xiaoxie(inv.get("价税合计小写", "0"))
    items = inv.get("项目明细") or []
    return {
        "invoice_id": inv.get("发票号码"),
        "date": _normalize_cn_date(inv.get("开票日期", "")),
        "vendor": inv.get("销售方名称"),
        "amount": round(amt, 2) if amt else None,
        "tax": None,
        "currency": "CNY",
        "category": inv.get("发票类型"),
        "items": items if isinstance(items, list) else [],
        "raw_text": None,
        "source_file": inv.get("文件路径") or "",
        "status": "success",
        "warnings": [],
    }


def _en_record_to_cn(rec: dict) -> dict:
    amt = float(rec.get("amount") or 0)
    raw_items = rec.get("items") or []
    lines: list[dict] = []
    if isinstance(raw_items, list):
        for it in raw_items:
            if isinstance(it, dict):
                lines.append(
                    {
                        "项目名称": str(it.get("desc", "")),
                        "金额": str(it.get("unit_price", amt)),
                    }
                )
    if not lines:
        lines = [{"项目名称": "上传发票", "金额": str(amt)}]
    return {
        "发票类型": rec.get("category") or "其他",
        "发票号码": str(rec.get("invoice_id") or ""),
        "开票日期": rec.get("date") or "",
        "购买方名称": "",
        "销售方名称": rec.get("vendor") or "",
        "项目明细": lines,
        "价税合计小写": f"¥{amt}",
        "备注": "",
        "文件路径": str(rec.get("source_file") or "session://upload"),
    }


def _new_invoice_record(**overrides: Any) -> InvoiceRecord:
    base: InvoiceRecord = {
        "invoice_id": None,
        "date": None,
        "vendor": None,
        "amount": None,
        "tax": None,
        "currency": "USD",
        "category": None,
        "items": [],
        "raw_text": None,
        "source_file": "",
        "status": "success",
        "warnings": [],
    }
    merged = {**base, **overrides}
    return merged  # type: ignore[return-value]


def _all_records_en() -> list[InvoiceRecord]:
    disk = [_cn_invoice_to_record(x) for x in _load_disk_cn()]
    return disk + list(_SESSION_EN)


def parse_invoice(file_path: str) -> tuple[InvoiceRecord, dict]:
    """
    Parse a PDF invoice using Module A (Qwen-VL multimodal LLM).

    Returns:
        (InvoiceRecord, cn_dict): The English-keyed record for UI display,
        and the original Chinese-keyed dict for Module B/C indexing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Call the real PDF parser (Qwen-VL via DashScope)
    cn_dict = _parse_pdf(str(path.resolve()))

    # Convert Chinese-keyed dict → English InvoiceRecord for UI
    record = _cn_invoice_to_record(cn_dict)

    return record, cn_dict


def index_invoice(invoice_record: dict, cn_dict: dict | None = None) -> IndexResult:
    """
    Index an invoice into Module B (hybrid search) and Module C (session store).

    Args:
        invoice_record: English-keyed InvoiceRecord dict for UI/session.
        cn_dict: Optional Chinese-keyed dict (from parser). If provided,
                 used directly for Module B and Module C indexing.
    """
    try:
        rec = _coerce_to_invoice_record(invoice_record)
        iid = rec.get("invoice_id")
        if not iid:
            iid = str(uuid.uuid4())
            rec = _new_invoice_record(**{**dict(rec), "invoice_id": iid})

        _SESSION_EN.append(rec)

        # Store Chinese-keyed version for Module C
        if cn_dict:
            _SESSION_CN.append(cn_dict)
        else:
            _SESSION_CN.append(_en_record_to_cn(dict(rec)))

        # Index into Module B for hybrid search
        try:
            svc = _get_b_service()
            if cn_dict:
                # Use the real Chinese dict for accurate indexing
                b_rec = _BInvoiceRecord.from_chinese_dict(cn_dict)
            else:
                b_rec = _BInvoiceRecord(
                    invoice_id=iid,
                    date=rec.get("date"),
                    vendor=rec.get("vendor"),
                    amount=float(rec.get("amount") or 0),
                    tax=float(rec["tax"]) if rec.get("tax") else None,
                    currency=rec.get("currency", "CNY"),
                    category=rec.get("category"),
                    raw_text=rec.get("raw_text"),
                    source_file=rec.get("source_file"),
                    status="active",
                )
            svc.index_invoice(b_rec)
        except Exception as e:
            print(f"[adapter] Module B index warning: {e}")

        return {
            "status": "success",
            "indexed_id": iid,
            "message": f"Invoice {iid} indexed.",
        }
    except Exception as e:
        return {
            "status": "error",
            "indexed_id": None,
            "message": str(e),
        }


def _coerce_to_invoice_record(data: dict) -> InvoiceRecord:
    if not isinstance(data, dict):
        raise TypeError("invoice_record must be a dict")
    allowed = set(InvoiceRecord.__annotations__.keys())
    filtered = {k: v for k, v in data.items() if k in allowed}
    return _new_invoice_record(**filtered)


def _b_record_to_ui_dict(rec: _BInvoiceRecord) -> dict:
    """Convert Module B's Pydantic InvoiceRecord to a plain dict for UI."""
    return {
        "invoice_id": rec.invoice_id,
        "date": rec.date,
        "vendor": rec.vendor,
        "amount": rec.amount,
        "tax": rec.tax,
        "currency": rec.currency,
        "category": rec.category,
        "raw_text": rec.raw_text,
        "source_file": rec.source_file,
        "status": rec.status,
        "items": [],
        "warnings": [],
    }


def query_invoices(user_query: str) -> RetrievalResult:
    """
    Natural-language hybrid search powered by Module B.

    Uses SQLite (structured filters) + ChromaDB (vector semantic search)
    to find relevant invoices.
    """
    raw_q = user_query or ""
    q = raw_q.strip()

    if not q:
        return {
            "query": raw_q,
            "matched_invoices": [],
            "summary_context": "请输入查询内容，例如：2024年7月的发票、金额超过500的发票、和差旅相关的票据。",
            "query_type": "empty",
            "status": "empty",
        }

    try:
        svc = _get_b_service()
        b_result: _BRetrievalResult = svc.query_invoices(q)

        # Convert Module B's Pydantic models to plain dicts for UI
        matched_dicts = [_b_record_to_ui_dict(inv) for inv in b_result.matched_invoices]

        return {
            "query": b_result.query,
            "matched_invoices": matched_dicts,
            "summary_context": b_result.summary_context,
            "query_type": b_result.query_type,
            "status": b_result.status,
        }
    except Exception as e:
        return {
            "query": raw_q.strip(),
            "matched_invoices": [],
            "summary_context": f"检索出错: {e}",
            "query_type": "error",
            "status": "error",
        }


def get_all_invoices() -> list[InvoiceRecord]:
    return [dict(x) for x in _all_records_en()]  # type: ignore[return-value]


def run_task(command: str) -> TaskResult:
    invoices_data = list(_load_disk_cn()) + list(_SESSION_CN)
    try:
        raw = module_c_agent.run_task(command or "", invoices_data)
        res = raw.get("result")
        if res is None:
            res = {}
        elif not isinstance(res, (dict, list)):
            res = {}
        st = raw.get("status") or "success"
        if st not in ("success", "warning", "error"):
            st = "success"
        return {
            "task_type": str(raw.get("task_type") or "generic"),
            "status": st,  # type: ignore[typeddict-item]
            "message": str(raw.get("message") or ""),
            "result": res,  # type: ignore[typeddict-item]
        }
    except Exception as e:
        return {
            "task_type": "generic",
            "status": "error",
            "message": str(e),
            "result": {},
        }


__all__ = [
    "InvoiceRecord",
    "IndexResult",
    "RetrievalResult",
    "TaskResult",
    "parse_invoice",
    "index_invoice",
    "query_invoices",
    "get_all_invoices",
    "run_task",
]
