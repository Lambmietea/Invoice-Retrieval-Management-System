"""SQLite structured storage — SQLAlchemy CRUD for invoices."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import Column, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .models import InvoiceRecord

# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class InvoiceRow(Base):
    __tablename__ = "invoices"

    invoice_id = Column(String, primary_key=True)
    date = Column(String, nullable=True, index=True)
    vendor = Column(String, nullable=True, index=True)
    amount = Column(Float, default=0.0)
    tax = Column(Float, nullable=True)
    currency = Column(String, default="CNY")
    category = Column(String, nullable=True, index=True)
    raw_text = Column(Text, nullable=True)
    source_file = Column(String, nullable=True)
    status = Column(String, default="active")


# ---------------------------------------------------------------------------
# Storage service
# ---------------------------------------------------------------------------

_DEFAULT_DB_URL = "sqlite:///module_b_invoices.db"


class SQLiteStorage:
    """Thin wrapper around SQLAlchemy for invoice CRUD."""

    def __init__(self, db_url: str | None = None) -> None:
        url = db_url or os.getenv("MODULE_B_DB_URL", _DEFAULT_DB_URL)
        self.engine = create_engine(url, echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    # -- helpers --

    def _session(self) -> Session:
        return self._Session()

    @staticmethod
    def _row_to_record(row: InvoiceRow) -> InvoiceRecord:
        return InvoiceRecord(
            invoice_id=row.invoice_id,
            date=row.date,
            vendor=row.vendor,
            amount=row.amount or 0.0,
            tax=row.tax,
            currency=row.currency or "CNY",
            category=row.category,
            raw_text=row.raw_text,
            source_file=row.source_file,
            status=row.status or "active",
        )

    @staticmethod
    def _record_to_row(rec: InvoiceRecord) -> InvoiceRow:
        return InvoiceRow(
            invoice_id=rec.invoice_id,
            date=rec.date,
            vendor=rec.vendor,
            amount=rec.amount,
            tax=rec.tax,
            currency=rec.currency,
            category=rec.category,
            raw_text=rec.raw_text,
            source_file=rec.source_file,
            status=rec.status,
        )

    # -- public API --------------------------------------------------------

    def insert(self, rec: InvoiceRecord) -> None:
        """Insert or replace a single invoice."""
        with self._session() as s:
            existing = s.get(InvoiceRow, rec.invoice_id)
            if existing:
                for col in ("date", "vendor", "amount", "tax", "currency",
                            "category", "raw_text", "source_file", "status"):
                    setattr(existing, col, getattr(rec, col))
            else:
                s.add(self._record_to_row(rec))
            s.commit()

    def insert_many(self, records: list[InvoiceRecord]) -> int:
        """Bulk insert/replace. Returns count of records processed."""
        with self._session() as s:
            for rec in records:
                existing = s.get(InvoiceRow, rec.invoice_id)
                if existing:
                    for col in ("date", "vendor", "amount", "tax", "currency",
                                "category", "raw_text", "source_file", "status"):
                        setattr(existing, col, getattr(rec, col))
                else:
                    s.add(self._record_to_row(rec))
            s.commit()
        return len(records)

    def get_all(self) -> list[InvoiceRecord]:
        """Return every invoice in the database."""
        with self._session() as s:
            rows = s.query(InvoiceRow).all()
            return [self._row_to_record(r) for r in rows]

    def filter(self, filters: dict[str, Any]) -> list[InvoiceRecord]:
        """
        Apply structured filters and return matching records.

        Supported filter keys:
          - invoice_id: str         → invoice_id = value (exact match)
          - month: "YYYY-MM"        → date LIKE 'YYYY-MM%'
          - vendor: str             → vendor LIKE '%value%'
          - category: str           → category LIKE '%value%'
          - currency: str           → currency = value (exact match, case-insensitive)
          - status: str             → status = value (exact match)
          - amount_gt: float        → amount > value
          - amount_lt: float        → amount < value
          - date_from: "YYYY-MM-DD" → date >= value
          - date_to: "YYYY-MM-DD"   → date <= value
        """
        with self._session() as s:
            q = s.query(InvoiceRow)

            if filters.get("invoice_id"):
                q = q.filter(InvoiceRow.invoice_id == filters["invoice_id"])
            if filters.get("month"):
                q = q.filter(InvoiceRow.date.like(f"{filters['month']}%"))
            if filters.get("vendor"):
                q = q.filter(InvoiceRow.vendor.like(f"%{filters['vendor']}%"))
            if filters.get("category"):
                q = q.filter(InvoiceRow.category.like(f"%{filters['category']}%"))
            if filters.get("currency"):
                q = q.filter(InvoiceRow.currency == filters["currency"].upper())
            if filters.get("status"):
                q = q.filter(InvoiceRow.status == filters["status"])
            if filters.get("amount_gt") is not None:
                q = q.filter(InvoiceRow.amount > float(filters["amount_gt"]))
            if filters.get("amount_lt") is not None:
                q = q.filter(InvoiceRow.amount < float(filters["amount_lt"]))
            if filters.get("date_from"):
                q = q.filter(InvoiceRow.date >= filters["date_from"])
            if filters.get("date_to"):
                q = q.filter(InvoiceRow.date <= filters["date_to"])

            rows = q.all()
            return [self._row_to_record(r) for r in rows]

    def count(self) -> int:
        with self._session() as s:
            return s.query(InvoiceRow).count()

    def clear(self) -> None:
        """Delete all rows — useful for tests."""
        with self._session() as s:
            s.query(InvoiceRow).delete()
            s.commit()
