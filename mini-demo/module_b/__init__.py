"""Module B — Invoice Hybrid-RAG Retrieval Microservice.

Public interface:
    from module_b.service import InvoiceService
    from module_b.models import InvoiceRecord, RetrievalResult
"""

from .models import InvoiceRecord, RetrievalResult
from .service import InvoiceService

__all__ = ["InvoiceRecord", "RetrievalResult", "InvoiceService"]
