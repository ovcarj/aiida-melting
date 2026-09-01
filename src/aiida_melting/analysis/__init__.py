"""Read-only analysis helpers for completed melting calculations."""

from .query import query_results
from .records import ResultRecord

__all__ = ("ResultRecord", "query_results")
