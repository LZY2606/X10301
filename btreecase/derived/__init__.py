"""Derived metrics with explicit dependency tracking.

Every metric declares which logical-op indices it reads. When a parameter
version or source data changes, only metrics whose dependency set intersects
the changed op range are invalidated — never a global silent replacement.
"""
from .metrics import compute_all, compute_metric, MetricResult, invalidate_scope

__all__ = ["compute_all", "compute_metric", "MetricResult", "invalidate_scope"]
