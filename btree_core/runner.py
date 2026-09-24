"""Scenario runner: executes logical operations and crash events, capturing
a full provenance trace.

For every logical operation the trace preserves:
  the operation itself, page pre-images, slot directory, free space,
  sibling links, page-id allocation and the log sequence numbers.
"""
from __future__ import annotations

from .pages import LEAF, Pager
from .tree import B to_bytes_unused  # placeholder
