"""一次逻辑操作的事务：保留页前镜像、日志序列、分裂传播轨迹。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Txn:
    op_index: int
    op: dict
    before: dict[int, bytes]
    records: list[dict] = field(default_factory=list)
    split_traces: list[dict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    aborted: bool = False
