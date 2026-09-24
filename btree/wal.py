"""Write-ahead log records.

Each record is a single LSN and carries full (before/after) page images so the
replay pass can both redo an interrupted write and keep recovery idempotent.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .page import Page


@dataclass
class WalRecord:
    lsn: int
    op: str                      # page_write | leaf_split | internal_split | root_split | free_page | set_root
    pages_before: dict[int, bytes] = field(default_factory=dict)
    pages_after: dict[int, bytes] = field(default_factory=dict)
    new_pages: list[int] = field(default_factory=list)
    root_before: int = 0
    root_after: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    committed: bool = False

    def to_dict(self) -> dict[str, Any]:
        def enc(images: dict[int, bytes]) -> dict[str, str]:
            import base64
            return {str(pid): base64.b64encode(img).decode() for pid, img in images.items()}

        d = asdict(self)
        d["pages_before"] = enc(self.pages_before)
        d["pages_after"] = enc(self.pages_after)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WalRecord":
        import base64

        def dec(images: dict[str, str]) -> dict[int, bytes]:
            return {int(pid): base64.b64decode(img) for pid, img in images.items()}

        return cls(
            lsn=data["lsn"],
            op=data["op"],
            pages_before=dec(data["pages_before"]),
            pages_after=dec(data["pages_after"]),
            new_pages=list(data["new_pages"]),
            root_before=data["root_before"],
            root_after=data["root_after"],
            detail=data["detail"],
            committed=data["committed"],
        )
