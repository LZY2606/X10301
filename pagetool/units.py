"""领域量：携带单位或坐标基准，导入层保留原始表示，派生层才做规范化。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Q:
    """带单位的量。value 为规范值，raw 为原始表示，unit 为单位标识。"""

    raw: str
    value: float | int
    unit: str
    diag: str | None = None

    def as_int(self) -> int:
        return int(self.value)

    def to_dict(self) -> dict[str, Any]:
        out = {"raw": self.raw, "value": self.value, "unit": self.unit}
        if self.diag is not None:
            out["diag"] = self.diag
        return out


@dataclass(frozen=True)
class Offset:
    """页内字节坐标基准（base=page_start 表示自页首起的字节偏移）。"""

    value: int
    base: str = "page_start"
    unit: str = "byte"

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "base": self.base, "unit": self.unit}


def parse_bytes(raw: Any, *, default: int | None = None) -> Q:
    """把 '128 B' / '1 KiB' 等表示规范化为字节；无法转换时保留原值并附诊断。"""
    if isinstance(raw, int):
        return Q(raw=str(raw), value=raw, unit="byte")
    text = str(raw).strip()
    units = {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024 * 1024,
        "mib": 1024 * 1024,
    }
    parts = text.split()
    try:
        num = float(parts[0])
        mult = 1
        if len(parts) > 1:
            key = parts[1].lower()
            if key not in units:
                raise ValueError(f"unknown unit {parts[1]!r}")
            mult = units[key]
        val = num * mult
        if val != int(val):
            raise ValueError("non-integral byte count")
        return Q(raw=text, value=int(val), unit="byte")
    except (ValueError, IndexError):
        if default is None:
            return Q(raw=text, value=0, unit="byte", diag=f"unconvertible:{text!r}")
        return Q(raw=text, value=default, unit="byte", diag=f"unconvertible:{text!r};default={default}")
