"""领域量：原始表示、单位/坐标基准、仅在派生层做规范化。

所有领域量（free space、键长、页大小、槽位移……）都携带单位或坐标基准。
导入层保留原始表示；无法转换的值留在原位置并附诊断，不抛异常丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 支持的长度单位及其到字节的换算
_UNIT_TO_BYTES = {
    "B": 1.0,
    "byte": 1.0,
    "bytes": 1.0,
    "页": None,  # 特殊：依赖 page_size，解析时需上下文
}


@dataclass(frozen=True)
class Quantity:
    """带单位的领域量。raw 保留导入时的原始表示（字符串/数字）。"""

    value: float
    unit: str
    raw: Any = None
    page_size: int | None = None  # 以“页”为单位时的换算基准
    diagnostic: str | None = None

    def to_bytes(self) -> float:
        if self.diagnostic:
            raise ValueError(f"无法转换的领域量: {self.diagnostic}")
        if self.unit in ("B", "byte", "bytes"):
            return self.value
        if self.unit == "页":
            if self.page_size is None:
                raise ValueError("单位为“页”但缺少 page_size 坐标基准")
            return self.value * self.page_size
        raise ValueError(f"未知单位: {self.unit}")

    def to_int_bytes(self) -> int:
        return int(self.to_bytes())

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "raw": self.raw,
            "page_size": self.page_size,
            "diagnostic": self.diagnostic,
            "bytes": None if self.diagnostic else self._safe_bytes(),
        }

    def _safe_bytes(self) -> int | None:
        try:
            return int(self.to_bytes())
        except ValueError:
            return None


def parse_quantity(raw: Any, *, page_size: int | None = None, default_unit: str = "B") -> Quantity:
    """把原始表示解析成 Quantity；失败时返回带 diagnostic 的原值，不抛出。"""
    if isinstance(raw, Quantity):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return Quantity(value=float(raw), unit=default_unit, raw=raw, page_size=page_size)
    if isinstance(raw, dict):
        value = raw.get("value")
        unit = raw.get("unit", default_unit)
        q = parse_quantity(value, page_size=page_size, default_unit=unit)
        return Quantity(value=q.value, unit=unit, raw=raw, page_size=page_size, diagnostic=q.diagnostic)
    if isinstance(raw, str):
        s = raw.strip()
        unit = default_unit
        body = s
        for u in ("bytes", "byte", "B", "页"):
            if s.endswith(u):
                body = s[: -len(u)].strip()
                unit = u
                break
        try:
            value = float(body)
        except ValueError:
            return Quantity(value=float("nan"), unit=unit, raw=raw, diagnostic=f"无法解析数值: {raw!r}")
        if unit == "页" and page_size is None:
            return Quantity(value=value, unit=unit, raw=raw, diagnostic="单位为“页”但缺少 page_size 坐标基准")
        return Quantity(value=value, unit=unit, raw=raw, page_size=page_size)
    return Quantity(value=float("nan"), unit=default_unit, raw=raw, diagnostic=f"不支持的表示: {raw!r}")


@dataclass
class Coordinate:
    """坐标基准：页内字节偏移/长度都以 page 坐标为基准（0 = 页首）。"""

    offset: int
    basis: str = "page_offset_0"
    length: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"offset": self.offset, "basis": self.basis, "length": self.length}
