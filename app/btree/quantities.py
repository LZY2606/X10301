"""领域量：所有数值都携带单位或坐标基准。

原始表示在导入层保留；本模块提供规范化（派生层才使用）。
无法转换为标准单位的值停留在原位置并附诊断，而不是被丢弃。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# 可转换的单位族：基准单位在第一位
UNIT_FAMILIES: dict[str, list[str]] = {
    "byte": ["byte", "bytes", "B", "KiB", "MiB"],
    "key": ["key", "keys", "entry", "entries"],
    "page": ["page", "pages"],
    "lsn": ["lsn"],
}

TO_BASE: dict[str, float] = {
    "byte": 1.0, "bytes": 1.0, "B": 1.0,
    "KiB": 1024.0, "MiB": 1024.0 * 1024.0,
    "key": 1.0, "keys": 1.0, "entry": 1.0, "entries": 1.0,
    "page": 1.0, "pages": 1.0,
    "lsn": 1.0,
}


def unit_family(unit: str) -> str | None:
    for family, units in UNIT_FAMILIES.items():
        if unit in units:
            return family
    return None


@dataclass(frozen=True)
class Quantity:
    """带单位的量；raw 保留导入时的原始表示。"""

    raw_value: Any
    unit: str
    basis: str = "page-coordinate/0"
    raw_repr: str | None = None
    diagnosis: str | None = None

    @property
    def convertible(self) -> bool:
        return self.diagnosis is None and unit_family(self.unit) is not None

    def normalized(self) -> float:
        """派生层规范化到基准单位；不可转换时抛出。"""
        if not self.convertible:
            raise ValueError(f"quantity not convertible: {self!r}")
        return float(self.raw_value) * TO_BASE[self.unit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_value": self.raw_value,
            "unit": self.unit,
            "basis": self.basis,
            "raw_repr": self.raw_repr,
            "diagnosis": self.diagnosis,
            "family": unit_family(self.unit),
            "normalized": self.normalized() if self.convertible else None,
        }


def parse_quantity(obj: Any, default_unit: str, basis: str = "page-coordinate/0") -> Quantity:
    """解析 {"value": .., "unit": ..} 或裸数值。未知单位附诊断、保留原值。"""

    if isinstance(obj, dict):
        value = obj.get("value")
        unit = obj.get("unit", default_unit)
        raw_repr = obj.get("raw_repr") or json.dumps(obj, ensure_ascii=False, sort_keys=True)
    else:
        value = obj
        unit = default_unit
        raw_repr = json.dumps(obj, ensure_ascii=False)
    diagnosis: str | None = None
    if unit_family(unit) is None:
        diagnosis = f"unknown unit '{unit}'; value kept at original location, no normalization"
    else:
        try:
            float(value)
        except (TypeError, ValueError):
            diagnosis = f"non-numeric value {value!r} for unit '{unit}'"
    return Quantity(raw_value=value, unit=unit, basis=basis,
                    raw_repr=raw_repr, diagnosis=diagnosis)


def bytes_quantity(n: int) -> Quantity:
    return Quantity(raw_value=n, unit="byte", basis="page-coordinate/0", raw_repr=str(n))


@dataclass(frozen=True)
class Coordinate:
    """页内坐标基准：页号 + 字节偏移（0 基于页起始）。"""

    page_id: int
    offset: int
    basis: str = "page-coordinate/0"

    def to_dict(self) -> dict[str, Any]:
        return {"page_id": self.page_id, "offset": self.offset, "basis": self.basis}
