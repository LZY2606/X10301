"""领域量：原始表示 + 单位/坐标基准；派生层才规范化，转换失败留在原位并附诊断。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 规范单位：page_size/cell/slot/free_space 用 byte；split_ratio/overhead 用 fraction；
# lsn 为单调日志序号（无物理单位）；page_id/slot_index/offset 是坐标基准。
_ALLOWED = {"byte", "fraction", "count", "lsn", "page_id", "slot_index", "offset", "text"}
_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z%_/]*)\s*$")

_UNIT_ALIAS = {
    "b": "byte",
    "bytes": "byte",
    "byte": "byte",
    "": "count",
    "count": "count",
    "n": "count",
    "%": "fraction",
    "fraction": "fraction",
    "ratio": "fraction",
    "lsn": "lsn",
    "page_id": "page_id",
    "pid": "page_id",
    "slot_index": "slot_index",
    "slot": "slot_index",
    "offset": "offset",
    "bytes_offset": "offset",
}


@dataclass
class Quantity:
    raw: str
    value: float | int | None = None
    unit: str | None = None
    coordinate: str | None = None  # 坐标基准，如 "page:3/offset:10"
    diagnostics: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.diagnostics

    def normalized(self, target_unit: str) -> float | int | None:
        if self.value is None or self.unit != target_unit:
            return None
        return self.value


def parse_quantity(raw: object, coordinate: str | None = None) -> Quantity:
    """解析时保留原始表示；无法转换的值留在原位置并附诊断，绝不静默丢弃。"""
    if isinstance(raw, (int, float)):
        return Quantity(raw=repr(raw), value=raw, unit="count", coordinate=coordinate)
    if not isinstance(raw, str):
        return Quantity(raw=repr(raw), diagnostics=[f"unsupported representation {type(raw).__name__}"],
                        coordinate=coordinate)
    m = _NUM.match(raw)
    if not m:
        return Quantity(raw=raw, diagnostics=["unparseable quantity"], coordinate=coordinate)
    num_s, unit_s = m.group(1), m.group(2).lower()
    unit = _UNIT_ALIAS.get(unit_s)
    if unit is None:
        return Quantity(raw=raw, diagnostics=[f"unknown unit {unit_s!r}"], coordinate=coordinate)
    if unit not in _ALLOWED:
        return Quantity(raw=raw, diagnostics=[f"unit {unit!r} not allowed"], coordinate=coordinate)
    val: float | int = float(m.group(1)) if "." in num_s else int(m.group(1))
    if unit == "fraction" and unit_s == "%":
        val = val / 100.0
    return Quantity(raw=raw, value=val, unit=unit, coordinate=coordinate)


def to_bytes(q: Quantity, name: str = "value") -> tuple[int | None, list[str]]:
    """派生层规范化到 byte；不可转换时返回诊断。"""
    if q.value is None:
        return None, q.diagnostics or [f"{name}: missing value"]
    if q.unit == "byte":
        return int(q.value), []
    if q.unit == "fraction":
        return None, [f"{name}: fraction requires a coordinate byte base"]
    return None, [f"{name}: unit {q.unit!r} cannot normalize to bytes (raw={q.raw!r})"]
