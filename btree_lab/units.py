"""领域量：携带单位或坐标基准。

规则（需求）：
- 所有领域量携带单位或坐标基准；
- 导入时保留原始表示（raw），派生层才做规范化；
- 无法转换的值留在原位置并附诊断（convertible=false），不能为画图丢掉。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

UNITS = {
    "bytes": ("byte", "字节"),
    "page_id": ("coordinate:page", "页坐标"),
    "slot_no": ("coordinate:slot", "槽位坐标（0 基）"),
    "byte_offset": ("coordinate:offset", "页内字节偏移（0 基，自页首）"),
    "count": ("count", "个数"),
    "lsn": ("sequence:lsn", "日志序号（单调递增）"),
    "op_seq": ("sequence:op", "逻辑操作序号"),
}


@dataclass(frozen=True)
class Quantity:
    """一个带来源坐标的领域量。"""

    guid: str
    label: str
    raw: Any
    unit: str
    value: float | int | str | None
    basis: str
    convertible: bool = True
    diagnostic: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "guid": self.guid,
            "label": self.label,
            "raw": self.raw,
            "unit": self.unit,
            "value": self.value,
            "basis": self.basis,
            "convertible": self.convertible,
            "diagnostic": self.diagnostic,
        }


def parse_int_quantity(
    guid: str,
    label: str,
    raw: Any,
    unit: str,
    basis: str,
) -> Quantity:
    """从原始表示规范化为整型量；失败时保留原值与诊断（不抛异常）。"""
    try:
        if isinstance(raw, bool):
            raise ValueError("bool 不是页坐标")
        value = int(raw)
        if isinstance(raw, str) and str(value) != raw.strip():
            raise ValueError(f"非纯整数表示: {raw!r}")
        if value < 0:
            raise ValueError("坐标/长度不能为负")
        return Quantity(guid, label, raw, unit, value, basis, True, None)
    except (TypeError, ValueError) as exc:
        return Quantity(guid, label, raw, unit, None, basis, False, f"UNCONVERTIBLE: {exc}")


def bytes_quantity(guid: str, label: str, value: int, basis: str) -> Quantity:
    return Quantity(guid, label, str(value), "bytes", value, basis, True, None)
