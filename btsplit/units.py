"""单位与坐标基准。

领域量在导入层保留原始表示（raw + raw_unit），派生层才做规范化。
无法转换的值留在原位置并附诊断（raw 保留、normalized=None、diagnostic 非空）。
"""
from __future__ import annotations

from dataclasses import dataclass

BYTE_MULTIPLIERS = {"B": 1, "bytes": 1, "byte": 1, "KB": 1024, "K": 1024}


@dataclass(frozen=True)
class Quantity:
    raw: str
    raw_unit: str | None
    normalized: float | None
    unit: str | None
    diagnostic: str | None = None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "raw_unit": self.raw_unit,
            "normalized": self.normalized,
            "unit": self.unit,
            "diagnostic": self.diagnostic,
        }


def parse_bytes(value, unit: str | None = None) -> Quantity:
    """把字节空间量规范化为 int bytes；无法转换时保留原值与诊断。"""
    if isinstance(value, bool):
        return Quantity(str(value), unit, None, "B", "布尔值不是合法字节空间")
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            return Quantity(str(value), unit, None, "B", "字节空间不能为负")
        return Quantity(str(value), unit or "B", float(value), "B")
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return Quantity(str(value), unit or "B", float(value), "B")
    text = str(value).strip()
    mult = BYTE_MULTIPLIERS.get((unit or "B"))
    try:
        number = int(text)
    except ValueError:
        return Quantity(text, unit, None, "B", f"无法把 {text!r} 转换为字节数")
    if mult is None:
        return Quantity(text, unit, None, "B", f"未知字节单位 {unit!r}")
    if number < 0:
        return Quantity(text, unit, None, "B", "字节空间不能为负")
    return Quantity(text, unit, float(number * mult), "B")


def require_bytes(value, field: str) -> int:
    q = parse_bytes(value)
    if q.normalized is None:
        raise ValueError(f"{field}: {q.diagnostic}")
    return int(q.normalized)
