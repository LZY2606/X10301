"""领域量与坐标基准。

所有领域量都携带单位或坐标基准。导入时保留原始表示，派生层才做规范化；
无法转换的值留在原位置并附诊断（见 :func:`coerce_bytes`）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BYTE = "byte"
SLOT = "slot"
PAGE = "page_id"
LSN = "wal_byte_offset"
KEY_BYTES = "byte"
LOGICAL_OP = "logical_op_index"


@dataclass(frozen=True)
class Q:
    """带单位/坐标基准的领域量。raw 保留导入时的原始表示。"""

    value: Any
    unit: str
    raw: Any = None
    diagnostic: str | None = None

    @property
    def convertible(self) -> bool:
        return self.diagnostic is None

    def normalized(self) -> Any:
        """派生层规范化；无法转换时抛错，调用方应先检查 convertible。"""
        if not self.convertible:
            raise ValueError(self.diagnostic)
        return self.value


def coerce_bytes(raw: Any, *, field_name: str, op_index: int) -> Q:
    """把导入的原始值规范化成字节；不可转换时留在原位并附诊断，绝不丢弃。"""
    if isinstance(raw, (bytes, bytearray)):
        return Q(bytes(raw), BYTE, raw=raw)
    if isinstance(raw, str):
        try:
            return Q(raw.encode("utf-8"), BYTE, raw=raw)
        except UnicodeEncodeError as exc:  # pragma: no cover - 防御
            return Q(None, BYTE, raw=raw, diagnostic=f"{field_name}: {exc}")
    if isinstance(raw, int):
        return Q(str(raw).encode("ascii"), BYTE, raw=raw)
    if raw is None:
        return Q(b"", BYTE, raw=raw)
    return Q(
        None,
        BYTE,
        raw=raw,
        diagnostic=(
            f"op#{op_index} 字段 {field_name} 无法转换为字节，"
            f"类型={type(raw).__name__}，已保留原始表示并跳过规范化"
        ),
    )
