"""领域量的单位 / 坐标基准。

所有领域量都携带单位或坐标基准。导入层保留原始表示，
只有派生层才做规范化；无法转换的值留在原位置并附诊断，
绝不为了画图而丢数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BYTE = "byte"
KEY_COUNT = "key_count"
OCTET = "octet"
PAGE_OFFSET = "page_offset[byte]"
LSN = "lsn"
PAGE_ID = "page_id"
RAW = "raw"

_KNOWN_UNITS = {BYTE, KEY_COUNT, OCTET, PAGE_OFFSET, LSN, PAGE_ID, RAW}


class UnitError(ValueError):
    """原始值无法转换为目标单位时抛出；调用方必须保留原值并附诊断。"""


@dataclass(frozen=True)
class Quantity:
    """带单位 / 坐标基准的领域量，保留原始表示。

    raw 是导入时的原始（字符串）表示；value 是规范化后的数值。
    规范化失败时 value 为 None，诊断写入 diagnostic。
    """

    raw: Any
    unit: str
    value: float | int | None = None
    coordinate_base: str | None = None
    diagnostic: str | None = None

    @property
    def ok(self) -> bool:
        return self.diagnostic is None

    def display(self) -> str:
        if self.value is None:
            return f"{self.raw!r} (无法转换)"
        return f"{self.value} {self.unit}"


def _parse_int(raw: Any) -> int:
    if isinstance(raw, bool):
        raise UnitError("布尔值不是合法的领域整数")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        raise UnitError(f"期望整数，得到浮点 {raw!r}")
    if isinstance(raw, str):
        s = raw.strip().lower()
        try:
            if s.startswith("0x"):
                return int(s, 16)
            return int(s)
        except ValueError as exc:
            raise UnitError(f"无法把 {raw!r} 解析为整数") from exc
    raise UnitError(f"无法把 {raw!r} 解析为整数")


def to_bytes(raw: Any, *, coordinate_base: str = "page_start") -> Quantity:
    """规范化为字节空间。接受显式后缀（B/bytes）；无后缀按字节。"""
    if isinstance(raw, str):
        s = raw.strip()
        low = s.lower()
        if low.endswith("bytes"):
            body = s[: -len("bytes")].strip()
        elif low.endswith("byte"):
            body = s[: -len("byte")].strip()
        elif low.endswith("b") and not low.startswith("0x"):
            body = s[:-1].strip()
        else:
            body = s
        try:
            n = _parse_int(body)
        except UnitError as exc:
            return Quantity(raw=raw, unit=BYTE, coordinate_base=coordinate_base,
                           diagnostic=str(exc))
    else:
        try:
            n = _parse_int(raw)
        except UnitError as exc:
            return Quantity(raw=raw, unit=BYTE, coordinate_base=coordinate_base,
                           diagnostic=str(exc))
    if n < 0:
        return Quantity(raw=raw, unit=BYTE, value=n, coordinate_base=coordinate_base,
                        diagnostic="字节空间不能为负")
    return Quantity(raw=raw, unit=BYTE, value=n, coordinate_base=coordinate_base)


def to_key_count(raw: Any) -> Quantity:
    try:
        n = _parse_int(raw)
    except UnitError as exc:
        return Quantity(raw=raw, unit=KEY_COUNT, diagnostic=str(exc))
    if n < 0:
        return Quantity(raw=raw, unit=KEY_COUNT, value=n, diagnostic="键数不能为负")
    return Quantity(raw=raw, unit=KEY_COUNT, value=n)


def to_lsn(raw: Any) -> Quantity:
    try:
        n = _parse_int(raw)
    except UnitError as exc:
        return Quantity(raw=raw, unit=LSN, diagnostic=str(exc))
    if n < 0:
        return Quantity(raw=raw, unit=LSN, value=n, diagnostic="日志序号不能为负")
    return Quantity(raw=raw, unit=LSN, value=n)


def to_page_id(raw: Any) -> Quantity:
    try:
        n = _parse_int(raw)
    except UnitError as exc:
        return Quantity(raw=raw, unit=PAGE_ID, diagnostic=str(exc))
    if n < 0:
        return Quantity(raw=raw, unit=PAGE_ID, value=n, diagnostic="page id 不能为负")
    return Quantity(raw=raw, unit=PAGE_ID, value=n)


def qty_to_dict(q: Quantity) -> dict[str, Any]:
    return {
        "raw": q.raw,
        "unit": q.unit,
        "value": q.value,
        "coordinate_base": q.coordinate_base,
        "diagnostic": q.diagnostic,
        "ok": q.ok,
        "display": q.display(),
    }


def key_byte_len(key: Any) -> int:
    """变长键按字节计长（UTF-8），这是分裂判定的唯一空间基准。"""
    if isinstance(key, bytes):
        return len(key)
    if isinstance(key, str):
        return len(key.encode("utf-8"))
    if isinstance(key, int):
        return len(str(key).encode("utf-8"))
    raise UnitError(f"键类型无法按字节计量: {type(key).__name__}")


def normalize_key(raw: Any) -> tuple[bytes | None, str | None]:
    """把导入的键规范化为字节；失败返回 (None, 诊断)，原值留在原处。"""
    if isinstance(raw, bytes):
        return raw, None
    if isinstance(raw, str):
        return raw.encode("utf-8"), None
    if isinstance(raw, int):
        return str(raw).encode("utf-8"), None
    if isinstance(raw, dict) and "hex" in raw:
        try:
            return bytes.fromhex(raw["hex"]), None
        except ValueError:
            return None, f"非法十六进制键: {raw['hex']!r}"
    return None, f"键无法转换为字节: {raw!r}"
