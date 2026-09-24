"""Units / coordinates.

All domain quantities keep their original representation.  Normalisation into a
canonical unit is a *derived* layer; values that cannot be converted stay at
their original position with a diagnostic instead of being dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

# Byte-based units used for page layout.
BYTE_UNITS = {"B": 1, "byte": 1, "bytes": 1, "KB": 1024, "KiB": 1024}


class UnitError(ValueError):
    pass


@dataclass(frozen=True)
class Quantity:
    raw: str
    value: float | None
    unit: str | None
    canonical_bytes: int | None
    diagnostic: str | None = None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "value": self.value,
            "unit": self.unit,
            "canonical_bytes": self.canonical_bytes,
            "diagnostic": self.diagnostic,
        }


def parse_quantity(raw: object) -> Quantity:
    """Parse ``"128 B"`` / ``128`` / ``"0x80"`` into a byte-based Quantity.

    Non-convertible values are preserved with a diagnostic (never dropped).
    """
    if isinstance(raw, bool):
        return Quantity(repr(raw), None, None, None, "boolean is not a quantity")
    if isinstance(raw, int):
        return Quantity(str(raw), float(raw), "B", raw, None)
    if not isinstance(raw, str):
        return Quantity(repr(raw), None, None, None, f"unsupported type {type(raw).__name__}")
    text = raw.strip()
    unit = None
    token = text
    for u in sorted(BYTE_UNITS, key=len, reverse=True):
        if text.endswith(u):
            token = text[: -len(u)].strip()
            unit = u
            break
    try:
        value = int(token, 0)
    except ValueError:
        return Quantity(raw, None, unit, None, f"cannot interpret {text!r} as byte size")
    if value < 0:
        return Quantity(raw, float(value), unit, None, "negative size")
    scale = BYTE_UNITS[unit] if unit else 1
    return Quantity(raw, float(value), unit or "B", value * scale, None)


def page_bytes(raw: object) -> int:
    q = parse_quantity(raw)
    if q.canonical_bytes is None:
        raise UnitError(q.diagnostic or "unconvertible page size")
    return q.canonical_bytes
