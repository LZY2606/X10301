"""Domain quantities with explicit units / coordinate basis.

Raw fragments keep their original representation. Only the derivation layer
normalizes values; values that cannot be converted stay at their origin with a
diagnostic instead of being dropped for display convenience.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BYTE_UNIT = "B"          # byte space on a page
KEY_COUNT_UNIT = "key"   # number of keys (kept distinct from bytes on purpose)
LSN_UNIT = "lsn"         # log sequence number
PAGE_ID_UNIT = "pid"     # page identifier coordinate
OFFSET_UNIT = "offset_B" # byte offset inside a page, origin = page start

_KILO = 1024
_BYTE_FACTORS = {
    "B": 1,
    "byte": 1,
    "bytes": 1,
    "KB": _KILO,
    "K": _KILO,
    "MB": _KILO * _KILO,
}


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    raw: str | None = None
    basis: str | None = None  # coordinate basis, e.g. "page_start"

    def to_bytes(self) -> "Quantity":
        factor = _BYTE_FACTORS.get(self.unit)
        if factor is None:
            raise UnitError(f"cannot convert unit {self.unit!r} to bytes")
        return Quantity(self.value * factor, BYTE_UNIT, raw=self.raw, basis=self.basis)


class UnitError(ValueError):
    pass


@dataclass
class ConversionDiagnostic:
    raw: str
    declared_unit: str | None
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "declared_unit": self.declared_unit,
            "message": self.message,
            "severity": self.severity,
        }


def parse_size(token: Any) -> tuple[Quantity | None, ConversionDiagnostic | None]:
    """Parse a raw token into a byte-space Quantity, preserving the raw text.

    Returns ``(None, diagnostic)`` instead of raising so callers can keep the
    unconvertible value at its origin.
    """
    if isinstance(token, (int, float)) and not isinstance(token, bool):
        return Quantity(float(token), BYTE_UNIT, raw=str(token),
                        basis="page_start"), None
    if not isinstance(token, str):
        return None, ConversionDiagnostic(str(token), None,
                                          "size token is neither number nor string")
    text = token.strip()
    digits = ""
    i = 0
    while i < len(text) and (text[i].isdigit() or text[i] == "."):
        digits += text[i]
        i += 1
    unit = text[i:].strip() or BYTE_UNIT
    if not digits or unit not in _BYTE_FACTORS:
        return None, ConversionDiagnostic(text, unit or None,
                                          f"unparseable size with unit {unit!r}")
    try:
        value = float(digits)
    except ValueError:
        return None, ConversionDiagnostic(text, unit, "non-numeric magnitude")
    q = Quantity(value, unit, raw=text, basis="page_start")
    return q.to_bytes(), None
