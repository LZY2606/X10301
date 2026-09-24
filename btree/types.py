"""Domain values with units / coordinate basis.

Raw representation is preserved verbatim at import time. Normalisation only
happens in the derived layer; a value that cannot be converted stays in place
together with a diagnostic instead of being dropped for display purposes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# coordinate bases understood by the normaliser
UNITLESS = "unitless"
BYTES = "bytes"
PAGES = "pages"
KEYS = "keys"
LSN = "lsn"
PAGE_ID = "page_id"
SLOT_IDX = "slot_index"
BYTE_OFFSET = "byte_offset"
RATIO = "ratio"


@dataclass
class DomainValue:
    """A measurement that always carries its unit / coordinate basis."""

    raw: str
    unit: str
    basis: str = ""
    value: float | int | None = None
    diagnostic: str | None = None

    @property
    def convertible(self) -> bool:
        return self.diagnostic is None and self.value is not None

    def normalised(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "basis": self.basis,
            "value": self.value,
            "diagnostic": self.diagnostic,
            "convertible": self.convertible,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainValue":
        return cls(**{k: data.get(k) for k in ("raw", "unit", "basis", "value", "diagnostic")})


def measure(raw: str, unit: str, basis: str = "") -> DomainValue:
    """Parse a raw measurement into a DomainValue without discarding the raw form."""

    text = str(raw).strip()
    try:
        if unit in (BYTES, PAGES, KEYS):
            value: float | int = int(text)
        elif unit == LSN:
            value = int(text)
        elif unit in (PAGE_ID, SLOT_IDX, BYTE_OFFSET):
            value = int(text)
        elif unit == RATIO:
            if text.endswith("%"):
                value = float(text[:-1]) / 100.0
            else:
                value = float(text)
        elif unit == UNITLESS:
            value = text
        else:
            return DomainValue(text, unit, basis, None, f"unknown unit {unit!r}")
        return DomainValue(text, unit, basis, value, None)
    except (ValueError, TypeError):
        return DomainValue(text, unit, basis, None, f"cannot convert {text!r} as {unit}")


def bytes_val(n: int, basis: str = "page_payload") -> DomainValue:
    return DomainValue(str(n), BYTES, basis, n, None)


def keys_val(n: int) -> DomainValue:
    return DomainValue(str(n), KEYS, "logical_keys", n, None)


def lsn_val(n: int) -> DomainValue:
    return DomainValue(str(n), LSN, "wal_sequence", n, None)
