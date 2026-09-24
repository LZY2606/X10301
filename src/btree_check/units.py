"""Domain quantities with explicit units / coordinate basis.

Raw representations are preserved verbatim; normalization happens only in the
derived layer. A value that cannot be converted stays at its original location
and gains a diagnostic instead of being silently dropped for the drawing.
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical (derived-layer) units and how to convert into them.
_LENGTH_TO_BYTES = {"B": 1, "byte": 1, "bytes": 1, "KB": 1024, "MB": 1024 * 1024}
# Offsets are byte coordinates inside one fixed-size page.
_COORD_UNITS = {"B": 1, "byte": 1, "bytes": 1}


class UnitError(ValueError):
    """Raised when a raw quantity cannot be converted to its canonical unit."""


@dataclass(frozen=True)
class Quantity:
    """A raw scalar with the unit it was authored in.

    ``raw`` keeps the original textual representation forever; only ``canon``
    is the normalized derived value.
    """

    raw: str
    value: float
    unit: str
    canonical_unit: str
    canonical: float
    diagnosable: str | None = None

    def as_int(self) -> int:
        if float(int(self.canonical)) != self.canonical:
            raise UnitError(f"{self.raw} {self.unit} is not an integral number of {self.canonical_unit}")
        return int(self.canonical)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "value": self.value,
            "unit": self.unit,
            "canonical_unit": self.canonical_unit,
            "canonical": self.canonical,
            "diagnostic": self.diagnosable,
        }


def _split(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if not text:
        raise UnitError("empty quantity")
    cut = 0
    while cut < len(text) and (text[cut].isdigit() or text[cut] in ".+-"):
        cut += 1
    if cut == 0:
        raise UnitError(f"no numeric prefix in {raw!r}")
    return text[:cut], text[cut:].strip()


def length(raw: str | int) -> Quantity:
    """Parse a length quantity into canonical bytes."""
    if isinstance(raw, int):
        return Quantity(str(raw), float(raw), "B", "B", float(raw))
    number, unit = _split(raw)
    try:
        value = float(number)
    except ValueError as exc:
        raise UnitError(f"bad number in {raw!r}") from exc
    factor = _LENGTH_TO_BYTES.get(unit)
    if factor is None:
        return Quantity(raw, value, unit, "B", float("nan"),
                        diagnosable=f"unknown length unit {unit!r}; raw value kept in place")
    return Quantity(raw, value, unit, "B", value * factor)


def coordinate(raw: str | int) -> Quantity:
    """Parse a byte offset inside a page (coordinate basis: byte index)."""
    q = length(raw)
    if q.diagnosable is None:
        return Quantity(q.raw, q.value, q.unit, "byte_offset", q.canonical)
    return Quantity(raw, q.value, q.unit, "byte_offset", float("nan"), diagnosable=q.diagnosable)


def page_size_bytes(raw: str | int) -> int:
    q = length(raw)
    if q.diagnosable:
        raise UnitError(q.diagnosable)
    n = q.as_int()
    if n <= HEADER_SIZE:
        raise UnitError(f"page size {raw} must exceed header {HEADER_SIZE} bytes")
    return n


HEADER_SIZE = 28
LEAF = "leaf"
INTERNAL = "internal"
META = "meta"

MAGIC = b"PSAB"  # Page-Split Archive B-tree
