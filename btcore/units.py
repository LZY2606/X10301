"""Domain quantities that carry units or a coordinate basis.

Raw representations are preserved at the import layer (``raw`` attribute);
only the derived layer performs normalization. A value that cannot be
converted stays where it was, carrying a diagnostic, instead of being
silently dropped for the sake of a diagram.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


BYTE_PREFIXES = {
    "B": 1,
    "byte": 1,
    "bytes": 1,
    "KiB": 1024,
    "MiB": 1024 ** 2,
}


class InconvertibleError(ValueError):
    """Raised when a raw value cannot be converted to the requested unit."""

    def __init__(self, raw: str, unit: str, reason: str = ""):
        super().__init__(f"inconvertible value {raw!r} -> {unit}: {reason}".rstrip(": "))
        self.raw = raw
        self.unit = unit
        self.reason = reason


@dataclass(frozen=True)
class Coordinate:
    """Coordinate basis of a measurement on a page."""

    origin: str = "page_start"  # page_start | slot_end | cell_start
    unit: str = "byte"

    def key(self) -> str:
        return f"{self.origin}/{self.unit}"


@dataclass(frozen=True)
class Quantity:
    """A measurement with a value, unit and coordinate basis.

    ``raw`` is the original textual representation from the case pack.
    Diagnostics stay attached instead of deleting the value.
    """

    value: float
    unit: str
    coord: Coordinate = field(default_factory=Coordinate)
    raw: Optional[str] = None
    diagnostic: Optional[str] = None

    def to(self, unit: str) -> "Quantity":
        factor = _conversion(self.unit, unit)
        return Quantity(
            value=self.value * factor,
            unit=unit,
            coord=self.coord,
            raw=self.raw,
            diagnostic=self.diagnostic,
        )

    def bytes_value(self) -> int:
        return int(self.to("B").value)

    def with_diagnostic(self, message: str) -> "Quantity":
        return Quantity(self.value, self.unit, self.coord, self.raw, message)


def _conversion(src: str, dst: str) -> float:
    if src == dst:
        return 1.0
    if src in BYTE_PREFIXES and dst in BYTE_PREFIXES:
        return BYTE_PREFIXES[src] / BYTE_PREFIXES[dst]
    raise InconvertibleError("", dst, f"no conversion {src} -> {dst}")


def parse_quantity(text: Any, default_unit: str = "B") -> Quantity:
    """Parse ``"4096 B"`` / ``"1 KiB"`` / ``"128"`` into a byte Quantity.

    Unconvertible input yields a Quantity carrying a diagnostic (value None
    is represented as NaN) rather than raising; the import layer decides
    whether the problem is fatal.
    """

    raw = text if isinstance(text, str) else str(text)
    token = raw.strip()
    if not token:
        raise InconvertibleError(raw, default_unit, "empty")
    parts = token.split()
    if len(parts) == 1:
        number, unit = parts[0], default_unit
    else:
        number, unit = parts[0], parts[1]
    try:
        value = int(number)
    except ValueError:
        try:
            value = float(number)
        except ValueError as exc:
            raise InconvertibleError(raw, unit, "not numeric") from exc
    if unit not in BYTE_PREFIXES:
        q = Quantity(float("nan"), unit, raw=raw)
        return q.with_diagnostic(f"unknown unit {unit!r}; kept raw representation")
    return Quantity(value, unit, raw=raw)
