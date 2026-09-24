"""Domain quantities carrying explicit units / coordinate bases.

Every numeric value that originates from the domain (byte offsets, slot
indices, page ids, LSN sequence numbers) is represented with its unit so the
UI never has to guess whether "12" means twelve bytes, twelve keys or page
number twelve.  Values that cannot be converted are kept verbatim by the
importer (see :mod:`btree_lab.importer`); nothing here silently drops data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class Quantity:
    """A scalar domain value with an explicit unit and coordinate base."""

    value: int
    unit: str
    base: str = "zero"

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "base": self.base}


def bytes_at(value: int) -> Quantity:
    return Quantity(int(value), "byte", "zero")


def slots(value: int) -> Quantity:
    return Quantity(int(value), "slot", "zero")


def keys(value: int) -> Quantity:
    return Quantity(int(value), "key", "zero")


def lsn(value: int) -> Quantity:
    return Quantity(int(value), "lsn", "one")


def page_id(value: int) -> Quantity:
    return Quantity(int(value), "page_id", "zero")


def quantity_to_dict(obj: Any) -> Any:
    if isinstance(obj, Quantity):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: quantity_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [quantity_to_dict(v) for v in obj]
    return obj
