"""Domain quantities keep their raw representation; normalization happens here,
in the derived layer. Unconvertible values stay in place with a diagnostic."""
import re

_UNIT_TABLE = {
    "B": ("byte", 1), "byte": ("byte", 1), "bytes": ("byte", 1),
    "KB": ("byte", 1024), "KiB": ("byte", 1024), "MB": ("byte", 1024 * 1024),
    "page": ("page", 1), "pages": ("page", 1),
    "slot": ("slot", 1), "slots": ("slot", 1),
}

_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")


def parse_quantity(raw):
    """Return {'value','unit','raw'} or {'value':None,'unit':None,'raw','diagnostic'}."""
    if isinstance(raw, (int, float)):
        return {"value": raw, "unit": "count", "raw": raw}
    m = _PATTERN.match(str(raw))
    if not m:
        return {"value": None, "unit": None, "raw": raw,
                "diagnostic": f"cannot parse quantity: {raw!r}"}
    number, suffix = float(m.group(1)), m.group(2)
    if suffix not in _UNIT_TABLE:
        return {"value": None, "unit": None, "raw": raw,
                "diagnostic": f"unknown unit {suffix!r} in {raw!r}; value left unconverted"}
    unit, factor = _UNIT_TABLE[suffix]
    value = number * factor
    if value == int(value):
        value = int(value)
    return {"value": value, "unit": unit, "raw": raw}
