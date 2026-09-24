"""Parser for the offline page-split case-pack scripts.

Grammar (one directive per line; ``#`` starts a comment)::

    page_size: 128 B
    dup_policy: replace
    set <key_utf8> = <value_utf8>
    del <key_utf8>
    set <key> = <value>  crash=<point>   # scheduled crash during commit
    checkpoint  <label>
    expect      <finding_code> <status>  # informational expectation
    label <name>

The raw text and 1-based line number of every directive are kept; values
that cannot be parsed stay in ``diagnostics`` and never delete data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedLine:
    lineno: int
    raw: str
    directive: str
    error: Optional[str] = None
    page_size_raw: Optional[str] = None
    page_size_bytes: Optional[int] = None
    dup_policy: Optional[str] = None
    op: Optional[str] = None
    key: Optional[str] = None
    value: Optional[str] = None
    crash: Optional[str] = None
    label: Optional[str] = None
    expect_code: Optional[str] = None
    expect_status: Optional[str] = None
    diagnostic: Optional[str] = None


@dataclass
class ParsedScript:
    lines: List[ParsedLine] = field(default_factory=list)
    page_size_raw: str = "128 B"
    dup_policy: str = "replace"
    diagnostics: List[ParsedLine] = field(default_factory=list)

    def operations(self) -> List[ParsedLine]:
        return [l for l in self.lines if l.op]


def parse_case_script(text: str) -> ParsedScript:
    ps = ParsedScript()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = ParsedLine(lineno=lineno, raw=raw, directive=line)
        try:
            _parse_line(parsed, ps)
        except Exception as exc:  # import-layer failures are diagnoses, kept
            parsed.error = str(exc)
            ps.diagnostics.append(parsed)
        ps.lines.append(parsed)
        if parsed.diagnostic:
            ps.diagnostics.append(parsed)
    return ps


def _parse_line(p: ParsedLine, ps: ParsedScript) -> None:
    if p.directive.startswith("page_size:"):
        token = p.directive.split(":", 1)[1].strip()
        p.page_size_raw = token
        from .units import parse_quantity
        q = parse_quantity(token, default_unit="B")
        if q.diagnostic:
            p.diagnostic = q.diagnostic
        else:
            p.page_size_bytes = q.bytes_value()
            ps.page_size_raw = token
        return
    if p.directive.startswith("dup_policy:"):
        p.dup_policy = p.directive.split(":", 1)[1].strip()
        if p.dup_policy not in ("replace", "reject", "multiset"):
            p.diagnostic = f"unknown dup_policy {p.dup_policy!r}; retained"
        else:
            ps.dup_policy = p.dup_policy
        return
    if p.directive.startswith("label "):
        p.label = p.directive.split(None, 1)[1].strip()
        return
    if p.directive.startswith("checkpoint"):
        rest = p.directive.split(None, 1)
        p.label = rest[1].strip() if len(rest) > 1 else None
        p.op = "checkpoint"
        return
    if p.directive.startswith("expect"):
        parts = p.directive.split()
        if len(parts) >= 3:
            p.op = "expect"
            p.expect_code = parts[1]
            p.expect_status = parts[2]
        else:
            p.diagnostic = "expect needs <code> <status>"
        return
    if p.directive.startswith("set ") or p.directive.startswith("del "):
        _parse_op(p)
        return
    p.error = f"unknown directive: {p.directive}"


def _parse_op(p: ParsedLine) -> None:
    body = p.directive
    crash = None
    if "crash=" in body:
        body, tail = body.split("crash=", 1)
        crash = tail.strip().split()[0]
    body = body.strip()
    if body.startswith("set "):
        rest = body[4:]
        if "=" not in rest:
            p.error = "set needs key = value"
            return
        key, value = rest.split("=", 1)
        p.op = "set"
        p.key = key.strip()
        p.value = value.strip()
    elif body.startswith("del "):
        p.op = "del"
        p.key = body[4:].strip()
    p.crash = crash
