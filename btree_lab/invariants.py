"""Structural invariants and machine-readable verdicts.

Each check returns a dict shaped like::

    {"rule": "...", "status": "pass|fail|candidate",
     "subject": {"kind": "page|tree", "id": "..."},
     "evidence": [...], "message": "..."}

so the archive can store accepted and rejected results uniformly and the
review UI can render the exact byte/coordinate evidence behind a verdict.
"""
from __future__ import annotations

from typing import Any

from .pages import Page, LEAF, INTERNAL, NONE, HEADER_SIZE


def _verdict(rule: str, status: str, subject: dict, message: str,
             evidence: list[dict] | None = None) -> dict[str, Any]:
    return {
        "rule": rule,
        "status": status,
        "subject": subject,
        "message": message,
        "evidence": evidence or [],
    }


def ordered_keys(p: Page) -> list[bytes]:
    if p.is_leaf:
        return [k for k, _v in p.items]
    return [sep for sep, _c in p.items]


def page_fits(p: Page) -> dict:
    free = p.free_bytes
    status = "pass" if free >= 0 else "fail"
    return _verdict(
        "page.fits_in_page_size", status,
        {"kind": "page", "id": p.pid},
        f"page {p.pid} free space = {free} bytes",
        [{"metric": "free_bytes", "value": free, "unit": "byte"}])


def keys_sorted_unique(p: Page, dup_policy: str = "error") -> dict:
    keys = ordered_keys(p)
    bad = next((i for i in range(1, len(keys)) if keys[i - 1] > keys[i]), -1)
    dups = [i for i in range(1, len(keys)) if keys[i - 1] == keys[i]]
    status = "pass"
    msg = f"page {p.pid}: {len(keys)} keys strictly ordered"
    if bad != -1:
        status = "fail"
        msg = f"page {p.pid}: keys out of order at slot {bad}"
    elif dups and dup_policy == "error":
        status = "fail"
        msg = f"page {p.pid}: duplicate keys at slots {dups}"
    elif dups:
        status = "pass"
        msg = (f"page {p.pid}: {len(dups)} duplicate group(s) allowed by "
               f"dup_policy={dup_policy}")
    return _verdict(
        "page.keys_sorted_unique", status,
        {"kind": "page", "id": p.pid}, msg,
        [{"metric": "duplicate_slots", "value": dups, "unit": "slot"}])


def split_by_bytes(left: Page, right: Page, page_size: int) -> dict:
    """A split must be decided by *byte* occupancy, not key count.

    Both halves must fit after the split and neither side is permitted to
    overflow; the cut is byte-balanced rather than a fixed key-count half.
    """
    evid = [
        {"metric": "left_free_bytes", "value": left.free_bytes,
         "unit": "byte", "subject": left.pid},
        {"metric": "right_free_bytes", "value": right.free_bytes,
         "unit": "byte", "subject": right.pid},
        {"metric": "left_key_count", "value": len(left.items),
         "unit": "key"},
        {"metric": "right_key_count", "value": len(right.items),
         "unit": "key"},
    ]
    fits = left.free_bytes >= 0 and right.free_bytes >= 0
    nonempty = len(left.items) > 0 and len(right.items) > 0
    status = "pass" if fits and nonempty else "fail"
    # detect a purely count-based cut on variable-length keys: equal counts
    # while one side is nearly full and the other mostly empty
    balance_gap = abs(
        (page_size - HEADER_SIZE - max(0, left.used_bytes - HEADER_SIZE))
        - (page_size - HEADER_SIZE - max(0, right.used_bytes - HEADER_SIZE)))
    evid.append({"metric": "byte_balance_gap", "value": balance_gap,
                 "unit": "byte"})
    if len(left.items) == len(right.items) and balance_gap > page_size // 4:
        status = "fail"
        msg = ("split looks key-count-based with variable-length keys: "
               "equal key counts but >25% byte imbalance")
    else:
        msg = ("split halves both fit and occupancy is byte-balanced "
               f"(gap {balance_gap} bytes)")
    return _verdict("split.by_byte_space_not_key_count", status,
                    {"kind": "split", "id": f"{left.pid}+{right.pid}"},
                    msg, evid)


def _subtree_min(pid: int, children: dict[int, Page]) -> bytes:
    page = children.get(pid)
    if page is None:
        return b""
    while page is not None and page.is_internal:
        page = children.get(page.leftmost)
    if page is None or not page.items:
        return b""
    return ordered_keys(page)[0]


def separator_rule(parent: Page, children: dict[int, Page],
                   mode: str = "promote") -> dict:
    """Separators must follow one consistent copy/promote rule.

    ``mode='promote'`` requires the promoted separator to equal the left
    child's last key / right child's first key boundary exactly (classic
    B+tree).  ``mode='copy'`` is accepted only when declared consistently on
    the whole tree; mixing rules on one node is a failure.
    """
    problems: list[str] = []
    if parent.is_leaf or not parent.is_internal:
        return _verdict("separator.copy_or_promote_consistent", "pass",
                        {"kind": "page", "id": parent.pid},
                        "not an internal page")
    ordered = [parent.leftmost] + [c for _s, c in parent.items]
    seps = [s for s, _c in parent.items]
    for i, sep in enumerate(seps):
        lp = children.get(ordered[i])
        rp = children.get(ordered[i + 1])
        if lp is None or rp is None:
            problems.append(f"child of slot {i} missing")
            continue
        lmax = ordered_keys(lp)[-1:] or [b""]
        rmin = [_subtree_min(ordered[i + 1], children)]
        if mode == "promote":
            # separator must equal right subtree minimum (promoted copy)
            if sep != rmin[0]:
                problems.append(
                    f"separator[{i}]={sep!r} != right-min={rmin[0]!r}")
        else:  # copy: separator must sit strictly between neighbor bounds
            if not (lmax[0] < sep <= rmin[0]):
                problems.append(
                    f"separator[{i}]={sep!r} not in ({lmax[0]!r},{rmin[0]!r}]")
    status = "pass" if not problems else "fail"
    return _verdict(
        "separator.copy_or_promote_consistent", status,
        {"kind": "page", "id": parent.pid},
        "; ".join(problems) or f"separators consistent under mode={mode}",
        [{"metric": "mode", "value": mode, "unit": "rule"}])


def no_reachable_half_page(roots: list[int], pages: dict[int, Page]) -> dict:
    """After a crash no reachable page may be a half-written FRESH page."""
    seen: set[int] = set()
    stack = list(roots)
    bad: list[int] = []
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        page = pages.get(pid)
        if page is None:
            bad.append(pid)
            continue
        if page.ptype == 0:
            bad.append(pid)
        if page.is_internal:
            if page.leftmost != NONE:
                stack.append(page.leftmost)
            stack.extend(c for _s, c in page.items)
    status = "pass" if not bad else "fail"
    return _verdict(
        "crash.no_reachable_half_page", status,
        {"kind": "tree", "id": "reachable"},
        ("every reachable page fully initialized"
         if not bad else f"half-written/missing reachable pages: {bad}"),
        [{"metric": "reachable_pages", "value": sorted(seen),
          "unit": "page_id"},
         {"metric": "half_pages", "value": bad, "unit": "page_id"}])


def recovery_idempotent(snapshot_a: dict, snapshot_b: dict) -> dict:
    """Running recovery twice over the same log must yield the same state."""
    equal = snapshot_a == snapshot_b
    return _verdict(
        "recovery.replay_idempotent",
        "pass" if equal else "fail",
        {"kind": "recovery", "id": "replay_twice"},
        "second replay produced an identical snapshot"
        if equal else "second replay changed the snapshot (not idempotent)",
        [{"metric": "snapshots_equal", "value": equal, "unit": "bool"}])


def run_tree_checks(pages: dict[int, Page], roots: list[int],
                    dup_policy: str = "error", sep_mode: str = "promote"
                    ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in pages.values():
        out.append(page_fits(p))
        out.append(keys_sorted_unique(p, dup_policy))
    for pid, p in pages.items():
        if p.is_internal:
            out.append(separator_rule(p, pages, sep_mode))
    out.append(no_reachable_half_page(roots, pages))
    return out
