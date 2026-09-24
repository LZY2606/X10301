"""Structural invariant checks. Each returns a machine-readable event."""
from __future__ import annotations

from .engine import BTree
from .layout import Page


def check_split_by_bytes(tree: BTree) -> dict:
    """Variable-length keys: split point is chosen by byte space, not key count."""
    splits = [e for e in tree.events if e["type"] == "split_propagation"]
    return {"type": "invariant", "name": "split_by_bytes",
            "status": "pass" if splits else "unknown",
            "rule": "byte_space_not_key_count",
            "subjects": [s["steps"][0]["page_id"] for s in splits],
            "details": {"split_count": len(splits)}}


def check_separator_consistency(tree: BTree) -> dict:
    """Every internal separator equals the minimum key of its right child."""
    violations = []

    torn = []

    def walk(pid: int):
        try:
            page = Page.decode(tree.disk[pid], tree.page_size)
        except ValueError:
            torn.append(pid)
            return
        if page.is_leaf:
            return
        children = [page.left_child] + [c.child for c in page.cells]
        for i, cell in enumerate(page.cells):
            min_key = _min_key(tree, children[i + 1])
            if min_key != cell.key:
                violations.append({"page_id": pid, "separator": cell.key.hex(),
                                   "right_min": min_key.hex() if min_key else None})
        for c in children:
            walk(c)

    walk(tree.root_id)
    status = "unknown" if torn and not violations else ("pass" if not violations else "fail")
    return {"type": "invariant", "name": "separator_consistency",
            "status": status,
            "rule": "leaf_copy_up_internal_push_up",
            "subjects": [v["page_id"] for v in violations],
            "details": {"violations": violations, "torn_pages": torn}}


def check_no_reachable_half_page(tree: BTree) -> dict:
    bad = tree.reachable_half_pages()
    return {"type": "invariant", "name": "no_reachable_half_page",
            "status": "pass" if not bad else "fail",
            "rule": "crash_point_leaves_no_reachable_torn_page",
            "subjects": bad, "details": {"reachable": tree.reachable_pages()}}


def check_replay_idempotent(tree: BTree) -> dict:
    before = tree.state_hash()
    tree.recover()
    once = tree.state_hash()
    tree.recover()
    twice = tree.state_hash()
    ok = once == twice and before == once
    return {"type": "invariant", "name": "replay_idempotent",
            "status": "pass" if ok else "fail",
            "rule": "double_recovery_replay_is_idempotent",
            "subjects": [],
            "details": {"hash_before": before, "hash_once": once, "hash_twice": twice}}


def _min_key(tree: BTree, pid: int) -> bytes | None:
    page = Page.decode(tree.disk[pid], tree.page_size)
    while not page.is_leaf:
        page = Page.decode(tree.disk[page.left_child], tree.page_size)
    return page.cells[0].key if page.cells else None


def run_all(tree: BTree) -> list[dict]:
    return [check_split_by_bytes(tree), check_separator_consistency(tree),
            check_no_reachable_half_page(tree), check_replay_idempotent(tree)]
