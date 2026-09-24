"""Deterministic B+tree engine used to *generate* the teaching archive.

The engine is deliberately small but real: fixed-size pages from
:mod:`btree_lab.pages`, WAL from :mod:`btree_lab.wal`, byte-balanced splits,
root and cascading splits, pluggable duplicate-key and empty-page policies,
and crash injection for write-page interruption experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .pages import (
    Page, LEAF, INTERNAL, FRESH, NONE, DEFAULT_PAGE_SIZE,
    leaf_page, internal_page, fresh_page,
)
from .wal import Wal, PageStore, BEGIN, UPDATE, ALLOC, ROOT, COMMIT

# duplicate-key policies
DUP_ERROR = "error"
DUP_LAST = "last_wins"
DUP_MULTI = "multi_value"
# empty-leaf policies after the last key is removed
EMPTY_KEEP = "keep_empty_page"
EMPTY_FREE = "free_empty_page"
EMPTY_UNLINK = "unlink_empty_leaf"


@dataclass
class EngineConfig:
    page_size: int = DEFAULT_PAGE_SIZE
    split_metric: str = "bytes"   # "bytes" (correct) or "keys" (counterexample)
    separator_rule: str = "promote"
    dup_policy: str = DUP_ERROR
    empty_policy: str = EMPTY_KEEP

    def to_dict(self) -> dict[str, str]:
        return {
            "page_size": str(self.page_size),
            "split_metric": self.split_metric,
            "separator_rule": self.separator_rule,
            "dup_policy": self.dup_policy,
            "empty_policy": self.empty_policy,
        }


@dataclass
class Change:
    kind: str
    page: Page
    before: Optional[bytes] = None  # raw before-image (None if new)
    torn: bool = False

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {"pid": self.page.pid}
        if self.kind == ALLOC:
            body["image"] = {"bytes": self.page.serialize().hex()}
        else:
            body["before"] = (None if self.before is None
                              else {"bytes": self.before.hex()})
            body["after"] = {"bytes": self.page.serialize().hex()}
        return body


@dataclass
class OpTrace:
    """Everything reviewers need for one logical operation."""

    seq: int
    op: dict[str, Any]
    before_snapshot: dict[int, dict[str, Any]]
    changes: list[dict[str, Any]] = field(default_factory=list)
    root_before: int = NONE
    root_after: int = NONE
    lsn_range: list[int] = field(default_factory=list)
    pages_allocated: list[int] = field(default_factory=list)
    split_events: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""


def _cell_bytes_for(page: Page, n: int) -> np.ndarray:
    """Cumulative bytes consumed by the first ``n`` items (cells + slots)."""
    sizes = [page.cell_size(it) + 2 for it in page.items[:n]]
    return np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])


def byte_split_point(page: Page, config: EngineConfig) -> int:
    """Choose the number of items that move to the new right page.

    Correct rule (``split_metric='bytes'``): cut where the page header plus
    one side no longer overflows, choosing the most byte-balanced feasible
    cut.  Counterexample rule (``'keys'``): a fixed count-based half, which
    misbalances badly for variable-length keys.
    """
    n = len(page.items)
    if config.split_metric == "keys":
        return n // 2
    header = 40
    capacity = config.page_size - header
    cum = _cell_bytes_for(page, n)
    total = int(cum[n])
    feasible = [k for k in range(1, n)
                if int(cum[k]) <= capacity and total - int(cum[k]) <= capacity]
    if not feasible:  # a single item cannot fit: still force a cut
        feasible = [k for k in range(1, n)]
    if not feasible:
        return max(1, n // 2)
    k = min(feasible, key=lambda kk: (abs(int(cum[kk]) - (total - int(cum[kk]))),
                                      kk))
    return k


class Engine:
    def __init__(self, config: Optional[EngineConfig] = None,
                 store: Optional[PageStore] = None, wal: Optional[Wal] = None):
        self.config = config or EngineConfig()
        self.store = store or PageStore(self.config.page_size)
        self.wal = wal or Wal()
        self.pages: dict[int, Page] = {}
        self.root = NONE
        self.next_pid = 0
        self.txn = 0

    # ------------------------------------------------------------- loading
    def _load_all(self) -> None:
        self.pages = {
            pid: Page.parse(raw, self.config.page_size)
            for pid, raw in self.store.pages.items()
        }
        if self.pages:
            self.next_pid = max(self.pages) + 1

    def _alloc(self, page: Page, changes: list[Change]) -> Page:
        page.pid = self.next_pid
        self.next_pid += 1
        self.pages[page.pid] = page
        changes.append(Change(ALLOC, page))
        return page

    def _mut(self, page: Page, changes: list[Change]) -> None:
        raw = self.store.read(page.pid)
        changes.append(Change(UPDATE, page, before=raw))

    def _snapshot_meta(self) -> dict[int, dict[str, Any]]:
        out = {}
        for pid, p in self.pages.items():
            out[pid] = self._page_meta(p)
        return out

    @staticmethod
    def _page_meta(p: Page) -> dict[str, Any]:
        return {
            "pid": p.pid, "type": ("leaf" if p.is_leaf
                                   else "internal" if p.is_internal
                                   else "fresh"),
            "parent": p.parent, "prev": p.prev_sibling, "next": p.next_sibling,
            "slot_count": p.slot_count, "free_bytes": p.free_bytes,
            "used_bytes": p.used_bytes, "page_lsn": p.page_lsn,
            "empty": p.empty_flag,
            "keys": [k.hex() for k in (
                [x[0] for x in p.items] if p.is_leaf
                else [x[0] for x in p.items])],
        }

    def _subtree_min(self, pid: int) -> bytes:
        node = self.pages[pid]
        while node.is_internal:
            node = self.pages[node.leftmost]
        return node.items[0][0]

    def _subtree_max_key(self, pid: int) -> bytes:
        node = self.pages[pid]
        while node.is_internal:
            node = self.pages[node.items[-1][1]]
        return node.items[-1][0]

    # ------------------------------------------------------------- commits
    def _commit(self, changes: list[Change], op_trace: OpTrace,
                root_changed: bool,
                crash: Optional[dict[str, Any]] = None) -> None:
        self.txn += 1
        txn = self.txn
        first = self.wal.append(BEGIN, txn, {"op_seq": op_trace.seq})
        pids = {c.page.pid for c in changes}
        if root_changed:
            self.wal.append(ROOT, txn, {"root": self.root})
        for change in changes:
            change.page.page_lsn = self.wal.records[-1].lsn + 1
            if change.kind == ALLOC:
                self.wal.append(ALLOC, txn, change.payload())
            else:
                self.wal.append(UPDATE, txn, change.payload())
        op_trace.lsn_range = [first.lsn, self.wal.records[-1].lsn]
        op_trace.pages_allocated = [c.page.pid for c in changes
                                    if c.kind == ALLOC]
        for c in changes:
            self._write_change(c, crash)
        if crash and crash.get("before_commit"):
            raise CrashInjected(op_trace,
                                "crash before COMMIT (undo path)")
        commit_rec = self.wal.append(COMMIT, txn, {})
        op_trace.lsn_range[1] = commit_rec.lsn
        if crash and crash.get("after_commit"):
            raise CrashInjected(op_trace,
                                "crash after COMMIT (redo path)")

    def _write_change(self, change: Change,
                      crash: Optional[dict[str, Any]]) -> None:
        torn = False
        if crash and crash.get("pid") == change.page.pid:
            torn = True
        raw = change.page.serialize()
        if change.kind == ALLOC:
            self.store.write(change.page.pid, raw, torn=torn,
                             torn_at=crash and crash.get("torn_at"))
        else:
            self.store.write(change.page.pid, raw, torn=torn,
                             torn_at=crash and crash.get("torn_at"))


class CrashInjected(RuntimeError):
    def __init__(self, trace: OpTrace, where: str):
        super().__init__(where)
        self.trace = trace
        self.where = where


def _dup_value(existing: bytes, new: bytes, policy: str) -> bytes:
    if policy == DUP_LAST:
        return new
    return existing + b"\x00" + new  # multi_value: append versions


class Engine(Engine):  # noqa: F811 - reopen class with operational methods

    # =============================================================== INSERT
    def insert(self, key: bytes, value: bytes,
               seq: int, crash: Optional[dict[str, Any]] = None) -> OpTrace:
        trace = OpTrace(seq=seq, op={"op": "insert", "key": key.hex(),
                                    "value": value.hex()},
                        before_snapshot=self._snapshot_meta(),
                        root_before=self.root)
        changes: list[Change] = []
        if self.root == NONE:
            leaf = self._alloc(
                leaf_page(-1, self.config.page_size,
                          items=[(key, value)], parent=NONE), changes)
            self.root = leaf.pid
            self._commit(changes, trace, root_changed=True, crash=crash)
            trace.root_after = self.root
            trace.changes = [c.payload() for c in changes]
            return trace

        path: list[tuple[int, int]] = []  # (pid, child index chosen)
        node = self.pages[self.root]
        while node.is_internal:
            child = node.child_for(key)
            idx = ([node.leftmost] + [c for _s, c in node.items]).index(child)
            path.append((node.pid, idx))
            node = self.pages[child]

        leaf = node
        pos, exact = leaf.leaf_index(key)
        if exact:
            if self.config.dup_policy == DUP_ERROR:
                raise DuplicateKeyError(key, seq)
            old_key, old_val = leaf.items[pos]
            leaf.items[pos] = (old_key,
                               _dup_value(old_val, value,
                                          self.config.dup_policy))
            self._mut(leaf, changes)
        else:
            leaf.items.insert(pos, (key, value))
            self._mut(leaf, changes)
            if leaf.free_bytes < 0:
                self._split_leaf(leaf, changes, trace, path, crash)
        self._commit(changes, trace,
                     root_changed=self.root != trace.root_before,
                     crash=crash)
        trace.root_after = self.root
        trace.changes = [c.payload() for c in changes]
        return trace

    # ---------------------------------------------------------------- SPLIT
    def _split_leaf(self, leaf: Page, changes: list[Change], trace: OpTrace,
                    path: list[tuple[int, int]],
                    crash: Optional[dict[str, Any]]) -> None:
        k = byte_split_point(leaf, self.config)
        left_items, right_items = leaf.items[:k], leaf.items[k:]
        old_next = leaf.next_sibling
        right = self._alloc(
            leaf_page(-1, self.config.page_size, items=right_items,
                      parent=leaf.parent, prev_sibling=leaf.pid,
                      next_sibling=old_next), changes)
        leaf.items = left_items
        leaf.next_sibling = right.pid
        if old_next != NONE:
            old = self.pages[old_next]
            old.prev_sibling = right.pid
            self._mut(old, changes)
        sep = self._subtree_min(right.pid)
        trace.split_events.append({
            "kind": "leaf_split", "parent_was": leaf.parent,
            "left_pid": leaf.pid, "right_pid": right.pid,
            "separator": sep.hex(),
            "left_keys": len(left_items), "right_keys": len(right_items),
            "left_bytes": left_items and sum(
                leaf.cell_size(x) + 2 for x in left_items),
            "right_bytes": sum(right.cell_size(x) + 2
                               for x in right_items),
            "metric": self.config.split_metric,
        })
        self._insert_in_parent(leaf.pid, right.pid, sep,
                               changes, trace, path, crash)

    def _insert_in_parent(self, left_pid: int, right_pid: int, sep: bytes,
                          changes: list[Change], trace: OpTrace,
                          path: list[tuple[int, int]],
                          crash: Optional[dict[str, Any]]) -> None:
        if not path:  # leaf was the root: grow a new root
            new_root = self._alloc(
                internal_page(-1, self.config.page_size,
                              items=[(sep, right_pid)],
                              leftmost=left_pid, parent=NONE), changes)
            self.pages[left_pid].parent = new_root.pid
            self.pages[right_pid].parent = new_root.pid
            self._mut(self.pages[left_pid], changes)
            self._mut(self.pages[right_pid], changes)
            self.root = new_root.pid
            trace.split_events.append({"kind": "root_split",
                                       "new_root": new_root.pid,
                                       "separator": sep.hex()})
            self._record_root_change(trace, new_root)
            return

        parent_pid, _idx = path.pop()
        parent = self.pages[parent_pid]
        pos = 0
        while pos < len(parent.items) and parent.items[pos][0] <= sep:
            pos += 1
        parent.items.insert(pos, (sep, right_pid))
        self.pages[right_pid].parent = parent_pid
        self._mut(parent, changes)
        if parent.free_bytes < 0:
            self._split_internal(parent, changes, trace, path, crash)

    def _split_internal(self, node: Page, changes: list[Change],
                        trace: OpTrace, path: list[tuple[int, int]],
                        crash: Optional[dict[str, Any]]) -> None:
        k = byte_split_point(node, self.config)
        # promote separator k: left keeps items[:k], right keeps items[k+1:]
        promote = node.items[k][0]
        left_items = node.items[:k]
        right_items = node.items[k + 1:]
        right_children = [node.items[k][1]] + [c for _s, c in right_items]
        left_children = [node.leftmost] + [c for _s, c in left_items]
        right = self._alloc(
            internal_page(-1, self.config.page_size,
                          items=right_items, leftmost=right_children[0],
                          parent=node.parent), changes)
        node.items = left_items
        for c in right_children:
            self.pages[c].parent = right.pid
            self._mut(self.pages[c], changes)
        for c in left_children:
            if self.pages[c].parent != node.pid:
                self.pages[c].parent = node.pid
                self._mut(self.pages[c], changes)
        trace.split_events.append({
            "kind": "internal_split", "parent_was": node.parent,
            "left_pid": node.pid, "right_pid": right.pid,
            "promoted_separator": promote.hex(),
            "rule": self.config.separator_rule,
        })
        self._insert_in_parent(node.pid, right.pid, promote,
                               changes, trace, path, crash)

    def _record_root_change(self, trace: OpTrace, new_root: Page) -> None:
        trace.root_after = new_root.pid


class DuplicateKeyError(ValueError):
    def __init__(self, key: bytes, seq: int):
        super().__init__(f"duplicate key {key!r} at op {seq}")
        self.key = key
        self.seq = seq


class Engine(Engine):  # noqa: F811

    # =============================================================== DELETE
    def delete(self, key: bytes, seq: int,
               crash: Optional[dict[str, Any]] = None) -> OpTrace:
        trace = OpTrace(seq=seq, op={"op": "delete", "key": key.hex()},
                        before_snapshot=self._snapshot_meta(),
                        root_before=self.root)
        changes: list[Change] = []
        leaf = self._find_leaf(key)
        note = "key absent; no page change"
        if leaf is not None:
            pos, exact = leaf.leaf_index(key)
            if exact:
                leaf.items.pop(pos)
                self._mut(leaf, changes)
                note = self._after_delete(leaf, changes, trace, crash)
        trace.note = note
        self._commit(changes, trace,
                     root_changed=self.root != trace.root_before, crash=crash)
        trace.root_after = self.root
        trace.changes = [c.payload() for c in changes]
        return trace

    def _find_leaf(self, key: bytes) -> Optional[Page]:
        if self.root == NONE:
            return None
        node = self.pages[self.root]
        while node.is_internal:
            node = self.pages[node.child_for(key)]
        return node

    def _after_delete(self, leaf: Page, changes: list[Change],
                      trace: OpTrace, crash) -> str:
        if leaf.items:
            return "key removed; leaf still has keys"
        policy = self.config.empty_policy
        leaf.mark_empty(True)
        if policy == EMPTY_KEEP:
            return "leaf became empty and is retained (keep_empty_page)"
        if policy == EMPTY_FREE:
            leaf.items = []
            leaf.mark_empty(True)
            self.pages.pop(leaf.pid, None)
            self.store.pages.pop(leaf.pid, None)
            changes[:] = [c for c in changes if c.page.pid != leaf.pid]
            trace.split_events.append({"kind": "empty_page_freed",
                                       "pid": leaf.pid})
            return "empty leaf page freed"
        # unlink: only legal when the empty leaf is not the leftmost child
        if leaf.parent == NONE:
            leaf.mark_empty(True)
            return ("empty leaf is the only root; unlink illegal, kept as "
                    "reachable empty page [candidate retained]")
        parent = self.pages[leaf.parent]
        child_order = [parent.leftmost] + [c for _s, c in parent.items]
        idx = child_order.index(leaf.pid)
        if idx == 0:
            leaf.mark_empty(True)
            return ("empty leaf is leftmost child; unlink would lose the "
                    "left subtree, so it is kept [candidate retained]")
        # remove separator idx-1 and pointer idx from parent
        sep = parent.items.pop(idx - 1)[0]
        if leaf.prev_sibling != NONE:
            self.pages[leaf.prev_sibling].next_sibling = leaf.next_sibling
            self._mut(self.pages[leaf.prev_sibling], changes)
        if leaf.next_sibling != NONE:
            self.pages[leaf.next_sibling].prev_sibling = leaf.prev_sibling
            self._mut(self.pages[leaf.next_sibling], changes)
        self._mut(parent, changes)
        self.pages.pop(leaf.pid, None)
        self.store.pages.pop(leaf.pid, None)
        changes[:] = [c for c in changes if c.page.pid != leaf.pid]
        trace.split_events.append({"kind": "empty_page_unlinked",
                                   "pid": leaf.pid, "separator": sep.hex()})
        return "empty leaf unlinked from parent and sibling chain"

    # ============================================================== DRIVER
    def execute(self, op: dict[str, Any], seq: int,
                crash: Optional[dict[str, Any]] = None) -> OpTrace:
        kind = op["op"]
        key = bytes.fromhex(op["key"])
        if kind == "insert":
            value = bytes.fromhex(op.get("value", ""))
            return self.insert(key, value, seq, crash=crash)
        if kind == "delete":
            return self.delete(key, seq, crash=crash)
        raise ValueError(f"unknown op {kind!r}")

    def run_ops(self, ops: list[dict[str, Any]],
                crash_plan: Optional[dict[str, Any]] = None
                ) -> list[OpTrace]:
        traces: list[OpTrace] = []
        for i, op in enumerate(ops, start=1):
            crash = crash_plan if (crash_plan and
                                   crash_plan.get("op_seq") == i) else None
            traces.append(self.execute(op, i, crash=crash))
        return traces

    # ----------------------------------------------------------- snapshots
    def reachable_pages(self) -> list[int]:
        seen: list[int] = []
        if self.root == NONE:
            return seen
        stack = [self.root]
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.append(pid)
            page = self.pages.get(pid)
            if page is not None and page.is_internal:
                if page.leftmost != NONE:
                    stack.append(page.leftmost)
                stack.extend(c for _s, c in page.items)
        return sorted(seen)

    def tree_shape(self) -> dict[str, Any]:
        def node(pid: int) -> dict[str, Any]:
            p = self.pages[pid]
            if p.is_leaf:
                return {"pid": pid, "type": "leaf",
                        "keys": [k.hex() for k, _v in p.items],
                        "next": p.next_sibling, "prev": p.prev_sibling,
                        "free_bytes": p.free_bytes, "empty": p.empty_flag}
            return {"pid": pid, "type": "internal",
                    "separators": [s.hex() for s, _c in p.items],
                    "children": [node(p.leftmost)]
                    + [node(c) for _s, c in p.items]}
        return {"root": self.root, "tree": (node(self.root)
                                            if self.root != NONE else None)}
