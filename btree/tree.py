"""B+tree over the paged store: byte-space splits, cascade, root split.

Split decisions use NumPy and are made on *byte space* (free space), never on
key counts. Separator handling follows one declared policy consistently:

* ``bplus_copy``: leaf keeps the promote key; internal separators >= left keys
  and < right keys.
* ``btree_promote``: the promote key is removed from the leaf level.

Duplicate keys: ``version`` (multiversion), ``reject``, ``replace``.
Empty leaf pages after deletion: ``retain`` or ``collapse`` (leaf removed,
parent entry deleted, collapse cascades while parents become empty).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import page as P
from .pager import Pager, Txn, META_PID
from .record import UPDATE_PAGE


class TreeError(RuntimeError):
    pass


class DuplicateKeyError(TreeError):
    pass


@dataclass
class TreeConfig:
    page_size: int = 256
    separator_policy: str = "bplus_copy"   # bplus_copy | btree_promote
    duplicate_policy: str = "version"      # version | reject | replace
    empty_leaf_policy: str = "retain"      # retain | collapse
    split_tolerance_bytes: int = 0         # ambiguity window for legal plans


@dataclass
class SplitEvent:
    op_index: int
    pid: int
    new_pid: int
    parent_pid: int | None
    level: str
    byte_basis: bool
    left_bytes: int
    right_bytes: int
    total_cell_bytes: int
    promote_key: str
    key_count_before: int
    legal_plans: list[dict] = field(default_factory=list)
    chosen_index: int = 0


@dataclass
class OpTrace:
    index: int
    logical: dict
    before_image: str | None = None
    pages_before: dict = field(default_factory=dict)
    touched: list[int] = field(default_factory=list)
    split_events: list[SplitEvent] = field(default_factory=list)
    pages_after: dict = field(default_factory=list)
    slots_after: dict = field(default_factory=dict)
    free_after: dict = field(default_factory=dict)
    sibling_links: dict = field(default_factory=dict)
    root_before: int | None = None
    root_after: int | None = None
    next_pid_before: int | None = None
    next_pid_after: int | None = None
    lsn_before: int = 0
    lsn_after: int = 0
    status: str = "applied"
    diagnostic: str | None = None


class BPlusTree:
    def __init__(self, pager: Pager, config: TreeConfig | None = None):
        self.pager = pager
        self.cfg = config or TreeConfig(page_size=pager.page_size)
        self.traces: list[OpTrace] = []

    # ---------- frame helpers ----------
    def _meta(self) -> dict:
        return self.pager.meta()

    def _root_pid(self) -> int:
        return self._meta()["root_pid"]

    def _h(self, txn: Txn, pid: int) -> P.Header:
        return P.header(txn.read(pid))

    def _leaf_cells(self, txn: Txn, pid: int) -> list[bytes]:
        frame = txn.read(pid)
        return [cell for _, cell in P.cell_bytes(frame)]

    def _int_cells(self, txn: Txn, pid: int) -> list[bytes]:
        frame = txn.read(pid)
        return [cell for _, cell in P.cell_bytes(frame)]

    def _rewrite_leaf(self, txn: Txn, pid: int, cells: list[bytes],
                      **overrides) -> bytes:
        cur = self._h(txn, pid)
        root = overrides.get("root", cur.is_root)
        parent = overrides.get("parent", cur.parent)
        prev_sib = overrides.get("prev_sib", cur.prev_sib)
        next_sib = overrides.get("next_sib", cur.next_sib)
        frame = txn.modify(pid, lambda lsn: P.frame(
            self.cfg.page_size, P.KIND_LEAF, pid, parent=parent,
            prev_sib=prev_sib, next_sib=next_sib, root=root,
            page_lsn=lsn, cells=cells))
        return frame

    def _rewrite_internal(self, txn: Txn, pid: int, cells: list[bytes],
                          **overrides) -> bytes:
        cur = self._h(txn, pid)
        root = overrides.get("root", cur.is_root)
        parent = overrides.get("parent", cur.parent)
        frame = txn.modify(pid, lambda lsn: P.frame(
            self.cfg.page_size, P.KIND_INTERNAL, pid, parent=parent,
            prev_sib=-1, next_sib=-1, root=root,
            page_lsn=lsn, cells=cells))
        return frame

    # ---------- traversal ----------
    def _descend(self, txn: Txn, key: bytes) -> tuple[list[int], int]:
        """Return (path, leaf_pid); path = root..parent of leaf."""
        pid = self._root_pid()
        path: list[int] = []
        while self._h(txn, pid).kind == P.KIND_INTERNAL:
            cells = self._int_cells(txn, pid)
            nxt = P.decode_internal_cell(cells[-1])["child"]
            for cell in cells:
                d = P.decode_internal_cell(cell)
                if d["key"] is None:
                    continue
                go_left = key < d["key"]
                if self.cfg.separator_policy == "btree_promote" and key == d["key"]:
                    go_left = True
                if go_left:
                    nxt = d["child"]
                    break
            path.append(pid)
            pid = nxt
        return path, pid

    def _find_leaf_version(self, txn: Txn, pid: int, key: bytes,
                           version: int) -> int | None:
        for idx, cell in enumerate(self._leaf_cells(txn, pid)):
            d = P.decode_leaf_cell(cell)
            if d["key"] == key and d["version"] == version:
                return idx
        return None

    def _find_leaf_any(self, txn: Txn, pid: int, key: bytes) -> list[int]:
        out = []
        for idx, cell in enumerate(self._leaf_cells(txn, pid)):
            if P.decode_leaf_cell(cell)["key"] == key:
                out.append(idx)
        return out

    # ---------- split planning: byte space, NumPy ----------
    def _legal_leaf_splits(self, cells: list[bytes]) -> list[dict]:
        """All cut points whose *byte* occupation fits each new page.

        Byte occupation = sum(cell lengths) + 2 slots per cell. A plan is legal
        only when both sides fit one page (leaving the minimum free reserve of
        zero); near-tie plans within ``split_tolerance_bytes`` are all retained.
        """
        body = np.array([len(c) + 2 for c in cells], dtype=np.int64)
        n = len(cells)
        plans: list[dict] = []
        usable = self.cfg.page_size - P.HEADER_LEN
        # s = index of the separator key among sorted cells.
        # bplus_copy:    left=cells[:s],   right=cells[s:]   (sep stays right)
        # btree_promote: left=cells[:s+1], right=cells[s+1:] (sep moves left)
        for s in range(1, n - 1):
            if self.cfg.separator_policy == "btree_promote":
                left = int(body[:s + 1].sum())
                right = int(body[s + 1:].sum())
            else:
                left = int(body[:s].sum())
                right = int(body[s:].sum())
            if left <= usable and right <= usable:
                plans.append({
                    "cut": s,
                    "separator_index": s,
                    "left_bytes": left,
                    "right_bytes": right,
                    "imbalance": abs(left - right),
                    "metric": "byte_space",
                    "units": "B",
                })
        if not plans:
            raise TreeError("no legal byte-space split: cells too large for page")
        best = min(plans, key=lambda pl: pl["imbalance"])
        tol = self.cfg.split_tolerance_bytes
        legal = [pl for pl in plans if pl["imbalance"] <= best["imbalance"] + tol]
        for pl in legal:
            pl["legal"] = True
        return legal

    def _legal_internal_splits(self, cells: list[bytes]) -> list[dict]:
        """Internal split in cell units, converted to byte space for storage.

        Internal cells = n separators + 1 sentinel; the split cut k means left
        keeps k separator cells + sentinel(child k), right keeps the remainder
        (cell k+1..n-1), and separator index k is the promoted key.
        """
        body = np.array([len(c) + 2 for c in cells], dtype=np.int64)
        n = len(cells)
        usable = self.cfg.page_size - P.HEADER_LEN
        plans: list[dict] = []
        for sep in range(1, n - 1):
            left_units = sep + 1
            right_units = n - sep - 1
            left_bytes = int(body[:sep].sum() + 2)
            right_bytes = int(body[sep + 1:].sum() + 2)
            if left_bytes <= usable and right_bytes <= usable:
                plans.append({"cut": sep, "left_cells": left_units,
                              "right_cells": right_units,
                              "left_bytes": left_bytes,
                              "right_bytes": right_bytes,
                              "imbalance": abs(left_bytes - right_bytes),
                              "metric": "byte_space", "units": "B"})
        if not plans:
            raise TreeError("no legal byte-space split for internal page")
        best = min(plans, key=lambda pl: pl["imbalance"])
        tol = self.cfg.split_tolerance_bytes
        legal = [pl for pl in plans if pl["imbalance"] <= best["imbalance"] + tol]
        for pl in legal:
            pl["legal"] = True
        return legal

    # ---------- leaf split + cascade ----------
    def _fits(self, cells: list[bytes]) -> bool:
        need = 2 * len(cells) + sum(len(c) for c in cells)
        return need <= self.cfg.page_size - P.HEADER_LEN

    def _snapshot(self, txn: Txn) -> dict:
        return {pid: frame for pid, frame in self.pager.disk.items()}

    def _split_leaf(self, txn: Txn, pid: int, path: list[int],
                    trace: OpTrace, op_index: int) -> None:
        cells = self._leaf_cells(txn, pid)
        plans = self._legal_leaf_splits(cells)
        chosen = plans[trace.chosen_plan_index] if hasattr(
            trace, "chosen_plan_index") else plans[0]
        cut = chosen["cut"]
        if self.cfg.separator_policy == "btree_promote":
            left_cells = cells[:cut + 1]
            right_cells = cells[cut + 1:]
        elif self.cfg.separator_policy == "bplus_copy":
            left_cells, right_cells = cells[:cut], cells[cut:]
        else:
            raise TreeError(f"unknown separator policy {self.cfg.separator_policy}")
        promote = P.decode_leaf_cell(cells[cut])["key"]

        cur = self._h(txn, pid)
        new_pid, _ = txn.alloc(P.KIND_LEAF)
        self._rewrite_leaf(txn, pid, left_cells, next_sib=new_pid)
        self._rewrite_leaf(txn, new_pid, right_cells, parent=cur.parent,
                           prev_sib=pid, next_sib=cur.next_sib)
        if cur.next_sib >= 0:
            nxt_cells = self._leaf_cells(txn, cur.next_sib)
            self._rewrite_leaf(txn, cur.next_sib, nxt_cells,
                               prev_sib=new_pid)
        trace.split_events.append(SplitEvent(
            op_index=op_index, pid=pid, new_pid=new_pid,
            parent_pid=path[-1] if path else None, level="leaf",
            byte_basis=True, left_bytes=chosen["left_bytes"],
            right_bytes=chosen["right_bytes"],
            total_cell_bytes=chosen["left_bytes"] + chosen["right_bytes"],
            promote_key=promote.decode("utf-8", "replace"),
            key_count_before=len(cells), legal_plans=plans))
        self._insert_in_parent(txn, pid, promote, new_pid, path, trace,
                               op_index, "internal")

    def _insert_in_parent(self, txn: Txn, left_pid: int, key: bytes,
                          right_pid: int, path: list[int], trace: OpTrace,
                          op_index: int, level: str) -> None:
        if not path:
            return self._root_split(txn, left_pid, key, right_pid, trace,
                                    op_index, level)
        parent_pid = path[-1]
        cells = self._int_cells(txn, parent_pid)
        idx = next(i for i, c in enumerate(cells)
                   if P.decode_internal_cell(c)["child"] == left_pid)
        old = P.decode_internal_cell(cells[idx])
        new_cells = (cells[:idx]
                     + [P.internal_cell(left_pid, key),
                        P.internal_cell(right_pid, old["key"])]
                     + cells[idx + 1:])
        if self._fits(new_cells):
            self._rewrite_internal(txn, parent_pid, new_cells)
            return
        plans = self._legal_internal_splits(new_cells)
        chosen = plans[0]
        sep = chosen["cut"]
        promoted = P.decode_internal_cell(new_cells[sep])["key"]
        left_cells = new_cells[:sep] + [
            P.internal_cell(P.decode_internal_cell(new_cells[sep])["child"], None)]
        right_cells = new_cells[sep + 1:]
        new_pid, _ = txn.alloc(P.KIND_INTERNAL)
        self._rewrite_internal(txn, parent_pid, left_cells)
        self._rewrite_internal(txn, new_pid, right_cells,
                               parent=self._h(txn, parent_pid).parent)
        self._fix_child_parents(txn, parent_pid, new_pid, left_cells, right_cells)
        trace.split_events.append(SplitEvent(
            op_index=op_index, pid=parent_pid, new_pid=new_pid,
            parent_pid=path[-2] if len(path) >= 2 else None, level=level,
            byte_basis=True, left_bytes=chosen["left_bytes"],
            right_bytes=chosen["right_bytes"],
            total_cell_bytes=chosen["left_bytes"] + chosen["right_bytes"],
            promote_key=promoted.decode("utf-8", "replace"),
            key_count_before=len(new_cells), legal_plans=plans))
        self._insert_in_parent(txn, parent_pid, promoted, new_pid,
                               path[:-1], trace, op_index,
                               "internal_cascade" if level != "root" else level)

    def _fix_child_parents(self, txn: Txn, left_pid: int, right_pid: int,
                           left_cells: list[bytes], right_cells: list[bytes]) -> None:
        for cells, parent in ((left_cells, left_pid), (right_cells, right_pid)):
            for cell in cells:
                child = P.decode_internal_cell(cell)["child"]
                frame = txn.read(child)
                h = P.header(frame)
                if h.parent != parent:
                    if h.kind == P.KIND_LEAF:
                        leaf_cells = self._leaf_cells(txn, child)
                        self._rewrite_leaf(txn, child, leaf_cells, parent=parent)
                    else:
                        ic = self._int_cells(txn, child)
                        self._rewrite_internal(txn, child, ic, parent=parent)

    def _root_split(self, txn: Txn, left_pid: int, key: bytes, right_pid: int,
                    trace: OpTrace, op_index: int, level: str) -> None:
        root_pid, _ = txn.alloc(P.KIND_INTERNAL)
        cells = [P.internal_cell(left_pid, key),
                 P.internal_cell(right_pid, None)]
        txn.modify(root_pid, lambda lsn: P.frame(
            self.cfg.page_size, P.KIND_INTERNAL, root_pid, parent=-1,
            root=True, page_lsn=lsn, cells=cells))
        self._clear_root(txn, left_pid, root_pid)
        self._clear_root(txn, right_pid, root_pid)
        txn.set_root(root_pid)
        cur_pid = trace.pid_of_split if hasattr(trace, "pid_of_split") else left_pid
        trace.split_events.append(SplitEvent(
            op_index=op_index, pid=cur_pid, new_pid=root_pid,
            parent_pid=None, level="root", byte_basis=True,
            left_bytes=0, right_bytes=0, total_cell_bytes=0,
            promote_key=key.decode("utf-8", "replace"),
            key_count_before=0))

    def _clear_root(self, txn: Txn, pid: int, parent: int) -> None:
        frame = txn.read(pid)
        h = P.header(frame)
        if h.kind == P.KIND_LEAF:
            cells = self._leaf_cells(txn, pid)
            self._rewrite_leaf(txn, pid, cells, parent=parent, root=False)
        else:
            cells = self._int_cells(txn, pid)
            self._rewrite_internal(txn, pid, cells, parent=parent, root=False)

    # ---------- public ops ----------
    def init_tree(self) -> None:
        txn = Txn(self.pager.next_txn_id, self.pager)
        self.pager.next_txn_id += 1
        txn.begin()
        pid, _ = txn.alloc(P.KIND_LEAF)  # pid 1, the initial root leaf
        self._clear_root(txn, pid, -1) if False else None
        txn.set_root(pid)
        txn.commit()

    def insert(self, key: bytes, value: bytes, op_index: int = 0,
               version: int = 0) -> OpTrace:
        logical = {"op": "insert", "key": key.decode("utf-8", "replace"),
                   "value": value.decode("utf-8", "replace"), "version": version}
        trace = OpTrace(index=op_index, logical=logical)
        txn = Txn(self.pager.next_txn_id, self.pager)
        self.pager.next_txn_id += 1
        trace.lsn_before = self.pager._lsn
        trace.root_before = self._root_pid()
        trace.next_pid_before = self._meta()["next_pid"]
        trace.pages_before = self._snapshot(txn)
        trace.before_image = None
        txn.begin()
        path, leaf_pid = self._descend(txn, key)
        cells = self._leaf_cells(txn, leaf_pid)
        existing = [P.decode_leaf_cell(c) for c in cells
                    if P.decode_leaf_cell(c)["key"] == key]
        insert_version = version
        if existing:
            if self.cfg.duplicate_policy == "reject":
                trace.status = "rejected_duplicate"
                trace.diagnostic = f"duplicate key {key!r} under reject policy"
                self._finish_noop(txn, trace)
                self.traces.append(trace)
                return trace
            if self.cfg.duplicate_policy == "replace":
                ex = existing[0]
                cells = [c for c in cells
                         if not (P.decode_leaf_cell(c)["key"] == key
                                 and P.decode_leaf_cell(c)["version"]
                                 == ex["version"])]
                insert_version = ex["version"]
            else:
                insert_version = 1 + max(e["version"] for e in existing)
        new_cell = P.leaf_cell(key, value, insert_version)
        new_cells = sorted(cells + [new_cell],
                           key=lambda c: (P.decode_leaf_cell(c)["key"],
                                          P.decode_leaf_cell(c)["version"]))
        trace.before_image = txn.read(leaf_pid).hex()
        if self._fits(new_cells):
            self._rewrite_leaf(txn, leaf_pid, new_cells)
        else:
            txn.modify(leaf_pid, lambda lsn: P.frame(
                self.cfg.page_size, P.KIND_LEAF, leaf_pid,
                parent=self._h(txn, leaf_pid).parent,
                prev_sib=self._h(txn, leaf_pid).prev_sib,
                next_sib=self._h(txn, leaf_pid).next_sib,
                root=self._h(txn, leaf_pid).is_root,
                page_lsn=lsn, cells=new_cells))
            self._split_leaf(txn, leaf_pid, path, trace, op_index)
        txn.commit()
        self._finalize_trace(txn, trace)
        self.traces.append(trace)
        return trace

    def delete(self, key: bytes, op_index: int = 0,
               version: int | None = None) -> OpTrace:
        logical = {"op": "delete", "key": key.decode("utf-8", "replace"),
                   "version": version}
        trace = OpTrace(index=op_index, logical=logical)
        txn = Txn(self.pager.next_txn_id, self.pager)
        self.pager.next_txn_id += 1
        trace.lsn_before = self.pager._lsn
        trace.root_before = self._root_pid()
        trace.next_pid_before = self._meta()["next_pid"]
        trace.pages_before = self._snapshot(txn)
        txn.begin()
        path, leaf_pid = self._descend(txn, key)
        cells = self._leaf_cells(txn, leaf_pid)
        match = [c for c in cells
                 if (d := P.decode_leaf_cell(c))["key"] == key
                 and (version is None or d["version"] == version)]
        trace.before_image = txn.read(leaf_pid).hex()
        if not match:
            trace.status = "not_found"
            self._finish_noop(txn, trace)
            self.traces.append(trace)
            return trace
        removed = match[0]
        remaining = [c for c in cells if c != removed]
        if remaining or self.cfg.empty_leaf_policy == "retain":
            self._rewrite_leaf(txn, leaf_pid, remaining)
        else:
            self._collapse_leaf(txn, leaf_pid, path, trace)
        txn.commit()
        self._finalize_trace(txn, trace)
        self.traces.append(trace)
        return trace

    def _collapse_leaf(self, txn: Txn, pid: int, path: list[int],
                       trace: OpTrace) -> None:
        h = self._h(txn, pid)
        if not path:  # deleting last entry of a root leaf: keep an empty root
            self._rewrite_leaf(txn, pid, [])
            trace.status = "applied_empty_root_retained"
            return
        parent_pid = path[-1]
        if h.prev_sib >= 0:
            prev_cells = self._leaf_cells(txn, h.prev_sib)
            self._rewrite_leaf(txn, h.prev_sib, prev_cells,
                               next_sib=h.next_sib)
        if h.next_sib >= 0:
            nxt_cells = self._leaf_cells(txn, h.next_sib)
            self._rewrite_leaf(txn, h.next_sib, nxt_cells,
                               prev_sib=h.prev_sib)
        self._delete_parent_entry(txn, parent_pid, pid, path[:-1], trace)
        txn.free(pid)

    def _delete_parent_entry(self, txn: Txn, parent_pid: int, child_pid: int,
                             path: list[int], trace: OpTrace) -> None:
        cells = self._int_cells(txn, parent_pid)
        idx = next(i for i, c in enumerate(cells)
                   if P.decode_internal_cell(c)["child"] == child_pid)
        if idx == len(cells) - 1:  # rightmost child; separator is previous cell
            idx = len(cells) - 2
        new_cells = cells[:idx] + cells[idx + 2:]
        # reattach sentinel: the cell before new rightmost must become sentinel
        if new_cells:
            last_child = P.decode_internal_cell(new_cells[-1])["child"]
            new_cells[-1] = P.internal_cell(last_child, None)
        if self._h(txn, parent_pid).is_root:
            if len(new_cells) <= 1:
                if new_cells:
                    only_child = P.decode_internal_cell(new_cells[0])["child"]
                    self._promote_child_to_root(txn, only_child, parent_pid)
                else:
                    self._rewrite_internal(txn, parent_pid, new_cells)
                return
        if not new_cells:
            txn.free(parent_pid)
            if path:
                gp = path[-1]
                self._delete_parent_entry(txn, gp, parent_pid, path[:-1], trace)
            return
        self._rewrite_internal(txn, parent_pid, new_cells)

    def _promote_child_to_root(self, txn: Txn, child_pid: int,
                               old_root: int) -> None:
        frame = txn.read(child_pid)
        h = P.header(frame)
        if h.kind == P.KIND_LEAF:
            cells = self._leaf_cells(txn, child_pid)
            self._rewrite_leaf(txn, child_pid, cells, root=True, parent=-1)
        else:
            cells = self._int_cells(txn, child_pid)
            self._rewrite_internal(txn, child_pid, cells, root=True, parent=-1)
            self._fix_child_parents(txn, child_pid, child_pid, cells, cells)
        txn.free(old_root)
        txn.set_root(child_pid)

    def _finish_noop(self, txn: Txn, trace: OpTrace) -> None:
        txn.commit()
        self._finalize_trace(txn, trace)

    def _finalize_trace(self, txn: Txn, trace: OpTrace) -> None:
        trace.lsn_after = self.pager._lsn
        trace.root_after = self._root_pid()
        trace.next_pid_after = self._meta()["next_pid"]
        trace.touched = sorted(txn.dirty)
        pids = [pid for pid in self.pager.disk if pid != META_PID]
        trace.pages_after = {}
        for pid in pids:
            frame = self.pager.disk[pid]
            trace.slots_after[str(pid)] = [
                {"slot": i, "offset": off} for i, off in
                enumerate(P.slot_offsets(frame))]
            trace.free_after[str(pid)] = P.free_bytes_of(
                frame, self.cfg.page_size)
            h = P.header(frame)
            trace.sibling_links[str(pid)] = {
                "prev": h.prev_sib, "next": h.next_sib,
                "parent": h.parent, "kind": h.kind, "root": h.is_root}
            trace.pages_after[str(pid)] = frame.hex()
