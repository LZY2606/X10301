"""B+tree engine over btree.wal with an append-only event stream."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .encoding import (
    FLAG_HALF_PAGE,
    NO_PAGE,
    TYPE_INTERNAL,
    TYPE_LEAF,
    Cell,
    Page,
    PageFormatError,
    byte_split_plan,
    cell_size,
)
from .wal import CrashInjected, CrashPoint, Store

DUP_REPLACE = "replace"
DUP_REJECT = "reject"
DUP_MULTISET = "multiset"
DUP_POLICIES = (DUP_REPLACE, DUP_REJECT, DUP_MULTISET)

EMPTY_KEEP = "keep"
EMPTY_DETACH = "detach"


@dataclass
class Params:
    page_size: int = 256
    duplicate_policy: str = DUP_REPLACE
    split_bias: str = "balanced"
    separator_policy: str = "copy"
    empty_leaf_policy: str = EMPTY_KEEP

    def version_id(self) -> str:
        return (f"ps={self.page_size};dup={self.duplicate_policy};"
                f"bias={self.split_bias};sep={self.separator_policy};"
                f"empty={self.empty_leaf_policy}")


@dataclass
class Event:
    seq: int
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)


class Engine:
    def __init__(self, params: Optional[Params] = None):
        self.params = params or Params()
        self.store = Store(self.params.page_size)
        self.events: List[Event] = []
        self._seq = 0
        self.op_index = 0
        self._cur_tx = 0

    def emit(self, event_kind: str, **payload: Any) -> Event:
        self._seq += 1
        ev = Event(self._seq, event_kind, payload)
        self.events.append(ev)
        return ev

    @staticmethod
    def _cell_view(cell: Cell) -> Dict[str, Any]:
        view = {
            "key_hex": cell.key.hex(),
            "key_repr": _safe_repr(cell.key),
            "key_len_bytes": len(cell.key),
        }
        if cell.value:
            view["value_hex"] = cell.value.hex()
            view["value_len_bytes"] = len(cell.value)
        if cell.child != NO_PAGE:
            view["left_child"] = cell.child
        return view

    def _page_summary(self, page: Page) -> Dict[str, Any]:
        return {
            "page_id": page.page_id,
            "type": "leaf" if page.page_type == TYPE_LEAF else "internal",
            "nslots": len(page.slots),
            "free_space_bytes": page.free_space,
            "used_bytes": page.used_bytes,
            "space_unit": "byte",
            "slot_directory": [
                {"slot": i, "offset_bytes": s.offset, "length_bytes": s.length}
                for i, s in enumerate(page.slots)],
            "prev_page": page.prev_page,
            "next_page": page.next_page,
            "lsn": page.lsn,
            "flags": page.flags,
            "rightmost_child": page.rightmost_child,
            "cells": [self._cell_view(c) for c in page.cells],
        }

    @staticmethod
    def _clone(page: Page) -> Page:
        return Page.decode(page.encode())

    # ---------------------------------------------------------- execute
    def execute(self, op: Dict[str, Any]) -> List[Event]:
        start = len(self.events)
        kind = op["op"]
        self.op_index += 1
        self.emit("logical_operation", index=self.op_index, op=op,
                  unit_note="all key/value lengths are byte counts")
        if kind == "put":
            key = _as_bytes(op["key"])
            value = _as_bytes(op.get("value", b""))
            self._put(key, value, op.get("crash"))
        elif kind == "delete":
            self._delete(_as_bytes(op["key"]), op.get("crash"))
        elif kind == "crash":
            self.emit("crash_point", name=op.get("name"),
                      at=op.get("at", "during_write"), explicit=True)
        elif kind == "recover":
            self.recover()
        else:
            raise ValueError(f"unknown op {kind!r}")
        return self.events[start:]

    # --------------------------------------------------------------- put
    def _put(self, key: bytes, value: bytes,
             crash_spec: Optional[Dict[str, Any]]) -> None:
        crash = CrashPoint(crash_spec["at"]) if crash_spec else None
        tx = self.store.begin()
        self._cur_tx = tx
        if self.store.root_page_id == NO_PAGE:
            page = self._alloc_leaf(tx)
            page.cells.append(Cell(key, value))
            page.rebuild()
            page.lsn = self.store.lsn
            self._write_and_commit(tx, [(None, page)], page.page_id, crash,
                                   new_pages=[page.page_id])
            return
        path: List[Tuple[Page, int]] = []
        pid = self.store.root_page_id
        while True:
            page = self.store.read_page(pid)
            if page.page_type == TYPE_LEAF:
                break
            slot = self._internal_child_slot(page, key)
            path.append((page, slot))
            pid = self._child_at(page, slot)
        leaf = page
        pre = self._clone(leaf)
        self.emit("page_pre_image", **self._page_summary(pre))
        dup = self._find_dup(leaf, key)
        if dup is not None:
            self._handle_duplicate(tx, leaf, pre, dup, key, value, path, crash)
            return
        cell = Cell(key, value)
        need = cell_size(key, value)
        if leaf.can_fit(need):
            leaf.cells.append(cell)
            leaf.cells.sort(key=lambda c: (c.key, c.child))
            leaf.rebuild()
            leaf.lsn = self.store.lsn
            self._write_and_commit(tx, [(pre, leaf)],
                                   self.store.root_page_id, crash)
            return
        self._leaf_split_insert(tx, path, leaf, pre, cell, crash)

    @staticmethod
    def _child_at(page: Page, slot: int) -> int:
        if slot < len(page.cells):
            return page.cells[slot].child
        return page.rightmost_child

    @staticmethod
    def _internal_child_slot(page: Page, key: bytes) -> int:
        for i, cell in enumerate(page.cells):
            if key < cell.key:
                return i
        return len(page.cells)

    @staticmethod
    def _find_dup(leaf: Page, key: bytes) -> Optional[int]:
        if leaf.page_type != TYPE_LEAF:
            return None
        for i, cell in enumerate(leaf.cells):
            if cell.key == key:
                return i
        return None

    def _alloc_leaf(self, tx: int) -> Page:
        pid = self.store.allocate_page_id()
        page = Page(pid, self.params.page_size, TYPE_LEAF)
        self.emit("page_id_allocated", page_id=pid, tx=tx,
                  next_page_id=self.store.next_page_id, id_reused=False)
        return page

    def _alloc_internal(self, tx: int) -> Page:
        pid = self.store.allocate_page_id()
        page = Page(pid, self.params.page_size, TYPE_INTERNAL)
        self.emit("page_id_allocated", page_id=pid, tx=tx,
                  next_page_id=self.store.next_page_id, id_reused=False)
        return page

    # -------------------------------------------------------- duplicates
    def _handle_duplicate(self, tx: int, leaf: Page, pre: Page, dup: int,
                          key: bytes, value: bytes,
                          path: List[Tuple[Page, int]],
                          crash: Optional[CrashPoint]) -> None:
        policy = self.params.duplicate_policy
        if policy == DUP_REJECT:
            self.store.abort(tx)
            self.emit("duplicate_decision", policy=policy, key_hex=key.hex(),
                      accepted=False,
                      reason="reject policy: existing key retained")
            return
        if policy == DUP_REPLACE:
            old = leaf.cells[dup]
            need = cell_size(key, value)
            fits = leaf.can_fit(need - cell_size(old.key, old.value))
            verdict = "accepted_replace_in_place"
            if fits:
                leaf.cells[dup] = Cell(key, value)
                leaf.cells.sort(key=lambda c: (c.key, c.child))
                leaf.rebuild()
                leaf.lsn = self.store.lsn
                self.emit("duplicate_decision", policy=policy,
                          key_hex=key.hex(), accepted=True,
                          verdict=verdict)
                self._write_and_commit(tx, [(pre, leaf)],
                                       self.store.root_page_id, crash)
            else:
                self.emit("duplicate_decision", policy=policy,
                          key_hex=key.hex(), accepted=True,
                          verdict="accepted_replace_forces_split")
                leaf.cells[dup] = Cell(key, value)
                leaf.cells.sort(key=lambda c: (c.key, c.child))
                self._leaf_split_insert(tx, path, leaf, pre, None, crash)
            return
        # multiset: keep both copies
        cell = Cell(key, value)
        self.emit("duplicate_decision", policy=policy, key_hex=key.hex(),
                  accepted=True, verdict="accepted_extra_occurrence",
                  multiplicity=sum(1 for c in leaf.cells if c.key == key) + 1)
        if leaf.can_fit(cell_size(key, value)):
            leaf.cells.append(cell)
            leaf.cells.sort(key=lambda c: (c.key, c.child))
            leaf.rebuild()
            leaf.lsn = self.store.lsn
            self._write_and_commit(tx, [(pre, leaf)],
                                   self.store.root_page_id, crash)
        else:
            self._leaf_split_insert(tx, path, leaf, pre, cell, crash)

    # ---------------------------------------------------- write / commit
    def _write_and_commit(self, tx: int,
                          writes: List[Tuple[Optional[Page], Page]],
                          root_page_id: int,
                          crash: Optional[CrashPoint],
                          new_pages: Optional[List[int]] = None,
                          next_page_id: Optional[int] = None,
                          context: Optional[Dict[str, Any]] = None) -> bool:
        torn_pid: Optional[int] = None
        for i, (before, after) in enumerate(writes):
            after.lsn = self.store.lsn
            if crash is not None and crash.at == "during_write" and i == 0:
                torn_pid = after.page_id
                self.store.write_page_torn(tx, before, after)
            else:
                self.store.write_page(tx, before, after)
                self.store.materialize(after)
        if torn_pid is not None:
            self.emit("page_write_interrupted", page_id=torn_pid,
                      flags=FLAG_HALF_PAGE,
                      reachable_after_crash=self._still_reachable(torn_pid),
                      verdict="half_page_durable_until_recovery")
        try:
            self.store.commit(tx, root_page_id,
                              next_page_id or self.store.next_page_id, crash)
        except CrashInjected:
            self.emit("crash_fired",
                      at=crash.at,
                      root_published=False,
                      dirty_page_ids=[a.page_id for _b, a in writes],
                      reachable_half_page=torn_pid is not None
                      and self._still_reachable(torn_pid))
            return False
        for _before, after in writes:
            self.emit("page_after_image", **self._page_summary(after))
        self.emit("commit", tx=tx, root_page_id=root_page_id,
                  next_page_id=self.store.next_page_id,
                  lsn=self.store.meta_lsn,
                  new_pages=new_pages or [])
        self._check_invariants(context=context)
        return True

    def _still_reachable(self, page_id: int) -> bool:
        """Reachability uses the last *published* root (never uncommitted)."""
        if self.store.root_page_id == NO_PAGE:
            return False
        seen = set()
        stack = [self.store.root_page_id]
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                page = self.store.read_page(pid)
            except PageFormatError:
                if pid == page_id:
                    return True
                continue
            if page.page_type == TYPE_INTERNAL:
                stack.append(page.rightmost_child)
                stack.extend(c.child for c in page.cells)
        return page_id in seen

    # ------------------------------------------------------- leaf split
    def _leaf_split_insert(self, tx: int,
                           path: List[Tuple[Page, int]],
                           leaf: Page, pre: Page,
                           extra: Optional[Cell],
                           crash: Optional[CrashPoint]) -> None:
        merged = list(leaf.cells)
        if extra is not None:
            merged.append(extra)
        merged.sort(key=lambda c: (c.key, c.child))
        sizes = [cell_size(c.key, c.value) for c in merged]
        plan = byte_split_plan(sizes, bias=self.params.split_bias)
        right = self._alloc_leaf(tx)
        left_cells = merged[:plan.index]
        right_cells = merged[plan.index:]
        leaf.cells = left_cells
        leaf.rebuild()
        right.cells = right_cells
        right.rebuild()
        sep_key = bytes(right_cells[0].key)
        old_next = pre.next_page
        leaf.next_page = right.page_id
        right.prev_page = leaf.page_id
        right.next_page = old_next
        split_group = f"op{self.op_index}-split-leaf-{leaf.page_id}"
        self.emit("split", group_id=split_group, kind="leaf",
                  page_id=leaf.page_id, new_page_id=right.page_id,
                  decision_byte_space=True,
                  left_bytes=sum(cell_size(c.key, c.value) for c in left_cells),
                  right_bytes=sum(cell_size(c.key, c.value) for c in right_cells),
                  byte_unit="byte",
                  cut_slots=plan.candidates,
                  chosen_cut=plan.index,
                  ambiguous=plan.ambiguous,
                  legal_candidates=[
                      {"cut": c,
                       "left_bytes": int(sum(sizes[:c])),
                       "right_bytes": int(sum(sizes[c:]))}
                      for c in plan.candidates],
                  separator_key_hex=sep_key.hex(),
                  separator_policy="copy",
                  promoted=False,
                  sibling_links={"left_next": right.page_id,
                                 "right_prev": leaf.page_id,
                                 "right_next": old_next})
        if not path:
            new_root = self._alloc_internal(tx)
            new_root.cells = [Cell(sep_key, b"", child=leaf.page_id)]
            new_root.rightmost_child = right.page_id
            new_root.rebuild()
            self.emit("root_change", old_root=leaf.page_id,
                      new_root=new_root.page_id, kind="root_leaf_split",
                      separator_key_hex=sep_key.hex())
            self._write_and_commit(
                tx,
                [(pre, leaf), (None, right), (None, new_root)],
                new_root.page_id, crash,
                new_pages=[right.page_id, new_root.page_id],
                context={"split_group": split_group})
            return
        self._insert_in_parent(tx, path, leaf.page_id, right.page_id,
                               sep_key, leaf, pre, right, crash, split_group,
                               promoted=False,
                               base_writes=[(pre, leaf), (None, right)],
                               base_new_ids=[right.page_id])

    # --------------------------------------------------- parent insert
    def _entry_size(self, key: bytes) -> int:
        return cell_size(key, internal=True)

    def _insert_in_parent(self, tx: int,
                          path: List[Tuple[Page, int]],
                          left_pid: int, right_pid: int,
                          sep_key: bytes,
                          parent_left: Page, parent_left_pre: Optional[Page],
                          parent_right: Page,
                          crash: Optional[CrashPoint],
                          split_group: str,
                          promoted: bool,
                          base_writes: List[Tuple[Optional[Page], Page]],
                          base_new_ids: List[int]) -> None:
        parent, slot = path[-1]
        pre_parent = self._clone(parent)
        # Canonical child sequence: [child_0, child_1, ..., child_m] where
        # child_i is cells[i].child (i<m) and child_m is rightmost_child.
        # Leaf split at routing `slot` replaces child_slot by [left,right]
        # and inserts the separator before `right`.
        children = [c.child for c in parent.cells] + [parent.rightmost_child]
        children[slot:slot + 1] = [left_pid, right_pid]
        keys = [c.key for c in parent.cells]
        keys.insert(slot, bytes(sep_key))
        m = len(keys)
        new_cells = [Cell(keys[i], b"", child=children[i])
                     for i in range(m)]
        new_rightmost = children[m]
        needed = sum(self._entry_size(k) for k in keys)
        writes = list(base_writes)
        if needed + 4 * len(new_cells) <= parent.page_size - 38:
            parent.cells = new_cells
            parent.rightmost_child = new_rightmost
            parent.rebuild()
            self.emit("parent_insert", page_id=parent.page_id, slot=slot,
                      separator_key_hex=sep_key.hex(),
                      right_child=right_pid,
                      policy=self.params.separator_policy
                      if promoted else "copy")
            writes.append((pre_parent, parent))
            self._write_and_commit(
                tx, writes, self.store.root_page_id, crash,
                new_pages=base_new_ids,
                context={"split_group": split_group})
            return
        self._internal_split_insert(
            tx, path, pre_parent, new_cells, new_rightmost,
            writes, base_new_ids, crash, split_group)

    def _internal_split_insert(self, tx: int,
                               path: List[Tuple[Page, int]],
                               pre_parent: Page,
                               entries: List[Cell], rightmost: int,
                               base_writes: List[Tuple[Optional[Page], Page]],
                               base_new_ids: List[int],
                               crash: Optional[CrashPoint],
                               split_group: str) -> None:
        sizes = [self._entry_size(c.key) for c in entries]
        plan = byte_split_plan(sizes, bias=self.params.split_bias,
                               internal=True)
        idx = plan.index
        up_key = bytes(entries[idx].key)
        promoted = self.params.separator_policy == "promote"
        new_right = self._alloc_internal(tx)

        left_cells = [Cell(c.key, b"", child=c.child) for c in entries[:idx]]
        left_rightmost = entries[idx].child
        if promoted:
            right_cells = [Cell(c.key, b"", child=c.child)
                           for c in entries[idx + 1:]]
        else:
            right_cells = [Cell(c.key, b"", child=c.child)
                           for c in entries[idx:]]
        left_page_id = pre_parent.page_id
        right_page_id = new_right.page_id

        left_internal = Page(left_page_id, self.params.page_size,
                             TYPE_INTERNAL, pre_parent.prev_page,
                             pre_parent.next_page, NO_PAGE,
                             pre_parent.lsn, pre_parent.flags)
        left_internal.cells = left_cells
        left_internal.rightmost_child = left_rightmost
        left_internal.rebuild()
        new_right.cells = right_cells
        new_right.rightmost_child = rightmost
        new_right.rebuild()

        self.emit("split", group_id=split_group, kind="internal",
                  page_id=left_page_id, new_page_id=right_page_id,
                  decision_byte_space=True,
                  left_bytes=int(sum(sizes[:idx])),
                  right_bytes=int(sum(sizes[idx:])),
                  byte_unit="byte",
                  cut_slots=plan.candidates, chosen_cut=idx,
                  ambiguous=plan.ambiguous,
                  legal_candidates=[
                      {"cut": c, "left_bytes": int(sum(sizes[:c])),
                       "right_bytes": int(sum(sizes[c:]))}
                      for c in plan.candidates],
                  separator_key_hex=up_key.hex(),
                  separator_policy=self.params.separator_policy,
                  promoted=promoted,
                  cascade_depth=len(path))
        writes = list(base_writes) + [(pre_parent, left_internal),
                                      (None, new_right)]
        new_ids = list(base_new_ids) + [right_page_id]
        if len(path) == 1:
            new_root = self._alloc_internal(tx)
            new_root.cells = [Cell(up_key, b"", child=left_page_id)]
            new_root.rightmost_child = right_page_id
            new_root.rebuild()
            self.emit("root_change", old_root=left_page_id,
                      new_root=new_root.page_id, kind="root_internal_split",
                      separator_key_hex=up_key.hex())
            writes.append((None, new_root))
            new_ids.append(new_root.page_id)
            self._write_and_commit(
                tx, writes, new_root.page_id, crash,
                new_pages=new_ids,
                context={"split_group": split_group})
            return
        self.emit("cascade_split", group_id=split_group,
                  from_page=left_page_id, depth=len(path))
        self._insert_in_parent(
            tx, path[:-1], left_page_id, right_page_id, up_key,
            left_internal, pre_parent, new_right, crash, split_group,
            promoted=promoted,
            base_writes=writes, base_new_ids=new_ids)


    # ------------------------------------------------------------ delete
    def _delete(self, key: bytes,
                crash_spec: Optional[Dict[str, Any]]) -> None:
        crash = CrashPoint(crash_spec["at"]) if crash_spec else None
        if self.store.root_page_id == NO_PAGE:
            self.emit("delete_miss", key_hex=key.hex(), accepted=False)
            return
        tx = self.store.begin()
        self._cur_tx = tx
        path: List[Tuple[Page, int]] = []
        pid = self.store.root_page_id
        while True:
            page = self.store.read_page(pid)
            if page.page_type == TYPE_LEAF:
                leaf = page
                break
            slot = self._internal_child_slot(page, key)
            path.append((page, slot))
            pid = self._child_at(page, slot)
        pre = self._clone(leaf)
        self.emit("page_pre_image", **self._page_summary(pre))
        idx = next((i for i, c in enumerate(leaf.cells) if c.key == key),
                   None)
        if idx is None:
            self.store.abort(tx)
            self.emit("delete_miss", key_hex=key.hex(), accepted=False)
            return
        removed = leaf.cells.pop(idx)
        if not leaf.cells and leaf.page_id != self.store.root_page_id and \
                self.params.empty_leaf_policy == EMPTY_DETACH:
            leaf.rebuild()
            self._detach_empty_leaf(tx, path, leaf, pre, crash,
                                    removed_key=removed.key)
            return
        leaf.rebuild()
        leaf.lsn = self.store.lsn
        self.emit("delete_applied", page_id=leaf.page_id,
                  removed_key_hex=removed.key.hex(),
                  empty=not leaf.cells,
                  empty_policy=self.params.empty_leaf_policy)
        self._write_and_commit(tx, [(pre, leaf)],
                               self.store.root_page_id, crash)

    def _detach_empty_leaf(self, tx: int,
                           path: List[Tuple[Page, int]],
                           leaf: Page, pre: Page,
                           crash: Optional[CrashPoint],
                           removed_key: bytes) -> None:
        parent, slot = path[-1]
        pre_parent = self._clone(parent)
        writes: List[Tuple[Optional[Page], Page]] = [(pre, leaf)]
        left = right = None
        if leaf.prev_page != NO_PAGE:
            left = self.store.read_page(leaf.prev_page)
            writes.append((self._clone(left), left))
        if leaf.next_page != NO_PAGE:
            right = self.store.read_page(leaf.next_page)
            writes.append((self._clone(right), right))
        if left is not None:
            left.next_page = leaf.next_page
            left.rebuild()
        if right is not None:
            right.prev_page = leaf.prev_page
            right.rebuild()
        new_cells = list(parent.cells)
        new_rightmost = parent.rightmost_child
        if slot < len(parent.cells) and new_cells[slot].child == leaf.page_id:
            del new_cells[slot]
        elif slot > 0 and new_cells[slot - 1].child == leaf.page_id:
            del new_cells[slot - 1]
        elif slot == len(parent.cells):
            new_rightmost = leaf.prev_page
        parent.cells = [Cell(c.key, b"", child=c.child) for c in new_cells]
        parent.rightmost_child = new_rightmost
        parent.rebuild()
        writes.append((pre_parent, parent))
        self.emit("empty_page_detached", page_id=leaf.page_id,
                  reachable=False, page_id_reused=False,
                  removed_key_hex=removed_key.hex())
        new_root = self.store.root_page_id
        if parent.page_id == self.store.root_page_id:
            if not parent.cells and parent.rightmost_child == NO_PAGE:
                new_root = NO_PAGE
                self.emit("root_change", old_root=parent.page_id,
                          new_root=NO_PAGE, kind="root_emptied")
            elif not parent.cells:
                new_root = parent.rightmost_child
                self.emit("root_change", old_root=parent.page_id,
                          new_root=new_root, kind="root_collapse_internal")
        self._write_and_commit(tx, writes, new_root, crash)

    # ---------------------------------------------------------- recovery
    def recover(self) -> None:
        result = self.store.recover()
        self.emit("recovery_replay", run=self.store.recoveries,
                  redone=result["redone"], undone=result["undone"],
                  losers=result["losers"], winners=result["winners"],
                  root_page_id=self.store.root_page_id,
                  idempotent_note="second replay must redo/undo nothing")
        reachable_torn = [pid for pid in self._reachable_page_ids()
                          if self.store.page_is_torn(pid)]
        self.emit("recovery_verdict",
                  reachable_half_pages=reachable_torn,
                  passed=not reachable_torn,
                  rule="no reachable half page after replay")
        self._check_invariants(context={"recovery_run": self.store.recoveries})

    # -------------------------------------------------------- invariants
    def _check_invariants(self,
                          context: Optional[Dict[str, Any]] = None) -> list:
        checks = list(self._invariant_checks())
        self.emit("invariants",
                  checks=checks,
                  passed=all(c["passed"] for c in checks),
                  context=context or {})
        return checks

    def _invariant_checks(self) -> list:
        yield self._check_leaf_chain()
        yield self._check_free_space()
        yield self._check_separator_consistency()
        yield self._check_no_torn_reachable()
        yield self._check_page_ids()

    def _check_leaf_chain(self) -> dict:
        if self.store.root_page_id == NO_PAGE:
            return {"name": "leaf_sibling_chain", "passed": True}
        leaves = self._collect_leaves()
        for i, p in enumerate(leaves):
            prev_exp = leaves[i - 1].page_id if i else NO_PAGE
            next_exp = leaves[i + 1].page_id if i + 1 < len(leaves) else NO_PAGE
            if p.prev_page != prev_exp or p.next_page != next_exp:
                return {"name": "leaf_sibling_chain", "passed": False,
                        "detail": f"page {p.page_id} links "
                                  f"{p.prev_page}->{p.next_page}"}
        return {"name": "leaf_sibling_chain", "passed": True,
                "leaf_count": len(leaves)}

    def _check_free_space(self) -> dict:
        for pid, raw in self.store.disk.items():
            try:
                p = Page.decode(raw)
            except PageFormatError:
                continue
            if p.free_start - p.slot_end != p.free_space or p.free_space < 0:
                return {"name": "free_space_accounting", "passed": False,
                        "detail": f"page {pid}"}
        return {"name": "free_space_accounting", "passed": True}

    def _check_separator_consistency(self) -> dict:
        if self.store.root_page_id == NO_PAGE:
            return {"name": "separator_rule_consistency", "passed": True}

        def walk(pid: int, lo: Optional[bytes], hi: Optional[bytes]) -> str:
            p = self.store.read_page(pid)
            if p.page_type == TYPE_LEAF:
                keys = [c.key for c in p.cells]
                if keys != sorted(keys):
                    return f"leaf {pid} unsorted"
                if lo is not None and keys and keys[0] < lo:
                    return f"leaf {pid} key below separator {lo.hex()}"
                if hi is not None and keys and keys[-1] >= hi \
                        and self.params.separator_policy == "promote":
                    return f"leaf {pid} key >= separator {hi.hex()}"
                return ""
            for i, c in enumerate(p.cells):
                child_lo = lo if i == 0 else p.cells[i - 1].key
                msg = walk(c.child, child_lo, c.key)
                if msg:
                    return msg
            if p.cells:
                msg = walk(p.rightmost_child, p.cells[-1].key, hi)
                if msg:
                    return msg
            return ""

        error = walk(self.store.root_page_id, None, None)
        return {"name": "separator_rule_consistency",
                "passed": not error, "detail": error,
                "policy": self.params.separator_policy}

    def _check_no_torn_reachable(self) -> dict:
        for pid in self._reachable_page_ids():
            if self.store.page_is_torn(pid):
                return {"name": "no_reachable_half_page", "passed": False,
                        "detail": f"page {pid} torn and reachable"}
        return {"name": "no_reachable_half_page", "passed": True}

    def _check_page_ids(self) -> dict:
        max_id = max(self.store.disk.keys(), default=-1)
        if max_id >= self.store.next_page_id:
            return {"name": "page_id_monotonic", "passed": False,
                    "detail": f"id {max_id} >= allocator {self.store.next_page_id}"}
        return {"name": "page_id_monotonic", "passed": True}

    # ---------------------------------------------------------- read back
    def _reachable_page_ids(self) -> List[int]:
        if self.store.root_page_id == NO_PAGE:
            return []
        seen: List[int] = []
        stack = [self.store.root_page_id]
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.append(pid)
            try:
                p = self.store.read_page(pid)
            except PageFormatError:
                continue
            if p.page_type == TYPE_INTERNAL:
                stack.append(p.rightmost_child)
                stack.extend(c.child for c in p.cells)
        return seen

    def _collect_leaves(self) -> List[Page]:
        if self.store.root_page_id == NO_PAGE:
            return []
        pid = self.store.root_page_id
        while True:
            p = self.store.read_page(pid)
            if p.page_type == TYPE_LEAF:
                break
            pid = p.cells[0].child if p.cells else p.rightmost_child
        leaves = []
        seen = set()
        while pid != NO_PAGE and pid not in seen:
            seen.add(pid)
            p = self.store.read_page(pid)
            leaves.append(p)
            pid = p.next_page
        return leaves

    def read_all(self) -> List[Dict[str, Any]]:
        rows = []
        for leaf in self._collect_leaves():
            for c in leaf.cells:
                rows.append({"key_hex": c.key.hex(),
                             "value_hex": c.value.hex(),
                             "leaf_page_id": leaf.page_id})
        return rows

    def tree_view(self) -> Dict[str, Any]:
        if self.store.root_page_id == NO_PAGE:
            return {"root": None}

        def node(pid: int) -> Dict[str, Any]:
            p = self.store.read_page(pid)
            view = self._page_summary(p)
            if p.page_type == TYPE_INTERNAL:
                view["children"] = [node(c.child) for c in p.cells]
                view["children"].append(node(p.rightmost_child))
            return view

        return {"root": node(self.store.root_page_id),
                "reachable_page_ids": self._reachable_page_ids(),
                "leaf_count": len(self._collect_leaves())}


def _safe_repr(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.hex()


def _as_bytes(value) -> bytes:
    if isinstance(value, str):
        if value.startswith("hex:"):
            return bytes.fromhex(value[4:])
        return value.encode("utf-8")
    return bytes(value)
