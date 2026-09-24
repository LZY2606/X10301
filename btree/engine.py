"""B+ tree engine over fixed-size pages with WAL, crash and recovery.

Split rule: variable-length keys split by *byte space*, never by key count.
Separator rule: leaf split copies the separator up (it stays in the right
leaf); internal split pushes the separator up (it leaves both children).
"""
from __future__ import annotations

import hashlib

from .layout import Cell, NO_PAGE, Page, crc16


class BTree:
    def __init__(self, page_size: int = 256, duplicate_policy: str = "reject"):
        if duplicate_policy not in ("reject", "overwrite"):
            raise ValueError(f"unknown duplicate_policy {duplicate_policy}")
        self.page_size = page_size
        self.duplicate_policy = duplicate_policy
        self.disk: dict[int, bytes] = {}
        self.wal: list[dict] = []
        self.lsn = 0
        self.next_page_id = 1
        self.retired: list[int] = []
        self.events: list[dict] = []
        self.root_id = self._alloc_page(is_leaf=True)

    # ---- page allocation & WAL-backed writes -------------------------------
    def _alloc_page(self, is_leaf: bool) -> int:
        pid = self.next_page_id
        self.next_page_id += 1
        page = Page(page_id=pid, is_leaf=is_leaf)
        self._write_page(page, op="alloc_page")
        self.events.append({"type": "page_id_alloc", "page_id": pid, "leaf": is_leaf})
        return pid

    def _write_page(self, page: Page, op: str) -> None:
        self.lsn += 1
        page.lsn = self.lsn
        before = self.disk.get(page.page_id)
        after = page.encode(self.page_size)
        self.wal.append({"lsn": self.lsn, "page_id": page.page_id, "op": op,
                         "before": before, "after": after})
        self.disk[page.page_id] = after

    def _read_page(self, pid: int) -> Page:
        return Page.decode(self.disk[pid], self.page_size)

    # ---- navigation ---------------------------------------------------------
    def _descend(self, key: bytes) -> list[Page]:
        path = []
        page = self._read_page(self.root_id)
        while not page.is_leaf:
            path.append(page)
            page = self._read_page(self._child_for(page, key))
        path.append(page)
        return path

    @staticmethod
    def _child_for(page: Page, key: bytes) -> int:
        for i, cell in enumerate(page.cells):
            if key < cell.key:
                return page.left_child if i == 0 else page.cells[i - 1].child
        return page.cells[-1].child if page.cells else page.left_child

    # ---- insert -------------------------------------------------------------
    def insert(self, key: bytes, value: bytes) -> dict:
        path = self._descend(key)
        leaf = path[-1]
        for i, cell in enumerate(leaf.cells):
            if cell.key == key:
                if self.duplicate_policy == "reject":
                    self.events.append({"type": "duplicate_rejected", "key": key.decode("utf8", "replace")})
                    return {"status": "rejected", "key": key}
                leaf.cells[i] = Cell(key=key, value=value)
                self._write_page(leaf, op="overwrite_duplicate")
                self.events.append({"type": "duplicate_overwritten", "key": key.decode("utf8", "replace")})
                return {"status": "overwritten", "key": key}
        pos = sum(1 for c in leaf.cells if c.key < key)
        leaf.cells.insert(pos, Cell(key=key, value=value))
        if leaf.free_bytes(self.page_size) >= 0:
            self._write_page(leaf, op="insert")
            return {"status": "inserted", "key": key}
        self._split_and_propagate(path)
        return {"status": "inserted_with_split", "key": key}

    def _split_and_propagate(self, path: list[Page]) -> None:
        page = path.pop()
        sep, right = self._split_page(page)
        self._write_page(page, op="split_left")
        propagation = [{"page_id": page.page_id, "new_page_id": right.page_id,
                        "separator": sep.decode("utf8", "replace"),
                        "rule": "copy_up" if page.is_leaf else "push_up"}]
        while True:
            if not path:
                new_root = Page(page_id=self.next_page_id, is_leaf=False,
                                left_child=page.page_id,
                                cells=[Cell(key=sep, child=right.page_id)])
                self.next_page_id += 1
                self._write_page(new_root, op="root_create")
                self.root_id = new_root.page_id
                propagation.append({"root_split": True, "new_root_id": new_root.page_id})
                self.events.append({"type": "root_split", "new_root_id": new_root.page_id})
                break
            parent = path.pop()
            pos = self._separator_slot(parent, page.page_id)
            parent.cells.insert(pos, Cell(key=sep, child=right.page_id))
            if parent.free_bytes(self.page_size) >= 0:
                self._write_page(parent, op="separator_insert")
                break
            sep, right = self._split_page(parent)
            self._write_page(parent, op="cascade_split_left")
            propagation.append({"page_id": parent.page_id, "new_page_id": right.page_id,
                                "separator": sep.decode("utf8", "replace"), "rule": "push_up"})
            self.events.append({"type": "cascade_split", "page_id": parent.page_id})
            page = parent
        self.events.append({"type": "split_propagation", "steps": propagation})

    @staticmethod
    def _separator_slot(parent: Page, child_id: int) -> int:
        if parent.left_child == child_id:
            return 0
        for i, cell in enumerate(parent.cells):
            if cell.child == child_id:
                return i + 1
        raise ValueError(f"child {child_id} not found in parent {parent.page_id}")

    def _split_page(self, page: Page) -> tuple[bytes, Page]:
        """Split by byte space: left keeps cells until >= half of used bytes."""
        cells = page.cells
        total = sum(c.encoded_size() for c in cells)
        half = total / 2
        acc, idx = 0, 0
        while idx < len(cells) - 1 and acc < half:
            acc += cells[idx].encoded_size()
            idx += 1
        idx = max(1, min(idx, len(cells) - 1))
        right = Page(page_id=self.next_page_id, is_leaf=page.is_leaf)
        self.next_page_id += 1
        self.events.append({"type": "page_id_alloc", "page_id": right.page_id, "leaf": page.is_leaf})
        if page.is_leaf:
            right.cells = cells[idx:]
            page.cells = cells[:idx]
            sep = right.cells[0].key            # copy-up: stays in right leaf
            right.right_sibling = page.right_sibling
            page.right_sibling = right.page_id
        else:
            mid = idx
            push = cells[mid]
            right.left_child = push.child
            right.cells = cells[mid + 1:]
            page.cells = cells[:mid]
            sep = push.key                      # push-up: leaves both children
        self._write_page(right, op="split_right")
        return sep, right

    # ---- delete ---------------------------------------------------------------
    def delete(self, key: bytes) -> dict:
        path = self._descend(key)
        leaf = path[-1]
        before = len(leaf.cells)
        leaf.cells = [c for c in leaf.cells if c.key != key]
        if len(leaf.cells) == before:
            return {"status": "absent", "key": key}
        self._write_page(leaf, op="delete")
        if not leaf.cells and leaf.page_id != self.root_id:
            self._retire_empty_leaf(leaf, path[:-1])
            return {"status": "deleted_page_retired", "key": key}
        return {"status": "deleted", "key": key}

    def _retire_page(self, pid: int) -> None:
        if pid in self.disk:
            del self.disk[pid]
        self.retired.append(pid)
        self.events.append({"type": "page_retired", "page_id": pid})

    def _retire_empty_leaf(self, leaf: Page, ancestors: list[Page]) -> None:
        pred = self._leftmost_leaf()
        while pred.right_sibling not in (leaf.page_id, NO_PAGE):
            pred = self._read_page(pred.right_sibling)
        if pred.right_sibling == leaf.page_id:
            pred.right_sibling = leaf.right_sibling
            self._write_page(pred, op="unlink_empty_leaf")
        self._retire_page(leaf.page_id)
        if ancestors:
            self._drop_child(ancestors[-1], leaf.page_id, ancestors[:-1])

    def _drop_child(self, parent: Page, child_id: int, ancestors: list[Page]) -> None:
        if parent.left_child == child_id:
            parent.left_child = parent.cells[0].child
            parent.cells = parent.cells[1:]
        else:
            parent.cells = [c for c in parent.cells if c.child != child_id]
        self._write_page(parent, op="drop_empty_child")
        if parent.cells or parent.is_leaf:
            return
        if parent.page_id == self.root_id:
            self.root_id = parent.left_child
            self._retire_page(parent.page_id)
            self.events.append({"type": "root_shrink", "new_root_id": self.root_id})
            return
        if not ancestors:
            return
        gp = ancestors[-1]
        if gp.left_child == parent.page_id:
            gp.left_child = parent.left_child
        else:
            for cell in gp.cells:
                if cell.child == parent.page_id:
                    cell.child = parent.left_child
        self._write_page(gp, op="bypass_empty_parent")
        self._retire_page(parent.page_id)

    def _leftmost_leaf(self) -> Page:
        page = self._read_page(self.root_id)
        while not page.is_leaf:
            page = self._read_page(page.left_child)
        return page

    # ---- crash & recovery -----------------------------------------------------
    def simulate_crash(self, torn_page_ids: list[int]) -> dict:
        """Interrupt page writes: zero the tail of each torn page on disk."""
        torn = []
        for pid in torn_page_ids:
            raw = bytearray(self.disk[pid])
            cut = len(raw) // 2
            for i in range(cut, len(raw)):
                raw[i] = 0
            self.disk[pid] = bytes(raw)
            torn.append(pid)
        event = {"type": "crash", "torn_pages": torn, "at_lsn": self.lsn}
        self.events.append(event)
        return event

    def recover(self) -> dict:
        restored, replayed = [], []
        for pid, raw in list(self.disk.items()):
            try:
                Page.decode(raw, self.page_size)
            except ValueError:
                candidates = [r for r in self.wal if r["page_id"] == pid]
                if candidates:
                    self.disk[pid] = candidates[-1]["after"]
                    restored.append(pid)
        for rec in self.wal:
            raw = self.disk.get(rec["page_id"])
            if raw is None:
                continue
            page = Page.decode(raw, self.page_size)
            if page.lsn < rec["lsn"]:
                self.disk[rec["page_id"]] = rec["after"]
                replayed.append(rec["lsn"])
        report = {"type": "recovery", "restored_pages": restored, "replayed_lsns": replayed}
        self.events.append(report)
        return report

    # ---- structural views -----------------------------------------------------
    def reachable_pages(self) -> list[int]:
        seen, order = set(), []
        stack = [self.root_id]
        while stack:
            pid = stack.pop()
            if pid in seen or pid == NO_PAGE or pid not in self.disk:
                continue
            seen.add(pid)
            order.append(pid)
            page = Page.decode(self.disk[pid], self.page_size)
            if page.is_leaf:
                if page.right_sibling != NO_PAGE:
                    stack.append(page.right_sibling)
            else:
                stack.append(page.left_child)
                stack.extend(c.child for c in page.cells)
        return order

    def reachable_half_pages(self) -> list[int]:
        bad = []
        for pid in self.reachable_pages():
            try:
                Page.decode(self.disk[pid], self.page_size)
            except ValueError:
                bad.append(pid)
        return bad

    def state_hash(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.root_id).encode())
        for pid in sorted(self.disk):
            h.update(pid.to_bytes(4, "little"))
            h.update(self.disk[pid])
        return h.hexdigest()

    def tree_structure(self) -> dict:
        def node(pid: int) -> dict:
            page = self._read_page(pid)
            n = {"page_id": pid, "leaf": page.is_leaf, "lsn": page.lsn,
                 "keys": [c.key.decode("utf8", "replace") for c in page.cells],
                 "used_bytes": page.used_bytes(), "free_bytes": page.free_bytes(self.page_size)}
            if page.is_leaf:
                n["right_sibling"] = None if page.right_sibling == NO_PAGE else page.right_sibling
                n["values"] = [c.value.decode("utf8", "replace") for c in page.cells]
            else:
                n["children"] = [node(page.left_child)] + [node(c.child) for c in page.cells]
            return n
        return {"root": node(self.root_id), "page_size": self.page_size,
                "retired_pages": list(self.retired)}

    def wal_view(self) -> list[dict]:
        return [{"lsn": r["lsn"], "page_id": r["page_id"], "op": r["op"],
                 "before_crc": crc16(r["before"]) if r["before"] else None,
                 "after_crc": crc16(r["after"])} for r in self.wal]
