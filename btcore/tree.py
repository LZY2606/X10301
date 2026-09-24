"""B+tree over fixed-size pages with byte-driven splits.

Split policy invariants exercised here:

* feasibility is computed from packed *byte space* (header + slot dir +
  cells), not key count; every legal point is enumerated;
* leaf split **copies** the separator (left max key stays in the leaf and
  also appears in the parent); internal split **promotes** it (the key
  disappears from both children and exists only in the parent).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pages import (
    LEAF,
    INTERNAL,
    NONE_PID,
    Cell,
    LeafPage,
    InternalPage,
    BasePage,
    PageFormatError,
    encode_page,
    page_free_bytes,
    legal_split_points,
    best_split_point,
    cells_payload_bytes,
)


class DuplicateKey(ValueError):
    pass


class KeyTooLarge(PageFormatError):
    pass


@dataclass
class SplitRecord:
    lsn: int
    level: str                       # 'leaf' | 'internal' | 'root'
    split_pid: int
    new_pid: int
    cut: int
    legal_points: List[int]
    chosen_by: str                   # 'min_byte_imbalance'
    separator: bytes
    separator_mode: str              # 'copy' (leaf) | 'promote' (internal)
    parent_pid: Optional[int]
    new_root: bool
    free_before: int
    free_left: int
    free_right: int
    links_before: Dict[int, Tuple[Optional[int], Optional[int]]]
    links_after: Dict[int, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)
    before_image: bytes = b""
    after_left_image: bytes = b""
    after_right_image: bytes = b""
    root_before: int = -1
    root_after: int = -1
    key_count_before: int = 0
    key_count_left: int = 0
    key_count_right: int = 0
    byte_payload_before: int = 0
    byte_payload_left: int = 0
    byte_payload_right: int = 0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["separator"] = self.separator.hex()
        return d


class BTree:
    def __init__(self, page_size: int = 128, dup_policy: str = "replace"):
        self.page_size = page_size
        self.dup_policy = dup_policy
        self.pages: Dict[int, BasePage] = {}
        self._next_pid = 0
        root = self._new_leaf()
        self.root = root.page_id
        self.lsn = 0
        self._touched: set[int] = set()

    def _new_leaf(self) -> LeafPage:
        pid = self._next_pid
        self._next_pid += 1
        page = LeafPage(page_id=pid)
        self.pages[pid] = page
        return page

    def _new_internal(self, hi_child: int) -> InternalPage:
        pid = self._next_pid
        self._next_pid += 1
        page = InternalPage(page_id=pid, hi_child=hi_child)
        self.pages[pid] = page
        return page

    def allocated_pids(self) -> List[int]:
        return list(range(self._next_pid))

    def reachable_pids(self) -> List[int]:
        seen: List[int] = []

        def walk(pid: int) -> None:
            if pid in seen or pid not in self.pages:
                return
            seen.append(pid)
            page = self.pages[pid]
            if isinstance(page, InternalPage):
                for c in page.cells:
                    walk(c.child)
                walk(page.hi_child)

        walk(self.root)
        return seen

    def leaf_sibling_links(self) -> Dict[int, Tuple[Optional[int], Optional[int]]]:
        leaves: List[int] = []

        def walk(pid: int) -> None:
            page = self.pages[pid]
            if isinstance(page, LeafPage):
                leaves.append(pid)
            else:
                for c in page.cells:
                    walk(c.child)
                walk(page.hi_child)

        walk(self.root)
        out: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        for i, pid in enumerate(leaves):
            out[pid] = (leaves[i - 1] if i else None,
                        leaves[i + 1] if i + 1 < len(leaves) else None)
        return out

    def snapshot(self) -> Dict[int, bytes]:
        return {pid: encode_page(p, self.page_size) for pid, p in self.pages.items()}

    def find(self, key: bytes) -> Optional[bytes]:
        page = self.pages[self.root]
        while isinstance(page, InternalPage):
            pid = page.hi_child
            for c in page.cells:
                # Leaf separators are copied: left child owns keys <= sep.
                if key <= c.key:
                    pid = c.child
                    break
            page = self.pages[pid]
        for c in page.cells:
            if c.key == key:
                return c.value
        return None

    def page_free(self, pid: int) -> int:
        page = self.pages[pid]
        kind = LEAF if isinstance(page, LeafPage) else INTERNAL
        return page_free_bytes(self.page_size, page.cells, kind)

    # ------------------------------------------------------------------
    def _leaf_insert(self, page: LeafPage, key: bytes, value: bytes) -> str:
        for i, c in enumerate(page.cells):
            if c.key == key:
                if self.dup_policy == "reject":
                    raise DuplicateKey(f"duplicate key {key!r}")
                if self.dup_policy == "replace":
                    c.value = value
                    return "replaced"
            if key < c.key:
                page.cells.insert(i, Cell(key=key, value=value))
                return "inserted"
        page.cells.append(Cell(key=key, value=value))
        return "inserted"

    def insert(self, key: bytes, value: bytes) -> List[SplitRecord]:
        self.lsn += 1
        lsn = self.lsn
        self._touched = set()
        root_before = self.root
        records: List[SplitRecord] = []
        result = self._insert(self.root, key, value, lsn, records, parent=None)
        if result is not None:
            new_root = self._new_internal(hi_child=result["right"])
            new_root.cells = [Cell(key=result["sep"], child=result["left"])]
            self._touched.add(new_root.page_id)
            old_root = self.root
            self.root = new_root.page_id
            for rec in records:
                if rec.split_pid == old_root:
                    rec.new_root = True
                    rec.level = "root"
                    rec.parent_pid = None
                    rec.root_after = self.root
                    break
        for pid in self._touched:
            self.pages[pid].page_lsn = lsn
        return records

    def _insert(
        self,
        pid: int,
        key: bytes,
        value: bytes,
        lsn: int,
        records: List[SplitRecord],
        parent: Optional[int],
    ) -> Optional[dict]:
        """Return {'sep','mode','left','right'} when ``pid`` split."""
        page = self.pages[pid]
        if isinstance(page, LeafPage):
            before = encode_page(page, self.page_size)
            before_links = self.leaf_sibling_links()
            free_before = self.page_free(pid)
            n_before = len(page.cells)
            payload_before = cells_payload_bytes(page.cells, LEAF)
            outcome = self._leaf_insert(page, key, value)
            self._touched.add(pid)
            if outcome == "replaced":
                return None
            if self.page_free(pid) >= 0:
                return None
            cut = best_split_point(page.cells, self.page_size, LEAF)
            legal = legal_split_points(page.cells, self.page_size, LEAF)
            if cut is None:
                raise KeyTooLarge(f"key {key!r} cannot fit page of {self.page_size}B")
            new_leaf = self._new_leaf()
            new_leaf.cells = page.cells[cut:]
            page.cells = page.cells[:cut]
            self._touched.add(new_leaf.page_id)
            rec = SplitRecord(
                lsn=lsn,
                level="root" if pid == self.root else "leaf",
                split_pid=pid,
                new_pid=new_leaf.page_id,
                cut=cut,
                legal_points=legal,
                chosen_by="min_byte_imbalance",
                separator=page.cells[-1].key,
                separator_mode="copy",
                parent_pid=parent,
                new_root=False,
                free_before=free_before,
                free_left=self.page_free(pid),
                free_right=self.page_free(new_leaf.page_id),
                links_before=before_links,
                links_after=self.leaf_sibling_links(),
                before_image=before,
                after_left_image=encode_page(page, self.page_size),
                after_right_image=encode_page(new_leaf, self.page_size),
                root_before=self.root,
                root_after=self.root,
                key_count_before=n_before,
                key_count_left=len(page.cells),
                key_count_right=len(new_leaf.cells),
                byte_payload_before=payload_before,
                byte_payload_left=cells_payload_bytes(page.cells, LEAF),
                byte_payload_right=cells_payload_bytes(new_leaf.cells, LEAF),
            )
            records.append(rec)
            return {"sep": rec.separator, "mode": "copy",
                    "left": pid, "right": new_leaf.page_id}

        target = page.hi_child
        slot = len(page.cells)
        for i, c in enumerate(page.cells):
            if key <= c.key:
                target = c.child
                slot = i
                break
        result = self._insert(target, key, value, lsn, records, parent=pid)
        if result is None:
            return None
        before = encode_page(page, self.page_size)
        if slot == len(page.cells):
            page.cells.append(Cell(key=result["sep"], child=result["left"]))
            page.hi_child = result["right"]
        else:
            page.cells.insert(slot, Cell(key=result["sep"], child=result["left"]))
        self._touched.add(pid)

        if self.page_free(pid) >= 0:
            return None

        free_before = self.page_free(pid)
        n_before = len(page.cells)
        payload_before = cells_payload_bytes(page.cells, INTERNAL)
        cut = best_split_point(page.cells, self.page_size, INTERNAL)
        legal = legal_split_points(page.cells, self.page_size, INTERNAL)
        if cut is None:
            raise KeyTooLarge(f"internal pid {pid}: no feasible split point")
        promoted = page.cells[cut].key
        left_hi = page.cells[cut].child
        new_right = self._new_internal(page.hi_child)
        new_right.cells = page.cells[cut + 1:]
        page.hi_child = left_hi
        page.cells = page.cells[:cut]
        self._touched.add(new_right.page_id)
        rec = SplitRecord(
            lsn=lsn,
            level="root" if pid == self.root else "internal",
            split_pid=pid,
            new_pid=new_right.page_id,
            cut=cut,
            legal_points=legal,
            chosen_by="min_byte_imbalance",
            separator=promoted,
            separator_mode="promote",
            parent_pid=parent,
            new_root=False,
            free_before=free_before,
            free_left=self.page_free(pid),
            free_right=self.page_free(new_right.page_id),
            links_before={},
            links_after={},
            before_image=before,
            after_left_image=encode_page(page, self.page_size),
            after_right_image=encode_page(new_right, self.page_size),
            root_before=self.root,
            root_after=self.root,
            key_count_before=n_before,
            key_count_left=len(page.cells),
            key_count_right=len(new_right.cells),
            byte_payload_before=payload_before,
            byte_payload_left=cells_payload_bytes(page.cells, INTERNAL),
            byte_payload_right=cells_payload_bytes(new_right.cells, INTERNAL),
        )
        records.append(rec)
        return {"sep": promoted, "mode": "promote",
                "left": pid, "right": new_right.page_id}

    # ------------------------------------------------------------------
    def delete(self, key: bytes) -> bool:
        self.lsn += 1
        leaf = self._descend_leaf(key)
        for i, c in enumerate(leaf.cells):
            if c.key == key:
                del leaf.cells[i]
                leaf.page_lsn = self.lsn
                return True
        return False

    def _descend_leaf(self, key: bytes) -> LeafPage:
        page = self.pages[self.root]
        while isinstance(page, InternalPage):
            target = page.hi_child
            for c in page.cells:
                if key <= c.key:
                    target = c.child
                    break
            page = self.pages[target]
        return page

    def empty_leaf_pids(self) -> List[int]:
        return [pid for pid in self.reachable_pids()
                if isinstance(self.pages[pid], LeafPage) and not self.pages[pid].cells]
