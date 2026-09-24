"""B-tree over the fixed-size page layout.

Split rule (R1): a page splits when the *byte space* it needs exceeds the
free bytes on the page -- never by counting keys.  Split points are chosen
by byte balance, not cell count.

Separator rule (R2): leaf splits copy the first key of the right page up
(copy-up, the key stays in the leaf); internal splits promote the middle
key (push-up, the key leaves the splitting page).
"""
from __future__ import annotations

from bisect import bisect_right

from .pages import LEAF, INTERNAL, Cell, Page, Pager, SLOT_SIZE


class DuplicateKey(Exception):
    def __init__(self, key):
        self.key = key
        super().__init__(f"duplicate key {key!r}")


def _split_point_by_bytes(page: Page) -> int:
    """First index whose accumulated byte cost reaches half of the used bytes."""
    half = page.used_bytes() / 2.0
    acc = 0.0
    for i, cell in enumerate(page.cells):
        acc += cell.size(page.ptype) + SLOT_SIZE
        if acc >= half and i + 1 < len(page.cells):
            return i + 1
    return max(1, len(page.cells) // 2)


def legal_split_points(page: Page, page_size: int, tolerance: float = 0.25) -> list:
    """All split indices whose left/right byte balance stays within tolerance.

    Used to expose every legal candidate when the split point is not
    uniquely decidable; the archive keeps all of them.
    """
    total = page.used_bytes()
    acc = 0
    points = []
    for i, cell in enumerate(page.cells[:-1]):
        acc += cell.size(page.ptype) + SLOT_SIZE
        left = acc
        right = total - acc
        if abs(left - right) <= tolerance * total:
            points.append(i + 1)
    return points or [_split_point_by_bytes(page)]


class BTree:
    def __init__(self, pager: Pager, duplicate_policy: str = "reject"):
        assert duplicate_policy in ("reject", "overwrite")
        self.pager = pager
        self.duplicate_policy = duplicate_policy
        self.root = pager.alloc(LEAF)
        self.height = 1
        self.events = []
        self.dirty = set()

    # -- helpers ---------------------------------------------------------
    def _emit(self, kind, **payload):
        self.events.append({"kind": kind, **payload})

    def _touch(self, page):
        self.dirty.add(page.page_id)

    def _find_leaf(self, key: bytes):
        path = []  # (internal page, child index or -1 for left_child)
        page = self.root
        while page.ptype == INTERNAL:
            idx = bisect_right(page.keys(), key) - 1
            if idx < 0:
                path.append((page, -1))
                page = self.pager.pages[page.left_child]
            else:
                path.append((page, idx))
                page = self.pager.pages[page.cells[idx].child]
        return page, path

    # -- insert ----------------------------------------------------------
    def insert(self, key: bytes, value: bytes):
        leaf, path = self._find_leaf(key)
        pos = bisect_right(leaf.keys(), key)
        if pos > 0 and leaf.cells[pos - 1].key == key:
            if self.duplicate_policy == "reject":
                self._emit("duplicate", key=key.hex(), policy="reject", decision="rejected")
                raise DuplicateKey(key)
            self._emit("duplicate", key=key.hex(), policy="overwrite", decision="overwritten")
            leaf.cells[pos - 1].value = value
            self._touch(leaf)
            self._overflow_check(leaf, path)
            return
        leaf.cells.insert(pos, Cell(key=key, value=value))
        self._touch(leaf)
        self._overflow_check(leaf, path)

    def _overflow_check(self, page: Page, path):
        """Byte-space rule: split iff free_space < 0 after the insert."""
        if page.free_space(self.pager.page_size) >= 0:
            return
        self._split(page, path)

    def _split(self, page: Page, path):
        ps = self.pager.page_size
        cut = _split_point_by_bytes(page)
        candidates = legal_split_points(page, ps)
        right = self.pager.alloc(page.ptype)
        right.cells = page.cells[cut:]
        page.cells = page.cells[:cut]
        if page.ptype == LEAF:
            separator = right.cells[0].key
            rule = "copy_up"
            right.right_sibling = page.right_sibling
            page.right_sibling = right.page_id
        else:
            middle = page.cells.pop()  # placeholder, replaced below
            # internal split: promote the last key of the left half
            promoted_cell = middle
            separator = promoted_cell.key
            rule = "promote"
            right.left_child = promoted_cell.child
        self._touch(page)
        self._touch(right)
        self._emit(
            "split",
            scope="leaf" if page.ptype == LEAF else "internal",
            left=page.page_id, right=right.page_id,
            separator=separator.hex(), separator_rule=rule,
            free_space_before=page.free_space(ps),
            split_point=cut, legal_split_points=candidates,
        )
        if not path:
            # root split: tree grows one level
            new_root = self.pager.alloc(INTERNAL)
            new_root.left_child = page.page_id
            new_root.cells = [Cell(key=separator, child=right.page_id)]
            self.root = new_root
            self.height += 1
            self._touch(new_root)
            self._emit("root_split", new_root=new_root.page_id, height=self.height,
                       separator=separator.hex())
            return
        parent, idx = path.pop()
        cell = Cell(key=separator, child=right.page_id)
        if idx < 0:
            # split page was the leftmost child: separator becomes first cell
            parent.cells.insert(0, cell)
        else:
            parent.cells.insert(idx + 1, cell)
        self._touch(parent)
        self._overflow_check(parent, path)

    # -- delete ----------------------------------------------------------
    def delete(self, key: bytes) -> bool:
        leaf, path = self._find_leaf(key)
        pos = bisect_right(leaf.keys(), key)
        if pos == 0 or leaf.cells[pos - 1].key != key:
            return False
        leaf.cells.pop(pos - 1)
        self._touch(leaf)
        if leaf.cells or leaf.page_id == self.root.page_id:
            return True
        self._unlink_empty_leaf(leaf, path)
        return True

    def _unlink_empty_leaf(self, leaf: Page, path):
        # detach from the sibling chain
        predecessor = None
        for page in self.pager.pages.values():
            if page.ptype == LEAF and page.right_sibling == leaf.page_id:
                predecessor = page
                page.right_sibling = leaf.right_sibling
                self._touch(page)
                break
        # detach from the parent
        if not path:
            return
        parent, idx = path[-1]
        if idx < 0:
            # empty page was the leftmost child
            parent.left_child = leaf.right_sibling
            if parent.cells:
                parent.cells.pop(0)
        else:
            parent.cells.pop(idx)
        self._touch(parent)
        self.pager.free(leaf.page_id)
        self._emit("empty_page_unlinked", page_id=leaf.page_id,
                   predecessor=predecessor.page_id if predecessor else 0,
                   parent=parent.page_id)
        # root shrink: empty internal root with a single child
        if parent.page_id == self.root.page_id and not parent.cells and parent.ptype == INTERNAL:
            old_root = parent
            self.root = self.pager.pages[old_root.left_child]
            self.height -= 1
            self.pager.free(old_root.page_id)
            self._emit("root_shrink", old_root=old_root.page_id,
                       new_root=self.root.page_id, height=self.height)

    # -- read ------------------------------------------------------------
    def get(self, key: bytes):
        leaf, _ = self._find_leaf(key)
        pos = bisect_right(leaf.keys(), key)
        if pos > 0 and leaf.cells[pos - 1].key == key:
            return leaf.cells[pos - 1].value
        return None

    def reachable_page_ids(self) -> list:
        """Pages reachable from the root via child pointers and sibling links."""
        seen = []
        stack = [self.root.page_id]
        while stack:
            pid = stack.pop()
            if pid in seen or pid not in self.pager.pages:
                continue
            seen.append(pid)
            page = self.pager.pages[pid]
            if page.ptype == INTERNAL:
                if page.left_child:
                    stack.append(page.left_child)
                stack.extend(c.child for c in page.cells)
        # follow the leaf sibling chain from the leftmost leaf
        page = self.root
        while page.ptype == INTERNAL:
            page = self.pager.pages[page.left_child]
        while page.right_sibling:
            nxt = page.right_sibling
            if nxt in seen or nxt not in self.pager.pages:
                break
            seen.append(nxt)
            page = self.pager.pages[nxt]
        return sorted(seen)
