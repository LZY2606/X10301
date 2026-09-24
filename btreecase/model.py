"""B-tree page simulator: fixed page size, byte-based split, crash points, WAL replay."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SLOT_SIZE = 2          # bytes per slot directory entry
LEAF_HEADER = 8        # page_id(4) + flags(1) + right_sibling(3)
INTERNAL_HEADER = 12   # page_id(4) + flags(1) + child_count(3) + reserved(4)


def _enc_key(key: str) -> bytes:
    return key.encode("utf-8")


def _enc_val(value: str) -> bytes:
    return value.encode("utf-8")


@dataclass
class Page:
    page_id: int
    is_leaf: bool
    keys: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)      # leaf only
    children: List[int] = field(default_factory=list)    # internal only
    right_sibling: Optional[int] = None

    def used_bytes(self) -> int:
        if self.is_leaf:
            body = sum(len(_enc_key(k)) + len(_enc_val(v)) + SLOT_SIZE
                       for k, v in zip(self.keys, self.values))
            return LEAF_HEADER + body
        body = sum(len(_enc_key(k)) + SLOT_SIZE for k in self.keys)
        return INTERNAL_HEADER + body + 4 * len(self.children)

    def slot_directory(self, page_size: int) -> List[Dict[str, int]]:
        """Offset/length of each record within the page image (bytes)."""
        slots = []
        if self.is_leaf:
            off = LEAF_HEADER
            for k, v in zip(self.keys, self.values):
                ln = len(_enc_key(k)) + len(_enc_val(v))
                slots.append({"offset": off, "length": ln})
                off += ln
        else:
            off = INTERNAL_HEADER
            for k in self.keys:
                ln = len(_enc_key(k))
                slots.append({"offset": off, "length": ln})
                off += ln
        return slots

    def free_space(self, page_size: int) -> int:
        return page_size - self.used_bytes()

    def to_image(self, page_size: int) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "is_leaf": self.is_leaf,
            "keys": list(self.keys),
            "values": list(self.values),
            "children": list(self.children),
            "right_sibling": self.right_sibling,
            "used_bytes": self.used_bytes(),
            "free_space": self.free_space(page_size),
            "slot_directory": self.slot_directory(page_size),
        }

    @staticmethod
    def from_image(img: Dict[str, Any]) -> "Page":
        return Page(
            page_id=img["page_id"], is_leaf=img["is_leaf"],
            keys=list(img["keys"]), values=list(img.get("values", [])),
            children=list(img.get("children", [])),
            right_sibling=img.get("right_sibling"),
        )


@dataclass
class OpResult:
    index: int
    op: Dict[str, Any]
    pages_before: Dict[int, Dict[str, Any]]
    pages_after: Dict[int, Dict[str, Any]]
    root_before: Optional[int]
    root_after: Optional[int]
    page_ids_allocated: List[int]
    splits: List[Dict[str, Any]]
    notes: List[str]


class BTree:
    """B-tree with byte-space-based splitting. duplicate_policy: 'update' | 'allow'."""

    def __init__(self, page_size: int, duplicate_policy: str = "update"):
        self.page_size = page_size
        self.duplicate_policy = duplicate_policy
        self.pages: Dict[int, Page] = {}
        self.root_id: Optional[int] = None
        self.next_page_id = 1
        self.lsn = 0
        self.wal: List[Dict[str, Any]] = []

    # ---- allocation / WAL ----
    def _alloc(self, is_leaf: bool) -> Page:
        p = Page(self.next_page_id, is_leaf)
        self.next_page_id += 1
        self.pages[p.page_id] = p
        return p

    def _log(self, kind: str, **kw):
        self.lsn += 1
        rec = {"lsn": self.lsn, "kind": kind}
        rec.update(kw)
        self.wal.append(rec)

    # ---- lookup helpers ----
    def _find_leaf(self, key: str) -> Page:
        node = self.pages[self.root_id]
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = self.pages[node.children[i]]
        return node

    def get(self, key: str) -> Optional[str]:
        if self.root_id is None:
            return None
        leaf = self._find_leaf(key)
        for k, v in zip(leaf.keys, leaf.values):
            if k == key:
                return v
        return None

    # ---- insert ----
    def insert(self, key: str, value: str):
        self._log("insert", key=key, value=value)
        if self.root_id is None:
            root = self._alloc(is_leaf=True)
            root.keys.append(key)
            root.values.append(value)
            self.root_id = root.page_id
            return
        leaf = self._find_leaf(key)
        for i, k in enumerate(leaf.keys):
            if k == key:
                if self.duplicate_policy == "update":
                    leaf.values[i] = value
                    return
                break  # allow: insert duplicate after existing
        i = 0
        while i < len(leaf.keys) and leaf.keys[i] <= key:
            i += 1
        leaf.keys.insert(i, key)
        leaf.values.insert(i, value)
        if leaf.used_bytes() > self.page_size:
            self._split_leaf(leaf)

    def _split_leaf(self, leaf: Page):
        """Split by byte space: move records to right sibling until left fits."""
        right = self._alloc(is_leaf=True)
        right.right_sibling = leaf.right_sibling
        leaf.right_sibling = right.page_id
        # Move from the tail while the left page still overflows and more than
        # one record remains on the left.
        while leaf.used_bytes() > self.page_size and len(leaf.keys) > 1:
            right.keys.insert(0, leaf.keys.pop())
            right.values.insert(0, leaf.values.pop())
        # separator copied up (copy-up for leaf)
        sep = right.keys[0]
        self._log("split_leaf", left=leaf.page_id, right=right.page_id, separator=sep)
        self._insert_into_parent(leaf, sep, right, copy_up=True)

    def _split_internal(self, node: Page):
        right = self._alloc(is_leaf=False)
        # Move tail keys/children until left fits; middle key is pushed up.
        while node.used_bytes() > self.page_size and len(node.keys) > 1:
            right.keys.insert(0, node.keys.pop())
            right.children.insert(0, node.children.pop())
        right.children.insert(0, node.children.pop())
        sep = node.keys.pop()  # push-up (not copied)
        self._log("split_internal", left=node.page_id, right=right.page_id, separator=sep)
        self._insert_into_parent(node, sep, right, copy_up=False)

    def _insert_into_parent(self, left: Page, sep: str, right: Page, copy_up: bool):
        parent = self._parent_of(left.page_id)
        if parent is None:
            new_root = self._alloc(is_leaf=False)
            new_root.keys = [sep]
            new_root.children = [left.page_id, right.page_id]
            self.root_id = new_root.page_id
            self._log("root_split", new_root=new_root.page_id, separator=sep)
            return
        i = 0
        while i < len(parent.keys) and parent.keys[i] <= sep:
            i += 1
        parent.keys.insert(i, sep)
        parent.children.insert(i + 1, right.page_id)
        if parent.used_bytes() > self.page_size:
            self._split_internal(parent)  # cascading split

    def _parent_of(self, page_id: int) -> Optional[Page]:
        def walk(node: Page) -> Optional[Page]:
            if node.is_leaf:
                return None
            for c in node.children:
                if c == page_id:
                    return node
                r = walk(self.pages[c])
                if r is not None:
                    return r
            return None
        if self.root_id is None or self.root_id == page_id:
            return None
        return walk(self.pages[self.root_id])

    # ---- delete ----
    def delete(self, key: str):
        self._log("delete", key=key)
        if self.root_id is None:
            return
        leaf = self._find_leaf(key)
        for i, k in enumerate(leaf.keys):
            if k == key:
                leaf.keys.pop(i)
                leaf.values.pop(i)
                break
        # empty non-root leaf: unlink from parent and sibling chain
        if not leaf.keys and leaf.page_id != self.root_id:
            parent = self._parent_of(leaf.page_id)
            if parent is not None:
                idx = parent.children.index(leaf.page_id)
                parent.children.pop(idx)
                if idx > 0:
                    parent.keys.pop(idx - 1)
                elif parent.keys:
                    parent.keys.pop(0)
            # fix sibling link
            prev = self._left_sibling_of(leaf.page_id)
            if prev is not None:
                prev.right_sibling = leaf.right_sibling
            self._log("unlink_empty_page", page_id=leaf.page_id)
            # collapse root if it lost all keys
            root = self.pages[self.root_id]
            while not root.is_leaf and not root.keys and root.children:
                self.root_id = root.children[0]
                self._log("root_shrink", new_root=self.root_id)
                root = self.pages[self.root_id]

    def _left_sibling_of(self, page_id: int) -> Optional[Page]:
        parent = self._parent_of(page_id)
        if parent is None:
            return None
        idx = parent.children.index(page_id)
        if idx == 0:
            return None
        return self.pages[parent.children[idx - 1]]

    # ---- snapshots / reachability ----
    def snapshot(self) -> Dict[int, Dict[str, Any]]:
        return {pid: p.to_image(self.page_size) for pid, p in self.pages.items()}

    def reachable_pages(self) -> List[int]:
        """Pages reachable from root via children + sibling chain."""
        if self.root_id is None:
            return []
        seen, out = set(), []
        def walk(pid: int):
            if pid in seen:
                return
            seen.add(pid)
            out.append(pid)
            p = self.pages[pid]
            for c in p.children:
                walk(c)
        walk(self.root_id)
        # leaf sibling chain
        leftmost = self.pages[self.root_id]
        while not leftmost.is_leaf:
            leftmost = self.pages[leftmost.children[0]]
        cur = leftmost
        while cur is not None and cur.page_id not in seen or (cur and cur.page_id in seen and cur.right_sibling):
            nxt = cur.right_sibling
            if nxt is None or nxt in seen:
                break
            seen.add(nxt)
            out.append(nxt)
            cur = self.pages[nxt]
        return out

    def check_no_reachable_half_page(self) -> List[Dict[str, Any]]:
        """Invariant: no reachable page (except root) uses less than half the page."""
        bad = []
        for pid in self.reachable_pages():
            if pid == self.root_id:
                continue
            p = self.pages[pid]
            if p.used_bytes() < self.page_size // 2 and (p.keys or p.is_leaf):
                bad.append({"page_id": pid, "used_bytes": p.used_bytes(),
                            "page_size": self.page_size})
        return bad


# ---------- crash / replay ----------

def replay(wal: List[Dict[str, Any]], page_size: int, duplicate_policy: str) -> BTree:
    """Rebuild a tree purely from WAL records (insert/delete)."""
    t = BTree(page_size, duplicate_policy)
    for rec in wal:
        if rec["kind"] == "insert":
            t.insert(rec["key"], rec["value"])
        elif rec["kind"] == "delete":
            t.delete(rec["key"])
    return t


def replay_twice_idempotent(wal: List[Dict[str, Any]], page_size: int,
                            duplicate_policy: str) -> Dict[str, Any]:
    """Recovery replay must be idempotent: replaying the replayed WAL again
    yields the same reachable structure."""
    t1 = replay(wal, page_size, duplicate_policy)
    t2 = replay(t1.wal, page_size, duplicate_policy)
    r1 = {pid: t1.pages[pid].to_image(page_size) for pid in t1.reachable_pages()}
    r2 = {pid: t2.pages[pid].to_image(page_size) for pid in t2.reachable_pages()}
    return {"idempotent": r1 == r2, "first": r1, "second": r2}


def run_operations(ops: List[Dict[str, Any]], page_size: int,
                   duplicate_policy: str = "update",
                   crash_at: Optional[int] = None) -> Dict[str, Any]:
    """Execute logical ops, capturing per-op pre/post page mirrors, slot dirs,
    free space, sibling links, page id allocation and LSNs.

    crash_at: if set, simulate a write-page interruption after that many ops:
    the tree state is discarded and recovery replays the WAL from scratch.
    """
    tree = BTree(page_size, duplicate_policy)
    results: List[OpResult] = []
    for idx, op in enumerate(ops):
        before = tree.snapshot()
        root_before = tree.root_id
        next_before = tree.next_page_id
        notes: List[str] = []
        splits_before = len([w for w in tree.wal if w["kind"].startswith("split")])
        kind = op["op"]
        if kind == "insert":
            tree.insert(op["key"], op["value"])
        elif kind == "delete":
            tree.delete(op["key"])
        else:
            raise ValueError(f"unknown op {kind}")
        allocated = list(range(next_before, tree.next_page_id))
        splits = [w for w in tree.wal if w["kind"].startswith("split")][splits_before:]
        results.append(OpResult(
            index=idx, op=op, pages_before=before, pages_after=tree.snapshot(),
            root_before=root_before, root_after=tree.root_id,
            page_ids_allocated=allocated, splits=splits, notes=notes))
        if crash_at is not None and idx + 1 == crash_at:
            wal_copy = list(tree.wal)
            recovered = replay(wal_copy, page_size, duplicate_policy)
            notes.append(f"crash after op {idx}: write interrupted; recovered by WAL replay")
            tree = recovered
            results.append(OpResult(
                index=idx, op={"op": "crash_recover"}, pages_before=before,
                pages_after=tree.snapshot(), root_before=root_before,
                root_after=tree.root_id, page_ids_allocated=[], splits=[],
                notes=["crash_recovery"]))
    return {
        "tree": tree,
        "steps": results,
        "wal": tree.wal,
        "reachable": tree.reachable_pages(),
        "half_page_violations": tree.check_no_reachable_half_page(),
    }
