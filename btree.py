"""B-tree engine: fixed-size pages, slot directory, WAL, crash recovery.

All space decisions are made in *bytes*, never in key counts.
Separator policy (fixed and consistent):
  * leaf split     -> copy-up: smallest key of the right page is copied up
                      and the key stays inside the right leaf;
  * internal split -> promote: the middle key moves up and is removed
                      from both children.
"""
from __future__ import annotations

import numpy as np

PAGE_SIZE = 256
HEADER_SIZE = 20
LEAF = "leaf"
INTERNAL = "internal"
SLOT_ENTRY = 2
NO_PAGE = 0xFFFFFFFF

KIND_CODE = {LEAF: 1, INTERNAL: 2}
CODE_KIND = {1: LEAF, 2: INTERNAL}

LEAF_OVERHEAD = 4        # key_len(2) + value_len(2)
INTERNAL_OVERHEAD = 6    # key_len(2) + child(4)

H_KIND = 0
H_NSLOTS = 1
H_FREE_START = 3
H_FREE_END = 5
H_SIBLING = 7
H_LEFT_CHILD = 11
H_FLAGS = 15
H_CHECKSUM = 18

FLAG_EMPTY = 0x01


class TornPageError(ValueError):
    """Raised when a page image fails its checksum (torn/interrupted write)."""


def leaf_record_size(key: bytes, value: bytes) -> int:
    return LEAF_OVERHEAD + len(key) + len(value)


def internal_record_size(key: bytes) -> int:
    return INTERNAL_OVERHEAD + len(key)


class Page:
    """One fixed-size page.

    Leaf records:     list[(key: bytes, value: bytes)]
    Internal records: list[(key: bytes, child_page_id: int)], ordered by key;
                      ``left_child`` covers keys smaller than the first record.
    """

    def __init__(self, page_id: int, kind: str):
        self.page_id = page_id
        self.kind = kind
        self.records = []
        self.sibling = None
        self.left_child = None
        self.flags = 0

    def used_bytes(self) -> int:
        total = HEADER_SIZE + SLOT_ENTRY * len(self.records)
        if self.kind == LEAF:
            total += sum(LEAF_OVERHEAD + len(k) + len(v) for k, v in self.records)
        else:
            total += sum(INTERNAL_OVERHEAD + len(k) for k, _ in self.records)
        return total

    def free_bytes(self, page_size: int = PAGE_SIZE) -> int:
        return page_size - self.used_bytes()

    def overfull(self, page_size: int = PAGE_SIZE) -> bool:
        return self.used_bytes() > page_size


def _encode_leaf_record(key: bytes, value: bytes) -> bytes:
    return len(key).to_bytes(2, "big") + key + len(value).to_bytes(2, "big") + value


def _encode_internal_record(key: bytes, child: int) -> bytes:
    return len(key).to_bytes(2, "big") + key + child.to_bytes(4, "big")


def serialize_page(page: Page, page_size: int = PAGE_SIZE) -> bytes:
    """Render a page as exactly ``page_size`` bytes (the archived image)."""
    buf = bytearray(page_size)
    pos = page_size
    offsets = []
    for record in page.records:
        if page.kind == LEAF:
            data = _encode_leaf_record(record[0], record[1])
        else:
            data = _encode_internal_record(record[0], record[1])
        pos -= len(data)
        buf[pos:pos + len(data)] = data
        offsets.append(pos)

    nslots = len(page.records)
    free_start = HEADER_SIZE + SLOT_ENTRY * nslots
    free_end = pos if nslots else page_size
    buf[H_KIND] = KIND_CODE[page.kind]
    buf[H_NSLOTS:H_NSLOTS + 2] = nslots.to_bytes(2, "big")
    buf[H_FREE_START:H_FREE_START + 2] = free_start.to_bytes(2, "big")
    buf[H_FREE_END:H_FREE_END + 2] = free_end.to_bytes(2, "big")
    sibling = page.sibling if page.sibling is not None else NO_PAGE
    left_child = page.left_child if page.left_child is not None else NO_PAGE
    buf[H_SIBLING:H_SIBLING + 4] = sibling.to_bytes(4, "big")
    buf[H_LEFT_CHILD:H_LEFT_CHILD + 4] = left_child.to_bytes(4, "big")
    buf[H_FLAGS] = page.flags
    for i, offset in enumerate(offsets):
        base = HEADER_SIZE + SLOT_ENTRY * i
        buf[base:base + 2] = offset.to_bytes(2, "big")

    checksum = int(np.frombuffer(bytes(buf), dtype=np.uint8).sum()) % 65536
    buf[H_CHECKSUM:H_CHECKSUM + 2] = checksum.to_bytes(2, "big")
    return bytes(buf)


def page_checksum(image: bytes) -> int:
    zeroed = bytearray(image)
    zeroed[H_CHECKSUM:H_CHECKSUM + 2] = b"\x00\x00"
    return int(np.frombuffer(bytes(zeroed), dtype=np.uint8).sum()) % 65536


def parse_page(image: bytes, page_id: "int | None" = None) -> Page:
    """Parse a page image, verifying its checksum."""
    stored = int.from_bytes(image[H_CHECKSUM:H_CHECKSUM + 2], "big")
    if page_checksum(image) != stored:
        raise TornPageError(f"page {page_id} checksum mismatch (torn write)")
    kind = CODE_KIND[image[H_KIND]]
    page = Page(page_id if page_id is not None else 0, kind)
    nslots = int.from_bytes(image[H_NSLOTS:H_NSLOTS + 2], "big")
    page.sibling = int.from_bytes(image[H_SIBLING:H_SIBLING + 4], "big")
    if page.sibling == NO_PAGE:
        page.sibling = None
    page.left_child = int.from_bytes(image[H_LEFT_CHILD:H_LEFT_CHILD + 4], "big")
    if page.left_child == NO_PAGE:
        page.left_child = None
    page.flags = image[H_FLAGS]
    offsets = [
        int.from_bytes(image[HEADER_SIZE + 2 * i:HEADER_SIZE + 2 * i + 2], "big")
        for i in range(nslots)
    ]
    for offset in offsets:
        key_len = int.from_bytes(image[offset:offset + 2], "big")
        key = bytes(image[offset + 2:offset + 2 + key_len])
        if kind == LEAF:
            value_start = offset + 2 + key_len
            value_len = int.from_bytes(image[value_start:value_start + 2], "big")
            value = bytes(image[value_start + 2:value_start + 2 + value_len])
            page.records.append((key, value))
        else:
            child = int.from_bytes(
                image[offset + 2 + key_len:offset + 2 + key_len + 4], "big"
            )
            page.records.append((key, child))
    return page


def layout_regions(image: bytes) -> list:
    """Return [(start, end, region_name)] byte regions for the layout view."""
    nslots = int.from_bytes(image[H_NSLOTS:H_NSLOTS + 2], "big")
    free_end = int.from_bytes(image[H_FREE_END:H_FREE_END + 2], "big")
    slot_end = HEADER_SIZE + 2 * nslots
    regions = [(0, HEADER_SIZE, "header")]
    if nslots:
        regions.append((HEADER_SIZE, slot_end, "slot_directory"))
    if slot_end < free_end:
        regions.append((slot_end, free_end, "free_space"))
    if free_end < len(image):
        regions.append((free_end, len(image), "records"))
    return regions


def split_index_bytes(sizes: list) -> int:
    """Pick the split point by *byte* space (about half the record bytes)."""
    total = sum(sizes)
    cumulative = 0
    for i, size in enumerate(sizes):
        cumulative += size
        if cumulative >= (total + 1) // 2:
            return i + 1
    return len(sizes)


class BTree:
    def __init__(self, page_size: int = PAGE_SIZE, duplicate_policy: str = "overwrite"):
        self.page_size = page_size
        self.duplicate_policy = duplicate_policy
        self.pages = {}
        self.root_id = None
        self.next_page_id = 1
        self.lsn = 0
        self.wal = []            # [{lsn, op_index, images: {pid: bytes}}]
        self.disk = {}           # pid -> committed page image
        self.checkpoint_lsn = 0
        self.disk_root_id = None
        self.op_log = []         # logical op records (the archive baseline)
        self.alloc_log = []      # page-id allocation sequence

    # -- allocation / descent -------------------------------------------
    def _alloc(self, kind: str, op: dict) -> Page:
        page_id = self.next_page_id
        self.next_page_id += 1
        self.alloc_log.append(page_id)
        page = Page(page_id, kind)
        self.pages[page_id] = page
        op["page_id_alloc"].append(page_id)
        return page

    def _descend(self, key: bytes) -> list:
        path = []
        node = self.pages[self.root_id]
        while node.kind == INTERNAL:
            path.append(node)
            chosen = node.left_child
            for record_key, child in node.records:
                if key >= record_key:
                    chosen = child
                else:
                    break
            node = self.pages[chosen]
        path.append(node)
        return path

    def find(self, key: bytes):
        if self.root_id is None:
            return None
        leaf = self._descend(key)[-1]
        for record_key, value in leaf.records:
            if record_key == key:
                return value
        return None

    # -- logical operations ---------------------------------------------
    def _new_op(self, name: str, key: bytes, value) -> dict:
        self.lsn += 1
        return {
            "op_index": len(self.op_log),
            "op": name,
            "key": key.hex(),
            "value": value.hex() if isinstance(value, bytes) else value,
            "lsn": self.lsn,
            "touched": [],
            "pre_images": {},
            "post_images": {},
            "slot_directory": {},
            "free_space": {},
            "sibling_links": {},
            "page_id_alloc": [],
            "splits": [],
            "root_changed": False,
            "status": "applied",
            "duplicate": False,
            "empty_page": None,
        }

    def _snapshot(self, op: dict, pages, phase: str):
        table = op["pre_images"] if phase == "pre" else op["post_images"]
        for page in pages:
            table[str(page.page_id)] = serialize_page(page, self.page_size).hex()
        if phase == "post":
            for page in pages:
                pid = str(page.page_id)
                image = bytes.fromhex(table[pid])
                parsed = parse_page(image, page.page_id)
                offsets = [
                    int.from_bytes(
                        image[HEADER_SIZE + 2 * i:HEADER_SIZE + 2 * i + 2], "big"
                    )
                    for i in range(len(page.records))
                ]
                op["slot_directory"][pid] = offsets
                op["free_space"][pid] = parsed.free_bytes(self.page_size)
                op["sibling_links"][pid] = page.sibling

    def _finish(self, op: dict, touched):
        touched_ids = []
        for page in touched:
            if page.page_id not in touched_ids:
                touched_ids.append(page.page_id)
        op["touched"] = touched_ids
        images = {pid: bytes.fromhex(op["post_images"].get(str(pid),
                    serialize_page(self.pages[pid], self.page_size).hex()))
                  for pid in touched_ids}
        self.wal.append({"lsn": op["lsn"], "op_index": op["op_index"],
                         "images": images})
        self.op_log.append(op)
        return op

    def insert(self, key: bytes, value: bytes) -> dict:
        op = self._new_op("insert", key, value)
        if self.root_id is None:
            root = self._alloc(LEAF, op)
            self.root_id = root.page_id
            op["root_changed"] = True

        path = self._descend(key)
        leaf = path[-1]
        insert_at = len(leaf.records)
        for i, (record_key, _) in enumerate(leaf.records):
            if record_key >= key:
                insert_at = i
                break
        if insert_at < len(leaf.records) and leaf.records[insert_at][0] == key:
            op["duplicate"] = True
            if self.duplicate_policy == "reject":
                op["status"] = "rejected_duplicate"
                return self._finish(op, [])
            if self.duplicate_policy == "overwrite":
                self._snapshot(op, [leaf], "pre")
                leaf.records[insert_at] = (key, value)
                self._snapshot(op, [leaf], "post")
                return self._finish(op, [leaf])
            insert_at += 1  # "allow": keep both, new duplicate sits after existing

        self._snapshot(op, path, "pre")
        leaf.records.insert(insert_at, (key, value))
        if leaf.overfull(self.page_size):
            self._split_leaf(path, op)
        self._snapshot_post_chain(op, path)
        return self._finish(op, self._path_plus(op, path))

    def _snapshot_post_chain(self, op, path):
        pages = {p.page_id: p for p in path}
        for pid in op["page_id_alloc"]:
            pages[pid] = self.pages[pid]
        self._snapshot(op, list(pages.values()), "post")

    def _path_plus(self, op, path):
        pages = []
        seen = set()
        for page in path:
            if page.page_id not in seen:
                pages.append(page)
                seen.add(page.page_id)
        for pid in op["page_id_alloc"]:
            if pid not in seen:
                pages.append(self.pages[pid])
                seen.add(pid)
        return pages

    # -- splits ----------------------------------------------------------
    def _split_leaf(self, path: list, op: dict):
        leaf = path[-1]
        sizes = [LEAF_OVERHEAD + len(k) + len(v) for k, v in leaf.records]
        overflow = leaf.used_bytes() - self.page_size
        idx = split_index_bytes(sizes)
        right = self._alloc(LEAF, op)
        right.flags = leaf.flags
        right.records = leaf.records[idx:]
        leaf.records = leaf.records[:idx]
        right.sibling = leaf.sibling
        leaf.sibling = right.page_id
        separator = right.records[0][0]  # copy-up: separator stays in right leaf
        op["splits"].append({
            "page": leaf.page_id, "right": right.page_id, "kind": LEAF,
            "separator": separator.hex(), "rule": "copy_up",
            "record_count_left": len(leaf.records),
            "record_count_right": len(right.records),
            "bytes_left": sum(sizes[:idx]),
            "bytes_right": sum(sizes[idx:]),
            "reason": {"used_bytes": self.page_size + overflow,
                       "overflow_bytes": overflow,
                       "page_size_bytes": self.page_size,
                       "record_count": len(right.records) + len(leaf.records)},
        })
        self._insert_in_parent(path, separator, right.page_id, op)

    def _split_internal(self, path: list, op: dict):
        node = path[-1]
        sizes = [INTERNAL_OVERHEAD + len(k) for k, _ in node.records]
        overflow = node.used_bytes() - self.page_size
        idx = split_index_bytes(sizes)
        promoted_key = node.records[idx][0]
        right = self._alloc(INTERNAL, op)
        right.left_child = node.records[idx][1]  # promoted record's child
        right.records = node.records[idx + 1:]
        node.records = node.records[:idx]
        op["splits"].append({
            "page": node.page_id, "right": right.page_id, "kind": INTERNAL,
            "separator": promoted_key.hex(), "rule": "promote",
            "record_count_left": len(node.records),
            "record_count_right": len(right.records),
            "bytes_left": sum(sizes[:idx]),
            "bytes_right": sum(sizes[idx + 1:]),
            "reason": {"used_bytes": self.page_size + overflow,
                       "overflow_bytes": overflow,
                       "page_size_bytes": self.page_size,
                       "record_count": len(right.records) + len(node.records) + 1},
        })
        self._insert_in_parent(path, promoted_key, right.page_id, op)

    def _insert_in_parent(self, path, separator: bytes, right_id: int, op: dict):
        leaf = path[-1]
        left_id = leaf.page_id
        if len(path) == 1:
            root = self._alloc(INTERNAL, op)
            root.left_child = left_id
            root.records = [(separator, right_id)]
            self.root_id = root.page_id
            op["root_changed"] = True
            return
        parent = path[-2]
        insert_at = len(parent.records)
        for i, (record_key, _) in enumerate(parent.records):
            if separator < record_key:
                insert_at = i
                break
        parent.records.insert(insert_at, (separator, right_id))
        if parent.overfull(self.page_size):
            self._split_internal(path[:-1], op)

    def delete(self, key: bytes) -> dict:
        op = self._new_op("delete", key, None)
        if self.root_id is None:
            op["status"] = "missing"
            return self._finish(op, [])
        path = self._descend(key)
        leaf = path[-1]
        self._snapshot(op, path, "pre")
        target = next((i for i, (k, _) in enumerate(leaf.records) if k == key), None)
        if target is None:
            op["status"] = "missing"
            self._snapshot(op, path, "post")
            return self._finish(op, path)
        leaf.records.pop(target)
        if not leaf.records:
            leaf.flags |= FLAG_EMPTY
            op["empty_page"] = leaf.page_id
        self._snapshot_post_chain(op, path)
        return self._finish(op, self._path_plus(op, path))

    # -- durability: flush / crash / recover -----------------------------
    def flush(self, through_lsn: "int | None" = None):
        """Apply WAL entries up to ``through_lsn`` to committed storage."""
        if through_lsn is None:
            through_lsn = self.lsn
        for entry in self.wal:
            if entry["lsn"] <= self.checkpoint_lsn:
                continue
            if entry["lsn"] > through_lsn:
                break
            self.disk.update(entry["images"])
            self.checkpoint_lsn = entry["lsn"]
            self.disk_root_id = self.root_id

    def reachable_pages(self, source: str = "memory") -> list:
        """All pages reachable from the root (children + leaf sibling chain)."""
        if source == "memory":
            root_id, store = self.root_id, self.pages
        else:
            root_id, store = self.disk_root_id, None
        if root_id is None:
            return []
        result = []
        seen = set()
        if source == "memory":
            stack = [root_id]
            while stack:
                pid = stack.pop()
                if pid in seen or pid not in self.pages:
                    continue
                seen.add(pid)
                result.append(pid)
                page = self.pages[pid]
                if page.kind == INTERNAL:
                    if page.left_child is not None:
                        stack.append(page.left_child)
                    stack.extend(child for _, child in page.records)
                elif page.sibling is not None:
                    stack.append(page.sibling)
            return sorted(result)
        stack = [root_id]
        while stack:
            pid = stack.pop()
            if pid in seen or pid not in self.disk:
                continue
            seen.add(pid)
            result.append(pid)
            page = parse_page(self.disk[pid], pid)
            if page.kind == INTERNAL:
                if page.left_child is not None:
                    stack.append(page.left_child)
                stack.extend(child for _, child in page.records)
            elif page.sibling is not None:
                stack.append(page.sibling)
        return sorted(result)

    def simulate_crash(self, torn_page_id: int) -> dict:
        """Interrupt a page write: persist the first half of the new image."""
        new_image = serialize_page(self.pages[torn_page_id], self.page_size)
        old_image = self.disk.get(torn_page_id, bytes(self.page_size))
        half = self.page_size // 2
        torn = new_image[:half] + old_image[half:]
        self.disk[torn_page_id] = torn
        return {
            "torn_page": torn_page_id,
            "half_bytes": half,
            "reachable": self.reachable_pages("disk"),
        }

    def recover(self) -> dict:
        """Replay WAL; fix torn pages from after-images. Safe to run twice."""
        torn_fixed = []
        for pid in list(self.disk.keys()):
            try:
                parse_page(self.disk[pid], pid)
            except TornPageError:
                image = self._wal_image(pid)
                if image is not None:
                    self.disk[pid] = image
                    torn_fixed.append(pid)
        replayed = []
        for entry in self.wal:
            if entry["lsn"] > self.checkpoint_lsn:
                self.disk.update(entry["images"])
                replayed.append(entry["lsn"])
                self.checkpoint_lsn = entry["lsn"]
                self.disk_root_id = self.root_id
        return {
            "torn_fixed": sorted(torn_fixed),
            "replayed_lsn": replayed,
            "reachable_after": self.reachable_pages("disk"),
        }

    def _wal_image(self, page_id: int):
        image = None
        for entry in self.wal:
            if page_id in entry["images"]:
                image = entry["images"][page_id]
        return image

    def disk_is_valid(self) -> tuple:
        """Return (ok, torn_or_missing_pages) over all *reachable* pages."""
        bad = []
        for pid in self.reachable_pages("disk"):
            image = self.disk.get(pid)
            if image is None:
                bad.append(pid)
                continue
            try:
                parse_page(image, pid)
            except TornPageError:
                bad.append(pid)
        return (not bad), bad

    def tree_snapshot(self) -> dict:
        pages = {}
        for pid, page in self.pages.items():
            pages[str(pid)] = {
                "kind": page.kind,
                "keys": [k.hex() for k, _ in page.records],
                "children": [c for _, c in page.records]
                              if page.kind == INTERNAL else [],
                "left_child": page.left_child,
                "sibling": page.sibling,
                "free_bytes": page.free_bytes(self.page_size),
                "empty": bool(page.flags & FLAG_EMPTY),
            }
        return {"root": self.root_id, "pages": pages,
                "next_page_id": self.next_page_id, "lsn": self.lsn}
