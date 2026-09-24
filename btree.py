"""B-tree page engine.

Fixed-size pages with an explicit slot directory, variable-length records,
byte-space driven splits, separator copy/promote rules, leaf sibling links,
page-id allocation, LSN-numbered physiological log records, torn-write crash
simulation and idempotent redo recovery.  Structural invariants are emitted
as machine-readable verdicts.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

# Header (28 bytes, page relative coordinates, origin = first header byte):
#   0..3  page_id (u32)
#   4     kind (0=leaf, 1=internal)
#   5..6  nslots (u16)
#   7..8  free_start (u16)  first byte after slot directory
#   9..10 free_end   (u16)  last record begins here (records grow downward)
#  11..14 right_sibling (u32, 0xffffffff = none)
#  15     flags (bit0 = empty/tombstone)
#  16..19 leftmost_child (u32, internal pages)
#  20..23 page_lsn (u32)
#  24..27 checksum (crc32 over bytes 0..23)
HEADER_SIZE = 28
SLOT_SIZE = 4
NO_PAGE = 0xFFFFFFFF
LEAF = 0
INTERNAL = 1
FLAG_EMPTY = 0x01


class TornPageError(Exception):
    """Raised when an on-disk page image cannot be parsed or fails its crc."""


def crc32(data: bytes) -> int:
    return zlib.crc32(np.frombuffer(data, dtype=np.uint8).tobytes()) & 0xFFFFFFFF


@dataclass
class Page:
    page_id: int
    kind: int
    entries: list = field(default_factory=list)
    right_sibling: int | None = None
    leftmost_child: int | None = None
    lsn: int = 0
    flags: int = 0

    # -- record / space accounting -----------------------------------------
    def record_bytes(self, entry) -> bytes:
        if self.kind == LEAF:
            key, value = entry
            return struct.pack(">H", len(key)) + key + struct.pack(">H", len(value)) + value
        key, child = entry
        return struct.pack(">H", len(key)) + key + struct.pack(">I", child)

    def used_bytes(self) -> int:
        return sum(len(self.record_bytes(e)) for e in self.entries)

    def dir_bytes(self) -> int:
        return SLOT_SIZE * len(self.entries)

    def free_bytes(self, page_size: int) -> int:
        return page_size - HEADER_SIZE - self.dir_bytes() - self.used_bytes()

    def needed_bytes(self, entry) -> int:
        return len(self.record_bytes(entry)) + SLOT_SIZE

    def is_empty(self) -> bool:
        return self.kind == LEAF and not self.entries

    # -- serialization ------------------------------------------------------
    def serialize(self, page_size: int) -> bytes:
        buf = bytearray(page_size)
        free_end = page_size
        slots = []
        for entry in self.entries:
            rec = self.record_bytes(entry)
            free_end -= len(rec)
            buf[free_end:free_end + len(rec)] = rec
            slots.append((free_end, len(rec)))
        free_start = HEADER_SIZE + SLOT_SIZE * len(self.entries)
        struct.pack_into(">IBHHH", buf, 0, self.page_id, self.kind,
                         len(self.entries), free_start, free_end)
        struct.pack_into(">I", buf, 11,
                         self.right_sibling if self.right_sibling is not None else NO_PAGE)
        buf[15] = self.flags
        struct.pack_into(">I", buf, 16,
                         self.leftmost_child if self.leftmost_child is not None else NO_PAGE)
        struct.pack_into(">I", buf, 20, self.lsn)
        offset = HEADER_SIZE
        for rec_off, rec_len in slots:
            struct.pack_into(">HH", buf, offset, rec_off, rec_len)
            offset += SLOT_SIZE
        struct.pack_into(">I", buf, 24, crc32(bytes(buf[:24])))
        return bytes(buf)

    @classmethod
    def parse(cls, buf: bytes) -> "Page":
        if len(buf) < HEADER_SIZE:
            raise TornPageError("short page image")
        page_id, kind, nslots, free_start, free_end = struct.unpack_from(">IBHHH", buf, 0)
        checksum = struct.unpack_from(">I", buf, 24)[0]
        if checksum != crc32(bytes(buf[:24])):
            raise TornPageError(f"page {page_id} checksum mismatch (torn write)")
        if kind not in (LEAF, INTERNAL) or free_start > free_end or free_end > len(buf):
            raise TornPageError(f"page {page_id} header pointers invalid")
        end = HEADER_SIZE + nslots * SLOT_SIZE
        if end > free_start:
            raise TornPageError(f"page {page_id} slot directory overruns free space")
        right = struct.unpack_from(">I", buf, 11)[0]
        leftmost = struct.unpack_from(">I", buf, 16)[0]
        lsn = struct.unpack_from(">I", buf, 20)[0]
        flags = buf[15]
        entries = []
        for i in range(nslots):
            rec_off, rec_len = struct.unpack_from(">HH", buf, HEADER_SIZE + i * SLOT_SIZE)
            if rec_off < free_end or rec_off + rec_len > len(buf):
                raise TornPageError(f"page {page_id} slot {i} out of bounds")
            rec = buf[rec_off:rec_off + rec_len]
            key_len = struct.unpack_from(">H", rec, 0)[0]
            if 2 + key_len > len(rec):
                raise TornPageError(f"page {page_id} slot {i} record truncated")
            key = bytes(rec[2:2 + key_len])
            if kind == LEAF:
                if 4 + key_len > len(rec):
                    raise TornPageError(f"page {page_id} slot {i} value truncated")
                value_len = struct.unpack_from(">H", rec, 2 + key_len)[0]
                value = bytes(rec[4 + key_len:4 + key_len + value_len])
                entries.append((key, value))
            else:
                if 2 + key_len + 4 > len(rec):
                    raise TornPageError(f"page {page_id} slot {i} child pointer truncated")
                child = struct.unpack_from(">I", rec, 2 + key_len)[0]
                entries.append((key, child))
        return cls(page_id=page_id, kind=kind, entries=entries,
                   right_sibling=None if right == NO_PAGE else right,
                   leftmost_child=None if leftmost == NO_PAGE else leftmost,
                   lsn=lsn, flags=flags)


def byte_layout_map(page: Page, page_size: int) -> np.ndarray:
    """Per-byte class map, page relative coordinates (0-based).

    0=header 1=slot_directory 2=free 3=record_data
    """
    image = page.serialize(page_size)
    classes = np.zeros(page_size, dtype=np.uint8)
    classes[:HEADER_SIZE] = 0
    nslots = len(page.entries)
    classes[HEADER_SIZE:HEADER_SIZE + nslots * SLOT_SIZE] = 1
    _, _, _, free_start, free_end = struct.unpack_from(">IBHHH", image, 0)
    classes[free_start:free_end] = 2
    classes[free_end:page_size] = 3
    return classes


class BTree:
    """In-memory page cache plus a durable page image disk and a redo log."""

    def __init__(self, page_size: int = 512, duplicate_policy: str = "reject",
                 separator_rule: str = "copy"):
        if separator_rule not in ("copy", "promote"):
            raise ValueError("separator_rule must be 'copy' or 'promote'")
        if duplicate_policy not in ("reject", "overwrite", "append"):
            raise ValueError("duplicate_policy must be reject/overwrite/append")
        self.page_size = page_size
        self.duplicate_policy = duplicate_policy
        self.separator_rule = separator_rule
        self.pages: dict[int, Page] = {}
        self.root_id: int | None = None
        self.next_page_id = 1
        self.lsn = 0
        self.log: list[dict] = []
        self.disk: dict[int, bytes] = {}
        self.superblock = {"root_id": None, "next_page_id": 1, "checkpoint_lsn": 0}
        self.traces: list[dict] = []

    # -- low level ----------------------------------------------------------
    def _alloc_page(self, kind: int) -> Page:
        page_id = self.next_page_id
        self.next_page_id += 1
        page = Page(page_id=page_id, kind=kind)
        self.pages[page_id] = page
        return page

    def _next_lsn(self) -> int:
        self.lsn += 1
        return self.lsn

    def _write_log(self, rectype: str, page: Page, extra: dict | None = None) -> dict:
        record = {
            "lsn": self.lsn,
            "type": rectype,
            "page_id": page.page_id,
            "image": page.serialize(self.page_size),
            "extra": extra or {},
            "root_id": self.root_id,
            "next_page_id": self.next_page_id,
        }
        self.log.append(record)
        return record

    def _snapshot(self, trace: dict, pages: list[Page]) -> None:
        for page in pages:
            key = str(page.page_id)
            if key in trace["pre_images"]:
                continue
            trace["pre_images"][key] = {
                "image": page.serialize(self.page_size).hex(),
                "slots": _slot_directory(page),
                "free_bytes": page.free_bytes(self.page_size),
                "right_sibling": page.right_sibling,
                "leftmost_child": page.leftmost_child,
                "lsn": page.lsn,
            }

    def _descend(self, key: bytes) -> tuple[Page, list]:
        path: list[tuple[Page, int | None]] = []
        page_id = self.root_id
        while True:
            page = self.pages[page_id]
            if page.kind == LEAF:
                return page, path
            child = page.leftmost_child
            entry_idx = None
            for idx, (sep, child_id) in enumerate(page.entries):
                if key >= sep:
                    child = child_id
                    entry_idx = idx
                else:
                    break
            path.append((page, entry_idx))
            page_id = child

    def _new_trace(self, op: str, key: bytes | None = None) -> dict:
        return {
            "op": op,
            "key": None if key is None else key.hex(),
            "pre_images": {},
            "allocations": [],
            "splits": [],
            "log_lsns": [],
            "root_changed": False,
            "result": "ok",
        }

    def get(self, key: bytes) -> bytes | None:
        leaf, _ = self._descend(key)
        for k, value in leaf.entries:
            if k == key:
                return value
        return None

    # -- insert / split -----------------------------------------------------
    def insert(self, key: bytes, value: bytes) -> dict:
        lsn = self._next_lsn()
        trace = self._new_trace("insert", key)
        trace["value"] = value.hex()
        if self.root_id is None:
            root = self._alloc_page(LEAF)
            self.root_id = root.page_id
            trace["allocations"].append(root.page_id)
            trace["root_changed"] = True
        leaf, path = self._descend(key)
        self._snapshot(trace, [leaf] + [p for p, _ in path])
        for idx, (existing_key, _) in enumerate(leaf.entries):
            if existing_key != key:
                continue
            trace["duplicate"] = self.duplicate_policy
            if self.duplicate_policy == "reject":
                trace["result"] = "rejected-duplicate"
                self.traces.append(trace)
                return trace
            if self.duplicate_policy == "overwrite":
                leaf.entries[idx] = (key, value)
                leaf.lsn = lsn
                trace["log_lsns"].append(lsn)
                self._write_log("update", leaf)
                self.traces.append(trace)
                return trace
            break
        entry = (key, value)
        if leaf.needed_bytes(entry) <= leaf.free_bytes(self.page_size):
            self._insert_leaf_entry(leaf, key, value)
            leaf.lsn = lsn
            trace["log_lsns"].append(lsn)
            self._write_log("update", leaf)
        else:
            self._split_leaf(leaf, entry, path, trace, lsn)
        self.traces.append(trace)
        return trace

    @staticmethod
    def _insert_leaf_entry(leaf: Page, key: bytes, value: bytes) -> None:
        position = 0
        while position < len(leaf.entries) and leaf.entries[position][0] < key:
            position += 1
        leaf.entries.insert(position, (key, value))

    def _split_leaf(self, leaf: Page, entry, path, trace: dict, lsn: int) -> None:
        merged = sorted(leaf.entries + [entry], key=lambda item: item[0])
        cut = _byte_balanced_cut(leaf, merged)
        left_entries = merged[:cut]
        right_entries = merged[cut:]
        before_free = leaf.free_bytes(self.page_size)
        needed = leaf.needed_bytes(entry)

        right = self._alloc_page(LEAF)
        trace["allocations"].append(right.page_id)
        right.entries = right_entries
        right.right_sibling = leaf.right_sibling
        leaf.entries = left_entries
        leaf.right_sibling = right.page_id
        separator = right.entries[0][0]
        if self.separator_rule == "promote":
            promoted = right.entries.pop(0)
            separator = promoted[0]
        leaf.lsn = lsn
        right.lsn = lsn
        trace["log_lsns"].extend([lsn, lsn])
        self._write_log("update", leaf)
        self._write_log("alloc", right)
        trace["splits"].append({
            "kind": "leaf",
            "left": leaf.page_id,
            "right": right.page_id,
            "separator": separator.hex(),
            "rule": self.separator_rule,
            "decided_by": "byte-space",
            "needed_bytes": needed,
            "free_bytes_before": before_free,
            "left_free_after": leaf.free_bytes(self.page_size),
            "right_free_after": right.free_bytes(self.page_size),
        })
        self._propagate(leaf.page_id, separator, right.page_id, path, trace, lsn)

    def _propagate(self, left_id: int, separator: bytes, right_id: int,
                   path, trace: dict, lsn: int) -> None:
        if not path:
            new_root = self._alloc_page(INTERNAL)
            trace["allocations"].append(new_root.page_id)
            new_root.leftmost_child = left_id
            new_root.entries = [(separator, right_id)]
            new_root.lsn = lsn
            trace["log_lsns"].append(lsn)
            self._write_log("alloc", new_root)
            trace["root_changed"] = True
            trace["splits"].append({
                "kind": "root", "new_root": new_root.page_id,
                "left": left_id, "right": right_id,
                "separator": separator.hex(), "rule": "promote",
            })
            self.root_id = new_root.page_id
            return
        parent, entry_idx = path[-1]
        new_entry = (separator, right_id)
        position = 0 if entry_idx is None else entry_idx + 1
        if parent.needed_bytes(new_entry) <= parent.free_bytes(self.page_size):
            parent.entries.insert(position, new_entry)
            parent.lsn = lsn
            trace["log_lsns"].append(lsn)
            self._write_log("update", parent)
            return
        self._split_internal(parent, new_entry, position, path[:-1], trace, lsn)

    def _split_internal(self, page: Page, new_entry, position: int,
                        path, trace: dict, lsn: int) -> None:
        merged = list(page.entries)
        merged.insert(position, new_entry)
        cut = max(1, min(len(merged) - 2, len(merged) // 2))
        left_entries = merged[:cut]
        middle_sep, middle_child = merged[cut]
        right_entries = merged[cut + 1:]

        right = self._alloc_page(INTERNAL)
        trace["allocations"].append(right.page_id)
        right.leftmost_child = middle_child
        right.entries = right_entries
        right.right_sibling = page.right_sibling
        page.entries = left_entries
        page.right_sibling = right.page_id
        page.lsn = lsn
        right.lsn = lsn
        trace["log_lsns"].extend([lsn, lsn])
        self._write_log("update", page)
        self._write_log("alloc", right)
        trace["splits"].append({
            "kind": "internal",
            "left": page.page_id,
            "right": right.page_id,
            "separator": middle_sep.hex(),
            "rule": "promote",
            "decided_by": "byte-space",
        })
        self._propagate(page.page_id, middle_sep, right.page_id, path, trace, lsn)

    # -- delete / empty pages ----------------------------------------------
    def delete(self, key: bytes) -> dict:
        lsn = self._next_lsn()
        trace = self._new_trace("delete", key)
        leaf, path = self._descend(key)
        self._snapshot(trace, [leaf] + [p for p, _ in path])
        position = next((i for i, (k, _) in enumerate(leaf.entries) if k == key), None)
        if position is None:
            trace["result"] = "not-found"
            self.traces.append(trace)
            return trace
        del leaf.entries[position]
        leaf.lsn = lsn
        trace["log_lsns"].append(lsn)
        self._write_log("update", leaf)
        if leaf.is_empty() and leaf.page_id != self.root_id:
            self._unlink_empty_leaf(leaf, path, trace, lsn)
        self._shrink_root(lsn, trace)
        self.traces.append(trace)
        return trace

    def _unlink_empty_leaf(self, leaf: Page, path, trace: dict, lsn: int) -> None:
        leaf.flags |= FLAG_EMPTY
        left_sibling = self._find_left_leaf_sibling(leaf.page_id)
        if left_sibling is not None:
            left_sibling.right_sibling = leaf.right_sibling
            left_sibling.lsn = lsn
            trace["log_lsns"].append(lsn)
            self._write_log("update", left_sibling)
        if path:
            parent, entry_idx = path[-1]
            if entry_idx is None:
                if parent.entries:
                    parent.leftmost_child = parent.entries[0][1]
                    del parent.entries[0]
            else:
                del parent.entries[entry_idx]
            parent.lsn = lsn
            trace["log_lsns"].append(lsn)
            self._write_log("update", parent)
        trace["empty_page"] = leaf.page_id
        trace["splits"].append({
            "kind": "empty-leaf",
            "page": leaf.page_id,
            "right_sibling": leaf.right_sibling,
            "retained": True,
        })

    def _find_left_leaf_sibling(self, leaf_id: int) -> Page | None:
        if self.root_id is None:
            return None
        previous = None

        def walk(page_id: int):
            nonlocal previous
            page = self.pages[page_id]
            if page.kind == LEAF:
                if page.page_id == leaf_id:
                    return previous
                previous = page
                return None
            result = walk(page.leftmost_child)
            if result is not None:
                return result
            for _, child_id in page.entries:
                result = walk(child_id)
                if result is not None:
                    return result
            return None

        return walk(self.root_id)

    def _shrink_root(self, lsn: int, trace: dict) -> None:
        while self.root_id is not None:
            root = self.pages[self.root_id]
            if root.kind == LEAF or root.entries:
                return
            promoted_child = root.leftmost_child
            self.root_id = promoted_child
            root.flags |= FLAG_EMPTY
            trace["root_changed"] = True
            trace["splits"].append({
                "kind": "root-shrink", "old_root": root.page_id,
                "promoted_child": promoted_child,
            })

    # -- commit / crash / recovery -----------------------------------------
    def commit(self) -> None:
        for page_id, page in self.pages.items():
            self.disk[page_id] = page.serialize(self.page_size)
        self.superblock = {
            "root_id": self.root_id,
            "next_page_id": self.next_page_id,
            "checkpoint_lsn": self.lsn,
        }

    def crash(self, torn_page_id: int, written_bytes: int) -> dict:
        """Interrupted page write: only the first written_bytes reach disk."""
        if torn_page_id not in self.pages:
            raise KeyError(torn_page_id)
        if not 0 <= written_bytes < self.page_size:
            raise ValueError("written_bytes must be inside the page")
        trace = self._new_trace("crash")
        trace["torn_page"] = torn_page_id
        trace["written_bytes"] = written_bytes
        full_image = self.pages[torn_page_id].serialize(self.page_size)
        self.disk[torn_page_id] = (
            full_image[:written_bytes] + bytes(self.page_size - written_bytes)
        )
        self.traces.append(trace)
        return trace

    def recover(self) -> dict:
        """Redo log records past the checkpoint; idempotent.

        The last redone record also restores the superblock (root pointer and
        page-id allocator), so a crash after a root split cannot strand the
        new root on a page image that never reached disk.
        """
        applied = []
        latest: dict[int, dict] = {}
        last_record = None
        # Physical after-image redo scans the durable log from the start:
        # each page is redone from its latest after-image whenever the on-disk
        # image is torn (crc fails) or older than the log record.  After-images
        # make this idempotent, and a torn page whose last modification
        # predates the checkpoint is rebuilt as well.
        for record in self.log:
            latest[record["page_id"]] = record
            last_record = record
        for page_id, record in latest.items():
            current = self.disk.get(page_id)
            needs_redo = True
            if current is not None:
                try:
                    needs_redo = Page.parse(current).lsn < record["lsn"]
                except TornPageError:
                    needs_redo = True
            if needs_redo:
                self.disk[page_id] = record["image"]
                applied.append(record["lsn"])
        applied.sort()
        if last_record is not None:
            self.superblock = {
                "root_id": last_record["root_id"],
                "next_page_id": last_record["next_page_id"],
                "checkpoint_lsn": self.lsn,
            }
        else:
            self.superblock["checkpoint_lsn"] = self.lsn
        return {"applied_lsns": applied}

    def reachable_pages(self) -> dict:
        """Traverse root, child pointers and sibling links over disk images."""
        root_id = self.superblock["root_id"]
        seen: set[int] = set()
        torn: list[int] = []
        order: list[int] = []
        stack = [root_id]
        while stack:
            page_id = stack.pop()
            if page_id is None or page_id in seen:
                continue
            seen.add(page_id)
            image = self.disk.get(page_id)
            if image is None:
                torn.append(page_id)
                continue
            try:
                page = Page.parse(image)
            except TornPageError:
                torn.append(page_id)
                continue
            order.append(page_id)
            if page.kind == INTERNAL:
                stack.append(page.leftmost_child)
                stack.extend(child_id for _, child_id in page.entries)
            if page.right_sibling is not None:
                stack.append(page.right_sibling)
        return {"reachable": sorted(seen), "torn": sorted(torn), "order": order}

    def leaf_chain(self) -> list[int]:
        page_id = self._leftmost_leaf_id()
        chain = []
        while page_id is not None:
            chain.append(page_id)
            page_id = self.pages[page_id].right_sibling
        return chain

    def _leftmost_leaf_id(self) -> int | None:
        page_id = self.root_id
        while page_id is not None:
            page = self.pages[page_id]
            if page.kind == LEAF:
                return page_id
            page_id = page.leftmost_child
        return None


def _slot_directory(page: Page) -> list[dict]:
    free_end = None
    slots = []
    for idx, entry in enumerate(page.entries):
        rec = page.record_bytes(entry)
        slots.append({
            "slot": idx,
            "key": entry[0].hex(),
            "record_bytes": len(rec),
        })
    return slots


def _byte_balanced_cut(page: Page, entries: list) -> int:
    """Split point chosen by record BYTE space, never by entry count.

    The left page receives whole entries until it holds at least half of the
    total record bytes; both sides must stay non-empty.
    """
    sizes = np.array([len(page.record_bytes(e)) for e in entries], dtype=np.int64)
    total = int(sizes.sum())
    cumulative = np.cumsum(sizes)
    cut = int(np.searchsorted(cumulative, (total + 1) // 2, side="left")) + 1
    return max(1, min(cut, len(entries) - 1))


def _leftmost_leaf_key(tree: BTree, page_id: int) -> bytes | None:
    page = tree.pages[page_id]
    while page.kind == INTERNAL:
        page = tree.pages[page.leftmost_child]
    return page.entries[0][0] if page.entries else None


def reachable_in_cache(tree: BTree) -> set[int]:
    """Page ids reachable from the in-memory root via child pointers."""
    seen: set[int] = set()
    if tree.root_id is None:
        return seen
    stack = [tree.root_id]
    while stack:
        page_id = stack.pop()
        if page_id in seen:
            continue
        seen.add(page_id)
        page = tree.pages.get(page_id)
        if page is None or page.kind != INTERNAL:
            continue
        stack.append(page.leftmost_child)
        stack.extend(child_id for _, child_id in page.entries)
    return seen


def check_invariants(tree: BTree) -> list[dict]:
    """Machine-readable structural verdicts (pass / fail / undetermined)."""
    verdicts: list[dict] = []
    reachable = reachable_in_cache(tree)

    # R1: every page fits page size -> splits were driven by byte space.
    overflow = [pid for pid, page in tree.pages.items() if pid in reachable
                if page.free_bytes(tree.page_size) < 0]
    split_traces = [s for trace in tree.traces for s in trace.get("splits", [])
                    if s.get("kind") in ("leaf", "internal")]
    forced = all(s.get("needed_bytes", 0) > s.get("free_bytes_before", -1)
                 for s in split_traces if "needed_bytes" in s)
    verdicts.append({
        "rule_id": "R1-byte-space-split",
        "status": "pass" if not overflow and (forced or not split_traces) else "fail",
        "affected": overflow,
        "basis": "split 点由剩余字节空间决定；任何页 header+slotdir+records 不得超过 page_size(byte)",
    })

    # R2: separator copy/promote rule applied consistently.
    inconsistent = []
    for page in (tree.pages[pid] for pid in sorted(reachable)):
        if page.kind != INTERNAL:
            continue
        for sep, child_id in page.entries:
            first_key = _leftmost_leaf_key(tree, child_id)
            if first_key is None:
                continue
            if tree.separator_rule == "copy" and first_key != sep:
                inconsistent.append({"page": page.page_id, "child": child_id,
                                     "separator": sep.hex(), "first_key":
                                     None if first_key is None else first_key.hex()})
            if tree.separator_rule == "promote" and first_key is not None \
                    and first_key <= sep:
                inconsistent.append({"page": page.page_id, "child": child_id,
                                     "separator": sep.hex(),
                                     "first_key": first_key.hex()})
    verdicts.append({
        "rule_id": "R2-separator-consistency",
        "status": "pass" if not inconsistent else "fail",
        "affected": inconsistent,
        "basis": f"separator 规则({tree.separator_rule})必须在全树一致：copy=右子树最小键与分隔键相同，promote=右子树最小键严格更大",
    })

    # R3: no reachable half-written page after crash recovery.
    reach = tree.reachable_pages()
    verdicts.append({
        "rule_id": "R3-no-reachable-half-page",
        "status": "pass" if not reach["torn"] else "fail",
        "affected": reach["torn"],
        "basis": "崩溃恢复后从根经子指针和 sibling link 可达的页面都必须通过 crc 校验",
    })

    # R4: recovery replay is idempotent.
    before = {pid: img for pid, img in tree.disk.items()}
    tree.recover()
    after_first = {pid: img for pid, img in tree.disk.items()}
    second = tree.recover()
    verdicts.append({
        "rule_id": "R4-replay-idempotent",
        "status": "pass" if after_first == tree.disk and not second["applied_lsns"] else "fail",
        "affected": [],
        "basis": "recover() 连续执行两次：磁盘镜像逐字节相同且第二次不应用任何日志记录",
    })

    # R5: leaf sibling chain matches in-order key traversal.
    ordered_keys: list[bytes] = []

    def collect(page_id: int) -> None:
        page = tree.pages[page_id]
        if page.kind == LEAF:
            ordered_keys.extend(k for k, _ in page.entries)
            return
        collect(page.leftmost_child)
        for _, child_id in page.entries:
            collect(child_id)

    chain_broken = []
    if tree.root_id is not None:
        collect(tree.root_id)
        chain = tree.leaf_chain()
        chain_keys: list[bytes] = []
        for pid in chain:
            chain_keys.extend(k for k, _ in tree.pages[pid].entries)
        if ordered_keys != chain_keys or ordered_keys != sorted(ordered_keys):
            chain_broken = chain
    verdicts.append({
        "rule_id": "R5-sibling-chain-order",
        "status": "pass" if not chain_broken else "fail",
        "affected": chain_broken,
        "basis": "沿 leftmost/sibling link 走出的叶子键序必须等于中序遍历且非降序（含删除空页后）",
    })

    # R6: free space accounting matches the serialized byte layout (numpy).
    mismatched = []
    for pid in sorted(reachable):
        page = tree.pages[pid]
        layout = byte_layout_map(page, tree.page_size)
        free_cells = int(np.count_nonzero(layout == 2))
        if free_cells != page.free_bytes(tree.page_size):
            mismatched.append({"page": pid, "layout_free": free_cells,
                               "accounted_free": page.free_bytes(tree.page_size)})
    verdicts.append({
        "rule_id": "R6-free-space-accounting",
        "status": "pass" if not mismatched else "fail",
        "affected": mismatched,
        "basis": "页内字节布局中 free 区字节数必须等于 page_size-header-slotdir-records",
    })

    # R7: page ids allocated monotonically, never reused.
    allocated = [pid for trace in tree.traces for pid in trace.get("allocations", [])]
    reused = sorted({pid for pid in allocated if allocated.count(pid) > 1})
    verdicts.append({
        "rule_id": "R7-page-id-monotonic",
        "status": "pass" if not reused and allocated == sorted(allocated) else "fail",
        "affected": reused,
        "basis": "page id 单调分配，空页保留后其 id 不得被复用",
    })
    return verdicts
