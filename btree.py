"""Page-based B-tree engine.

Fixed page size, variable-length keys, slot directory, free-space accounting,
sibling links, page-id allocation, log sequence numbers, torn-write crash
and idempotent recovery replay.
"""
from __future__ import annotations

import hashlib
import struct
from bisect import bisect_left, bisect_right

import numpy as np

LEAF = 1
INTERNAL = 2
HEADER_SIZE = 20
FLAG_TORN = 1

_O_TYPE = 0
_O_PID = 1
_O_PARENT = 3
_O_SIBLING = 5
_O_LSN = 7
_O_NSLOTS = 9
_O_FSTART = 11
_O_FEND = 13
_O_FLAGS = 15
_O_LEFTMOST = 16


def _u16(page, off):
    return struct.unpack_from("<H", page, off)[0]


def _w16(page, off, value):
    struct.pack_into("<H", page, off, value)


def new_page(page_size, page_type, page_id):
    page = np.zeros(page_size, dtype=np.uint8)
    page[_O_TYPE] = page_type
    _w16(page, _O_PID, page_id)
    _w16(page, _O_FSTART, HEADER_SIZE)
    _w16(page, _O_FEND, page_size)
    return page


def leaf_cell(key, value):
    return bytes([len(key)]) + key + bytes([len(value)]) + value


def internal_cell(key, child):
    return bytes([len(key)]) + key + struct.pack("<H", child)


def parse_cell(page, off, page_type):
    key_len = int(page[off])
    key = bytes(page[off + 1:off + 1 + key_len])
    if page_type == LEAF:
        val_len = int(page[off + 1 + key_len])
        value = bytes(page[off + 2 + key_len:off + 2 + key_len + val_len])
        return key, value, 2 + key_len + val_len
    child = _u16(page, off + 1 + key_len)
    return key, child, 3 + key_len


def slot_offsets(page):
    count = _u16(page, _O_NSLOTS)
    return [_u16(page, HEADER_SIZE + 2 * i) for i in range(count)]


def free_bytes(page):
    return int(_u16(page, _O_FEND) - _u16(page, _O_FSTART))


def read_cells(page):
    page_type = int(page[_O_TYPE])
    return [parse_cell(page, off, page_type)[:2] for off in slot_offsets(page)]


def read_key(page, off):
    key_len = int(page[off])
    return bytes(page[off + 1:off + 1 + key_len])


def insert_cell(page, cell, key, right_of_equal=True):
    need = len(cell) + 2
    if free_bytes(page) < need:
        raise PageFull("not enough free space")
    keys = [read_key(page, off) for off in slot_offsets(page)]
    pos = bisect_right(keys, key) if right_of_equal else bisect_left(keys, key)
    count = _u16(page, _O_NSLOTS)
    new_end = _u16(page, _O_FEND) - len(cell)
    page[new_end:new_end + len(cell)] = np.frombuffer(cell, dtype=np.uint8)
    for i in range(count, pos, -1):
        _w16(page, HEADER_SIZE + 2 * i, _u16(page, HEADER_SIZE + 2 * (i - 1)))
    _w16(page, HEADER_SIZE + 2 * pos, new_end)
    _w16(page, _O_FSTART, HEADER_SIZE + 2 * (count + 1))
    _w16(page, _O_FEND, new_end)
    _w16(page, _O_NSLOTS, count + 1)


def rebuild(page, cells):
    """Rewrite a page preserving its header but replacing all cells."""
    page_type = int(page[_O_TYPE])
    header = bytes(page[:HEADER_SIZE])
    page[:] = np.zeros(len(page), dtype=np.uint8)
    page[:HEADER_SIZE] = np.frombuffer(header, dtype=np.uint8)
    _w16(page, _O_NSLOTS, 0)
    _w16(page, _O_FSTART, HEADER_SIZE)
    _w16(page, _O_FEND, len(page))
    for key, value in cells:
        cell = leaf_cell(key, value) if page_type == LEAF else internal_cell(key, value)
        insert_cell(page, cell, key)


def page_keys(image_hex, page_type):
    raw = np.frombuffer(bytes.fromhex(image_hex), dtype=np.uint8)
    return [parse_cell(raw, off, page_type)[0] for off in slot_offsets(raw)]


def page_layout_segments(image_hex):
    """Byte-level segments of a page for layout rendering."""
    raw = np.frombuffer(bytes.fromhex(image_hex), dtype=np.uint8)
    page_type = int(raw[_O_TYPE])
    segments = [
        {"name": "type", "start": 0, "end": 1, "region": "header"},
        {"name": "page_id", "start": 1, "end": 3, "region": "header"},
        {"name": "parent", "start": 3, "end": 5, "region": "header"},
        {"name": "sibling", "start": 5, "end": 7, "region": "header"},
        {"name": "lsn", "start": 7, "end": 9, "region": "header"},
        {"name": "slot_count", "start": 9, "end": 11, "region": "header"},
        {"name": "free_start", "start": 11, "end": 13, "region": "header"},
        {"name": "free_end", "start": 13, "end": 15, "region": "header"},
        {"name": "flags", "start": 15, "end": 16, "region": "header"},
        {"name": "leftmost_child", "start": 16, "end": 18, "region": "header"},
        {"name": "reserved", "start": 18, "end": 20, "region": "header"},
    ]
    count = _u16(raw, _O_NSLOTS)
    if count:
        segments.append({
            "name": "slot_directory",
            "start": HEADER_SIZE,
            "end": HEADER_SIZE + 2 * count,
            "region": "slots",
        })
    for i, off in enumerate(slot_offsets(raw)):
        _, _, length = parse_cell(raw, off, page_type)
        segments.append({
            "name": "cell_%d" % i,
            "start": off,
            "end": off + length,
            "region": "cell",
        })
    used = sorted((s["start"], s["end"]) for s in segments if s["region"] == "cell")
    if used:
        first_cell = used[0][0]
        slot_end = HEADER_SIZE + 2 * count
        if slot_end < first_cell:
            segments.append({"name": "free_space", "start": slot_end,
                             "end": first_cell, "region": "free"})
    else:
        segments.append({"name": "free_space",
                         "start": HEADER_SIZE + 2 * count,
                         "end": len(raw), "region": "free"})
    return segments


class PageFull(Exception):
    pass


def _split_candidates(size_list, capacity):
    """All legal split indices decided purely by byte space."""
    total = len(size_list)
    legal = []
    for i in range(1, total):
        left_ok = sum(size_list[:i]) <= capacity
        right_ok = sum(size_list[i:]) <= capacity
        if left_ok and right_ok:
            legal.append(i)
    if not legal:
        raise PageFull("no legal byte-basis split point")
    chosen = min(legal, key=lambda i: abs(sum(size_list[:i]) - sum(size_list[i:])))
    return chosen, legal


class Engine:
    def __init__(self, page_size=256, duplicate_policy="allow_right",
                 empty_leaf_policy="unlink"):
        self.page_size = page_size
        self.duplicate_policy = duplicate_policy
        self.empty_leaf_policy = empty_leaf_policy
        self.disk = {}
        self.freed = set()
        self.log = []
        self.trace = []
        self.lsn = 0
        self.next_pid = 1
        self._cur_writes = None
        self.root = self._alloc(LEAF)
        pre = bytes(self.disk[self.root])
        self._write(self.root, "init_root", pre, allocated=self.root)
        self._emit_step("init", scenarios=[], extra={"root": self.root})

    def _alloc(self, page_type):
        page_id = self.next_pid
        self.next_pid += 1
        self.disk[page_id] = new_page(self.page_size, page_type, page_id)
        return page_id

    def _write(self, page_id, op, pre_image, allocated=None):
        page = self.disk[page_id]
        self.lsn += 1
        _w16(page, _O_LSN, self.lsn)
        post = bytes(page)
        record = {
            "lsn": self.lsn,
            "op": op,
            "page_id": page_id,
            "pre_image": pre_image.hex(),
            "post_image": post.hex(),
            "slot_directory": [
                {"slot": i, "offset": {"value": off, "basis": "page_offset"}}
                for i, off in enumerate(slot_offsets(page))
            ],
            "free_space": {"value": free_bytes(page), "unit": "byte"},
            "sibling_link": _u16(page, _O_SIBLING),
            "parent": _u16(page, _O_PARENT),
            "allocated_page_id": allocated,
            "torn": False,
        }
        self.log.append(record)
        if self._cur_writes is not None:
            self._cur_writes.append(record)
        return record

    def _emit_step(self, kind, scenarios, extra=None):
        first = self._cur_writes[0]["lsn"] if self._cur_writes else None
        last = self._cur_writes[-1]["lsn"] if self._cur_writes else None
        step = {
            "seq": len(self.trace),
            "kind": kind,
            "lsn_range": [first, last],
            "writes": list(self._cur_writes or []),
            "root": self.root,
            "freed": sorted(self.freed),
            "scenarios": list(scenarios),
        }
        if extra:
            step.update(extra)
        self.trace.append(step)
        return step

    def _descend(self, key):
        path = []
        page_id = self.root
        while True:
            page = self.disk[page_id]
            if int(page[_O_TYPE]) == LEAF:
                return path, page_id
            path.append(page_id)
            child = _u16(page, _O_LEFTMOST)
            for sep, child_id in read_cells(page):
                if key >= sep:
                    child = child_id
                else:
                    break
            page_id = child

    def _exists(self, key):
        _, leaf_id = self._descend(key)
        return any(k == key for k, _ in read_cells(self.disk[leaf_id]))

    # ---- inserts / splits -------------------------------------------------

    def insert(self, key, value):
        if isinstance(key, str):
            key = key.encode("utf-8")
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._cur_writes = []
        scenarios = ["variable_length_key"] if len(key) > 4 else []
        split_records = []
        if self.duplicate_policy == "reject" and self._exists(key):
            step = self._emit_step("insert", scenarios, {
                "logical_op": "insert", "key": key.decode("latin-1"),
                "rejected_duplicate": True,
            })
            self._cur_writes = None
            return step
        _, leaf_id = self._descend(key)
        page = self.disk[leaf_id]
        cell = leaf_cell(key, value)
        if free_bytes(page) >= len(cell) + 2:
            pre = bytes(page)
            insert_cell(page, cell, key)
            if any(k == key for k, _ in read_cells(page)[:-1]):
                scenarios.append("duplicate_key")
            self._write(leaf_id, "insert", pre)
        else:
            scenarios.append("leaf_split")
            if self._exists(key):
                scenarios.append("duplicate_key")
            split_records = self._split_leaf(leaf_id, key, value, self._descend(key)[0])
            tags = {s["type"] for s in split_records}
            if "root" in tags:
                scenarios.append("root_split")
            if "internal" in tags:
                scenarios.append("cascading_split")
        self._cur_writes = None
        return self._emit_step("insert", scenarios, {
            "logical_op": "insert",
            "key": key.decode("latin-1"),
            "value": value.decode("latin-1"),
            "splits": split_records,
        })

    def _split_leaf(self, leaf_id, key, value, path):
        page = self.disk[leaf_id]
        pre_left = bytes(page)
        cells = read_cells(page)
        pos = bisect_right([k for k, _ in cells], key)
        cells.insert(pos, (key, value))
        sizes = [len(leaf_cell(k, v)) + 2 for k, v in cells]
        chosen, candidates = _split_candidates(sizes, self.page_size - HEADER_SIZE)
        right_id = self._alloc(LEAF)
        right_page = self.disk[right_id]
        _w16(right_page, _O_PARENT, _u16(page, _O_PARENT))
        rebuild(page, cells[:chosen])
        rebuild(right_page, cells[chosen:])
        left_sib = _u16(page, _O_SIBLING)
        _w16(right_page, _O_SIBLING, left_sib)
        _w16(page, _O_SIBLING, right_id)
        record = {
            "type": "leaf", "left": leaf_id, "right": right_id,
            "separator": cells[chosen][0].decode("latin-1"),
            "separator_rule": "copy",
            "basis": "bytes",
            "chosen": chosen,
            "candidates": [
                {"index": i,
                 "left_bytes": {"value": sum(sizes[:i]), "unit": "byte"},
                 "right_bytes": {"value": sum(sizes[i:]), "unit": "byte"}}
                for i in candidates
            ],
            "undecided": len(candidates) > 1,
        }
        self._write(leaf_id, "split_leaf_left", pre_left)
        self._write(right_id, "split_leaf_right", bytes(HEADER_SIZE), allocated=right_id)
        records = [record]
        records += self._insert_into_parent(path, leaf_id, cells[chosen][0], right_id)
        return records

    def _insert_into_parent(self, path, left_id, sep_key, right_id):
        if not path:
            old_root = self.root
            new_root = self._alloc(INTERNAL)
            root_page = self.disk[new_root]
            _w16(root_page, _O_LEFTMOST, left_id)
            insert_cell(root_page, internal_cell(sep_key, right_id), sep_key,
                        right_of_equal=False)
            self.root = new_root
            for child in (left_id, right_id):
                pre = bytes(self.disk[child])
                _w16(self.disk[child], _O_PARENT, new_root)
                self._write(child, "set_parent", pre)
            record = {"type": "root", "new_root": new_root,
                      "old_root": old_root,
                      "separator": sep_key.decode("latin-1")}
            self._write(new_root, "new_root", bytes(HEADER_SIZE), allocated=new_root)
            return [record]
        parent_id = path[-1]
        page = self.disk[parent_id]
        cell = internal_cell(sep_key, right_id)
        if free_bytes(page) >= len(cell) + 2:
            pre = bytes(page)
            insert_cell(page, cell, sep_key, right_of_equal=False)
            self._write(parent_id, "insert_separator", pre)
            pre = bytes(self.disk[right_id])
            _w16(self.disk[right_id], _O_PARENT, parent_id)
            self._write(right_id, "set_parent", pre)
            return []
        return self._split_internal(parent_id, sep_key, right_id, path[:-1])

    def _split_internal(self, page_id, new_key, new_child, path):
        page = self.disk[page_id]
        pre_left = bytes(page)
        cells = read_cells(page)
        pos = bisect_left([k for k, _ in cells], new_key)
        cells.insert(pos, (new_key, new_child))
        sizes = [len(internal_cell(k, c)) + 2 for k, c in cells]
        chosen, candidates = _split_candidates(sizes, self.page_size - HEADER_SIZE)
        promoted_key, promoted_child = cells[chosen]
        right_id = self._alloc(INTERNAL)
        right_page = self.disk[right_id]
        _w16(right_page, _O_LEFTMOST, promoted_child)
        rebuild(page, cells[:chosen])
        rebuild(right_page, cells[chosen + 1:])
        _w16(right_page, _O_SIBLING, _u16(page, _O_SIBLING))
        _w16(page, _O_SIBLING, right_id)
        moved = [c for _, c in cells[chosen + 1:]] + [promoted_child]
        for child in moved:
            pre = bytes(self.disk[child])
            _w16(self.disk[child], _O_PARENT, right_id)
            self._write(child, "set_parent", pre)
        record = {
            "type": "internal", "left": page_id, "right": right_id,
            "promoted": promoted_key.decode("latin-1"),
            "separator_rule": "promote",
            "basis": "bytes",
            "chosen": chosen,
            "candidates": [
                {"index": i,
                 "left_bytes": {"value": sum(sizes[:i]), "unit": "byte"},
                 "right_bytes": {"value": sum(sizes[i:]), "unit": "byte"}}
                for i in candidates
            ],
            "undecided": len(candidates) > 1,
        }
        self._write(page_id, "split_internal_left", pre_left)
        self._write(right_id, "split_internal_right", bytes(HEADER_SIZE),
                    allocated=right_id)
        return [record] + self._insert_into_parent(path, page_id,
                                                    promoted_key, right_id)

    # ---- deletes ----------------------------------------------------------

    def delete(self, key):
        if isinstance(key, str):
            key = key.encode("utf-8")
        self._cur_writes = []
        scenarios = []
        path, leaf_id = self._descend(key)
        page = self.disk[leaf_id]
        cells = read_cells(page)
        idx = next((i for i, (k, _) in enumerate(cells) if k == key), None)
        if idx is None:
            self._cur_writes = None
            return self._emit_step("delete", scenarios, {
                "logical_op": "delete", "key": key.decode("latin-1"),
                "not_found": True,
            })
        pre = bytes(page)
        rebuild(page, [c for i, c in enumerate(cells) if i != idx])
        self._write(leaf_id, "delete", pre)
        emptied = False
        if not read_cells(page) and leaf_id != self.root and \
                self.empty_leaf_policy == "unlink":
            self._unlink_empty_leaf(leaf_id, path)
            scenarios.append("delete_empty_page")
            emptied = True
        self._cur_writes = None
        return self._emit_step("delete", scenarios, {
            "logical_op": "delete", "key": key.decode("latin-1"),
            "emptied_page": emptied,
        })

    def _unlink_empty_leaf(self, leaf_id, path):
        parent_id = path[-1]
        parent = self.disk[parent_id]
        pre_parent = bytes(parent)
        cells = read_cells(parent)
        leftmost = _u16(parent, _O_LEFTMOST)
        if leftmost == leaf_id:
            _w16(parent, _O_LEFTMOST, cells[0][1])
            rebuild(parent, cells[1:])
        else:
            idx = next(i for i, (_, c) in enumerate(cells) if c == leaf_id)
            rebuild(parent, cells[:idx] + cells[idx + 1:])
        self._write(parent_id, "unlink_empty_child", pre_parent)
        sib = _u16(self.disk[leaf_id], _O_SIBLING)
        for other_id, other in self.disk.items():
            if other_id != leaf_id and int(other[_O_TYPE]) == LEAF and \
                    _u16(other, _O_SIBLING) == leaf_id:
                pre = bytes(other)
                _w16(other, _O_SIBLING, sib)
                self._write(other_id, "relink_sibling", pre)
        self.freed.add(leaf_id)
        if int(self.disk[self.root][_O_TYPE]) == INTERNAL and \
                not read_cells(self.disk[self.root]):
            old_root = self.root
            self.root = _u16(self.disk[old_root], _O_LEFTMOST)
            self.freed.add(old_root)

    def delete_leaf_keys(self, anchor_key):
        """Delete every key currently stored in the leaf containing anchor."""
        if isinstance(anchor_key, str):
            anchor_key = anchor_key.encode("utf-8")
        _, leaf_id = self._descend(anchor_key)
        keys = [k.decode("latin-1") for k, _ in read_cells(self.disk[leaf_id])]
        steps = []
        for key in list(keys):
            steps.append(self.delete(key))
            _, still = self._descend(anchor_key)
            if still != leaf_id:
                break
        return steps

    # ---- crash / recovery -------------------------------------------------

    def crash(self, lsn=None):
        """Interrupt the page write of one log record (torn half page)."""
        self._cur_writes = []
        if lsn is None:
            candidates = [r["lsn"] for r in self.log if not r["op"].startswith("init")]
            lsn = candidates[len(candidates) // 2]
        record = next(r for r in self.log if r["lsn"] == lsn)
        record["torn"] = True
        page = self.disk[record["page_id"]]
        post = bytes.fromhex(record["post_image"])
        pre = bytes.fromhex(record["pre_image"])
        half = len(post) // 2
        page[:half] = np.frombuffer(post[:half], dtype=np.uint8)
        page[half:] = np.frombuffer(pre[half:], dtype=np.uint8)
        page[_O_FLAGS] |= FLAG_TORN
        self._cur_writes = None
        return self._emit_step("crash", ["torn_write"], {
            "logical_op": "crash", "torn_lsn": lsn,
            "torn_page": record["page_id"],
        })

    def recover(self):
        """Redo committed records, roll the torn record back; idempotent."""
        self._cur_writes = []
        for record in self.log:
            image_hex = record["pre_image"] if record.get("torn") \
                else record["post_image"]
            page = self.disk[record["page_id"]]
            page[:] = np.frombuffer(bytes.fromhex(image_hex), dtype=np.uint8)
            page[_O_FLAGS] = int(page[_O_FLAGS]) & ~FLAG_TORN
            self._cur_writes.append(record)
        self._cur_writes = None
        self._emit_step("recovery", ["recovery_replay"], {
            "logical_op": "recover",
            "state_hash": self.state_hash(),
        })
        return self.state_hash()

    def reachable(self):
        seen = set()
        stack = [self.root]
        while stack:
            page_id = stack.pop()
            if page_id in seen or page_id in self.freed or page_id not in self.disk:
                continue
            seen.add(page_id)
            page = self.disk[page_id]
            if int(page[_O_TYPE]) == INTERNAL:
                stack.append(_u16(page, _O_LEFTMOST))
                stack.extend(c for _, c in read_cells(page))
            else:
                sibling = _u16(page, _O_SIBLING)
                if sibling:
                    stack.append(sibling)
        return sorted(seen)

    def state_hash(self):
        digest = hashlib.sha256()
        digest.update(b"root=%d;" % self.root)
        digest.update(b"freed=%s;" % ",".join(map(str, sorted(self.freed))).encode())
        for page_id in sorted(self.disk):
            digest.update(b"p%d:" % page_id)
            digest.update(bytes(self.disk[page_id]))
        return digest.hexdigest()

    def tree_snapshot(self):
        nodes, edges = [], []

        def walk(page_id, depth, index):
            page = self.disk[page_id]
            page_type = int(page[_O_TYPE])
            node = {
                "id": "pg:%d" % page_id,
                "page_id": page_id,
                "depth": depth,
                "kind": "leaf" if page_type == LEAF else "internal",
                "sibling": _u16(page, _O_SIBLING),
                "lsn": _u16(page, _O_LSN),
                "free_space": {"value": free_bytes(page), "unit": "byte"},
                "slot_count": _u16(page, _O_NSLOTS),
                "torn": bool(int(page[_O_FLAGS]) & FLAG_TORN),
                "freed": page_id in self.freed,
                "keys": [k.decode("latin-1") for k, _ in read_cells(page)],
            }
            nodes.append(node)
            if page_type == INTERNAL:
                children = [_u16(page, _O_LEFTMOST)] + \
                    [c for _, c in read_cells(page)]
                for child in children:
                    edges.append({"from": "pg:%d" % page_id,
                                  "to": "pg:%d" % child})
                    walk(child, depth + 1, index)

        walk(self.root, 0, 0)
        return {"root": "pg:%d" % self.root, "nodes": nodes, "edges": edges,
                "reachable": ["pg:%d" % p for p in self.reachable()]}

    def run_script(self, ops):
        for spec in ops:
            op = spec["op"]
            if op == "insert":
                self.insert(spec["key"], spec.get("value", ""))
            elif op == "delete":
                self.delete(spec["key"])
            elif op == "delete_leaf_keys":
                self.delete_leaf_keys(spec["anchor"])
            elif op == "crash":
                self.crash(spec.get("lsn"))
            elif op == "recover":
                hashes = [self.recover() for _ in range(int(spec.get("times", 1)))]
                if len(hashes) > 1:
                    self.trace[-1]["replay_hashes"] = hashes
                    self.trace[-1]["idempotent"] = len(set(hashes)) == 1
        return self.trace
