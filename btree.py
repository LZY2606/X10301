"""Slotted-page B-tree engine.

Fixed page size, slot directory, free space, sibling links, page-id
allocation and monotonically increasing LSN.  Split points are chosen by
*byte space* (via NumPy), separators are copied up from leaves and pushed
up from internal pages.  A physical write-ahead log supports crash
simulation (torn / partial page writes) and idempotent redo recovery.
"""
from __future__ import annotations

import numpy as np

PAGE_SIZE = 128
HEADER = 16
NULL = 0xFFFF
F_LEAF = 1
F_ROOT = 2
LEAF_ESZ = 4
INT_ESZ = 6


def _u16(buf, off):
    return (buf[off] << 8) | buf[off + 1]


def _put_u16(buf, off, val):
    buf[off] = (val >> 8) & 0xFF
    buf[off + 1] = val & 0xFF


def _put_u32(buf, off, val):
    for i in range(4):
        buf[off + i] = (val >> (24 - 8 * i)) & 0xFF


def _checksum(buf):
    return sum(buf[4:]) & 0xFF


def page_ok(buf):
    """A reachable page must parse and match its byte checksum (no half page)."""
    if buf is None or len(buf) < HEADER:
        return False
    if buf[3] != _checksum(buf):
        return False
    fs, fe = _u16(buf, 6), _u16(buf, 8)
    return HEADER <= fs <= fe <= len(buf)


def split_candidates(lengths, page_size=PAGE_SIZE, esz=LEAF_ESZ):
    """Every cut index i (1..n-1) where both halves fit in one page.

    Feasibility is computed from consumed *bytes* (key data + slots),
    never from key count.
    """
    if len(lengths) < 2:
        return []
    arr = np.asarray(lengths, dtype=np.int64)
    total = int(arr.sum())
    cumsum = np.cumsum(arr)
    cap = page_size - HEADER
    n = len(lengths)
    out = []
    for i in range(1, n):
        left = int(cumsum[i - 1])
        right = total - left
        if left + i * esz <= cap and right + (n - i) * esz <= cap:
            out.append(i)
    return out


def choose_split(lengths, page_size=PAGE_SIZE, esz=LEAF_ESZ):
    """Byte-balanced cut: minimise |left_bytes - right_bytes|."""
    cands = split_candidates(lengths, page_size, esz)
    if not cands:
        return max(1, len(lengths) // 2), []
    total = sum(lengths)
    best = min(cands, key=lambda i: (abs(2 * sum(lengths[:i]) - total), i))
    return best, cands


def build_page(page_size, pid, flags, link, keys, children=None, lsn=0):
    """Rebuild a compacted page image from ordered keys.

    Leaf slot:  (offset u16, length u16).
    Internal slot: (offset u16, length u16, right-child u16); the left-most
    child rides in the link field.
    """
    buf = bytearray(page_size)
    _put_u16(buf, 0, pid)
    buf[2] = flags
    _put_u16(buf, 10, link)
    _put_u32(buf, 12, lsn)
    esz = LEAF_ESZ if flags & F_LEAF else INT_ESZ
    data = HEADER
    for i, key in enumerate(keys):
        buf[data:data + len(key)] = key
        slot = page_size - (i + 1) * esz
        _put_u16(buf, slot, data)
        _put_u16(buf, slot + 2, len(key))
        if children is not None:
            _put_u16(buf, slot + 4, children[i])
        data += len(key)
    _put_u16(buf, 4, len(keys))
    _put_u16(buf, 6, data)
    _put_u16(buf, 8, page_size - len(keys) * esz)
    buf[3] = _checksum(buf)
    return bytes(buf)


class PageView:
    def __init__(self, buf):
        self.buf = buf
        self.page_size = len(buf)

    @property
    def pid(self):
        return _u16(self.buf, 0)

    @property
    def flags(self):
        return self.buf[2]

    @property
    def is_leaf(self):
        return bool(self.flags & F_LEAF)

    @property
    def is_root(self):
        return bool(self.flags & F_ROOT)

    @property
    def nslots(self):
        return _u16(self.buf, 4)

    @property
    def free_start(self):
        return _u16(self.buf, 6)

    @property
    def free_end(self):
        return _u16(self.buf, 8)

    @property
    def link(self):
        return _u16(self.buf, 10)

    @property
    def lsn(self):
        return int.from_bytes(self.buf[12:16], "big")

    @property
    def esz(self):
        return LEAF_ESZ if self.is_leaf else INT_ESZ

    def free_space(self):
        return self.free_end - self.free_start

    def slot_off(self, i):
        return self.page_size - (i + 1) * self.esz

    def key(self, i):
        o = self.slot_off(i)
        off, ln = _u16(self.buf, o), _u16(self.buf, o + 2)
        return bytes(self.buf[off:off + ln])

    def child(self, i):
        return _u16(self.buf, self.slot_off(i) + 4)

    def keys(self):
        return [self.key(i) for i in range(self.nslots)]

    def children(self):
        return [self.link] + [self.child(i) for i in range(self.nslots)]


class Engine:
    def __init__(self, page_size=PAGE_SIZE, dup_policy="reject"):
        self.page_size = page_size
        self.dup_policy = dup_policy
        self.disk = {}                 # page_id -> committed bytes
        self.log = []                  # physical after-image log records
        self.op_log = []               # logical operation events
        self.alloc_log = []            # page id allocation / release events
        self.crash_log = []
        self.lsn = 0
        self.durable_lsn = 0
        self.root_id = None
        self.next_pid = 1
        self.free_pids = []
        self.pending = []              # staged writes: (pid, image|None, lsn)

    # ---------- storage / WAL ----------
    def _alloc(self):
        if self.free_pids:
            pid = self.free_pids.pop()
        else:
            pid = self.next_pid
            self.next_pid += 1
        self.alloc_log.append({"event": "alloc", "page_id": pid, "lsn": self.lsn})
        return pid

    def _delete_page(self, pid):
        self.lsn += 1
        self.log.append({"lsn": self.lsn, "op": "free", "page_id": pid, "image": None})
        self.pending.append((pid, None, self.lsn))
        self.free_pids.append(pid)
        self.alloc_log.append({"event": "free", "page_id": pid, "lsn": self.lsn})

    def _write(self, pid, buf):
        self.lsn += 1
        buf = bytearray(buf)
        _put_u32(buf, 12, self.lsn)
        buf[3] = _checksum(buf)
        img = bytes(buf)
        self.log.append({"lsn": self.lsn, "op": "write", "page_id": pid, "image": img})
        self.pending.append((pid, img, self.lsn))
        return img

    def _load(self, pid):
        for p, img, _ in reversed(self.pending):
            if p == pid and img is not None:
                return img
        return self.disk[pid]

    def commit(self):
        for pid, img, _ in self.pending:
            if img is None:
                self.disk.pop(pid, None)
            else:
                self.disk[pid] = img
        self.pending = []
        self.durable_lsn = self.lsn

    def crash_after(self, n, torn=False):
        """Interrupt the current multi-page write after n page writes.

        ``torn`` truncates the last flushed page (a half-written page); the
        rest of the write set is simply lost like a cache that was never
        flushed.
        """
        applied, lost = self.pending[:n], self.pending[n:]
        torn_pid = None
        for idx, (pid, img, _) in enumerate(applied):
            if img is None:
                self.disk.pop(pid, None)
            else:
                if torn and idx == len(applied) - 1:
                    half = len(img) // 2
                    img = img[:half] + bytes(len(img) - half)
                    torn_pid = pid
                self.disk[pid] = img
        self.durable_lsn = applied[-1][2] if applied else self.durable_lsn
        rec = {
            "at_lsn": self.lsn,
            "durable_lsn": self.durable_lsn,
            "applied": [p for p, _, _ in applied],
            "lost": [p for p, _, _ in lost],
            "torn_page": torn_pid,
        }
        self.crash_log.append(rec)
        self.pending = []
        return rec

    def recover(self):
        """Redo log replay; safe to run repeatedly (idempotent)."""
        for rec in self.log:
            cur = self.disk.get(rec["page_id"])
            if rec["image"] is None:
                if rec["lsn"] > self.durable_lsn and rec["page_id"] in self.disk:
                    self.disk.pop(rec["page_id"], None)
                continue
            if cur is None or not page_ok(cur) or rec["lsn"] > self.durable_lsn:
                self.disk[rec["page_id"]] = rec["image"]
        self.durable_lsn = self.lsn
        self.pending = []

    # ---------- traversal ----------
    def _descend(self, key):
        path, pid = [], self.root_id
        while True:
            path.append(pid)
            pv = PageView(self._load(pid))
            if pv.is_leaf:
                return path
            pid = self._child_for(pv, key)

    @staticmethod
    def _child_for(pv, key):
        for i in range(pv.nslots):
            if key < pv.key(i):
                return pv.link if i == 0 else pv.child(i - 1)
        return pv.child(pv.nslots - 1) if pv.nslots else pv.link

    def search(self, key):
        if self.root_id is None:
            return False
        path = self._descend(key)
        return key in PageView(self._load(path[-1])).keys()

    def count_key(self, key):
        if self.root_id is None:
            return 0
        path = self._descend(key)
        return PageView(self._load(path[-1])).keys().count(key)

    def leaves(self):
        if self.root_id is None:
            return []
        pid = self.root_id
        while not PageView(self._load(pid)).is_leaf:
            pid = PageView(self._load(pid)).link
        out = []
        while pid != NULL and pid in self.disk:
            out.append(pid)
            pid = PageView(self.disk[pid]).link
        return out

    def height(self):
        if self.root_id is None:
            return 0
        h, pid = 0, self.root_id
        while True:
            h += 1
            pv = PageView(self._load(pid))
            if pv.is_leaf:
                return h
            pid = pv.link

    def reachable(self):
        """BFS over child pointers and the leaf sibling chain (disk state)."""
        if self.root_id is None:
            return [], []
        if self.root_id not in self.disk:
            return [], ["root page missing"]
        seen, problems, stack, visited = [], [], [self.root_id], set()
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
            img = self.disk.get(pid)
            if img is None:
                problems.append("page %d referenced but missing" % pid)
                continue
            if not page_ok(img):
                problems.append("page %d corrupt or half-written" % pid)
                continue
            seen.append(pid)
            pv = PageView(img)
            if pv.is_leaf:
                if pv.link != NULL:
                    stack.append(pv.link)
            else:
                stack.extend(c for c in pv.children() if c != NULL)
        return sorted(seen), problems

    def verify(self):
        seen, problems = self.reachable()
        for pid in seen:
            keys = PageView(self.disk[pid]).keys()
            if keys != sorted(keys):
                problems.append("page %d keys out of order" % pid)
        return problems

    def snapshot(self):
        return {str(pid): img.hex() for pid, img in sorted(self.disk.items())}

    def log_digest(self):
        return [{"lsn": r["lsn"], "op": r["op"], "page_id": r["page_id"]} for r in self.log]

    # ---------- mutations ----------
    def insert(self, key, commit=True):
        ev = {"op": "insert", "key": key.hex(), "splits": []}
        if self.root_id is None:
            self.root_id = self._alloc()
            img = build_page(self.page_size, self.root_id, F_LEAF | F_ROOT, NULL, [key])
            self._write(self.root_id, img)
            ev["root_created"] = self.root_id
            if commit:
                self.commit()
            self.op_log.append(ev)
            return ev
        path = self._descend(key)
        leaf_pid = path[-1]
        pv = PageView(self._load(leaf_pid))
        keys = pv.keys()
        if key in keys and self.dup_policy == "reject":
            ev["result"] = "dup_rejected"
            self.op_log.append(ev)
            return ev
        pos = 0
        while pos < len(keys) and keys[pos] <= key:
            pos += 1
        new_keys = keys[:pos] + [key] + keys[pos:]
        if pv.free_space() >= len(key) + LEAF_ESZ:
            self._write(leaf_pid,
                        build_page(self.page_size, leaf_pid, pv.flags, pv.link, new_keys))
        else:
            self._split_leaf(path, new_keys, ev)
        if commit:
            self.commit()
        self.op_log.append(ev)
        return ev

    def _split_leaf(self, path, new_keys, ev):
        lengths = [len(k) for k in new_keys]
        idx, cands = choose_split(lengths, self.page_size, LEAF_ESZ)
        left_keys, right_keys = new_keys[:idx], new_keys[idx:]
        leaf_pid = path[-1]
        old = PageView(self._load(leaf_pid))
        right_pid = self._alloc()
        self._write(right_pid,
                    build_page(self.page_size, right_pid, F_LEAF, old.link, right_keys))
        self._write(leaf_pid,
                    build_page(self.page_size, leaf_pid, old.flags, right_pid, left_keys))
        sep = right_keys[0]
        ev["splits"].append({
            "kind": "leaf", "left": leaf_pid, "right": right_pid,
            "separator": sep.hex(), "separator_rule": "copy-up",
            "left_count": len(left_keys), "right_count": len(right_keys),
            "candidates": cands,
        })
        self._propagate(path, leaf_pid, sep, right_pid, ev)

    def _rebuild(self, pid, pv, flags=None, link=None, keys=None, children=None):
        self._write(
            pid,
            build_page(
                self.page_size, pid,
                pv.flags if flags is None else flags,
                pv.link if link is None else link,
                pv.keys() if keys is None else keys,
                None if pv.is_leaf else (
                    [pv.child(i) for i in range(pv.nslots)] if children is None else children),
            ),
        )

    def _clear_root_flag(self, pid):
        pv = PageView(self._load(pid))
        if not pv.is_root:
            return
        if pv.is_leaf:
            self._rebuild(pid, pv, flags=pv.flags & ~F_ROOT)
        else:
            self._rebuild(pid, pv, flags=pv.flags & ~F_ROOT,
                          children=[pv.child(i) for i in range(pv.nslots)])

    def _propagate(self, path, left_pid, sep, right_pid, ev, rule="copy-up"):
        if len(path) == 1:
            new_root = self._alloc()
            self._clear_root_flag(left_pid)
            self._write(new_root,
                        build_page(self.page_size, new_root, F_ROOT, left_pid, [sep], [right_pid]))
            self.root_id = new_root
            ev["splits"].append({
                "kind": "root", "left": left_pid, "right": right_pid,
                "new_root": new_root, "separator": sep.hex(),
                "separator_rule": rule,
            })
            return
        parent_pid = path[-2]
        pv = PageView(self._load(parent_pid))
        pkeys = pv.keys()
        pchildren = [pv.child(i) for i in range(pv.nslots)]
        j = ([pv.link] + pchildren).index(left_pid)
        new_keys = pkeys[:j] + [sep] + pkeys[j:]
        new_children = pchildren[:j] + [right_pid] + pchildren[j:]
        if pv.free_space() >= len(sep) + INT_ESZ:
            self._write(parent_pid,
                        build_page(self.page_size, parent_pid, pv.flags, pv.link,
                                   new_keys, new_children))
            return
        lengths = [len(k) for k in new_keys]
        idx, cands = choose_split(lengths, self.page_size, INT_ESZ)
        promote = new_keys[idx]
        right_link = new_children[idx]
        r_pid = self._alloc()
        self._write(r_pid,
                    build_page(self.page_size, r_pid, 0, right_link,
                               new_keys[idx + 1:], new_children[idx + 1:]))
        self._write(parent_pid,
                    build_page(self.page_size, parent_pid, pv.flags, pv.link,
                               new_keys[:idx], new_children[:idx]))
        ev["splits"].append({
            "kind": "internal", "left": parent_pid, "right": r_pid,
            "separator": promote.hex(), "separator_rule": "push-up",
            "candidates": cands,
        })
        self._propagate(path[:-1], parent_pid, promote, r_pid, ev, rule="push-up")

    @staticmethod
    def _leaf_pids(ev):
        return {s["left"] for s in ev["splits"] if s["kind"] == "leaf"}

    def delete(self, key, commit=True):
        ev = {"op": "delete", "key": key.hex()}
        if self.root_id is None:
            ev["result"] = "empty_tree"
            self.op_log.append(ev)
            return ev
        path = self._descend(key)
        leaf_pid = path[-1]
        pv = PageView(self._load(leaf_pid))
        keys = pv.keys()
        if key not in keys:
            ev["result"] = "not_found"
            self.op_log.append(ev)
            return ev
        keys.remove(key)
        self._rebuild(leaf_pid, pv, keys=keys)
        if not keys and not pv.is_root:
            left_sib = self._find_left_sibling(leaf_pid)
            if left_sib is not None:
                sv = PageView(self._load(left_sib))
                self._rebuild(left_sib, sv, link=pv.link)
            self._remove_child(path, leaf_pid, ev)
            self._delete_page(leaf_pid)
            ev["emptied_page"] = leaf_pid
        if commit:
            self.commit()
        self.op_log.append(ev)
        return ev

    def _find_left_sibling(self, pid):
        live = set(self.disk) | {p for p, img, _ in self.pending if img is not None}
        for other in live:
            img = self._load(other)
            pv = PageView(img)
            if pv.is_leaf and pv.link == pid:
                return other
        return None

    def _remove_child(self, path, dead_pid, ev):
        parent_pid = path[-2]
        pv = PageView(self._load(parent_pid))
        pkeys = pv.keys()
        pchildren = [pv.child(i) for i in range(pv.nslots)]
        seq = [pv.link] + pchildren
        j = seq.index(dead_pid)
        if j == 0:
            new_link = pchildren[0] if pchildren else NULL
            new_keys = pkeys[1:]
            new_children = pchildren[1:]
        else:
            new_link = pv.link
            new_keys = pkeys[:j - 1] + pkeys[j:]
            new_children = pchildren[:j - 1] + pchildren[j:]
        self._write(parent_pid,
                    build_page(self.page_size, parent_pid, pv.flags, new_link,
                               new_keys, new_children))
        if not new_keys:
            if new_link == NULL:
                self.root_id = None
                self._delete_page(parent_pid)
                ev["root_shrunk"] = None
            elif len(path) == 2:
                cv = PageView(self._load(new_link))
                if cv.is_leaf:
                    self._rebuild(new_link, cv, flags=cv.flags | F_ROOT)
                else:
                    self._rebuild(new_link, cv, flags=cv.flags | F_ROOT,
                                  children=[cv.child(i) for i in range(cv.nslots)])
                self.root_id = new_link
                self._delete_page(parent_pid)
                ev["root_shrunk"] = new_link
            else:
                self._remove_child(path[:-1], parent_pid, ev)
                self._delete_page(parent_pid)
