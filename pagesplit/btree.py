"""固定页大小的 B+tree 领域内核。

页布局（leaf/internal 通用，小端序）：
  页头 22 字节：magic(2) kind(1) page_id(2) parent(2) prev(2) next(2)
                 cell_count(2) free_space(2) checksum(4) torn(1) + pad(2)
  slot directory：自页头之后向下增长，每槽 4B = cell_offset(2)+cell_len(2)，按 key 升序
  cell 区：自页尾向上增长
leaf cell: keylen(2) vlen(2) key bytes value bytes
internal cell: flags(1) keylen(2) child(2) key bytes   （flags&1 => promote 后不重复存储）
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

from .txn import Txn
from typing import Any

import numpy as np

MAGIC = 0x5047  # "PG"
LEAF = 1
INTERNAL = 2
HEADER_SIZE = 22
SLOT_SIZE = 4
NONE = 0xFFFF
TORN_MARK = 0xA5

COPY = "copy"
PROMOTE = "promote"


class PageOverflow(Exception):
    pass


def crc(body: bytearray) -> int:
    """checksum 覆盖头(除自身 4B)、slot directory 与 cell 区。"""
    head = bytearray(body[:14]) + bytearray(body[18:HEADER_SIZE])
    return zlib.crc32(bytes(head + body[HEADER_SIZE:])) & 0xFFFFFFFF


@dataclass
class LeafCell:
    key: bytes
    value: bytes

    def size(self) -> int:
        return 4 + len(self.key) + len(self.value)


@dataclass
class InternalCell:
    key: bytes
    child: int
    promoted: bool = False  # True => key 已提升，不在子 leaf 中重复存储（promote 规则）
    value: bytes = b""      # 提升键自带负载（classic B-tree 语义）

    def size(self) -> int:
        return 7 + len(self.key) + (len(self.value) if self.promoted else 0)


@dataclass
class Page:
    page_id: int
    kind: int
    parent: int = NONE
    prev: int = NONE
    next: int = NONE
    leaf_cells: list[LeafCell] = field(default_factory=list)
    first_child: int = NONE
    internal_cells: list[InternalCell] = field(default_factory=list)
    torn: bool = False
    lsn: int = 0  # 最近一次写该页的日志序号

    def cells(self) -> list[LeafCell] | list[InternalCell]:
        return self.leaf_cells if self.kind == LEAF else self.internal_cells

    def used_bytes(self, page_size: int) -> int:
        body = self.encode(page_size)
        return page_size - self.free_space(page_size)

    def free_space(self, page_size: int) -> int:
        n = len(self.cells())
        top = page_size
        for c in self.cells():
            top -= c.size()
        return top - (HEADER_SIZE + n * SLOT_SIZE)

    def encode(self, page_size: int, *, allow_overflow: bool = False) -> bytes:
        cells = self.cells()
        n = len(cells)
        free = page_size - (HEADER_SIZE + n * SLOT_SIZE) - sum(c.size() for c in cells)
        if free < 0 and not allow_overflow:
            raise PageOverflow(f"page {self.page_id} overflow by {-free} bytes")
        body = bytearray(page_size)
        struct.pack_into(
            "<HHHHHHHH", body, 0,
            MAGIC, self.kind, self.page_id, self.parent, self.prev, self.next,
            n, max(free, 0),
        )
        # checksum 在 14..18
        struct.pack_into("<I", body, 14, 0)
        body[18] = TORN_MARK if self.torn else 0
        top = page_size
        for idx, c in enumerate(cells):
            if self.kind == LEAF:
                payload = struct.pack("<HH", len(c.key), len(c.value)) + c.key + c.value
            else:
                flags = (1 if c.promoted else 0)
                payload = struct.pack("<BHH", flags, len(c.key), c.child) + c.key
                if c.promoted:
                    payload += struct.pack("<H", len(c.value)) + c.value
            start = top - len(payload)
            body[start:top] = payload
            struct.pack_into("<HH", body, HEADER_SIZE + idx * SLOT_SIZE, start, len(payload))
            top = start
        checksum = crc(body)
        struct.pack_into("<I", body, 14, checksum)
        return bytes(body)

    def layout(self, page_size: int) -> dict[str, Any]:
        """页内字节布局，每个区域带稳定身份与字节坐标。"""
        cells = self.cells()
        n = len(cells)
        regions = [
            {"id": f"page{self.page_id}.header", "kind": "header",
             "start": 0, "end": HEADER_SIZE, "bytes": HEADER_SIZE},
        ]
        cursor = page_size
        entries: list[dict[str, Any]] = []
        for idx, c in enumerate(cells):
            size = c.size()
            start = cursor - size
            if self.kind == LEAF:
                cid = f"page{self.page_id}.cell.slot{idx}.key{c.key.hex()}"
                key = c.key
            else:
                tag = "p" if c.promoted else "k"
                cid = f"page{self.page_id}.cell.slot{idx}.{tag}{c.key.hex()}.child{c.child}"
                key = c.key
            entries.append({
                "id": cid, "slot": idx, "start": start, "end": cursor,
                "bytes": size, "key": _show(key),
            })
            cursor = start
        entries.sort(key=lambda e: e["slot"])
        regions.extend(entries)
        slot_end = HEADER_SIZE + n * SLOT_SIZE
        regions.append({
            "id": f"page{self.page_id}.slots", "kind": "slot_directory",
            "start": HEADER_SIZE, "end": slot_end, "bytes": n * SLOT_SIZE,
            "slot_count": n, "slot_size_bytes": SLOT_SIZE,
        })
        free_start = slot_end
        free_end = cursor
        regions.append({
            "id": f"page{self.page_id}.free", "kind": "free_space",
            "start": free_start, "end": free_end, "bytes": max(0, free_end - free_start),
            "unit": "byte",
        })
        regions.sort(key=lambda r: r["start"])
        return {
            "page_id": self.page_id,
            "kind": "leaf" if self.kind == LEAF else "internal",
            "page_size_bytes": page_size,
            "free_space_bytes": self.free_space(page_size),
            "torn": self.torn,
            "lsn": self.lsn,
            "regions": regions,
        }


def _show(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "0x" + b.hex()


@dataclass
class Params:
    page_size: int = 256
    separator_policy: str = COPY  # copy | promote，必须在 leaf/internal 间保持一致
    duplicate_policy: str = "overwrite"  # overwrite | reject
    empty_leaf: str = "retain"  # retain | remove
    min_fill_fraction: float = 0.25
    raw: dict = field(default_factory=dict)  # 保留导入时的原始表示（含单位）

    def rule_version(self) -> str:
        return (f"psize={self.page_size};sep={self.separator_policy};dup={self.duplicate_policy};"
                f"empty={self.empty_leaf};minfill={self.min_fill_fraction}")


def leaf_split_candidates(left_cells: list[LeafCell], page_size: int,
                          min_fill: float) -> list[dict[str, Any]]:
    """变长键：候选合法性按两侧字节空间判定，而不是键的个数。"""
    sizes = np.array([c.size() for c in left_cells], dtype=np.int64)
    usable = page_size - HEADER_SIZE - SLOT_SIZE
    out: list[dict[str, Any]] = []
    for i in range(1, len(left_cells)):
        l_bytes = int(sizes[:i].sum()) + i * SLOT_SIZE
        r_bytes = int(sizes[i:].sum()) + (len(sizes) - i) * SLOT_SIZE
        l_free, r_free = usable - l_bytes, usable - r_bytes
        feasible = l_free >= 0 and r_free >= 0
        enough = (min(l_bytes, r_bytes) >= min_fill * usable)
        out.append({
            "split_index": i,
            "left_bytes": l_bytes,
            "right_bytes": r_bytes,
            "left_free_bytes": l_free,
            "right_free_bytes": r_free,
            "balanced_penalty": abs(l_bytes - r_bytes),
            "feasible": feasible,
            "meets_min_fill": bool(enough),
            "legal": feasible and enough,
        })
    legals = [c for c in out if c["legal"]]
    chosen = min(legals, key=lambda c: (c["balanced_penalty"], c["split_index"])) if legals else None
    for c in out:
        c["chosen"] = chosen is not None and c["split_index"] == chosen["split_index"]
    return out


def decode_page(data: bytes) -> tuple[Page | None, list[str]]:
    """从字节读回页；checksum 不符 -> 半页（torn）。"""
    diagnostics: list[str] = []
    if len(data) < HEADER_SIZE:
        return None, ["short page"]
    body = bytearray(data)
    magic, kind, pid, parent, prev, nxt, count, free = struct.unpack_from("<HHHHHHHH", body, 0)
    (stored_crc,) = struct.unpack_from("<I", body, 14)
    torn_flag = body[18] == TORN_MARK
    ok_crc = crc(body) == stored_crc
    if magic != MAGIC:
        return None, [f"bad magic 0x{magic:04x}"]
    if kind not in (LEAF, INTERNAL):
        return None, [f"bad kind {kind}"]
    page = Page(page_id=pid, kind=kind, parent=parent, prev=prev, next=nxt,
                torn=torn_flag or not ok_crc)
    if not ok_crc:
        diagnostics.append(f"page {pid}: checksum mismatch (half-written page)")
    for idx in range(count):
        off, length = struct.unpack_from("<HH", body, HEADER_SIZE + idx * SLOT_SIZE)
        chunk = body[off:off + length]
        if kind == LEAF:
            if len(chunk) < 4:
                diagnostics.append(f"page {pid} slot {idx}: truncated cell")
                continue
            kl, vl = struct.unpack_from("<HH", chunk, 0)
            page.leaf_cells.append(LeafCell(bytes(chunk[4:4 + kl]), bytes(chunk[4 + kl:4 + kl + vl])))
        else:
            if len(chunk) < 5:
                diagnostics.append(f"page {pid} slot {idx}: truncated cell")
                continue
            flags, kl, child = struct.unpack_from("<BHH", chunk, 0)
            k = bytes(chunk[5:5 + kl])
            val = b""
            if flags & 1:
                base = 5 + kl
                (vl,) = struct.unpack_from("<H", chunk, base)
                val = bytes(chunk[base + 2:base + 2 + vl])
            page.internal_cells.append(InternalCell(k, child, promoted=bool(flags & 1), value=val))
    return page, diagnostics


@dataclass
class Tree:
    params: Params = field(default_factory=Params)
    pages: dict[int, Page] = field(default_factory=dict)
    root: int = NONE
    next_pid: int = 1  # page id 分配序列，单调递增、永不复用
    lsn: int = 0

    def new_pid(self) -> int:
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def _log(self, txn: "Txn", rec_type: str, **fields: Any) -> dict[str, Any]:
        self.lsn += 1
        rec = {"lsn": self.lsn, "type": rec_type, **fields}
        txn.records.append(rec)
        return rec

    def _write(self, txn: "Txn", page: Page, *, tag: str, action: str = "write") -> dict[str, Any]:
        rec = self._log(txn, "PAGE_WRITE", action=action, tag=tag, page_id=page.page_id,
                        parent=page.parent,
                        free_space_bytes=page.free_space(self.params.page_size))
        page.lsn = rec["lsn"]
        self.pages[page.page_id] = page
        return rec

    def snapshot_pages(self) -> dict[int, bytes]:
        return {pid: p.encode(self.params.page_size) for pid, p in self.pages.items()}

    def reachable_pids(self) -> list[int]:
        if self.root == NONE:
            return []
        seen: set[int] = set()
        stack = [self.root]
        while stack:
            pid = stack.pop()
            if pid in seen or pid not in self.pages:
                continue
            seen.add(pid)
            p = self.pages[pid]
            if p.kind == INTERNAL:
                if p.first_child != NONE:
                    stack.append(p.first_child)
                stack.extend(c.child for c in p.internal_cells)
        return sorted(seen)

    def get(self, key: bytes) -> bytes | None:
        if self.root == NONE:
            return None
        pid = self.root
        while self.pages[pid].kind == INTERNAL:
            p = self.pages[pid]
            keys = [c.key for c in p.internal_cells]
            if self.params.separator_policy == PROMOTE:
                i = int(np.searchsorted(keys, key, side="left"))
                pid = p.first_child if i == 0 else p.internal_cells[i - 1].child
            else:
                i = int(np.searchsorted(keys, key, side="right"))
                pid = p.first_child if i == 0 else p.internal_cells[i - 1].child
        leaf = self.pages[pid]
        for c in leaf.leaf_cells:
            if c.key == key:
                return c.value
        return None

    def leaf_for_insert(self, key: bytes) -> tuple[int, list[int]]:
        """返回目标 leaf 与内部路径（root->...->leaf 的父链）。"""
        path: list[int] = []
        pid = self.root
        while self.pages[pid].kind == INTERNAL:
            path.append(pid)
            p = self.pages[pid]
            keys = [c.key for c in p.internal_cells]
            i = int(np.searchsorted(keys, key, side="left")) \
                if self.params.separator_policy == PROMOTE \
                else int(np.searchsorted(keys, key, side="right"))
            pid = p.first_child if i == 0 else p.internal_cells[i - 1].child
        return pid, path

    # ------------------------------------------------------------------ 逻辑操作
    def apply_op(self, op_index: int, op: dict) -> Txn:
        txn = Txn(op_index=op_index, op=op, before=self.snapshot_pages())
        kind = op["op"]
        self._log(txn, "OP_BEGIN", seq=op_index, op=kind,
                  key=_show(op["key"]) if "key" in op else None)
        if kind == "insert":
            self._insert(txn, op["key"], op.get("value", b""))
        elif kind == "delete":
            self._delete(txn, op["key"])
        elif kind == "nop":
            pass
        else:
            txn.diagnostics.append(f"unknown op {kind!r}")
        self._log(txn, "OP_COMMIT", seq=op_index, aborted=txn.aborted,
                  root=self.root, next_pid=self.next_pid,
                  diagnostics=list(txn.diagnostics))
        return txn

    def _insert(self, txn: Txn, key: bytes, value: bytes) -> None:
        if self.root == NONE:
            pid = self.new_pid()
            self._log(txn, "PID_ALLOC", page_id=pid, reason="first-root")
            leaf = Page(pid, LEAF)
            leaf.leaf_cells.append(LeafCell(key, value))
            self._write(txn, leaf, tag="root-create")
            self.root = pid
            self._log(txn, "ROOT_SET", root=pid, reason="initial")
            return
        leaf_pid, path = self.leaf_for_insert(key)
        leaf = self.pages[leaf_pid]
        for idx, c in enumerate(leaf.leaf_cells):
            if c.key == key:
                if self.params.duplicate_policy == "reject":
                    txn.aborted = True
                    self._log(txn, "DUP_REJECT", page_id=leaf_pid, key=_show(key),
                              policy="reject")
                    txn.diagnostics.append(f"duplicate key {_show(key)!r} rejected")
                    return
                self._log(txn, "DUP_OVERWRITE", page_id=leaf_pid, key=_show(key),
                          policy="overwrite")
                leaf.leaf_cells[idx] = LeafCell(key, value)
                self._write(txn, leaf, tag="overwrite")
                return
        leaf.leaf_cells.append(LeafCell(key, value))
        leaf.leaf_cells.sort(key=lambda c: c.key)
        try:
            leaf.encode(self.params.page_size)
            self._write(txn, leaf, tag="leaf-insert")
            return
        except PageOverflow:
            leaf.leaf_cells.pop()  # 回退内存试探
        trace = self._split_leaf(txn, leaf_pid, key, value, path)
        return

    # ------------------------------------------------------------------ 分裂
    def _split_leaf(self, txn: Txn, leaf_pid: int, key: bytes, value: bytes,
                    path: list[int]) -> dict[str, Any]:
        leaf = self.pages[leaf_pid]
        merged = sorted(leaf.leaf_cells + [LeafCell(key, value)], key=lambda c: c.key)
        candidates = leaf_split_candidates(merged, self.params.page_size,
                                           self.params.min_fill_fraction)
        chosen = next((c for c in candidates if c["chosen"]), None)
        trace: dict[str, Any] = {
            "kind": "leaf_split", "page_id": leaf_pid, "path": list(path),
            "candidates": candidates, "key": _show(key),
        }
        if chosen is None:
            raise PageOverflow(f"no legal split for page {leaf_pid}")
        idx = chosen["split_index"]
        left_cells, right_cells = merged[:idx], merged[idx:]
        sep = right_cells[0].key
        new_pid = self.new_pid()
        self._log(txn, "PID_ALLOC", page_id=new_pid, reason="leaf-split", sibling_of=leaf_pid)
        # 顺序（采纳版）：先写新右页，再改左页，再写父连接，root 指针最后。
        right = Page(new_pid, LEAF, parent=NONE, prev=leaf_pid, next=leaf.next)
        right.leaf_cells = list(right_cells)
        if leaf.next != NONE:
            old_next = self.pages[leaf.next]
            old_next.prev = new_pid
            self._write(txn, old_next, tag="sibling-link")
        leaf.leaf_cells = list(left_cells)
        leaf.next = new_pid
        if self.params.separator_policy == PROMOTE:
            # promote：separator 从右叶移除，仅存于 internal（规则必须与查找一致）
            right.leaf_cells = [c for c in right.leaf_cells if c.key != sep]
        promoted_cell = None
        if self.params.separator_policy == PROMOTE and not right.leaf_cells:
            promoted_cell = LeafCell(sep, value)
        self._write(txn, right, tag="new-sibling")
        self._write(txn, leaf, tag="left-shrunk")
        trace.update({"new_page_id": new_pid, "separator": _show(sep),
                      "separator_policy": self.params.separator_policy,
                      "left_free_bytes": chosen["left_free_bytes"],
                      "right_free_bytes": chosen["right_free_bytes"],
                      "candidate_count": len(candidates),
                      "rejected_candidates": [c["split_index"] for c in candidates if not c["legal"]]})
        if not path:
            self._promote_new_root(txn, leaf_pid, new_pid, sep,
                                   promoted=(self.params.separator_policy == PROMOTE))
        else:
            self._insert_into_parent(txn, path, leaf_pid, new_pid, sep,
                                     promoted=(self.params.separator_policy == PROMOTE))
        txn.split_traces.append(trace)
        return trace

    def _promote_new_root(self, txn: Txn, left_pid: int, right_pid: int,
                          sep: bytes, *, promoted: bool) -> None:
        new_root_pid = self.new_pid()
        self._log(txn, "PID_ALLOC", page_id=new_root_pid, reason="new-root")
        node = Page(new_root_pid, INTERNAL)
        node.first_child = left_pid
        node.internal_cells = [InternalCell(sep, right_pid, promoted=promoted)]
        self._write(txn, node, tag="new-root")
        self.pages[left_pid].parent = new_root_pid
        self._write(txn, self.pages[left_pid], tag="reparent-left")
        self.pages[right_pid].parent = new_root_pid
        self._write(txn, self.pages[right_pid], tag="reparent-right")
        self.root = new_root_pid
        self._log(txn, "ROOT_SET", root=new_root_pid, reason="root-split",
                  left_child=left_pid, right_child=right_pid, separator=_show(sep))

    def _insert_into_parent(self, txn: Txn, path: list[int], left_pid: int,
                            right_pid: int, sep: bytes, *, promoted: bool) -> None:
        parent_pid = path[-1]
        node = self.pages[parent_pid]
        cell = InternalCell(sep, right_pid, promoted=promoted)
        # 新 cell 排在 left_pid 之后：按 separator 排序插入
        keys = [c.key for c in node.internal_cells]
        pos = int(np.searchsorted(keys, sep, side="left")) if promoted \
            else int(np.searchsorted(keys, sep, side="right"))
        node.internal_cells.insert(pos, cell)
        self.pages[left_pid].parent = parent_pid
        self.pages[right_pid].parent = parent_pid
        try:
            node.encode(self.params.page_size)
            self._write(txn, self.pages[left_pid], tag="reparent-left")
            self._write(txn, self.pages[right_pid], tag="reparent-right")
            self._write(txn, node, tag="parent-link")
            return
        except PageOverflow:
            node.internal_cells.pop(pos)
        self._split_internal(txn, parent_pid, path[:-1], cell)

    def _split_internal(self, txn: Txn, node_pid: int, upper: list[int],
                        incoming: InternalCell) -> None:
        node = self.pages[node_pid]
        merged = sorted(node.internal_cells + [incoming], key=lambda c: c.key)
        if self.params.separator_policy == PROMOTE:
            # promote 内分裂：中间键提升，两侧均不再保留
            mid = (len(merged) - 1) // 2
            up_key = merged[mid].key
            left_cells = merged[:mid]
            right_cells = merged[mid + 1:]
        else:
            mid = len(merged) // 2
            up_key = merged[mid].key
            left_cells = merged[:mid]
            right_cells = merged[mid:]
        children = [node.first_child] + [c.child for c in merged]
        left_children = children[:mid + 1]
        right_children = children[mid + 1:]
        new_pid = self.new_pid()
        self._log(txn, "PID_ALLOC", page_id=new_pid, reason="internal-split",
                  sibling_of=node_pid)
        left = Page(node_pid, INTERNAL, parent=node.parent)
        left.first_child = left_children[0]
        left.internal_cells = []
        for c, ch in zip(left_cells, left_children[1:]):
            left.internal_cells.append(InternalCell(c.key, ch, promoted=c.promoted))
        right = Page(new_pid, INTERNAL, parent=node.parent)
        right.first_child = right_children[0]
        right.internal_cells = []
        for c, ch in zip(right_cells, right_children[1:]):
            right.internal_cells.append(InternalCell(c.key, ch, promoted=c.promoted))
        self._write(txn, right, tag="new-internal")
        self._write(txn, left, tag="internal-shrunk")
        for ch in [c.child for c in left.internal_cells] + [left.first_child]:
            if ch in self.pages:
                self.pages[ch].parent = node_pid
                self._write(txn, self.pages[ch], tag="reparent", action="write")
        for ch in [c.child for c in right.internal_cells] + [right.first_child]:
            if ch in self.pages:
                self.pages[ch].parent = new_pid
                self._write(txn, self.pages[ch], tag="reparent")
        trace = {"kind": "internal_split", "page_id": node_pid, "new_page_id": new_pid,
                 "separator": _show(up_key),
                 "separator_policy": self.params.separator_policy,
                 "promoted_away": self.params.separator_policy == PROMOTE,
                 "path": list(upper)}
        txn.split_traces.append(trace)
        if not upper:
            self._promote_new_root(txn, node_pid, new_pid, up_key, promoted=False)
        else:
            self._insert_into_parent(txn, upper, node_pid, new_pid, up_key, promoted=False)
