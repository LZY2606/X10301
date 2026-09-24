"""B-tree 纯逻辑：变长键、按字节分裂、根/级联分裂、删除空页。

本模块不接触磁盘与日志；它只描述“一次逻辑操作”如何改变页集合。
所有长度判断均以 byte 为单位（pages.leaf_cell_size / internal_cell_size），
分裂点从不按“键数对半”决定，这是结构不变量 I1。

策略参数（来自规则版本，可修订）：
- duplicate_policy: "append"（重复键追加后缀后重插）| "overwrite"（覆盖旧值）
- split_tie:       "left"（字节差相同时偏左分）| "right"（偏右分）

separator 规则（不变量 I2）：
- leaf 分裂时，separator 是右页最小键的【复制】（右页保留该键）；
- internal 分裂时，中间键从子页中【提升】（不再留在任一子页）。
两条路径必须使用同一套 tie-break，由 BTree.split_candidates 统一产生。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pages import (
    INTERNAL,
    LEAF,
    INTERNAL_HEADER as INTERNAL_SPACE,
    InternalCell,
    LeafCell,
    Page,
    LEAF_HEADER as LEAF_SPACE,
    SLOT,
    internal_cell_size,
    leaf_cell_size,
)


class BTreeError(Exception):
    pass


class OversizeKeyError(BTreeError):
    """单个 cell 都无法放入一页：无法转换的值留在原位置并附诊断。"""


@dataclass
class SplitEvent:
    """一次页分裂的结构传播记录（供“分裂传播”视图）。"""
    kind: str                 # "leaf_split" | "internal_split" | "new_root"
    page_id: int
    new_page_id: int
    parent_id: int
    separator_hex: Optional[str]
    separator_rule: str       # "copied_to_parent" | "promoted_to_parent"
    left_bytes: int
    right_bytes: int
    depth: int                # 传播深度，0 = 叶层


@dataclass
class OpResult:
    op: str                   # insert | delete
    key_hex: str
    touched: List[int] = field(default_factory=list)
    allocated: List[int] = field(default_factory=list)
    freed: List[int] = field(default_factory=list)
    splits: List[SplitEvent] = field(default_factory=list)
    duplicate: bool = False
    note: str = ""


@dataclass
class BTree:
    page_size: int
    duplicate_policy: str = "append"
    split_tie: str = "left"
    pages: Dict[int, Page] = field(default_factory=dict)
    root: int = 0
    next_pid: int = 1

    # ---------- 基本管理 ----------
    @classmethod
    def create(cls, page_size: int, duplicate_policy: str = "append",
               split_tie: str = "left") -> "BTree":
        tree = cls(page_size=page_size, duplicate_policy=duplicate_policy,
                   split_tie=split_tie)
        leaf = Page.new_leaf(1, parent=0)
        tree.pages[1] = leaf
        tree.root = 1
        tree.next_pid = 2
        return tree

    def _alloc(self) -> int:
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def reachable_ids(self) -> List[int]:
        out: List[int] = []
        stack = [self.root]
        seen = set()
        while stack:
            pid = stack.pop()
            if pid in seen or pid not in self.pages:
                continue
            seen.add(pid)
            out.append(pid)
            page = self.pages[pid]
            if not page.is_leaf:
                stack.extend(page.child_ids())
        return sorted(seen)

    def leaf_chain(self) -> List[int]:
        """沿最左下叶 + sibling next 链，验证 sibling link 的可达闭环。"""
        pid = self.root
        page = self.pages[pid]
        while not page.is_leaf:
            pid = page.left_child
            page = self.pages[pid]
        chain = []
        seen = set()
        while pid and pid not in seen:
            seen.add(pid)
            chain.append(pid)
            pid = self.pages[pid].next
        return chain

    # ---------- 查找 ----------
    def find_leaf(self, key: bytes) -> Tuple[int, List[int]]:
        """返回 (leaf_pid, 访问路径 page_id 列表 root..leaf)。"""
        pid = self.root
        path: List[int] = [pid]
        page = self.pages[pid]
        while not page.is_leaf:
            nxt = page.left_child
            for cell in page.internal_cells:  # 已按键序
                if key < cell.key:
                    break
                nxt = cell.child
            pid = nxt
            path.append(pid)
            page = self.pages[pid]
        return pid, path

    def get(self, key: bytes) -> Optional[bytes]:
        pid, _ = self.find_leaf(key)
        for cell in self.pages[pid].leaf_cells:
            if cell.key == key:
                return cell.value
        return None

    # ---------- 分裂候选（统一 tie 规则，I1/I2）----------
    def _tie_choice(self, best_cuts: List[int]) -> int:
        if len(best_cuts) == 1:
            return best_cuts[0]
        return best_cuts[0] if self.split_tie == "left" else best_cuts[-1]

    def leaf_split_candidates(self, cells: List[LeafCell]) -> List[Tuple[int, int]]:
        """枚举所有合法叶分裂点 (cut, imbalance_bytes)，按字节空间而非键数。

        cut ∈ [1, n-1]：左页留前 cut 个键序 cell，右页留其余。
        合法条件：两侧 header+slot+body 都 <= page_size。
        排序键为两侧已用字节的绝对差，平局由 split_tie 决定。
        """
        n = len(cells)
        legal: List[Tuple[int, int]] = []
        if n < 2:
            return legal
        sizes = [c.size() for c in cells]
        total = sum(sizes)
        left = 0
        for cut in range(1, n):
            left += sizes[cut - 1]
            right = total - left
            lb = LEAF_SPACE + SLOT * cut + left
            rb = LEAF_SPACE + SLOT * (n - cut) + right
            if lb <= self.page_size and rb <= self.page_size:
                legal.append((cut, abs(lb - rb)))
        return legal

    def internal_split_candidates(self, cells: List[InternalCell]
                                  ) -> List[Tuple[int, int]]:
        """内部页分裂：n 个 entry + n+1 个 child。

        cut ∈ [0, n-1]：keys[:cut] 留左（left_child 为左最左孩子），
        keys[cut] 提升到父（promote，不在任一子页），keys[cut+1:] 留右，
        右页 left_child = cells[cut].child。
        左右至少一个孩子；cut=0 时左页 0 entry + 1 孩子（退化，随后由父折叠逻辑处理），
        为避免退化，本实现只接受 cut >= 1（保证左右各至少 1 个 entry）。
        """
        n = len(cells)
        legal: List[Tuple[int, int]] = []
        if n < 3:
            return legal
        sizes = [c.size() for c in cells]
        total = sum(sizes)
        left = 0
        for cut in range(1, n - 1):
            left += sizes[cut - 1]
            right = total - left - sizes[cut]
            lb = INTERNAL_SPACE + SLOT * cut + left
            rb = INTERNAL_SPACE + SLOT * (n - cut - 1) + right
            if lb <= self.page_size and rb <= self.page_size:
                legal.append((cut, abs(lb - rb)))
        return legal

    def _choose(self, candidates: List[Tuple[int, int]]) -> int:
        best = min(delta for _, delta in candidates)
        return self._tie_choice(sorted(c for c, d in candidates if d == best))

    # ---------- 插入 ----------
    def insert(self, key: bytes, value: bytes) -> OpResult:
        result = OpResult(op="insert", key_hex=key.hex())
        effective = key
        if self.get(key) is not None:
            result.duplicate = True
            if self.duplicate_policy == "overwrite":
                pid, _ = self.find_leaf(key)
                page = self.pages[pid]
                for cell in page.leaf_cells:
                    if cell.key == key:
                        cell.value = value
                result.touched.append(pid)
                result.note = "duplicate:overwrite"
                return result
            # append：加单调版本后缀直到唯一
            ver = 0
            while True:
                candidate = key + b"\x00" + bytes([ver & 0xFF])
                if self.get(candidate) is None:
                    effective = candidate
                    break
                ver += 1
                if ver > 255:
                    raise BTreeError("duplicate append exhausted")
            result.note = "duplicate:append"
        self._insert_leaf(effective, value, result)
        return result

    def _insert_leaf(self, key: bytes, value: bytes, result: OpResult) -> None:
        pid, path = self.find_leaf(key)
        leaf = self.pages[pid]
        leaf.leaf_cells.append(LeafCell(key, value))
        leaf.sort()
        result.touched.append(pid)
        if leaf.needs_split(self.page_size):
            sep_key, new_pid = self._split_leaf(pid, depth=len(path) - 1,
                                                result=result)
            parent_pid = path[-2] if len(path) >= 2 else 0
            self._attach(parent_pid, sep_key, new_pid, old_child=pid,
                         depth=len(path) - 1, result=result)

    def _split_leaf(self, pid: int, depth: int, result: OpResult
                    ) -> Tuple[bytes, int]:
        leaf = self.pages[pid]
        cells = sorted(leaf.leaf_cells, key=lambda c: c.key)
        candidates = self.leaf_split_candidates(cells)
        if not candidates:
            biggest = max(c.size() for c in cells)
            raise OversizeKeyError(
                f"leaf {pid} cannot split: largest cell {biggest} bytes "
                f"exceeds usable page {self.page_size - LEAF_SPACE - SLOT} bytes"
            )
        cut = self._choose(candidates)
        new_pid = self._alloc()
        sibling = Page.new_leaf(new_pid, parent=leaf.parent,
                                prev=pid, next=leaf.next)
        sibling.leaf_cells = cells[cut:]
        leaf.leaf_cells = cells[:cut]
        leaf.next = new_pid
        if sibling.next and sibling.next in self.pages:
            self.pages[sibling.next].prev = new_pid
        sibling.parent = leaf.parent
        self.pages[new_pid] = sibling
        result.allocated.append(new_pid)
        sep_key = sibling.leaf_cells[0].key  # 复制：右页保留该键
        result.splits.append(SplitEvent(
            kind="leaf_split", page_id=pid, new_page_id=new_pid,
            parent_id=leaf.parent, separator_hex=sep_key.hex(),
            separator_rule="copied_to_parent",
            left_bytes=leaf.used(), right_bytes=sibling.used(), depth=depth))
        return sep_key, new_pid

    def _attach(self, parent_pid: int, key: bytes, child_pid: int,
                old_child: int, depth: int, result: OpResult) -> None:
        """把 (key -> child_pid) 挂到父内部页；父溢出则递归分裂。"""
        if parent_pid == 0:
            return self._new_root(key, old_child, child_pid, depth, result)
        parent = self.pages[parent_pid]
        parent.internal_cells.append(InternalCell(key, child_pid))
        parent.sort()
        self.pages[child_pid].parent = parent_pid
        result.touched.append(parent_pid)
        if not parent.needs_split(self.page_size):
            return
        prom_key, prom_child, new_internal_pid = self._split_internal(
            parent_pid, depth=depth, result=result)
        grandparent = parent.parent
        self._attach(grandparent, prom_key, new_internal_pid,
                     old_child=parent_pid, depth=depth - 1, result=result)

    def _split_internal(self, pid: int, depth: int, result: OpResult
                        ) -> Tuple[bytes, int, int]:
        page = self.pages[pid]
        cells = sorted(page.internal_cells, key=lambda c: c.key)
        candidates = self.internal_split_candidates(cells)
        if not candidates:
            raise OversizeKeyError(
                f"internal {pid} cannot split within page_size {self.page_size}")
        cut = self._choose(candidates)
        new_pid = self._alloc()
        promoted = cells[cut]
        right = Page.new_internal(new_pid, parent=page.parent,
                                  left_child=promoted.child)
        right.internal_cells = cells[cut + 1:]
        left_cells = cells[:cut]
        page.internal_cells = left_cells
        for moved in right.internal_cells:
            self.pages[moved.child].parent = new_pid
        self.pages[promoted.child].parent = new_pid
        self.pages[new_pid] = right
        result.allocated.append(new_pid)
        result.splits.append(SplitEvent(
            kind="internal_split", page_id=pid, new_page_id=new_pid,
            parent_id=page.parent, separator_hex=promoted.key.hex(),
            separator_rule="promoted_to_parent",
            left_bytes=page.used(), right_bytes=right.used(), depth=depth))
        return promoted.key, new_pid, new_pid

    def _new_root(self, key: bytes, left_child: int, right_child: int,
                  depth: int, result: OpResult) -> None:
        new_pid = self._alloc()
        root_page = Page.new_internal(new_pid, parent=0, left_child=left_child)
        root_page.internal_cells = [InternalCell(key, right_child)]
        self.pages[new_pid] = root_page
        self.pages[left_child].parent = new_pid
        self.pages[right_child].parent = new_pid
        self.root = new_pid
        result.allocated.append(new_pid)
        result.splits.append(SplitEvent(
            kind="new_root", page_id=new_pid, new_page_id=right_child,
            parent_id=0, separator_hex=key.hex(),
            separator_rule="promoted_to_parent",
            left_bytes=0, right_bytes=0, depth=depth))

    # ---------- 删除 ----------
    def delete(self, key: bytes) -> OpResult:
        result = OpResult(op="delete", key_hex=key.hex())
        pid, path = self.find_leaf(key)
        leaf = self.pages[pid]
        before = len(leaf.leaf_cells)
        leaf.leaf_cells = [c for c in leaf.leaf_cells if c.key != key]
        result.touched.append(pid)
        if len(leaf.leaf_cells) == before:
            result.note = "not_found"
            return result
        if self.root == pid:
            return result  # 空根叶合法
        if leaf.leaf_cells:
            return result
        # 删除后空页：从叶链摘除并释放，再从父摘除孩子
        if leaf.prev and leaf.prev in self.pages:
            self.pages[leaf.prev].next = leaf.next
        if leaf.next and leaf.next in self.pages:
            self.pages[leaf.next].prev = leaf.prev
        parent_pid = leaf.parent
        del self.pages[pid]
        result.freed.append(pid)
        self._remove_child(parent_pid, pid, result)
        return result

    def _remove_child(self, parent_pid: int, child_pid: int,
                      result: OpResult) -> None:
        page = self.pages[parent_pid]
        children = page.child_ids()
        idx = children.index(child_pid)
        if idx == 0:
            # 删的是 left_child：无 separator 可删，把第一个 entry 降为 left_child
            if page.internal_cells:
                first = page.internal_cells[0]
                page.left_child = first.child
                page.internal_cells = page.internal_cells[1:]
        else:
            # children[idx] 对应 internal_cells[idx-1]：删该 entry
            page.internal_cells = [c for i, c in enumerate(page.internal_cells)
                                   if i != idx - 1]
        result.touched.append(parent_pid)
        if page.internal_cells:
            return
        # 内部页变空：折叠（只有一个孩子）
        only = page.left_child
        if parent_pid == self.root:
            self.pages[only].parent = 0
            self.root = only
            del self.pages[parent_pid]
            result.freed.append(parent_pid)
            return
        grandparent = page.parent
        del self.pages[parent_pid]
        result.freed.append(parent_pid)
        self._remove_child(grandparent, parent_pid, result)

    # ---------- 不变量 ----------
    def check_invariants(self) -> List[str]:
        """返回违反结构不变量的诊断列表；空列表 = 通过。"""
        problems: List[str] = []
        if self.root not in self.pages:
            problems.append(f"root {self.root} missing")
            return problems
        reachable = set(self.reachable_ids())
        for pid in reachable:
            page = self.pages[pid]
            if page.used() > self.page_size:
                problems.append(f"page {pid} overflow {page.used()} > {self.page_size} bytes")
            if not page.is_leaf:
                children = page.child_ids()
                for child in children:
                    if child not in self.pages:
                        problems.append(f"internal {pid} dangles -> {child}")
                    elif child not in reachable:
                        problems.append(f"internal {pid} points to unreachable half-page {child}")
                keys = page.keys()
                if keys != sorted(keys):
                    problems.append(f"internal {pid} separators unordered")
                if len(children) != len(keys) + 1:
                    problems.append(f"internal {pid} child count mismatch")
            else:
                keys = page.keys()
                if keys != sorted(keys):
                    problems.append(f"leaf {pid} keys unordered")
                if page.next and page.next in self.pages:
                    if self.pages[page.next].prev != pid:
                        problems.append(f"leaf {pid}.next backlink broken")
        # sibling 链从最左叶必须覆盖所有可达叶
        leaves = [pid for pid in reachable if self.pages[pid].is_leaf]
        chained = set(pid for pid in self.leaf_chain() if pid in reachable)
        if set(leaves) != chained:
            problems.append("leaf sibling chain does not cover all reachable leaves")
        return problems

    # ---------- 快照（页前镜像）----------
    def snapshot(self) -> Dict[int, bytes]:
        return {pid: page.serialize(self.page_size)
                for pid, page in sorted(self.pages.items())}
