"""B+ 树引擎：页内字节布局、分裂传播、根变化、WAL 与崩溃恢复。

关键设计（与需求一一对应）：
- 固定页大小，变长键；分裂点按 *字节空间*（pager.byte_split_candidates），
  并列时保留全部合法候选，选定规则记在 verdict 里。
- leaf 分裂 separator 复制到父页（right 首键同时留在 right leaf，I-SEPARATOR-COPY）；
  internal 分裂中间 separator 提升（不保留，I-SEPARATOR-PROMOTE）。
- 每次页写保留页前镜像；写页中断（torn）+ 崩溃走 ARIES-lite：
  redo 已提交事务、undo 未提交事务（CLR），恢复可重复执行且幂等
  （I-NO-REACHABLE-HALF / I-RECOVERY-IDEMPOTENT）。
- page id 由独立计数器单调分配，崩溃后也不重用（避免幽灵页）。
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from . import pager
SLOT_SIZE_DIR = 2

from .pager import (
    HEADER_SIZE,
    INTERNAL,
    LEAF,
    NONE,
    InternalCell,
    LeafCell,
    Page,
    TornPageError,
)

# ---- WAL 记录类型 ----
WAL_BEGIN = "BEGIN"
WAL_UPDATE = "UPDATE"      # redo: postimage；undo: preimage
WAL_NEW_PAGE = "NEW_PAGE"  # redo: postimage；undo: 释放 page id（标记为孤儿）
WAL_ALLOC = "ALLOC"        # page id 分配前进（崩溃不回退，避免 id 重用）
WAL_COMMIT = "COMMIT"
WAL_ABORT = "ABORT"
WAL_CLR = "CLR"            # undo 补偿记录（保证重复恢复幂等）


@dataclass
class WalRecord:
    lsn: int
    tx: int
    kind: str
    page_id: int | None = None
    preimage: bytes | None = None
    postimage: bytes | None = None
    undo_next: int = 0       # 本记录对应的上一条未补偿 UPDATE（CLR 链）

    def to_json(self) -> dict[str, Any]:
        return {
            "lsn": self.lsn, "tx": self.tx, "kind": self.kind,
            "page_id": self.page_id,
            "preimage": _b64(self.preimage),
            "postimage": _b64(self.postimage),
            "undo_next": self.undo_next,
        }


def _b64(b: bytes | None) -> str | None:
    if b is None:
        return None
    import base64
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str | None) -> bytes | None:
    if s is None:
        return None
    import base64
    return base64.b64decode(s.encode("ascii"))


# ---- 磁盘：页镜像 + WAL（崩溃只影响尚未“持久化”的一步）----
@dataclass
class Disk:
    page_size: int
    pages: dict[int, bytes] = field(default_factory=dict)
    wal: list[WalRecord] = field(default_factory=list)
    high_watermark: int = 0           # 已持久化的页 id 高水位（不重用）
    durable_lsn: int = 0              # 已持久化 WAL 序号
    _lsn: int = 0

    def next_lsn(self) -> int:
        self._lsn += 1
        return self._lsn

    def read_page(self, page_id: int) -> Page:
        return pager.parse_page(self.pages[page_id], self.page_size)

    def has_page(self, page_id: int) -> bool:
        return page_id in self.pages


@dataclass
class WriteStep:
    """一次逻辑操作内部的有序写页步骤（分裂传播顺序由此可见）。"""
    tag: str                         # NEW_RIGHT / UPDATE_PARENT / UPDATE_LEFT / WRITE_LEAF / ROOT
    page_id: int
    preimage: bytes | None
    postimage: bytes
    tx: int
    lsn: int


@dataclass
class OpTrace:
    """一次逻辑操作的完整轨迹（供复核界面与事件层使用）。"""
    op_seq: int
    op: str
    key: bytes | None
    value: bytes | None
    steps: list[WriteStep] = field(default_factory=list)
    split: dict[str, Any] | None = None
    root_before: int | None = None
    root_after: int | None = None
    next_pid_before: int = 0
    next_pid_after: int = 0
    notes: list[str] = field(default_factory=list)
    deleted: bool = False
    duplicated: bool = False


@dataclass
class BTree:
    disk: Disk
    root: int = NONE
    next_pid: int = 1
    duplicate_policy: str = "last"   # last | first | error（规则版本参数 dup_policy）
    tie_policy: str = "left"         # 字节均衡分裂并列时的默认选定方向
    op_seq: int = 0
    tx_counter: int = 0
    # crash injection
    crash_after_durable: int | None = None   # 在第 k 条 WAL 落盘后崩溃
    tear_next_write: bool = False            # 紧随其后的页写截断为半页
    on_step: Callable[[WriteStep], None] | None = None

    # ---------- 基础 ----------
    def clone(self) -> "BTree":
        return copy.deepcopy(self)

    def alloc_page_id(self, tx: int, trace: OpTrace) -> int:
        pid = self.next_pid
        self.next_pid += 1
        rec = WalRecord(self.disk.next_lsn(), tx, WAL_ALLOC, page_id=pid)
        self._emit_wal(rec, trace)
        return pid

    def _emit_wal(self, rec: WalRecord, trace: OpTrace | None) -> None:
        self.disk.wal.append(rec)
        self.disk.durable_lsn = len(self.disk.wal)
        if self.crash_after_durable is not None and self.disk.durable_lsn >= self.crash_after_durable:
            raise CrashInjected(f"crash after durable WAL #{self.disk.durable_lsn}")

    def _write_page(self, tx: int, page: Page, tag: str, trace: OpTrace,
                    kind_new: bool = False, undo_next: int = 0) -> WriteStep:
        pid = page.page_id
        preimage = self.disk.pages.get(pid)
        post = page.serialize()
        rec_kind = WAL_NEW_PAGE if kind_new else WAL_UPDATE
        rec = WalRecord(self.disk.next_lsn(), tx, rec_kind, page_id=pid,
                        preimage=preimage, postimage=post, undo_next=undo_next)
        self._emit_wal(rec, trace)
        if self.tear_next_write:
            self.tear_next_write = False
            self.disk.pages[pid] = post[: self.disk.page_size // 2]
            self.disk.high_watermark = max(self.disk.high_watermark, pid)
            raise CrashInjected(f"torn write on page {pid}")
        self.disk.pages[pid] = post
        self.disk.high_watermark = max(self.disk.high_watermark, pid)
        step = WriteStep(tag, pid, preimage, post, tx, rec.lsn)
        trace.steps.append(step)
        if self.on_step:
            self.on_step(step)
        return step

    # ---------- 查找 ----------
    def get(self, key: bytes) -> bytes | None:
        if self.root == NONE:
            return None
        leaf = self._descend(self.root, key)
        for c in leaf.cells:
            if c.key == key:
                return c.value
        return None

    def _descend(self, node_id: int, key: bytes, stack: list[int] | None = None) -> Page:
        node = self.disk.read_page(node_id)
        if stack is not None:
            stack.append(node_id)
        if node.kind == LEAF:
            return node
        i = 0
        while i < len(node.cells) and key >= node.cells[i].key:
            i += 1
        if i == len(node.cells):
            return self._descend(node.rightmost, key, stack)
        return self._descend(node.cells[i].left_child, key, stack)


class CrashInjected(Exception):
    """在指定崩溃点中止本次逻辑操作（事务未提交）。"""

    # ---------- 逻辑操作 ----------
    def begin_tx(self) -> int:
        self.tx_counter += 1
        return self.tx_counter

    def apply(self, op: str, key: bytes, value: bytes | None = None,
              crash_plan: "CrashPlan | None" = None) -> OpTrace:
        """执行一次逻辑操作。crash_plan 仅在该操作期间生效。"""
        self.op_seq += 1
        trace = OpTrace(self.op_seq, op, key, value)
        trace.root_before = self.root
        trace.next_pid_before = self.next_pid
        old_inj = (self.crash_after_durable, self.tear_next_write)
        if crash_plan is not None:
            self.crash_after_durable = crash_plan.after_durable
            self.tear_next_write = crash_plan.tear_next_write
        tx = self.begin_tx()
        try:
            begin = WalRecord(self.disk.next_lsn(), tx, WAL_BEGIN)
            self._emit_wal(begin, trace)
            if op == "put":
                self._op_put(tx, key, value or b"", trace)
            elif op == "delete":
                self._op_delete(tx, key, trace)
            else:
                raise ValueError(f"未知操作 {op}")
            commit = WalRecord(self.disk.next_lsn(), tx, WAL_COMMIT)
            self._emit_wal(commit, trace)
        except CrashInjected as exc:
            trace.notes.append(str(exc))
            return trace
        finally:
            self.crash_after_durable, self.tear_next_write = old_inj
        trace.root_after = self.root
        trace.next_pid_after = self.next_pid
        return trace

    def _op_delete(self, tx: int, key: bytes, trace: OpTrace) -> None:
        if self.root == NONE:
            trace.deleted = False
            trace.notes.append("树为空，删除无效")
            return
        leaf = self._descend(self.root, key)
        before = len(leaf.cells)
        leaf.cells = [c for c in leaf.cells if c.key != key]
        if len(leaf.cells) == before:
            trace.deleted = False
            trace.notes.append("键不存在")
            return
        trace.deleted = True
        leaf.page_lsn = self.disk._lsn
        if len(leaf.cells) == 0:
            trace.notes.append("EMPTY-LEAF: 删除后空页保留（可达），由 I-EMPTY-LEAF 诊断")
        self._write_page(tx, leaf, "WRITE_LEAF", trace)

    def _op_put(self, tx: int, key: bytes, value: bytes, trace: OpTrace) -> None:
        cell = LeafCell(key, value)
        if cell.size() + SLOT_SIZE_DIR > self.disk.page_size - HEADER_SIZE:
            raise ValueError(f"OVERSIZED: 单 cell 超过页容量 key={key!r}")
        if self.root == NONE:
            pid = self.alloc_page_id(tx, trace)
            leaf = Page(pid, LEAF, page_size=self.disk.page_size, page_lsn=self.disk._lsn)
            leaf.cells = [cell]
            self._write_page(tx, leaf, "ROOT", trace, kind_new=True)
            self.root = pid
            return
        stack: list[int] = []
        leaf = self._descend(self.root, key, stack)
        dup_idx = next((i for i, c in enumerate(leaf.cells) if c.key == key), None)
        if dup_idx is not None:
            trace.duplicated = True
            if self.duplicate_policy == "error":
                raise ValueError(f"重复键 {key!r}（dup_policy=error）")
            if self.duplicate_policy == "first":
                trace.notes.append("dup_policy=first: 保留首值，不覆盖")
                return
            leaf.cells[dup_idx] = cell  # last wins
            leaf.page_lsn = self.disk._lsn
            self._write_page(tx, leaf, "WRITE_LEAF", trace)
            return
        # 先按字节空间判断是否放得下（变长键按字节，不按键数）
        if leaf.fits_extra(cell.size()):
            inserted = False
            for i, c in enumerate(leaf.cells):
                if key < c.key:
                    leaf.cells.insert(i, cell)
                    inserted = True
                    break
            if not inserted:
                leaf.cells.append(cell)
            leaf.page_lsn = self.disk._lsn
            self._write_page(tx, leaf, "WRITE_LEAF", trace)
            return
        self._split_leaf_with(tx, leaf, cell, key, stack, trace)


    # ---------- leaf 分裂 ----------
    def _split_leaf(self, tx: int, leaf: Page, incoming_key: bytes,
                    stack: list[int], trace: OpTrace) -> None:
        sizes = [c.size() for c in leaf.cells]
        legal = pager.byte_split_candidates(sizes, self.disk.page_size)
        chosen, tied = pager.byte_balanced(sizes, legal)
        if chosen is None:
            raise ValueError("无合法分裂点（单键过大）")
        trace.split = {
            "kind": "leaf",
            "page_id": leaf.page_id,
            "legal_candidates": legal,
            "balanced_tied": tied,
            "chosen": chosen,
            "tie_policy": self.tie_policy,
            "decided_by": "rule" if len(tied) == 1 else "ambiguous",
            "byte_sizes": sizes,
        }
        # 并列无法唯一决定：全部合法并列候选保留，默认按 tie_policy 选；人工可在分支上改选
        if len(tied) > 1:
            trace.split["chosen"] = {"left": tied[0], "right": tied[-1]}.get(self.tie_policy, tied[0])
            chosen = trace.split["chosen"]
        sep = leaf.cells[chosen].key  # right 首键 = separator（复制）
        new_pid = self.alloc_page_id(tx, trace)
        right = Page(new_pid, LEAF, page_size=self.disk.page_size,
                     next=leaf.next, parent=leaf.parent, page_lsn=self.disk._lsn)
        right.cells = leaf.cells[chosen:]
        leaf.cells = leaf.cells[:chosen]
        old_next = leaf.next
        leaf.next = new_pid
        # 有序写：先全新右页（不可达），再父页，再左页 sibling —— 崩溃点不留下可达半页
        self._write_page(tx, right, "NEW_RIGHT", trace, kind_new=True)
        leaf.page_lsn = self.disk._lsn
        self._insert_in_parent(tx, stack[:-1], sep, leaf.page_id, new_pid, trace)
        self._write_page(tx, leaf, "UPDATE_LEFT", trace)
        trace.notes.append(f"leaf split p{leaf.page_id} -> p{leaf.page_id},p{new_pid}; "
                           f"sep={sep!r} copied to parent; old_next={old_next}")

    # ---------- 父页插入（含 internal 级联分裂与根变化）----------
    def _insert_in_parent(self, tx: int, stack: list[int], sep: bytes,
                          left_id: int, right_id: int, trace: OpTrace) -> None:
        if len(stack) < 2:
            self._grow_new_root(tx, sep, left_id, right_id, trace)
            return
        parent_id = stack[-2]
        parent = self.disk.read_page(parent_id)
        pos = next((i for i, c in enumerate(parent.cells) if c.left_child == left_id), None)
        if pos is not None:
            # 在 separator 槽位 pos 之后插入新 separator；其左孩子是新右页，
            # 原槽位 pos+0 之后的 separator 与其左孩子整体右移
            parent.cells.insert(pos + 1, InternalCell(sep, right_id))
        elif parent.rightmost == left_id:
            # 被分裂的是父页最右孩子：sep 成为最后一个 separator
            parent.cells.append(InternalCell(sep, right_id))
            parent.rightmost = right_id
        else:
            raise AssertionError("内部插入找不到左孩子指针")
        parent.page_lsn = self.disk._lsn
        if parent.free_bytes() >= 0:
            self._write_page(tx, parent, "UPDATE_PARENT", trace)
            return
        self._split_internal(tx, parent, stack, trace)

    # ---------- internal 分裂（级联）----------
    def _split_internal(self, tx: int, node: Page, stack: list[int],
                        trace: OpTrace) -> None:
        sizes = [c.size() for c in node.cells]
        # internal 页的容量核算包含固定 rightmost 保留区
        legal = pager.byte_split_candidates(sizes, self.disk.page_size)
        chosen, tied = pager.byte_balanced(sizes, legal)
        if chosen is None:
            raise ValueError("internal 无合法分裂点")
        promoted = node.cells[chosen].key  # 中间 separator 提升（不保留在子页）
        new_pid = self.alloc_page_id(tx, trace)
        right = Page(new_pid, INTERNAL, page_size=self.disk.page_size,
                     page_lsn=self.disk._lsn, parent=node.parent)
        # 左子页保留 [0..chosen)；其 rightmost = 被提升槽位原来的左孩子
        promoted_left_child = node.cells[chosen].left_child
        left_cells = node.cells[:chosen]
        right.cells = node.cells[chosen + 1:]
        right.rightmost = node.rightmost
        node.cells = left_cells
        node.rightmost = promoted_left_child
        if trace.split is not None and trace.split.get("cascade_from"):
            pass
        self._write_page(tx, right, "NEW_RIGHT", trace, kind_new=True)
        trace.notes.append(f"internal split p{node.page_id}: promote {promoted!r}")
        node.page_lsn = self.disk._lsn
        # 从 stack 中去掉当前层，再向上插入 promoted
        stack = [p for p in stack if p != node.page_id]
        self._insert_in_parent(tx, stack, promoted, node.page_id, new_pid, trace)
        self._write_page(tx, node, "UPDATE_LEFT", trace)

    def _grow_new_root(self, tx: int, sep: bytes, left_id: int, right_id: int,
                       trace: OpTrace) -> None:
        new_root_id = self.alloc_page_id(tx, trace)
        root = Page(new_root_id, INTERNAL, page_size=self.disk.page_size,
                    rightmost=right_id, page_lsn=self.disk._lsn)
        root.cells = [InternalCell(sep, left_id)]
        old_root = self.root
        self._write_page(tx, root, "ROOT", trace, kind_new=True)
        self.root = new_root_id
        trace.notes.append(f"root change p{old_root} -> new root p{new_root_id}")


@dataclass
class CrashPlan:
    """崩溃注入：在第 after_durable 条 WAL 落盘后崩溃，
    或把 tear_next_write 对应的那次页写截断成半页。"""
    after_durable: int | None = None
    tear_next_write: bool = False


@dataclass
class RecoveryReport:
    pass_no: int
    redone: list[int] = field(default_factory=list)
    undone: list[int] = field(default_factory=list)
    clrs: list[int] = field(default_factory=list)
    aborted_tx: list[int] = field(default_factory=list)
    orphan_pages: list[int] = field(default_factory=list)
    torn_detected: list[int] = field(default_factory=list)
    root: int = NONE
    reachable: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def recover(disk: Disk, root_selector: Callable[[Disk], int] | None = None,
            pass_no: int = 1) -> RecoveryReport:
    """ARIES-lite 恢复。可对同一 disk 重复执行（必须幂等）。

    - redo：重放 UPDATE/NEW_PAGE 后镜像（ torn 或落后页一律用 postimage 修复）；
    - undo：回滚没有 COMMIT 的事务；UPDATE 恢复 preimage，NEW_PAGE 把页标记为
      孤儿（写回 preimage 若存在，否则从 pages 删除并记 orphan）；
      每条 undo 追加 CLR，第二遍恢复时同一事务不会再被 undo。
    """
    report = RecoveryReport(pass_no)
    by_lsn = {r.lsn: r for r in disk.wal}
    begun = {r.tx for r in disk.wal if r.kind == WAL_BEGIN}
    committed = {r.tx for r in disk.wal if r.kind == WAL_COMMIT}
    clr_undone = {r.page_id and r.tx for r in disk.wal if r.kind == WAL_CLR}
    # 1) REDO
    for rec in disk.wal:
        if rec.kind in (WAL_UPDATE, WAL_NEW_PAGE):
            cur = disk.pages.get(rec.page_id)
            need = cur != rec.postimage
            if need:
                # torn（半页）检测
                if cur is not None:
                    try:
                        pager.parse_page(cur, disk.page_size)
                    except TornPageError:
                        report.torn_detected.append(rec.page_id)
                disk.pages[rec.page_id] = rec.postimage
                report.redone.append(rec.lsn)
    # 2) UNDO（未提交事务；已有 CLR 的跳过 -> 幂等）
    loser = sorted(t for t in begun if t not in committed)
    for tx in loser:
        if tx in clr_undone:
            continue
        recs = [r for r in disk.wal if r.tx == tx and
                r.kind in (WAL_UPDATE, WAL_NEW_PAGE)]
        for rec in reversed(recs):
            clr = WalRecord(disk.next_lsn(), tx, WAL_CLR, page_id=rec.page_id)
            disk.wal.append(clr)
            report.clrs.append(clr.lsn)
            if rec.kind == WAL_UPDATE and rec.preimage is not None:
                disk.pages[rec.page_id] = rec.preimage
            elif rec.kind == WAL_NEW_PAGE:
                if rec.page_id in disk.pages:
                    del disk.pages[rec.page_id]
                report.orphan_pages.append(rec.page_id)
            report.undone.append(rec.lsn)
        abort = WalRecord(disk.next_lsn(), tx, WAL_ABORT)
        disk.wal.append(abort)
        report.aborted_tx.append(tx)
    disk.durable_lsn = len(disk.wal)
    return report
