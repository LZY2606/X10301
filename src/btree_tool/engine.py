"""B-tree 回放引擎：逻辑操作、WAL、页 id 分配、崩溃注入与幂等恢复。

保留的原始材料（每次逻辑操作都会留下）：
  * 操作前后页的前镜像 / 后镜像；
  * slot directory、free space、sibling link；
  * page id 分配记录；日志序号（LSN）。

崩溃点（crash_after）：
  alloc_right / write_right / write_left / write_parent / set_root
写页中断（partial_write=True）：目标页只落盘一部分字节（其余为 0），
模拟真实 torn page；decode 时魔数或记账校验会失败。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .pages import (HEADER_SIZE, LEAF, INTERNAL, Cell, Page, PageFormatError,
                    decode_page, new_internal, new_leaf)
from .split import (TIE_LEFTMOST, TIE_UNSET, apply_split, choose_split,
                    legal_split_points)

DUP_REPLACE = "replace"
DUP_REJECT = "reject"
EMPTY_KEEP = "keep"
EMPTY_DEALLOCATE = "deallocate"

CRASH_POINTS = ("alloc_right", "write_right", "write_left",
                "write_parent", "set_root", None)


@dataclass
class WalRecord:
    lsn: int
    txn: int
    kind: str          # BEGIN / WRITE / ALLOC / COMMIT / ABORT / SETROOT / FREE
    pid: int | None = None
    image: bytes | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lsn": self.lsn, "txn": self.txn, "kind": self.kind,
            "pid": self.pid,
            "has_image": self.image is not None,
            "image_len": len(self.image) if self.image else 0,
            "detail": self.detail,
        }


@dataclass
class OpTrace:
    seq: int
    op: dict[str, Any]
    txn: int
    crash_after: str | None
    partial_write: bool
    pages_before: dict[int, dict[str, Any]]
    pages_after: dict[int, dict[str, Any]]
    page_allocs: list[dict[str, Any]]
    sibling_changes: list[dict[str, Any]]
    root_before: int | None
    root_after: int | None
    wal: list[dict[str, Any]]
    result: dict[str, Any]
    lsn_range: tuple[int, int] | None
    raw_key: Any
    raw_value: Any
    key_diag: str | None = None
    value_diag: str | None = None
    raw_page_size: Any = None
    page_size_diag: str | None = None


@dataclass
class ReplayResult:
    operations: list[OpTrace]
    reachable: dict[str, list[int]]
    invariants: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    recovery: list[dict[str, Any]]
    page_allocations: list[dict[str, Any]]
    root_history: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    params: dict[str, Any]
    root_id: int | None
    disk_size: int
    tree: dict[str, Any]
    leaf_order: list[int]
    raw_unconvertible: list[dict[str, Any]]


class _Crash(Exception):
    def __init__(self, result: dict[str, Any]):
        self.result = result


class _BadInput(Exception):
    pass


class Engine:
    def __init__(self, page_size: int, *,
                 split_policy: str = "byte_balance",
                 split_tiebreak: str = TIE_LEFTMOST,
                 dup_policy: str = DUP_REPLACE,
                 empty_leaf: str = EMPTY_KEEP):
        self.page_size = page_size
        self.split_policy = split_policy
        self.split_tiebreak = split_tiebreak
        self.dup_policy = dup_policy
        self.empty_leaf = empty_leaf

        # “磁盘”：pid -> bytes。EMPTY 表示页 id 已分配但内容从未写全。
        self.disk: dict[int, bytes] = {}
        self.wal: list[WalRecord] = []
        self.lsn = 0
        self.next_pid = 0
        self.txn = 0
        self.root: int | None = None
        self.free_pages: set[int] = set()
        self.allocation_log: list[dict[str, Any]] = []
        self.root_history: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        self.recovery_log: list[dict[str, Any]] = []
        self.raw_unconvertible: list[dict[str, Any]] = []

        # 分配根页
        self._alloc_new_page("initial_root")
        root = new_leaf(self.root)  # noqa: root 由 alloc 设置
        self._write_page(root, txn=0, logical=False)

    # ---- 低层原语 ------------------------------------------------------
    def _new_lsn(self) -> int:
        self.lsn += 1
        return self.lsn

    def _alloc_new_page(self, reason: str) -> int:
        pid = self.next_pid
        self.next_pid += 1
        lsn = self._new_lsn()
        self.allocation_log.append({
            "pid": pid, "reason": reason, "lsn": lsn,
            "next_pid_after": self.next_pid,
        })
        if self.root is None:
            self.root = pid
            self.root_history.append({"root": pid, "cause": "initial", "lsn": lsn})
        return pid

    def _wal(self, txn: int, kind: str, **kw: Any) -> WalRecord:
        rec = WalRecord(lsn=self._new_lsn(), txn=txn, kind=kind, **kw)
        self.wal.append(rec)
        return rec

    def _write_page(self, page: Page, *, txn: int, logical: bool = True,
                    torn: bool = False) -> int:
        page.lsn = self.lsn + 1
        image = page.encode(self.page_size)
        if torn:
            keep = max(1, self.page_size // 5)
            image = image[:keep] + b"\x00" * (self.page_size - keep)
        lsn = self._new_lsn()
        rec = self._wal(txn, "WRITE", pid=page.pid, image=image,
                        detail={"logical": logical, "torn": torn,
                                "page_lsn": page.lsn})
        self.disk[page.pid] = image
        return rec.lsn

    def get_page(self, pid: int) -> Page:
        return decode_page(pid, self.disk[pid], expect_size=self.page_size)

    def snapshot_pages(self, pids: list[int]) -> dict[int, dict[str, Any]]:
        out = {}
        for pid in pids:
            out[pid] = self._page_facts(pid)
        return out

    def _page_facts(self, pid: int) -> dict[str, Any]:
        """前/后镜像事实：slot directory、free space、sibling link、LSN。"""
        image = self.disk.get(pid)
        if image is None:
            return {"pid": pid, "present": False}
        try:
            page = decode_page(pid, image, expect_size=self.page_size)
        except PageFormatError as exc:
            return {"pid": pid, "present": True, "decodable": False,
                    "diagnostic": str(exc),
                    "raw_len_bytes": len(image)}
        layout = page.layout(self.page_size)
        return {
            "pid": pid,
            "present": True,
            "decodable": True,
            "type": "leaf" if page.is_leaf else "internal",
            "nslots": len(page.cells),
            "free_space_bytes": page.free_space(self.page_size),
            "used_bytes": page.used_bytes(),
            "slot_directory": layout["slots"],
            "sibling": {"prev": page.prev, "next": page.next},
            "high_child": page.high,
            "lsn": page.lsn,
            "keys_hex": [c.key.hex() for c in page.cells],
            "children": page.children() if not page.is_leaf else None,
            "coordinate_base": "page_start",
        }

    # ---- 逻辑操作 ------------------------------------------------------
    def apply_operation(self, seq: int, op: dict[str, Any]) -> OpTrace:
        kind = op.get("op")
        crash_after = op.get("crash_after")
        partial = bool(op.get("partial_write"))
        if crash_after not in CRASH_POINTS:
            raise ValueError(f"未知崩溃点: {crash_after!r}")

        self.txn += 1
        txn = self.txn
        lsn_start = self.lsn + 1
        self._wal(txn, "BEGIN", detail={"op": kind, "seq": seq})

        # 受影响页：根到叶路径 + 分裂新页，先抓前镜像
        path = self._find_path(op)
        touched = list(dict.fromkeys(path + [self.root]))
        pages_before = {pid: self._page_facts(pid) for pid in touched if pid is not None}
        allocs_before = len(self.allocation_log)
        siblings_before = self._sibling_links()

        raw_key, raw_value = op.get("key"), op.get("value")
        key = self._norm_key(raw_key, seq)
        value = self._norm_value(raw_value, seq)

        result: dict[str, Any]
        try:
            if kind == "put":
                result = self._op_put(txn, key, value, seq, crash_after, partial)
            elif kind == "delete":
                result = self._op_delete(txn, key, seq, crash_after, partial)
            else:
                raise ValueError(f"未知逻辑操作: {kind!r}")
        except _Crash as crash:
            result = crash.result
            self._wal(txn, "ABORT", detail={"crash_after": crash_after,
                                            "partial_write": partial,
                                            "reachable_half_page":
                                            result.get("reachable_half_page")})
            self.recovery_log.append(self._recover(note={
                "seq": seq, "crash_after": crash_after, "partial_write": partial}))
        else:
            self._wal(txn, "COMMIT",
                      detail={"committed": result.get("committed", True)})
        lsn_end = self.lsn

        touched_after = list(dict.fromkeys(
            [self.root] + list(pages_before.keys()) +
            [a["pid"] for a in self.allocation_log[allocs_before:]]))
        pages_after = {pid: self._page_facts(pid) for pid in touched_after
                       if pid is not None}
        sibling_changes = self._diff_siblings(siblings_before)
        wal = [r.to_dict() for r in self.wal if r.txn == txn]
        trace = OpTrace(
            seq=seq, op=op, txn=txn, crash_after=crash_after, partial_write=partial,
            pages_before=pages_before, pages_after=pages_after,
            page_allocs=self.allocation_log[allocs_before:],
            sibling_changes=sibling_changes,
            root_before=pages_before.get(self.root, {}).get("pid", self.root),
            root_after=self.root,
            wal=wal, result=result, lsn_range=(lsn_start, lsn_end),
            raw_key=raw_key, raw_value=raw_value,
        )
        return trace

    def _norm_key(self, raw: Any, seq: int) -> bytes:
        from .units import normalize_key
        key, diag = normalize_key(raw)
        if diag:
            self.raw_unconvertible.append({"seq": seq, "field": "key",
                                           "raw": raw, "diagnostic": diag})
            raise _BadInput(diag)
        return key  # type: ignore[return-value]

    def _norm_value(self, raw: Any, seq: int) -> bytes:
        if raw is None:
            return b""
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, str):
            try:
                return bytes.fromhex(raw) if raw.startswith("hex:") else raw.encode("utf-8")
            except ValueError:
                self.raw_unconvertible.append({"seq": seq, "field": "value",
                                               "raw": raw,
                                               "diagnostic": "非法 hex 值"})
                raise _BadInput("非法 hex 值")
        self.raw_unconvertible.append({"seq": seq, "field": "value", "raw": raw,
                                       "diagnostic": "值无法转换为字节"})
        raise _BadInput("值无法转换为字节")

    def _find_path(self, op: dict[str, Any]) -> list[int]:
        from .units import normalize_key
        key, _ = normalize_key(op.get("key"))
        if key is None:
            return []
        pids: list[int] = []
        pid = self.root
        while pid is not None:
            pids.append(pid)
            page = self.get_page(pid)
            if page.is_leaf:
                break
            pid = page.child_for(key)
        return pids

    def _sibling_links(self) -> dict[int, tuple[Any, Any]]:
        out = {}
        for pid in list(self.disk.keys()):
            try:
                page = self.get_page(pid)
            except PageFormatError:
                continue
            if page.is_leaf:
                out[pid] = (page.prev, page.next)
        return out

    def _diff_siblings(self, before: dict[int, tuple[Any, Any]]) -> list[dict[str, Any]]:
        after = self._sibling_links()
        changes = []
        for pid in set(before) | set(after):
            if before.get(pid) != after.get(pid):
                changes.append({"pid": pid,
                                "before": None if pid not in before else
                                {"prev": before[pid][0], "next": before[pid][1]},
                                "after": None if pid not in after else
                                {"prev": after[pid][0], "next": after[pid][1]}})
        return sorted(changes, key=lambda c: c["pid"])



    # ---- put / 分裂传播 -----------------------------------------------
    def _op_put(self, txn: int, key: bytes, value: bytes, seq: int,
                crash_after: str | None, partial: bool) -> dict[str, Any]:
        leaf = self.get_page(self.root) if self.root is not None else None
        chain = self._descend_to_leaf(key)
        leaf_pid, ancestors = chain[-1], chain[:-1]
        leaf = self.get_page(leaf_pid)
        existed = any(c.key == key for c in leaf.cells)
        if existed and self.dup_policy == DUP_REJECT:
            return {"committed": True, "action": "duplicate_rejected",
                    "key_hex": key.hex()}

        new_cell = Cell(key=key, value=value)
        split_happened = not leaf.can_insert(new_cell, self.page_size)
        if not split_happened:
            leaf.put_leaf(key, value, overwrite=True)
            self._write_page(leaf, txn=txn,
                             torn=(partial and crash_after == "write_left"))
            self._maybe_crash(
                crash_after, "write_left",
                self._crash_state(txn, seq, half_page=leaf_pid,
                                  reachable_half=True))
            return {"committed": True, "action": "updated" if existed else "inserted",
                    "leaf": leaf_pid, "split": False}

        # 需要分裂：从叶向上级联
        action, chosen = self._insert_with_split(
            txn, leaf_pid, ancestors, key, value, seq, crash_after, partial)
        return {"committed": True, "action": action, "split": True,
                "leaf_chain_len": len(chain), "chosen": chosen}

    def _descend_to_leaf(self, key: bytes) -> list[int]:
        chain: list[int] = []
        pid = self.root
        while True:
            chain.append(pid)
            page = self.get_page(pid)
            if page.is_leaf:
                return chain
            pid = page.child_for(key)

    def _insert_with_split(self, txn: int, leaf_pid: int,
                           ancestors: list[int], key: bytes, value: bytes,
                           seq: int, crash_after: str | None,
                           partial: bool) -> tuple[str, dict[str, Any] | None]:
        leaf = self.get_page(leaf_pid)
        leaf.put_leaf(key, value, overwrite=True)
        cand, all_cand = choose_split(leaf, self.page_size,
                                      self.split_policy, self.split_tiebreak)
        cand_record = {
            "seq": seq, "page": leaf_pid, "type": "leaf",
            "legal_points": legal_split_points(leaf, self.page_size),
            "candidates": [c.to_dict() for c in all_cand],
            "chosen": None if cand is None else cand.to_dict(),
            "policy": self.split_policy, "tiebreak": self.split_tiebreak,
            "decidable": cand is not None,
            "space_basis": "byte",
        }
        self.candidates.append(cand_record)
        if cand is None:
            # 无法唯一决定：全部合法候选已保留，参数未修订前不做物理分裂
            raise _BadInput("split_undecidable")

        new_pid = self._alloc_new_page("leaf_split_right")
        self._wal(txn, "ALLOC", pid=new_pid,
                  detail={"reason": "leaf_split_right", "parent_of": leaf_pid})
        self._maybe_crash(crash_after, "alloc_right", self._crash_state(txn, seq))

        left, right, sep, rule = apply_split(leaf, cand)
        right.pid = new_pid
        left.next = new_pid
        right.prev = leaf_pid
        if left.prev is not None and left.prev in self.disk:
            prev_page = self.get_page(left.prev)
            prev_page.next = leaf_pid
        # 写新右页（可能 torn）
        self._write_page(right, txn=txn, torn=(partial and crash_after == "write_right"))
        self._maybe_crash(crash_after, "write_right",
                          self._crash_state(txn, seq, half_page=new_pid,
                                            reachable_half=False))
        self._write_page(left, txn=txn,
                         torn=(partial and crash_after == "write_left"))
        self._maybe_crash(crash_after, "write_left",
                          self._crash_state(txn, seq, half_page=leaf_pid,
                                            reachable_half=True))
        # 修正左兄弟回链
        if left.prev is not None and left.prev in self.disk:
            prev_page = self.get_page(left.prev)
            prev_page.next = leaf_pid
            self._write_page(prev_page, txn=txn)

        self._propagate(txn, ancestors, leaf_pid, leaf_pid, new_pid, sep,
                        seq, crash_after, partial, depth=0)
        return "inserted_split", cand.to_dict()

    def _propagate(self, txn: int, ancestors: list[int],
                   old_child: int, left: int, right: int, sep: bytes,
                   seq: int, crash_after: str | None, partial: bool,
                   depth: int) -> None:
        if not ancestors:
            # 根分裂：新建根 internal
            new_root = self._alloc_new_page("root_split")
            self._wal(txn, "ALLOC", pid=new_root, detail={"reason": "root_split"})
            root_page = new_internal(new_root,
                                     [Cell(key=sep, child=left)], right)
            self._write_page(root_page, txn=txn,
                             torn=(partial and crash_after == "write_parent"))
            old_root = self.root
            self._maybe_crash(crash_after, "write_parent",
                              self._crash_state(txn, seq, half_page=new_root,
                                                reachable_half=False))
            self.root = new_root
            self._wal(txn, "SETROOT", detail={"old": old_root, "new": new_root})
            self.root_history.append({"root": new_root, "cause": "root_split",
                                      "lsn": self.lsn, "old": old_root})
            self._maybe_crash(crash_after, "set_root",
                              self._crash_state(txn, seq, half_page=None,
                                                reachable_half=False))
            return

        parent_pid = ancestors[-1]
        parent = self.get_page(parent_pid)
        parent.replace_child(old_child, left, sep, right)
        if parent.fits(self.page_size):
            self._write_page(parent, txn=txn,
                             torn=(partial and crash_after == "write_parent"))
            self._maybe_crash(crash_after, "write_parent",
                              self._crash_state(txn, seq,
                                                half_page=parent_pid,
                                                reachable_half=True))
            return

        # internal 级联分裂（separator 提升，不复制）
        cand, all_cand = choose_split(parent, self.page_size,
                                      self.split_policy, self.split_tiebreak)
        self.candidates.append({
            "seq": seq, "page": parent_pid, "type": "internal", "depth": depth + 1,
            "legal_points": legal_split_points(parent, self.page_size),
            "candidates": [c.to_dict() for c in all_cand],
            "chosen": None if cand is None else cand.to_dict(),
            "policy": self.split_policy, "tiebreak": self.split_tiebreak,
            "decidable": cand is not None, "space_basis": "byte",
        })
        if cand is None:
            raise _BadInput("internal_split_undecidable")
        new_pid = self._alloc_new_page("internal_split_right")
        self._wal(txn, "ALLOC", pid=new_pid,
                  detail={"reason": "internal_split_right"})
        pleft, pright, psep, rule = apply_split(parent, cand)
        assert rule == "promote"
        pright.pid = new_pid
        self._write_page(pright, txn=txn)
        self._write_page(pleft, txn=txn)
        self._propagate(txn, ancestors[:-1], parent_pid,
                        parent_pid, new_pid, psep, seq, crash_after, partial,
                        depth + 1)

    # ---- delete / 空页 -------------------------------------------------
    def _op_delete(self, txn: int, key: bytes, seq: int,
                   crash_after: str | None, partial: bool) -> dict[str, Any]:
        chain = self._descend_to_leaf(key)
        leaf_pid = chain[-1]
        leaf = self.get_page(leaf_pid)
        removed = leaf.remove_leaf(key)
        if not removed:
            return {"committed": True, "action": "delete_miss", "leaf": leaf_pid}
        if (not leaf.cells and self.empty_leaf == EMPTY_DEALLOCATE
                and (len(chain) > 1 or self.root != leaf_pid)):
            return self._delete_empty_leaf(txn, chain, leaf, seq,
                                           crash_after, partial)
        self._write_page(leaf, txn=txn,
                         torn=(partial and crash_after == "write_left"))
        self._maybe_crash(crash_after, "write_left",
                          self._crash_state(txn, seq, half_page=leaf_pid,
                                            reachable_half=True))
        return {"committed": True, "action": "deleted",
                "leaf": leaf_pid, "empty_leaf": not leaf.cells,
                "empty_leaf_policy": self.empty_leaf}

    def _delete_empty_leaf(self, txn: int, chain: list[int], leaf: Page,
                           seq: int, crash_after: str | None,
                           partial: bool) -> dict[str, Any]:
        parent_pid = chain[-2]
        parent = self.get_page(parent_pid)
        sep_index = None
        for i, c in enumerate(parent.cells):
            if c.child == leaf.pid:
                sep_index = i
                break
        removed_sep = None
        if sep_index is not None:
            removed_sep = parent.cells[sep_index].key
            del parent.cells[sep_index]
        elif parent.high == leaf.pid:
            if parent.cells:
                removed_sep = parent.cells[-1].key
                parent.high = parent.cells[-1].child
                del parent.cells[-1]
        # sibling 链接摘除
        if leaf.prev is not None and leaf.prev in self.disk:
            p = self.get_page(leaf.prev)
            p.next = leaf.next
            self._write_page(p, txn=txn)
        if leaf.next is not None and leaf.next in self.disk:
            n = self.get_page(leaf.next)
            n.prev = leaf.prev
            self._write_page(n, txn=txn)
        self._write_page(parent, txn=txn)
        self._wal(txn, "FREE", pid=leaf.pid,
                  detail={"reason": "empty_leaf_deallocated",
                          "removed_separator_hex":
                          removed_sep.hex() if removed_sep else None})
        self.free_pages.add(leaf.pid)
        self.disk.pop(leaf.pid, None)
        return {"committed": True, "action": "deallocated_empty_leaf",
                "leaf": leaf.pid, "removed_separator_hex":
                removed_sep.hex() if removed_sep else None}

    # ---- 崩溃 / 恢复 ----------------------------------------------------
    def _maybe_crash(self, crash_after: str | None, point: str,
                      state_factory: Any) -> None:
        if crash_after != point:
            return
        state = state_factory() if callable(state_factory) else state_factory
        raise _Crash(state)

    def _crash_state(self, txn: int, seq: int, *, half_page: int | None = None,
                     reachable_half: bool = False) -> dict[str, Any]:
        reachable_now, _ = self._reachable_ids()
        return {
            "committed": False,
            "action": "crash",
            "crashed_txn": txn,
            "seq": seq,
            "half_page": half_page,
            "reachable_half_page":
                bool(half_page is not None and half_page in reachable_now
                     and not self._page_decodable(half_page)),
            "reachable_at_crash": sorted(reachable_now),
        }

    def _page_decodable(self, pid: int) -> bool:
        try:
            self.get_page(pid)
        except PageFormatError:
            return False
        return True

    def _reachable_ids(self) -> tuple[set[int], list[str]]:
        problems: list[str] = []
        seen: set[int] = set()
        if self.root is None:
            return seen, problems
        stack = [self.root]
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                page = self.get_page(pid)
            except PageFormatError as exc:
                problems.append(f"reachable page {pid} 不可解码: {exc}")
                continue
            if not page.is_leaf:
                stack.extend(page.children())
        return seen, problems

    def recover(self) -> dict[str, Any]:
        """按 WAL 恢复：重做已 COMMIT 的物理写，回滚未提交事务。

        按页 LSN 去重，重复调用完全幂等（第二次为空操作）。
        """
        return self._recover(note={"explicit": True})

    def _recover(self, *, note: dict[str, Any]) -> dict[str, Any]:
        run = self.lsn + 1
        committed_txns = {r.txn for r in self.wal if r.kind == "COMMIT"}
        committed_txns.add(0)  # txn 0 = 系统初始化（建根页），视为已提交
        # 之前恢复轮次已经回滚过的事务 / 孤儿页；恢复点落 WAL，保证重复恢复幂等。
        already_undone_txns = {r.txn for r in self.wal if r.kind == "UNDO_DONE"}
        already_orphan = set()
        for r in self.wal:
            if r.kind == "ORPHAN_DONE":
                already_orphan.update(r.detail.get("pids", []))
        actions: list[dict[str, Any]] = []
        root_at_setroot = self.root
        # 先按顺序重做已提交事务的 ALLOC/WRITE/SETROOT/FREE
        for rec in self.wal:
            if rec.txn not in committed_txns:
                continue
            if rec.kind == "SETROOT":
                root_at_setroot = rec.detail["new"]
                if self.root != root_at_setroot:
                    actions.append({"lsn": rec.lsn, "action": "setroot",
                                    "root": root_at_setroot})
                    self.root = root_at_setroot
            elif (rec.kind == "WRITE" and rec.pid is not None and rec.image
                  and not rec.detail.get("torn")):
                current_lsn = self._safe_page_lsn(rec.pid)
                if current_lsn is None or current_lsn < rec.detail.get("page_lsn", 0):
                    self.disk[rec.pid] = rec.image
                    actions.append({"lsn": rec.lsn, "action": "redo_write",
                                    "pid": rec.pid})
            elif rec.kind == "FREE" and rec.pid is not None:
                if rec.pid in self.disk:
                    self.disk.pop(rec.pid, None)
                    actions.append({"lsn": rec.lsn, "action": "redo_free",
                                    "pid": rec.pid})
        # 回滚未提交、且此前未回滚过的事务。
        undone_pages: set[int] = set()
        orphan_pages: set[int] = set()
        for rec in self.wal:
            if rec.txn in committed_txns or rec.txn in already_undone_txns:
                continue
            if rec.kind == "WRITE" and rec.pid is not None:
                prior = self._last_good_image_before_uncommitted(rec.pid)
                if prior is None:
                    orphan_pages.add(rec.pid)
                else:
                    undone_pages.add(rec.pid)
        for pid in sorted(undone_pages):
            prior = self._last_good_image_before_uncommitted(pid)
            self.disk[pid] = prior  # type: ignore[assignment]
            actions.append({"lsn": None, "action": "undo_write", "pid": pid})
        for pid in sorted(orphan_pages):
            # 该页由未提交事务创建：分配保留为孤儿，但永不挂到可达树上
            if pid in self.disk and not self._page_decodable(pid):
                self.disk[pid] = b"\x00" * self.page_size
            actions.append({"lsn": None, "action": "orphan_alloc", "pid": pid})
        # 恢复点落 WAL（恢复本身也是日志事件），保证第二次恢复无动作。
        undone_txns = {r.txn for r in self.wal
                       if r.kind == "WRITE" and r.txn not in committed_txns
                       and r.txn not in already_undone_txns}
        if undone_txns:
            self._wal(0, "UNDO_DONE", detail={"txns": sorted(undone_txns)})
        if orphan_pages:
            self._wal(0, "ORPHAN_DONE", detail={"pids": sorted(orphan_pages)})
        reachable, problems = self._reachable_ids()
        report = {
            "recovery_run_lsn": run,
            "note": note,
            "actions": actions,
            "committed_txns": sorted(committed_txns),
            "reachable_after": sorted(reachable),
            "reachable_half_pages":
                sorted(pid for pid in reachable if not self._page_decodable(pid)),
            "problems": problems,
            "idempotent": len(actions) == 0,
        }
        self.recovery_log.append(report)
        return report

    def _safe_page_lsn(self, pid: int) -> int | None:
        if pid not in self.disk:
            return None
        try:
            return self.get_page(pid).lsn
        except PageFormatError:
            return None

    def _last_good_image_before_uncommitted(self, pid: int) -> bytes | None:
        committed = {r.txn for r in self.wal if r.kind == "COMMIT"}
        committed.add(0)
        good: bytes | None = None
        for rec in self.wal:
            if rec.kind != "WRITE" or rec.pid != pid:
                continue
            if rec.txn in committed and not rec.detail.get("torn"):
                good = rec.image
        return good

    # ---- 结构不变量 ----------------------------------------------------
    def check_invariants(self) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        reachable, decode_problems = self._reachable_ids()

        # I1: 可达页必须全部可解码（不允许可达半页）
        checks.append({
            "code": "reachable_pages_decodable",
            "passed": not decode_problems,
            "evidence": decode_problems or {"reachable": sorted(reachable)},
            "rule": "崩溃点不能留下可达半页",
        })

        stack = [self.root] if self.root is not None else []
        key_order_bad: list[Any] = []
        bounds_bad: list[Any] = []
        parent_bad: list[Any] = []
        balance_bad: list[Any] = []
        space_bad: list[Any] = []
        leaf_chain_bad: list[Any] = []

        def walk(pid: int, low: bytes | None, high: bytes | None,
                 parent: int | None, depth: int, heights: list[int]) -> None:
            page = self.get_page(pid)
            if not page.fits(self.page_size):
                space_bad.append({"pid": pid, "used": page.used_bytes(),
                                  "page_size": self.page_size})
            keys = [c.key for c in page.cells]
            if keys != sorted(keys):
                key_order_bad.append(pid)
            for k in keys:
                if low is not None and not (low <= k):
                    bounds_bad.append({"pid": pid, "key_hex": k.hex(),
                                       "violation": "key <= lower"})
                if high is not None and not (k < high):
                    bounds_bad.append({"pid": pid, "key_hex": k.hex(),
                                       "violation": "key >= upper"})
            if page.is_leaf:
                heights.append(depth)
                return
            children = page.children()
            bounds = [low] + keys + [high]
            for child, lo, hi in zip(children, bounds[:-1], bounds[1:]):
                if child not in self.disk:
                    parent_bad.append({"pid": pid, "missing_child": child})
                    continue
                walk(child, lo, hi, pid, depth + 1, heights)

        heights: list[int] = []
        if self.root is not None:
            try:
                walk(self.root, None, None, None, 0, heights)
            except PageFormatError as exc:
                key_order_bad.append(f"decode: {exc}")

        checks.append({"code": "keys_sorted", "passed": not key_order_bad,
                       "evidence": {"bad": key_order_bad},
                       "rule": "页内键 / separator 严格按键序"})
        checks.append({"code": "bounds_consistent", "passed": not bounds_bad,
                       "evidence": {"bad": bounds_bad},
                       "rule": "子页键区间与父 separator 一致"})
        checks.append({"code": "children_exist", "passed": not parent_bad,
                       "evidence": {"bad": parent_bad},
                       "rule": "internal 引用的子页必须存在"})
        checks.append({"code": "balanced_heights",
                       "passed": len(set(heights)) <= 1,
                       "evidence": {"leaf_depths": sorted(set(heights))},
                       "rule": "所有叶页深度相同"})
        checks.append({"code": "space_accounting", "passed": not space_bad,
                       "evidence": {"bad": space_bad},
                       "rule": "页 used_bytes <= 页大小（字节空间基准）"})

        # leaf sibling 链：可达叶页按 next 串成严格升序
        leaves: dict[int, Page] = {}
        if self.root is not None:
            st = [self.root]
            while st:
                pid = st.pop()
                try:
                    p = self.get_page(pid)
                except PageFormatError:
                    continue
                if p.is_leaf:
                    leaves[pid] = p
                else:
                    st.extend(p.children())
        ordered = self._leaf_chain(leaves)
        chain_keys: list[bytes] = []
        for pid in ordered:
            chain_keys.extend(c.key for c in leaves[pid].cells)
        chain_ok = chain_keys == sorted(chain_keys) and len(chain_keys) == len(set(chain_keys))
        link_ok = True
        for pid in ordered:
            p = leaves[pid]
            if p.next is not None and (p.next not in leaves
                                       or leaves[p.next].prev != pid):
                link_ok = False
                leaf_chain_bad.append({"pid": pid, "next": p.next})
            if p.prev is not None and (p.prev not in leaves
                                       or leaves[p.prev].next != pid):
                link_ok = False
                leaf_chain_bad.append({"pid": pid, "prev": p.prev})
        checks.append({"code": "leaf_sibling_chain",
                       "passed": chain_ok and link_ok,
                       "evidence": {"order": ordered, "bad": leaf_chain_bad},
                       "rule": "sibling link 串起全部可达叶页且键序严格"})
        return checks

    def _leaf_chain(self, leaves: dict[int, Page]) -> list[int]:
        heads = [pid for pid, p in leaves.items() if p.prev is None]
        if len(heads) != 1:
            return sorted(leaves)
        out: list[int] = []
        cur: int | None = heads[0]
        seen: set[int] = set()
        while cur is not None and cur not in seen:
            out.append(cur)
            seen.add(cur)
            cur = leaves[cur].next
        return out if set(out) == set(leaves) else sorted(leaves)

    # ---- 结构导出（SVG 树 / JSON） -------------------------------------
    def tree_structure(self) -> dict[str, Any]:
        nodes: dict[str, Any] = {}

        def build(pid: int) -> dict[str, Any]:
            page = self.get_page(pid)
            facts = self._page_facts(pid)
            node = {
                "id": f"page-{pid}",
                "pid": pid,
                "type": "leaf" if page.is_leaf else "internal",
                "keys_hex": [c.key.hex() for c in page.cells],
                "children": [f"page-{c}" for c in page.children()],
                "facts": facts,
            }
            nodes[f"page-{pid}"] = node
            if not page.is_leaf:
                for c in page.children():
                    if c in self.disk:
                        build(c)
            return node

        root_node = build(self.root) if self.root is not None else None
        return {"root_id": self.root,
                "root_node": f"page-{self.root}" if self.root is not None else None,
                "nodes": nodes, "root": root_node,
                "leaf_order": self._leaf_order_ids()}

    def _leaf_order_ids(self) -> list[int]:
        leaves: dict[int, Page] = {}
        if self.root is None:
            return []
        st = [self.root]
        while st:
            pid = st.pop()
            try:
                p = self.get_page(pid)
            except PageFormatError:
                continue
            if p.is_leaf:
                leaves[pid] = p
            else:
                st.extend(p.children())
        return self._leaf_chain(leaves)


def replay(operations: list[dict[str, Any]], page_size_raw: Any, *,
           split_policy: str = "byte_balance",
           split_tiebreak: str = TIE_LEFTMOST,
           dup_policy: str = DUP_REPLACE,
           empty_leaf: str = EMPTY_KEEP,
           run_recovery_twice: bool = True) -> ReplayResult:
    """从材料顺序回放全部逻辑操作，保留逐操作证据，并跑两次恢复。"""
    from .units import to_bytes
    pq = to_bytes(page_size_raw, coordinate_base="page_size")
    warnings: list[dict[str, Any]] = []
    if not pq.ok or pq.value is None:
        return ReplayResult(
            operations=[], reachable={}, invariants=[], candidates=[],
            recovery=[], page_allocations=[], root_history=[],
            warnings=[{"field": "page_size", "raw": page_size_raw,
                       "diagnostic": pq.diagnostic}],
            params={"page_size_raw": page_size_raw},
            root_id=None, disk_size=0, tree={}, leaf_order=[],
            raw_unconvertible=[])
    page_size = int(pq.value)
    eng = Engine(page_size, split_policy=split_policy, split_tiebreak=split_tiebreak,
                 dup_policy=dup_policy, empty_leaf=empty_leaf)
    traces: list[OpTrace] = []
    for seq, op in enumerate(operations):
        try:
            trace = eng.apply_operation(seq, op)
        except _BadInput as exc:
            # 不丢弃：留在原位置并附诊断
            traces.append(OpTrace(
                seq=seq, op=op, txn=eng.txn,
                crash_after=op.get("crash_after"),
                partial_write=bool(op.get("partial_write")),
                pages_before={}, pages_after={}, page_allocs=[],
                sibling_changes=[], root_before=eng.root, root_after=eng.root,
                wal=[], result={"committed": False, "action": "rejected",
                                "diagnostic": str(exc)},
                lsn_range=None, raw_key=op.get("key"), raw_value=op.get("value"),
                key_diag=str(exc) if "key" in str(exc) else None))
            continue
        traces.append(trace)

    # 崩溃操作内部已经恢复过；无论有无崩溃，结尾再整体恢复两次验证幂等
    rec1 = eng.recover()
    rec2 = eng.recover() if run_recovery_twice else None
    reachable, problems = eng._reachable_ids()
    inv = eng.check_invariants()
    tree = eng.tree_structure()
    return ReplayResult(
        operations=traces,
        reachable={"reachable": sorted(reachable),
                   "allocated": sorted(eng.disk.keys()),
                   "orphans": sorted(set(eng.disk) - reachable)},
        invariants=inv,
        candidates=eng.candidates,
        recovery=[r for r in eng.recovery_log],
        page_allocations=eng.allocation_log,
        root_history=eng.root_history,
        warnings=warnings,
        params={"page_size": page_size, "page_size_raw": page_size_raw,
                "split_policy": split_policy, "split_tiebreak": split_tiebreak,
                "dup_policy": dup_policy, "empty_leaf": empty_leaf,
                "second_recovery_idempotent":
                rec2["idempotent"] if rec2 else None,
                "first_end_recovery": rec1,
                "second_end_recovery": rec2},
        root_id=eng.root,
        disk_size=len(eng.disk),
        tree=tree,
        leaf_order=tree["leaf_order"],
        raw_unconvertible=eng.raw_unconvertible,
    )


def trace_to_dict(trace: OpTrace) -> dict[str, Any]:
    return {
        "seq": trace.seq, "op": trace.op, "txn": trace.txn,
        "crash_after": trace.crash_after, "partial_write": trace.partial_write,
        "pages_before": trace.pages_before, "pages_after": trace.pages_after,
        "page_allocs": trace.page_allocs,
        "sibling_changes": trace.sibling_changes,
        "root_before": trace.root_before, "root_after": trace.root_after,
        "wal": trace.wal, "result": trace.result,
        "lsn_range": trace.lsn_range,
        "raw_key": trace.raw_key, "raw_value": trace.raw_value,
        "key_diag": trace.key_diag, "value_diag": trace.value_diag,
    }
