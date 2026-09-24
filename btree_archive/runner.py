"""把案例材料（逻辑操作 + 参数 + 崩溃事件）重放成可复核的步骤记录。

每个步骤记录七类原始证据（来源与解释拆开保存）：
1. logic      —— 逻辑操作（op/key/value，保留原始表示与单位）；
2. before     —— 页前镜像（整页字节 hex + 解析出的布局坐标）；
3. slots      —— slot directory（序号/偏移/长度，单位 byte/slot）；
4. free_space —— 每页连续可用字节；
5. siblings   —— leaf sibling link（prev/next page id）；
6. pid_alloc  —— page id 分配（root_pid/next_pid/新分配/释放）；
7. lsn        —— 日志序号（每条 W/M/C/CLR/A 都带全局单调 LSN）。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .btree import BTree, OpResult, OversizeKeyError
from .pages import INTERNAL, LEAF, Page
from .store import (CP_COMMIT_FLUSH, CP_TORN_SPLIT, Crash, Disk,
                    RecoverReport)


@dataclass
class StepRecord:
    seq: int
    material_id: str
    logic: dict
    before_pages: Dict[int, dict]
    after_pages: Dict[int, dict]
    slots: Dict[int, list]
    free_space: Dict[str, int]          # {"p<pid>_bytes": n}
    siblings: Dict[str, dict]
    pid_alloc: dict
    lsn: dict
    splits: List[dict]
    touched: List[int]
    crash: Optional[dict]
    recovery: Optional[dict]
    status: str                         # applied | rejected | crashed_recovered
    diagnostics: List[str] = field(default_factory=list)


def _page_view(page: Page, page_size: int) -> dict:
    layout = page.layout(page_size)
    return {
        "page_id": page.page_id,
        "page_type": "leaf" if page.is_leaf else "internal",
        "parent": page.parent,
        "page_lsn": page.lsn,
        "raw_hex": (page.serialize(page_size).hex()
                    if page.used() <= page_size else None),
        "layout": layout,
        "note": None if page.used() <= page_size
        else f"pre-split transient overflow by {page.used() - page_size} bytes",
    }


def _siblings(tree: BTree) -> Dict[str, dict]:
    return {
        f"p{pid}": {"prev": p.prev, "next": p.next,
                    "unit": "page_id", "coordinate_basis": "leaf chain"}
        for pid, p in sorted(tree.pages.items()) if p.is_leaf
    }


def load_tree_from_disk(disk: Disk) -> BTree:
    root, nxt = disk.read_meta()
    tree = BTree(page_size=disk.page_size)
    tree.root = root
    tree.next_pid = nxt
    for idx, raw in disk._blocks.items():
        if idx == 0 or not raw or raw[0] not in (LEAF, INTERNAL):
            continue
        page = Page.parse(raw)
        tree.pages[page.page_id] = page
    return tree


class Replayer:
    """在一个临时磁盘目录上重放整个案例。"""

    def __init__(self, page_size: int, workdir: str,
                 duplicate_policy: str = "append",
                 split_tie: str = "left"):
        self.page_size = page_size
        self.workdir = workdir
        self.duplicate_policy = duplicate_policy
        self.split_tie = split_tie
        self.tree = BTree.create(page_size, duplicate_policy, split_tie)
        self.disk = Disk(workdir, page_size, create=True)
        self.steps: List[StepRecord] = []
        # 初次建库：把初始空叶落盘（非事务引导，保证磁盘可读）
        self._bootstrap()

    def _bootstrap(self) -> None:
        raw = self.tree.pages[1].serialize(self.page_size)
        self.disk._write_block_raw(1, raw)

    def replay(self, materials: List[dict]) -> List[StepRecord]:
        for seq, mat in enumerate(materials):
            self._step(seq, mat)
        return self.steps

    def _step(self, seq: int, mat: dict) -> None:
        op = mat["op"]
        before = {pid: _page_view(p, self.page_size)
                  for pid, p in sorted(self.tree.pages.items())}
        root_before, next_before = self.tree.root, self.tree.next_pid
        crash_point = mat.get("crash")

        if op == "insert":
            key, value, diag = self._decode_kv(mat)
            if key is None:
                self.steps.append(self._rejected(seq, mat, before, diag))
                return
            result = self._try_op(lambda: self.tree.insert(key, value),
                                  seq, mat, before, crash_point)
        elif op == "delete":
            key = bytes.fromhex(mat["key_hex"])
            result = self._try_op(lambda: self.tree.delete(key),
                                  seq, mat, before, crash_point)
        else:
            self.steps.append(
                StepRecord(seq, mat.get("id", f"m{seq:03d}"),
                           {"raw": mat}, before, {}, {}, {}, {}, {}, [], [],
                           None, None, "rejected",
                           [f"unknown op {op}"]))
            return
        if result is not None:
            self._record_applied(seq, mat, before, result,
                                 root_before, next_before)

    def _decode_kv(self, mat: dict):
        """导入层保留原始表示；无法转换的值留在原位置并附诊断，绝不丢值。"""
        try:
            key = bytes.fromhex(mat["key_hex"])
        except (KeyError, ValueError):
            return None, None, [f"unconvertible key_hex on {mat.get('id')}"]
        raw_value = mat.get("value", {"hex": "", "unit": "byte"})
        try:
            value = bytes.fromhex(raw_value["hex"])
        except (KeyError, ValueError, TypeError):
            return None, None, [f"unconvertible value on {mat.get('id')}"]
        declared = raw_value.get("length_bytes")
        if declared is not None and declared != len(value):
            return None, None, [
                f"length mismatch on {mat.get('id')}: declared {declared} "
                f"bytes != decoded {len(value)} bytes"]
        return key, value, []

    def _rejected(self, seq, mat, before, diag) -> StepRecord:
        return StepRecord(
            seq, mat.get("id", f"m{seq:03d}"),
            {"raw": mat, "units": {"length": "byte", "offset": "byte@page0"}},
            before, {}, {}, {},
            {"root_pid": self.tree.root, "next_pid": self.tree.next_pid},
            {}, [], [], None, None, "rejected", diag)

    def _try_op(self, op_fn, seq, mat, before, crash_point):
        old_raw: Dict[int, bytes] = {
            pid: page.serialize(self.page_size)
            for pid, page in self.tree.pages.items()
        }
        try:
            result = op_fn()
        except OversizeKeyError as exc:
            self.steps.append(
                StepRecord(seq, mat.get("id", f"m{seq:03d}"),
                           {"raw": mat}, before, {}, {}, {}, {}, {}, [], [],
                           None, None, "rejected", [str(exc)]))
            return None

        writes = []
        new_pids = set(result.allocated)
        pids_to_write = set(result.touched) | new_pids
        for pid in sorted(pids_to_write):
            page = self.tree.pages[pid]
            embed = (lambda lsn, _p=page: self._serialize_with_lsn(_p, lsn))
            writes.append({
                "pid": pid,
                "is_new": pid in new_pids,
                "before": old_raw.get(pid),
                "embed_lsn": embed,
            })

        crash_info = None
        recovery_info = None
        txn = f"t{seq:04d}"
        try:
            self.disk.flush_txn(txn, writes, self.tree.root,
                                self.tree.next_pid, crash_point)
            for w in writes:
                self.tree.pages[w["pid"]].lsn = w["lsn"]
        except Crash as crash:
            crash_info = {
                "point": crash.crash_point,
                "lsn_at_crash": crash.lsn,
                "torn_pages": crash.torn_pages,
                "unit": "lsn",
            }
            report1 = self.disk.recover()
            report2 = self.disk.recover()   # 重放两次：必须幂等
            recovery_info = {
                "first_pass": _report(report1),
                "second_pass": _report(report2),
                "idempotent": _report(report1) == _report(report2),
                "reachable_after": report1.reachable_after,
            }
            self.tree = load_tree_from_disk(self.disk)
            problems = self.tree.check_invariants()
            if problems:
                recovery_info["invariant_violations"] = problems
            result.touched = sorted(pids_to_write)
            self._record_crashed(seq, mat, before, result,
                                 crash_info, recovery_info)
            return None
        return result

    def _serialize_with_lsn(self, page: Page, lsn: int) -> bytes:
        page.lsn = lsn
        return page.serialize(self.page_size)

    def _record_applied(self, seq, mat, before, result,
                        root_before, next_before) -> None:
        after = {pid: _page_view(p, self.page_size)
                 for pid, p in sorted(self.tree.pages.items())}
        slots = {f"p{pid}": p.layout(self.page_size)["slots"]
                 for pid, p in sorted(self.tree.pages.items())}
        free_space = {
            f"p{pid}_bytes": p.free_space(self.page_size)
            for pid, p in sorted(self.tree.pages.items())
        }
        lsn_info = {"next_lsn": self.disk.next_lsn, "unit": "lsn",
                    "range_after_op": _lsn_range(self.disk)}
        rec = StepRecord(
            seq, mat.get("id", f"m{seq:03d}"),
            _logic_view(mat, result),
            before, after, slots, free_space, _siblings(self.tree),
            {"root_pid": self.tree.root, "next_pid": self.tree.next_pid,
             "root_before": root_before, "next_before": next_before,
             "allocated": result.allocated, "freed": result.freed,
             "unit": "page_id"},
            lsn_info,
            [vars(s) for s in result.splits],
            result.touched, None, None, "applied", [])
        self.steps.append(rec)

    def _record_crashed(self, seq, mat, before, result,
                        crash_info, recovery_info) -> None:
        after = {pid: _page_view(p, self.page_size)
                 for pid, p in sorted(self.tree.pages.items())}
        slots = {f"p{pid}": p.layout(self.page_size)["slots"]
                 for pid, p in sorted(self.tree.pages.items())}
        rec = StepRecord(
            seq, mat.get("id", f"m{seq:03d}"),
            _logic_view(mat, result),
            before, after, slots,
            {f"p{pid}_bytes": p.free_space(self.page_size)
             for pid, p in sorted(self.tree.pages.items())},
            _siblings(self.tree),
            {"root_pid": self.tree.root, "next_pid": self.tree.next_pid,
             "allocated": result.allocated, "freed": result.freed,
             "unit": "page_id"},
            {"next_lsn": self.disk.next_lsn, "unit": "lsn"},
            [vars(s) for s in result.splits],
            result.touched, crash_info, recovery_info,
            "crashed_recovered", [])
        self.steps.append(rec)


def _logic_view(mat: dict, result: OpResult) -> dict:
    return {
        "raw": mat,
        "material_id": mat.get("id"),
        "interpreted": {
            "op": result.op,
            "key_hex": result.key_hex,
            "duplicate": result.duplicate,
            "note": result.note,
            "length_unit": "byte",
        },
    }


def _lsn_range(disk: Disk) -> dict:
    recs = disk._read_wal()
    if not recs:
        return {"first": 0, "last": 0}
    return {"first": recs[0].lsn, "last": recs[-1].lsn}


def _report(report: RecoverReport) -> dict:
    return {
        "winner_txns": sorted(report.winner_txns),
        "loser_txns": sorted(report.loser_txns),
        "redone_lsn": report.redone,
        "undone_lsn": report.undone,
        "clr_lsn": report.clrs,
        "torn_detected": report.torn_detected,
        "reachable_after": report.reachable_after,
    }
