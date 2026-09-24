"""B-tree 重放引擎：固定页、按字节空间分裂、多候选、崩溃与幂等恢复。

设计要点
--------
* 输入是“逻辑操作账本”（原始表示单独保存）与一版规则参数；输出是
  带稳定身份的解释事件（页前镜像、slot directory、free space、
  sibling link、page id 分配、日志序号、候选与裁定）。
* 引擎是纯函数式重放：同一 (账本, 规则版本, 决定) 永远得到同一结果，
  因此历史事件引用的派生结论可以随时按当时版本重建，不被最新结果覆盖。
* 无法唯一决定的位置保留全部合法候选；非法候选（例如按键数分裂导致
  一侧溢出）也作为“被否定候选”保留并附判定依据。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import ids
from .pages import (
    Cell,
    PAGE_LEAF,
    PAGE_INTERNAL,
    NIL,
    Page,
    decode_page,
    new_internal,
    new_leaf,
)

# ---------------------------------------------------------------------------
# 规则参数（一版规则 = 一个不可变对象；参数修订 => 新规则版本）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rules:
    rule_id: str
    page_size_bytes: int = 128
    split_metric: str = "bytes"          # bytes | keys（后者保留为非法对照）
    split_choice: str = "balanced"      # balanced | first_fit
    separator_policy: str = "copy_right_min"  # copy_right_min | promote_left_max_zero
    duplicate_policy: str = "overwrite"  # overwrite | forbid | multiversion
    empty_leaf_policy: str = "keep"      # keep | deallocate
    recovery_policy: str = "redo_lsn"    # redo_lsn | discard_torn(非法对照)

    def asdict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "page_size_bytes": self.page_size_bytes,
            "split_metric": self.split_metric,
            "split_choice": self.split_choice,
            "separator_policy": self.separator_policy,
            "duplicate_policy": self.duplicate_policy,
            "empty_leaf_policy": self.empty_leaf_policy,
            "recovery_policy": self.recovery_policy,
        }


DEFAULT_RULES = Rules(rule_id="rules-v1")

# ---------------------------------------------------------------------------
# 账本解析：原始表示保留在 raw 层；无法转换留在原位并附诊断
# ---------------------------------------------------------------------------


@dataclass
class ParsedOp:
    seq: int
    raw: Dict[str, Any]
    op: Optional[Dict[str, Any]] = None
    diagnostic: Optional[str] = None


def parse_ledger(ledger: List[Dict[str, Any]]) -> List[ParsedOp]:
    out: List[ParsedOp] = []
    for i, entry in enumerate(ledger):
        seq = int(entry.get("seq", i))
        raw = copy.deepcopy(entry)
        kind = entry.get("op")
        try:
            if kind == "put":
                key = _as_bytes(entry["key"])
                val = _as_bytes(entry.get("value", b""))
                if key == b"":
                    raise ValueError("empty key")
                out.append(ParsedOp(seq, raw, {"op": "put", "key": key, "value": val}))
            elif kind == "delete":
                key = _as_bytes(entry["key"])
                out.append(ParsedOp(seq, raw, {"op": "delete", "key": key}))
            elif kind == "crash":
                out.append(ParsedOp(seq, raw, {
                    "op": "crash",
                    "after_logs": int(entry.get("after_logs", 1)),
                    "torn_page": entry.get("torn_page", "new_leaf"),
                }))
            elif kind == "recover":
                out.append(ParsedOp(seq, raw, {"op": "recover", "round": int(entry.get("round", 1))}))
            else:
                out.append(ParsedOp(seq, raw, None, f"unknown op: {kind!r}"))
        except (KeyError, ValueError, TypeError) as exc:
            out.append(ParsedOp(seq, raw, None, f"unconvertible op left in place: {exc}"))
    return out


def _as_bytes(v: Any) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        # 原始表示（hex 或 utf-8 文本）保留在 raw 层；这里只做派生层规范化
        if v.startswith("hex:"):
            return bytes.fromhex(v[4:])
        return v.encode("utf-8")
    if isinstance(v, int):
        return bytes([v & 0xFF])
    raise ValueError(f"cannot normalize value of type {type(v).__name__}")


# ---------------------------------------------------------------------------
# 磁盘 / WAL 模拟
# ---------------------------------------------------------------------------


@dataclass
class WalRec:
    lsn: int
    op_seq: int
    kind: str                 # split_propagate | leaf_write | root_replace ...
    pid: int
    fpi: bytes                # full page image（redo 的依据）
    new_lsn: int
    note: str = ""

    def view(self) -> Dict[str, Any]:
        return {
            "id": f"log-{self.lsn:04d}",
            "lsn": self.lsn,
            "unit": "log_sequence_number",
            "op_seq": self.op_seq,
            "kind": self.kind,
            "pid": self.pid,
            "page_lsn_after": self.new_lsn,
            "note": self.note,
            "fpi_size_bytes": len(self.fpi),
        }


class Simulator:
    def __init__(self, rules: Rules):
        self.rules = rules
        self.disk: Dict[int, bytes] = {}
        self.pool: Dict[int, Page] = {}
        self.wal: List[WalRec] = []
        self.lsn = 0
        self.next_pid = 0
        self.root_pid: Optional[int] = None
        self._init_meta_and_root()

    def _new_lsn(self) -> int:
        self.lsn += 1
        return self.lsn

    def alloc_pid(self) -> int:
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def _init_meta_and_root(self) -> None:
        # pid 0 = 分配器/根指针元数据页（简化为内存结构，仍保留 4B root 指针）
        meta = bytes([0] * 128)
        self.disk[0] = meta
        self.next_pid = 1
        pid = self.alloc_pid()
        leaf = new_leaf(pid, self._new_lsn())
        self.root_pid = pid
        self._install(leaf)
        self.flush_all(initial=True)
        self.pid_alloc_events = [
            {"pid": 1, "reason": "initial root leaf", "lsn": leaf.created_lsn}
        ]

    def _install(self, page: Page) -> None:
        self.pool[page.pid] = page

    def get(self, pid: int) -> Page:
        if pid not in self.pool:
            self.pool[pid] = decode_page(pid, self.disk[pid])
        return self.pool[pid]

    def touch_log(self, page: Page, op_seq: int, kind: str, note: str = "") -> WalRec:
        lsn = self._new_lsn()
        rec = WalRec(lsn=lsn, op_seq=op_seq, kind=kind, pid=page.pid,
                     fpi=page.encode(), new_lsn=lsn, note=note)
        page.page_lsn = lsn
        self.wal.append(rec)
        return rec

    def flush_all(self, initial: bool = False) -> None:
        order = sorted(self.pool.values(), key=lambda p: (p.pid == self.root_pid, p.pid))
        for page in order:
            self.disk[page.pid] = page.encode()
        self.pool.clear()

    def flush_plan(self, pids: List[int], op_seq: int) -> List[WalRec]:
        recs = [r for r in self.wal if r.op_seq == op_seq and r.pid in pids]
        recs.sort(key=lambda r: (r.pid == self.root_pid, r.lsn))
        return recs

    # ------------------------------------------------------------------
    def writeback(self, op_seq: int, crash: Optional[Dict[str, Any]] = None
                  ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """按 WAL 顺序把脏页写回；根最后写。可在指定日志后撕页中断。

        返回 (写入记录视图, 崩溃信息或 None)。
        """
        touched = sorted({r.pid for r in self.wal if r.op_seq == op_seq})
        plan = self.flush_plan(touched, op_seq)
        written: List[Dict[str, Any]] = []
        crash_info = None
        threshold = crash["after_logs"] if crash else None
        torn_target = self._resolve_torn_pid(crash) if crash else None
        aborted = False
        for idx, rec in enumerate(plan):
            if aborted:
                break
            if crash is not None and idx >= threshold:
                # 中断：目标页留下半写，其余脏页直接丢弃（从未落盘）
                if torn_target is not None and torn_target in self.pool:
                    half = bytearray(self.pool[torn_target].encode())
                    half[3] = max(1, half[3])  # 扰动一个字节使整页 CRC 失败
                    self.disk[torn_target] = bytes(half)
                crash_info = {
                    "op_seq": op_seq,
                    "after_logs": threshold,
                    "planned_logs": [r.lsn for r in plan],
                    "written_logs": [r.lsn for r in plan[:threshold]],
                    "torn_pid": torn_target,
                    "aborted_pids": [p for p in touched if p not in
                                     {r.pid for r in plan[:threshold]}],
                }
                aborted = True
                break
            self.disk[rec.pid] = rec.fpi
            written.append(rec.view())
        self.pool.clear()
        return written, crash_info

    def _resolve_torn_pid(self, crash: Dict[str, Any]) -> Optional[int]:
        target = crash.get("torn_page", "new_leaf")
        if isinstance(target, int):
            return target
        # 从本次 split 的新分配页里解析（默认取最新分配的非根页）
        candidates = [e["pid"] for e in self.pid_alloc_events]
        if not candidates:
            return None
        if target == "new_leaf":
            return max(candidates)
        return candidates[int(str(target).split(":")[-1]) - 1] if ":" in str(target) else max(candidates)

    # ------------------------------------------------------------------
    def recover(self) -> Dict[str, Any]:
        """从磁盘重建缓冲池并按 WAL 做 LSN 门控 redo；重复执行幂等。"""
        self.pool.clear()
        torn_before: List[int] = []
        for pid, raw in self.disk.items():
            if pid == 0:
                continue
            p = decode_page(pid, raw)
            if p.torn:
                torn_before.append(pid)
        reachable_before = self._reachable_pids_from_disk()
        reachable_torn_before = sorted(set(torn_before) & set(reachable_before))

        applied: List[int] = []
        skipped: List[int] = []
        if self.rules.recovery_policy == "discard_torn":
            for pid in torn_before:
                if pid in reachable_before:
                    # 非法对照：直接丢弃可达半页会造成指针悬空
                    self.disk.pop(pid, None)
            note = "discard_torn: 可达半页被直接丢弃（悬空指针，非法）"
        else:
            note = "redo_lsn: full-page redo，页 LSN >= 记录则跳过（幂等）"
            for rec in self.wal:
                if rec.pid not in self.disk:
                    self.disk[rec.pid] = rec.fpi
                    applied.append(rec.lsn)
                    continue
                cur = decode_page(rec.pid, self.disk[rec.pid])
                if cur.torn or cur.page_lsn < rec.new_lsn:
                    self.disk[rec.pid] = rec.fpi
                    applied.append(rec.lsn)
                else:
                    skipped.append(rec.lsn)

        torn_after: List[int] = []
        for pid, raw in self.disk.items():
            if pid != 0 and decode_page(pid, raw).torn:
                torn_after.append(pid)
        reachable_after = self._reachable_pids_from_disk()
        reachable_torn_after = sorted(set(torn_after) & set(reachable_after))
        self._reload_pool_and_pid()
        return {
            "torn_pids_at_crash": sorted(torn_before),
            "reachable_torn_before_recovery": reachable_torn_before,
            "logs_applied": applied,
            "logs_skipped_stale": skipped,
            "torn_pids_after_recovery": sorted(torn_after),
            "reachable_torn_after_recovery": reachable_torn_after,
            "reachable_pids": reachable_after,
            "method": note,
            "idempotent": len(applied) == 0,
        }

    def _reachable_pids_from_disk(self) -> List[int]:
        seen: set = set()
        if self.root_pid is None or self.root_pid not in self.disk:
            return []
        stack = [self.root_pid]
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            raw = self.disk.get(pid)
            if raw is None:
                seen.add(pid)  # 悬空指针也标出
                continue
            seen.add(pid)
            p = decode_page(pid, raw)
            if not p.torn:
                stack.extend(p.child_pids())
        return sorted(seen)

    def _reload_pool_and_pid(self) -> None:
        self.pool.clear()
        max_pid = max(self.disk) if self.disk else 0
        self.next_pid = max_pid + 1


# ---------------------------------------------------------------------------
# 分裂候选：按字节空间决定；keys 度量作为被否定对照保留
# ---------------------------------------------------------------------------


def leaf_split_candidates(left_cells: List[Cell], page_size: int,
                          metric: str) -> List[Dict[str, Any]]:
    """返回每个可能切分点的候选（含合法性判定依据）。

    切分点 i：排序后 [0..i) 留左页，[i..n) 去右页。
    合法依据 = 两页 free space(byte) >= 0 且两侧非空。
    """
    ordered = sorted(left_cells, key=lambda c: c.key)
    n = len(ordered)
    cands: List[Dict[str, Any]] = []
    from .pages import HEADER_END, SLOT_LEN, TRAILER_LEN

    def leaf_free(k: int, cell_bytes: int) -> int:
        return page_size - TRAILER_LEN - HEADER_END - SLOT_LEN * k - cell_bytes

    total_bytes = sum(c.encoded_len() for c in ordered)
    for i in range(1, n):
        lb = sum(c.encoded_len() for c in ordered[:i])
        rb = total_bytes - lb
        free_l = leaf_free(i, lb)
        free_r = leaf_free(n - i, rb)
        byte_legal = free_l >= 0 and free_r >= 0
        if metric == "keys":
            midpoint = n // 2
            chosen_by_metric = i == midpoint
            metric_note = f"按键数切半：n={n}，切点 {midpoint}"
        else:
            chosen_by_metric = True  # 所有字节合法点都是可选合法候选
            metric_note = "按字节空间评估两页 free space>=0"
        cands.append({
            "split_at": i,
            "left_key_count": i,
            "right_key_count": n - i,
            "left_cell_bytes": lb,
            "right_cell_bytes": rb,
            "free_space_left_bytes": free_l,
            "free_space_right_bytes": free_r,
            "legal": bool(byte_legal),
            "metric": metric,
            "metric_would_choose": bool(chosen_by_metric),
            "basis": (
                metric_note
                + f"；左页 {lb}B cell + {i} slots，free={free_l}B；"
                + f"右页 {rb}B + {n - i} slots，free={free_r}B"
                + ("；满足字节空间约束" if byte_legal else "；一侧溢出，非法")
            ),
        })
    return cands


def choose_candidate(cands: List[Dict[str, Any]], choice: str) -> int:
    legal = [c for c in cands if c["legal"]]
    pool = legal or cands
    if choice == "balanced":
        best = min(pool, key=lambda c: abs(
            c["free_space_left_bytes"] - c["free_space_right_bytes"]))
    else:  # first_fit：从最小切点开始第一个右侧放得下的点
        best = next(c for c in cands if c["legal"])
    return best["split_at"]


def make_separator(left_max: bytes, right_min: bytes, policy: str) -> bytes:
    if policy == "promote_left_max_zero":
        # 严格落在 (left_max, right_min) 的提升键：left_max + 0x00
        return left_max + b"\x00"
    return right_min  # copy_right_min：右页最小键复制到父层


def separator_consistent(sep: bytes, left_max: bytes, right_min: bytes,
                         policy: str) -> bool:
    if policy == "promote_left_max_zero":
        return sep == left_max + b"\x00" and sep > left_max and sep <= right_min
    return sep == right_min and sep > left_max


# ---------------------------------------------------------------------------
# 重放器
# ---------------------------------------------------------------------------


@dataclass
class DecisionPoint:
    dp_id: str
    op_seq: int
    kind: str
    candidates: List[Dict[str, Any]]
    default_choice: Any
    applied_choice: Any
    basis: str
    accepted: Optional[bool] = None  # 人工裁定：True 接受默认 / False 选 override
    override: Optional[Any] = None
    source_dp: Optional[str] = None  # 重新解释分支引用的原始决策点

    def view(self) -> Dict[str, Any]:
        return {
            "id": self.dp_id,
            "op_seq": self.op_seq,
            "kind": self.kind,
            "candidates": self.candidates,
            "default_choice": self.default_choice,
            "applied_choice": self.applied_choice,
            "basis": self.basis,
            "accepted": self.accepted,
            "override": self.override,
            "source_dp": self.source_dp,
        }


class Replayer:
    def __init__(self, rules: Rules, decisions: Optional[Dict[str, Any]] = None,
                 ledger_seq_for_id: int = 0):
        self.rules = rules
        self.sim = Simulator(rules)
        self.decisions = decisions or {}
        self.ledger_seq = ledger_seq_for_id  # 区分不同账本版本（源数据改动）
        self.pages_before: Dict[int, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.splits: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []
        self.pid_allocs: List[Dict[str, Any]] = list(self.sim.pid_alloc_events)
        self.dps: List[DecisionPoint] = []
        self.crashes: List[Dict[str, Any]] = []
        self.recoveries: List[Dict[str, Any]] = []
        self.diagnostics: List[Dict[str, Any]] = []
        self.ops_done = 0

    # ---- 稳定 id ----
    def _ev_id(self, op_seq: int, kind: str, tag: str = "") -> str:
        return ids.sid("ev", self.rules.rule_id, self.ledger_seq, op_seq, kind, tag)

    def _dp_id(self, op_seq: int, kind: str) -> str:
        return ids.sid("dp", self.rules.rule_id, self.ledger_seq, op_seq, kind)

    # ---- 页快照 ----
    def _snapshot(self, page: Page) -> Dict[str, Any]:
        enc = page.encode()
        return {
            "pid": page.pid,
            "ptype": "leaf" if page.is_leaf() else "internal",
            "page_lsn": page.page_lsn,
            "unit": "bytes",
            "coordinate_origin": "page_start_offset_0",
            "raw_hex": enc.hex(),
            "ncells": len(page.cells),
            "slots": [
                {
                    "id": f"slot-p{page.pid}-{i:02d}",
                    "index": i,
                    "offset_bytes": c.off,
                    "length_bytes": c.length,
                    "key_hex": c.key.hex(),
                    "key_text": _try_text(c.key),
                    "right_child": c.right_child,
                    "value_hex": c.value.hex() if c.right_child is None else None,
                }
                for i, c in enumerate(sorted(page.cells, key=lambda c: c.key))
            ],
            "free_space_bytes": page.free_space(),
            "free_start_offset_bytes": int.from_bytes(enc[4:6], "big"),
            "slots_end_offset_bytes": int.from_bytes(enc[6:8], "big"),
            "right_sibling": page.right_sibling,
            "leftmost_child": page.leftmost_child,
            "crc_ok": True,
        }

    def _capture_before(self, pids: List[int]) -> Dict[int, Dict[str, Any]]:
        return {pid: self._snapshot(self.sim.get(pid)) for pid in pids if pid in self.sim.disk}

    # ---- 决策点取裁定 ----
    def _resolve(self, dp: DecisionPoint) -> Any:
        prior = self.decisions.get(dp.dp_id)
        if prior is None:
            dp.applied_choice = dp.default_choice
            return dp.default_choice
        dp.accepted = prior.get("accepted")
        dp.override = prior.get("override")
        dp.source_dp = prior.get("source_dp")
        choice = dp.default_choice if prior.get("accepted", True) else prior.get("override")
        dp.applied_choice = choice
        return choice

    # ==================================================================
    # put
    # ==================================================================
    def put(self, op_seq: int, key: bytes, value: bytes) -> None:
        path = self._descend(key)
        leaf_pid = path[-1]
        before = self._capture_before(path)
        leaf = self.sim.get(leaf_pid)
        existing = next((c for c in leaf.cells if c.key == key), None)
        dup_kind = "leaf_cell"
        if existing is not None:
            choice = self._decision_duplicate(op_seq)
            if choice == "forbid":
                self._log_op(op_seq, "put", key, before, status="rejected_duplicate",
                             detail="duplicate policy=forbid")
                return
            if choice == "multiversion":
                # 多版本：同键追加带版本后缀的 cell（逻辑上保留多值）
                suffix = bytes([sum(1 for c in leaf.cells if c.key[:len(key)] == key) + 1])
                leaf.cells.append(Cell(key=key + suffix, value=value))
            else:
                existing.value = value
        else:
            leaf.cells.append(Cell(key=key, value=value))

        split_chain: List[Dict[str, Any]] = []
        if not leaf.fits():
            split_chain = self._split_leaf(leaf_pid, path, op_seq)
        else:
            self.sim.touch_log(leaf, op_seq, "leaf_write", "leaf insert, no split")

        self._writeback_and_record(op_seq, "put", key, before, split_chain)

    def _decision_duplicate(self, op_seq: int) -> str:
        cands = [
            {"choice": self.rules.duplicate_policy,
             "legal": True, "basis": f"当前规则版本 {self.rules.rule_id} 的重复键策略"},
            {"choice": "overwrite", "legal": True, "basis": "同键覆盖旧值"},
            {"choice": "forbid", "legal": True, "basis": "拒绝重复键并保留诊断"},
            {"choice": "multiversion", "legal": True, "basis": "同键多版本共存"},
        ]
        dp = DecisionPoint(
            dp_id=self._dp_id(op_seq, "duplicate"),
            op_seq=op_seq, kind="duplicate",
            candidates=cands,
            default_choice=self.rules.duplicate_policy,
            applied_choice=self.rules.duplicate_policy,
            basis="重复键策略无法由布局唯一决定，保留全部合法候选",
        )
        self.dps.append(dp)
        return self._resolve(dp)

    def _split_leaf(self, leaf_pid: int, path: List[int], op_seq: int) -> List[Dict[str, Any]]:
        leaf = self.sim.get(leaf_pid)
        ordered = sorted(leaf.cells, key=lambda c: c.key)
        cands = leaf_split_candidates(ordered, self.rules.page_size_bytes,
                                      self.rules.split_metric)
        dp = DecisionPoint(
            dp_id=self._dp_id(op_seq, "leaf_split"),
            op_seq=op_seq, kind="leaf_split",
            candidates=cands,
            default_choice=choose_candidate(cands, self.rules.split_choice),
            applied_choice=None,
            basis=(
                "变长键按字节空间而非键数决定分裂；保留全部切点候选，"
                "标注键数度量与字节溢出的非法依据"
            ),
        )
        self.dps.append(dp)
        split_at = self._resolve(dp)

        new_pid = self.sim.alloc_pid()
        lsn = self.sim._new_lsn()
        right = new_leaf(new_pid, lsn, right_sibling=leaf.right_sibling)
        left_cells = ordered[:split_at]
        right_cells = ordered[split_at:]
        leaf.cells = left_cells
        leaf.right_sibling = new_pid
        right.cells = right_cells
        self.sim._install(right)
        self.sim._install(leaf)
        self.pid_allocs.append({"pid": new_pid, "reason": "leaf split", "lsn": lsn})

        left_max = left_cells[-1].key
        right_min = right_cells[0].key
        sep = make_separator(left_max, right_min, self.rules.separator_policy)
        sep_ok = separator_consistent(sep, left_max, right_min, self.rules.separator_policy)
        self.sim.touch_log(right, op_seq, "split_new_leaf", "new leaf full image")
        self.sim.touch_log(leaf, op_seq, "split_old_leaf", "old leaf + sibling link")

        chain = [{
            "level": 0,
            "kind": "leaf_split",
            "left_pid": leaf_pid,
            "right_pid": new_pid,
            "split_at": split_at,
            "separator_hex": sep.hex(),
            "separator_text": _try_text(sep),
            "separator_policy": self.rules.separator_policy,
            "separator_consistent": sep_ok,
            "candidate_count": len(cands),
            "decision_point": dp.dp_id,
            "free_after_left_bytes": leaf.free_space(),
            "free_after_right_bytes": right.free_space(),
        }]
        # 向父层传播
        self._insert_in_parent(path, leaf_pid, new_pid, sep, op_seq, chain)
        self.splits.append({"op_seq": op_seq, "chain": chain})
        return chain

    def _insert_in_parent(self, path: List[int], left_pid: int, right_pid: int,
                          sep: bytes, op_seq: int, chain: List[Dict[str, Any]]) -> None:
        if len(path) == 1:
            self._new_root_internal(left_pid, right_pid, sep, op_seq, chain)
            return
        parent_pid = path[-2]
        parent = self.sim.get(parent_pid)
        parent.cells.append(Cell(key=sep, right_child=right_pid))
        parent.cells.sort(key=lambda c: c.key)
        if parent.fits():
            self.sim.touch_log(parent, op_seq, "internal_insert", "separator insert")
            return
        self._split_internal(parent_pid, path[:-1], op_seq, chain)

    def _split_internal(self, node_pid: int, parent_path: List[int],
                        op_seq: int, chain: List[Dict[str, Any]]) -> None:
        node = self.sim.get(node_pid)
        ordered = sorted(node.cells, key=lambda c: c.key)
        # internal 页同样按字节空间评估
        from .pages import HEADER_END, SLOT_LEN, TRAILER_LEN
        n = len(ordered)
        cands = []
        total = sum(c.encoded_len() for c in ordered)
        for i in range(1, n):
            lb = sum(c.encoded_len() for c in ordered[:i])
            fl = (self.rules.page_size_bytes - TRAILER_LEN - HEADER_END
                  - SLOT_LEN * i - lb)
            fr = (self.rules.page_size_bytes - TRAILER_LEN - HEADER_END
                  - SLOT_LEN * (n - i) - (total - lb))
            cands.append({"split_at": i,
                          "free_space_left_bytes": fl,
                          "free_space_right_bytes": fr,
                          "legal": fl >= 0 and fr >= 0,
                          "metric": "bytes"})
        dp = DecisionPoint(
            dp_id=self._dp_id(op_seq, f"internal_split_p{node_pid}"),
            op_seq=op_seq, kind="internal_split",
            candidates=cands,
            default_choice=choose_candidate(cands, self.rules.split_choice),
            applied_choice=None,
            basis="内部节点分裂同样按字节空间选择切点",
        )
        self.dps.append(dp)
        i = self._resolve(dp)
        # 提升键 = 切点前一个 separator（键从分裂节点移除、提升到父层）。
        # 左新节点保留它之前的 separator；右新节点的 leftmost 指向
        # 被提升 separator 原来的右孩子，右侧 separator 整体后移。
        pivot = ordered[i - 1]
        promoted = pivot.key
        left_seps = ordered[: i - 1]
        right_seps = ordered[i:]
        right_leftmost = pivot.right_child
        new_pid = self.sim.alloc_pid()
        lsn = self.sim._new_lsn()
        right_node = new_internal(new_pid, lsn, right_leftmost)
        right_node.cells = right_seps
        node.cells = left_seps
        self.sim._install(right_node)
        self.sim._install(node)
        self.pid_allocs.append({"pid": new_pid, "reason": "internal split", "lsn": lsn})
        self.sim.touch_log(right_node, op_seq, "split_new_internal", "promote child")
        self.sim.touch_log(node, op_seq, "split_old_internal", "retained child")
        chain.append({
            "level": len(parent_path),
            "kind": "internal_split",
            "left_pid": node_pid,
            "right_pid": new_pid,
            "promoted_separator_hex": promoted.hex(),
            "promoted_separator_text": _try_text(promoted),
            "decision_point": dp.dp_id,
        })
        self._insert_in_parent(parent_path, node_pid, new_pid, promoted, op_seq, chain)

    def _new_root_internal(self, left_pid: int, right_pid: int, sep: bytes,
                           op_seq: int, chain: List[Dict[str, Any]]) -> None:
        new_pid = self.sim.alloc_pid()
        lsn = self.sim._new_lsn()
        root = new_internal(new_pid, lsn, left_pid)
        root.cells = [Cell(key=sep, right_child=right_pid)]
        self.sim._install(root)
        old_root = self.sim.root_pid
        self.sim.root_pid = new_pid
        self.pid_allocs.append({"pid": new_pid, "reason": "new root (height +1)", "lsn": lsn})
        self.sim.touch_log(root, op_seq, "root_replace",
                           f"root leaf {old_root} -> internal root {new_pid}")
        chain.append({
            "level": 1,
            "kind": "root_split",
            "old_root_pid": old_root,
            "new_root_pid": new_pid,
            "separator_hex": sep.hex(),
            "height_change": "+1",
        })

    # ==================================================================
    # delete
    # ==================================================================
    def delete(self, op_seq: int, key: bytes) -> None:
        path = self._descend(key)
        leaf_pid = path[-1]
        before = self._capture_before(path)
        leaf = self.sim.get(leaf_pid)
        leaf.cells = [c for c in leaf.cells if c.key != key]
        self.sim.touch_log(leaf, op_seq, "leaf_write", "delete")
        empty_action = None
        if not leaf.cells and leaf_pid != self.sim.root_pid:
            empty_action = self._decision_empty_leaf(op_seq, leaf_pid, path)
            if empty_action == "deallocate":
                # 从父层摘掉指针，页保留在磁盘上作为已释放镜像（不再可达）
                self._deallocate_leaf(leaf_pid, path, op_seq)
        self._writeback_and_record(op_seq, "delete", key, before, [],
                                   extra={"empty_leaf_action": empty_action})

    def _decision_empty_leaf(self, op_seq: int, leaf_pid: int) -> str:
        cands = [
            {"choice": self.rules.empty_leaf_policy, "legal": True,
             "basis": f"当前规则 {self.rules.rule_id} 的空页策略"},
            {"choice": "keep", "legal": True,
             "basis": "保留可达空 leaf：0 slot 是合法结构状态"},
            {"choice": "deallocate", "legal": True,
             "basis": "摘除父指针；页镜像仍保留但不可达"},
        ]
        dp = DecisionPoint(
            dp_id=self._dp_id(op_seq, "empty_leaf"),
            op_seq=op_seq, kind="empty_leaf",
            candidates=cands,
            default_choice=self.rules.empty_leaf_policy,
            applied_choice=self.rules.empty_leaf_policy,
            basis="删除后空页的处置策略保留全部合法候选",
        )
        self.dps.append(dp)
        return self._resolve(dp)

    def _deallocate_leaf(self, leaf_pid: int, path: List[int], op_seq: int) -> None:
        if len(path) < 2:
            return
        parent = self.sim.get(path[-2])
        if parent.leftmost_child == leaf_pid:
            # 左most 摘除：用第一个 cell 的 right_child 顶替，separator 随之删除
            if parent.cells:
                parent.leftmost_child = parent.cells[0].right_child
                parent.cells = parent.cells[1:]
        else:
            parent.cells = [c for c in parent.cells if c.right_child != leaf_pid]
        self.sim.touch_log(parent, op_seq, "internal_delete", "drop empty leaf pointer")

    # ==================================================================
    # 路径 / 写回 / 崩溃 / 恢复
    # ==================================================================
    def _descend(self, key: bytes) -> List[int]:
        path: List[int] = []
        pid = self.sim.root_pid
        while True:
            path.append(pid)
            page = self.sim.get(pid)
            if page.is_leaf():
                return path
            pid = page.find_leaf(key)

    def _writeback_and_record(self, op_seq: int, opname: str, key: bytes,
                              before: Dict[int, Dict[str, Any]],
                              split_chain: List[Dict[str, Any]],
                              extra: Optional[Dict[str, Any]] = None) -> None:
        written, crash_info = self.sim.writeback(op_seq, crash=self._pending_crash)
        self._pending_crash = None
        for r in self.sim.wal:
            if r.op_seq == op_seq:
                self.logs.append(r.view())
        inv = self.invariants()
        status = "crashed" if crash_info else "applied"
        ev = {
            "id": self._ev_id(op_seq, opname),
            "seq": op_seq,
            "op": opname,
            "key_hex": key.hex(),
            "key_text": _try_text(key),
            "before_images": [before[pid] for pid in sorted(before)],
            "split_chain": split_chain,
            "wal_written": written,
            "crash": crash_info,
            "invariants_after": inv,
            "status": status,
            "extra": extra or {},
        }
        self.events.append(ev)
        if crash_info:
            self.crashes.append(crash_info)

    def crash(self, op_seq: int, spec: Dict[str, Any]) -> None:
        # 崩溃点附着到“下一条数据操作”的写回阶段
        self._pending_crash = spec
        self.events.append({
            "id": self._ev_id(op_seq, "crash-arm"),
            "seq": op_seq, "op": "crash",
            "spec": spec, "status": "armed",
        })

    def recover(self, op_seq: int, rnd: int) -> None:
        info = self.sim.recover()
        info["round"] = rnd
        info["op_seq"] = op_seq
        info["id"] = self._ev_id(op_seq, "recover", f"r{rnd}")
        self.recoveries.append(info)
        self.events.append({
            "id": info["id"], "seq": op_seq, "op": "recover",
            "round": rnd, "recovery": info, "status": "recovered",
            "invariants_after": self.invariants(),
        })

    # ==================================================================
    # 结构不变量（机器可读取的判定）
    # ==================================================================
    def invariants(self) -> List[Dict[str, Any]]:
        checks: List[Dict[str, Any]] = []
        sim = self.sim
        reachable = self._reachable()

        # I1 每页 free space >= 0（无溢出），且编码 CRC 有效
        overflow, bad_crc = [], []
        for pid in reachable:
            if pid not in sim.disk:
                continue
            page = sim.get(pid)
            if not page.fits():
                overflow.append(pid)
            raw = sim.disk[pid]
            from .pages import _crc16
            stored = int.from_bytes(raw[126:128], "big")
            if _crc16(raw[:126]) != stored:
                bad_crc.append(pid)
        checks.append(self._chk("I1", "每页 free space(byte)>=0 且 CRC16 有效",
                                not overflow and not bad_crc,
                                {"overflow_pids": overflow, "bad_crc_pids": bad_crc}))

        # I2 可达性：从根出发无悬空指针；无可达半页
        dangling = [pid for pid in reachable if pid not in sim.disk]
        checks.append(self._chk("I2", "无悬空子指针，无可达半页(torn)",
                                not dangling and not bad_crc,
                                {"dangling_pids": dangling,
                                 "reachable_torn_pids": bad_crc}))

        # I3 键序与 separator 覆盖
        sep_violations, link_violations = self._order_checks(reachable)
        checks.append(self._chk(
            "I3", "internal separator 覆盖规则一致；leaf sibling 链键序单调",
            not sep_violations and not link_violations,
            {"separator_violations": sep_violations,
             "sibling_link_violations": link_violations}))

        # I4 sibling link：可达 leaf 通过 right_sibling 串成链，无环、无回跳
        checks.append(self._chk("I4", "leaf sibling 链一致（无环/无重复）",
                                not link_violations, {"detail": link_violations}))

        # I5 恢复幂等：最近一次 recovery 若无日志重放则稳定
        last_rec = self.recoveries[-1] if self.recoveries else None
        idem = True if last_rec is None else (
            len(last_rec["logs_applied"]) == 0 if last_rec.get("round", 1) >= 2
            else last_rec["reachable_torn_after_recovery"] == [])
        checks.append(self._chk(
            "I5", "重放两次后重复恢复幂等；崩溃点不留下可达半页",
            idem, {"last_recovery": last_rec}))
        return checks

    def _chk(self, cid: str, desc: str, ok: bool, detail: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": ids.sid("chk", self.rules.rule_id, cid,
                              len(self.events), len(self.recoveries)),
                "check": cid, "description": desc, "holds": bool(ok), "detail": detail}

    def _reachable(self) -> List[int]:
        seen: set = set()
        root = self.sim.root_pid
        if root is None:
            return []
        stack = [root]
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            if pid not in self.sim.pool and pid not in self.sim.disk:
                continue
            page = self.sim.get(pid)
            if not page.is_leaf():
                stack.extend(page.child_pids())
        return sorted(seen)

    def _order_checks(self, reachable: List[int]) -> Tuple[list, list]:
        sep_violations, link_violations = [], []
        root = self.sim.root_pid
        leaves: List[Page] = []

        def walk(pid: int, lo: Optional[bytes], hi: Optional[bytes]) -> None:
            page = self.sim.get(pid)
            if page.is_leaf():
                leaves.append(page)
                keys = [c.key for c in sorted(page.cells, key=lambda c: c.key)]
                if keys != sorted(keys):
                    sep_violations.append({"pid": pid, "reason": "leaf keys unsorted"})
                if lo is not None and keys and not keys[0] >= lo:
                    sep_violations.append({"pid": pid, "reason": "key below lower bound"})
                if hi is not None and keys and not keys[-1] < hi:
                    sep_violations.append({"pid": pid, "reason": "key at/above upper bound"})
                return
            bounds = []
            seps = [c.key for c in sorted(page.cells, key=lambda c: c.key)]
            children = page.child_pids()
            for j, child in enumerate(children):
                child_lo = lo if j == 0 else seps[j - 1]
                child_hi = hi if j == len(children) - 1 else seps[j]
                walk(child, child_lo, child_hi)

        if root is not None and root in self.sim.disk:
            walk(root, None, None)

        seen_pids: set = set()
        by_pid = {p.pid: p for p in leaves}
        chain_start = [p.pid for p in leaves
                       if not any(q.right_sibling == p.pid for q in leaves)]
        heads = chain_start or [leaves[0].pid if leaves else None]
        visited: set = set()
        cur = heads[0] if heads else None
        order: List[int] = []
        while cur is not None:
            if cur in visited:
                link_violations.append({"reason": "cycle", "pid": cur})
                break
            visited.add(cur)
            order.append(cur)
            cur = by_pid.get(cur).right_sibling if cur in by_pid else None
        if len(visited) != len(leaves) and leaves:
            link_violations.append({
                "reason": "sibling chain does not cover all reachable leaves",
                "chain": order, "leaves": [p.pid for p in leaves]})
        for i in range(len(order) - 1):
            a, b = by_pid[order[i]], by_pid[order[i + 1]]
            if a.cells and b.cells and max(c.key for c in a.cells) >= min(c.key for c in b.cells):
                link_violations.append({"reason": "non-monotonic sibling keys",
                                        "pids": [a.pid, b.pid]})
        return sep_violations, link_violations

    # ==================================================================
    # 重放入口与视图
    # ==================================================================
    def run(self, parsed: List[ParsedOp]) -> Dict[str, Any]:
        self._pending_crash = None
        for p in parsed:
            if p.diagnostic:
                self.diagnostics.append({"seq": p.seq, "raw": p.raw,
                                         "diagnostic": p.diagnostic})
                self.events.append({"id": self._ev_id(p.seq, "unconvertible"),
                                    "seq": p.seq, "op": "unconvertible",
                                    "raw": p.raw, "diagnostic": p.diagnostic,
                                    "status": "left_in_place"})
                continue
            op = p.op
            if op["op"] == "put":
                self.put(p.seq, op["key"], op["value"])
            elif op["op"] == "delete":
                self.delete(p.seq, op["key"])
            elif op["op"] == "crash":
                self.crash(p.seq, {"after_logs": op["after_logs"],
                                   "torn_page": op["torn_page"]})
            elif op["op"] == "recover":
                self.recover(p.seq, op["round"])
        return self.result()

    def result(self) -> Dict[str, Any]:
        final = {}
        for pid in sorted(self.sim.disk):
            if pid == 0:
                continue
            page = self.sim.get(pid)
            final[pid] = self._snapshot(page)
            final[pid]["on_disk_crc_ok"] = not decode_page(
                pid, self.sim.disk[pid]).torn
        return {
            "rules": self.rules.asdict(),
            "ledger_seq": self.ledger_seq,
            "root_pid": self.sim.root_pid,
            "next_pid": self.sim.next_pid,
            "tree": self._tree_view(),
            "events": self.events,
            "decision_points": [d.view() for d in self.dps],
            "splits": self.splits,
            "logs": self.logs,
            "wal": [r.view() for r in self.sim.wal],
            "pid_allocations": self.pid_allocs,
            "crashes": self.crashes,
            "recoveries": self.recoveries,
            "diagnostics": self.diagnostics,
            "final_invariants": self.invariants(),
            "final_pages": final,
        }

    def _tree_view(self) -> Dict[str, Any]:
        def node(pid: int, depth: int) -> Optional[Dict[str, Any]]:
            page = self.sim.get(pid)
            return {
                "id": f"node-p{pid}",
                "pid": pid,
                "depth": depth,
                "kind": "leaf" if page.is_leaf() else "internal",
                "keys": [{"id": f"key-p{pid}-{i:02d}",
                          "text": _try_text(c.key), "hex": c.key.hex()}
                         for i, c in enumerate(sorted(page.cells, key=lambda c: c.key))],
                "right_sibling": page.right_sibling,
                "free_space_bytes": page.free_space(),
                "page_lsn": page.page_lsn,
                "children": [] if page.is_leaf() else [
                    n for n in (node(ch, depth + 1) for ch in page.child_pids()) if n],
            }
        root = self.sim.root_pid
        return {"id": "tree-root", "root": node(root, 0) if root is not None else None}


def _try_text(b: bytes) -> str:
    try:
        t = b.decode("utf-8")
        if all(32 <= ord(ch) < 127 or ch in "\t" for ch in t):
            return t
    except UnicodeDecodeError:
        pass
    return "hex:" + b.hex()


def replay(ledger: List[Dict[str, Any]], rules: Rules,
           decisions: Optional[Dict[str, Any]] = None,
           ledger_seq: int = 0) -> Dict[str, Any]:
    parsed = parse_ledger(ledger)
    return Replayer(rules, decisions, ledger_seq).run(parsed)
