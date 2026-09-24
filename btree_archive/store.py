"""磁盘页 + WAL + 写页中断 + 崩溃恢复。

磁盘模型：每个页是定长文件块（page_size byte），0 号块保存元信息
(root_pid, next_pid)。WAL 只追加，记录类型：

- W  = page write：{pid, lsn, after(全页后镜像), before(本事务内页前镜像)}
- M  = meta write（page id 分配器/根变化）
- C  = txn commit
- A  = txn abort
- X  = CLR（补偿日志，undoNextLSN 指向下一条待 undo 的 W）

崩溃点：
- CP_COMMIT_FLUSH：全部页写完、commit 落盘前崩溃 → 恢复时回滚（undo）。
- CP_TORN_SPLIT：分裂新页落盘时制造半页（torn）→ 恢复必须消除该可达性风险。
- CP_AFTER_REDO？ 我们通过“恢复重放两次”验证幂等。

关键保证（结构不变量 I3）：
分析阶段按 LSN 建 winner/loser 表；redo 阶段只重放 page.lsn < rec.lsn 的
已提交更新；undo 阶段逆序处理 loser 事务，写回 before 并追加 CLR。
因此第二次重放：已回滚 W 由 CLR 跳过，已重做页由 page.lsn 跳过 —— 幂等。
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

W = "W"
M = "M"
C = "C"
A = "A"
X = "X"

CP_COMMIT_FLUSH = "commit_flush"
CP_TORN_SPLIT = "torn_split"


class Crash(Exception):
    def __init__(self, crash_point: str, torn_pages: List[int], lsn: int):
        super().__init__(f"crash@{crash_point} lsn={lsn} torn={torn_pages}")
        self.crash_point = crash_point
        self.torn_pages = torn_pages
        self.lsn = lsn


@dataclass
class WriteRec:
    kind: str
    lsn: int
    txn: str
    pid: int = 0
    after_hex: str = ""
    before_hex: str = ""
    root_pid: int = 0
    next_pid: int = 0
    prev_lsn: int = 0      # 同一事务内上一条记录的 LSN（undo 链）
    undo_next: int = 0     # CLR：下一步 undo 的 LSN
    before_root_pid: int = 0
    before_next_pid: int = 0


@dataclass
class RecoverReport:
    winner_txns: List[str] = field(default_factory=list)
    loser_txns: List[str] = field(default_factory=list)
    redone: List[int] = field(default_factory=list)      # 重放的 W.lsn
    undone: List[int] = field(default_factory=list)      # 回滚的 W.lsn
    clrs: List[int] = field(default_factory=list)
    torn_detected: List[int] = field(default_factory=list)
    reachable_after: List[int] = field(default_factory=list)


def _meta_block(root_pid: int, next_pid: int, page_size: int) -> bytes:
    buf = bytearray(page_size)
    buf[0:4] = b"META"
    struct.pack_into(">II", buf, 4, root_pid, next_pid)
    return bytes(buf)


def _read_meta(raw: bytes) -> Tuple[int, int]:
    if raw[:4] != b"META":
        return 0, 0
    return struct.unpack_from(">II", raw, 4)


class Disk:
    def __init__(self, path: str, page_size: int, create: bool = False):
        self.path = path
        self.page_size = page_size
        self.pages_file = os.path.join(path, "pages.bin")
        self.wal_file = os.path.join(path, "wal.jsonl")
        self.next_lsn = 1
        if create:
            os.makedirs(path, exist_ok=True)
            with open(self.pages_file, "wb") as fh:
                fh.write(_meta_block(1, 2, page_size))
            open(self.wal_file, "wb").close()
        self._blocks: Dict[int, bytes] = {}
        self._load_blocks()
        records = self._read_wal()
        if records:
            self.next_lsn = 1 + max(r.lsn for r in records)

    # ---------- 底层块 IO ----------
    def _load_blocks(self) -> None:
        if not os.path.exists(self.pages_file):
            return
        with open(self.pages_file, "rb") as fh:
            data = fh.read()
        for i in range(0, len(data), self.page_size):
            self._blocks[i // self.page_size] = bytes(data[i:i + self.page_size])

    def read_block(self, idx: int) -> Optional[bytes]:
        return self._blocks.get(idx)

    def _write_block_raw(self, idx: int, raw: bytes, torn: bool = False) -> None:
        if torn:
            # 半页：仅前半段写入，其余置 0 —— 模拟写页中断
            half = self.page_size // 2
            raw = raw[:half] + b"\x00" * (self.page_size - half)
        assert len(raw) == self.page_size
        self._blocks[idx] = raw
        with open(self.pages_file, "r+b") as fh:
            fh.seek(idx * self.page_size)
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())

    def _read_wal(self) -> List[WriteRec]:
        if not os.path.exists(self.wal_file):
            return []
        out: List[WriteRec] = []
        with open(self.wal_file, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(WriteRec(**json.loads(line)))
        return out

    def _append_wal(self, rec: WriteRec) -> None:
        with open(self.wal_file, "a") as fh:
            fh.write(json.dumps(asdict(rec), separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def alloc_lsn(self) -> int:
        lsn = self.next_lsn
        self.next_lsn += 1
        return lsn

    def read_meta(self) -> Tuple[int, int]:
        return _read_meta(self._blocks[0])

    # ---------- 事务性 flush ----------
    def flush_txn(self, txn: str, writes: List[dict], root_pid: int,
                 next_pid: int, crash_point: Optional[str] = None
                 ) -> None:
        """writes: [{pid, after(bytes), before(bytes or None), is_new}]。

        写盘顺序（WAL-first）：
          1) 全部 W 记录追加并 fsync；
          2) 按写计划落页；新分配页先落，父/根最后落（保证不会留下可达半页）；
          3) meta（page id 分配 + 根变化）落盘；
          4) commit 记录落盘。
        """
        prev = 0
        recs: List[WriteRec] = []
        for w in writes:
            lsn = self.alloc_lsn()
            w["lsn"] = lsn
            rec = WriteRec(kind=W, lsn=lsn, txn=txn, pid=w["pid"],
                           after_hex="",
                           before_hex=(w["before"].hex() if w["before"] else ""),
                           prev_lsn=prev)
            prev = lsn
            recs.append(rec)
        # 后镜像必须携带本页 pageLSN（redo 判据），LSN 先分配后序列化
        for w, rec in zip(writes, recs):
            w["after"] = w["embed_lsn"](w["lsn"])
            rec.after_hex = w["after"].hex()
            self._append_wal(rec)
        old_root, old_next = self.read_meta()
        meta_lsn = self.alloc_lsn()
        meta_rec = WriteRec(kind=M, lsn=meta_lsn, txn=txn,
                            root_pid=root_pid, next_pid=next_pid,
                            prev_lsn=prev,
                            before_root_pid=old_root, before_next_pid=old_next)
        self._append_wal(meta_rec)

        # 新页先落（此时无父页指向它，是孤儿）；已有页（父/根）最后落，
        # 因此任何写页中断时刻都不存在“可达半页”（结构不变量 I3）。
        ordered = sorted(writes, key=lambda w: 1 if not w.get("is_new") else 0)
        for w in ordered:
            torn = (crash_point == CP_TORN_SPLIT and w.get("is_new"))
            if torn:
                self._write_block_raw(w["pid"], w["after"], torn=True)
                raise Crash(CP_TORN_SPLIT, [w["pid"]], w["lsn"])
            self._write_block_raw(w["pid"], w["after"])
        self._write_block_raw(0, _meta_block(root_pid, next_pid, self.page_size))

        if crash_point == CP_COMMIT_FLUSH:
            raise Crash(CP_COMMIT_FLUSH, [], meta_lsn)
        commit = WriteRec(kind=C, lsn=self.alloc_lsn(), txn=txn,
                          prev_lsn=meta_lsn)
        self._append_wal(commit)

    # ---------- 崩溃恢复 ----------
    def recover(self) -> RecoverReport:
        """分析 → redo → undo，返回报告。再次调用必须幂等。"""
        report = RecoverReport()
        records = self._read_wal()

        # 分析：winner = 有 C；loser = 其余事务。已存在 A/X 链的记录幂等跳过。
        txns: Dict[str, List[WriteRec]] = {}
        committed: Dict[str, bool] = {}
        compensated: set = set()
        for rec in records:
            if rec.txn:
                txns.setdefault(rec.txn, []).append(rec)
            if rec.kind == C:
                committed[rec.txn] = True
            if rec.kind in (X, A):
                compensated.add(rec.undo_next) if rec.kind == X and rec.undo_next else None
        for txn in txns:
            (report.winner_txns if committed.get(txn) else report.loser_txns).append(txn)

        # redo（仅 winner 的 W/M，page.lsn 判据使其天然幂等）
        for rec in records:
            if rec.kind not in (W, M) or not committed.get(rec.txn):
                continue
            if rec.kind == W:
                raw = bytes.fromhex(rec.after_hex)
                current = self.read_block(rec.pid)
                cur_lsn = _page_lsn_or_zero(current)
                if cur_lsn is None:
                    # 半页/不可解析：记录检测，仍由 redo 覆盖
                    if rec.pid not in report.torn_detected:
                        report.torn_detected.append(rec.pid)
                if cur_lsn is None or cur_lsn < rec.lsn:
                    self._write_block_raw(rec.pid, raw)
                    report.redone.append(rec.lsn)
            else:
                root, nxt = self.read_meta()
                if (root, nxt) != (rec.root_pid, rec.next_pid):
                    self._write_block_raw(
                        0, _meta_block(rec.root_pid, rec.next_pid, self.page_size))

        # undo（loser 逆序；CLR 幂等 + undoNextLSN 链）
        already_clr: set = set()
        for rec in records:
            if rec.kind == X:
                already_clr.add(rec.undo_next)
        for txn in report.loser_txns:
            chain = [r for r in txns[txn] if r.kind in (W, M)]
            next_undo = chain[-1].lsn if chain else 0
            wrote_abort = any(r.kind == A for r in txns[txn])
            while next_undo:
                rec = next(r for r in records if r.lsn == next_undo)
                nxt = rec.prev_lsn
                if rec.lsn not in already_clr:
                    if rec.kind == W:
                        if rec.before_hex:
                            before = bytes.fromhex(rec.before_hex)
                            self._write_block_raw(rec.pid, before)
                        else:
                            # 本事务新分配页：恢复后标记为零页（不可达、不可解析为树页）
                            self._write_block_raw(rec.pid, b"\x00" * self.page_size)
                        report.undone.append(rec.lsn)
                    else:
                        self._write_block_raw(
                            0, _meta_block(rec.before_root_pid,
                                           rec.before_next_pid, self.page_size))
                        report.undone.append(rec.lsn)
                    clr_lsn = self.alloc_lsn()
                    self._append_wal(WriteRec(
                        kind=X, lsn=clr_lsn, txn=txn, undo_next=rec.lsn))
                    report.clrs.append(clr_lsn)
                    already_clr.add(rec.lsn)
                next_undo = nxt
            if not wrote_abort:
                self._append_wal(WriteRec(
                    kind=A, lsn=self.alloc_lsn(), txn=txn))

        root, nxt = self.read_meta()
        report.reachable_after = self._reachable_blocks(root)
        return report

    def _reachable_blocks(self, root: int) -> List[int]:
        from .pages import INTERNAL, LEAF, Page
        seen: set = set()
        stack = [root]
        while stack:
            pid = stack.pop()
            if pid in seen or pid == 0:
                continue
            raw = self.read_block(pid)
            if raw is None or raw[0] not in (LEAF, INTERNAL):
                continue
            seen.add(pid)
            try:
                page = Page.parse(raw)
            except Exception:
                continue
            if not page.is_leaf:
                stack.extend(page.child_ids())
        return sorted(seen)


def _page_lsn_or_zero(raw: Optional[bytes]):
    from .pages import INTERNAL, LEAF
    if raw is None or not raw or raw[0] not in (LEAF, INTERNAL):
        return None
    # torn 检测：全零尾或 free_end 明显越界视为不可解析
    if raw[0] == LEAF:
        lsn = struct.unpack(">I", raw[17:21])[0]
        free_end = struct.unpack(">H", raw[21:23])[0]
        ok = LEAF <= free_end <= len(raw) or free_end == 0
    else:
        lsn = struct.unpack(">I", raw[13:17])[0]
        free_end = struct.unpack(">H", raw[17:19])[0]
        ok = INTERNAL <= free_end <= len(raw) or free_end == 0
    # 半页特征：后半段全 0 但头部声称有内容
    half = len(raw) // 2
    if raw[half:] == b"\x00" * (len(raw) - half) and free_end > half:
        return None
    return lsn if ok else None
