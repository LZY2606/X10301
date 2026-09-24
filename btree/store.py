"""页面存储 + 事务 + 写中断注入。

磁盘是 ``pid -> 页字节`` 的字典；所有修改先写 WAL（含页前镜像），再落盘，
最后写提交记录。崩溃点由测试显式制造，不依赖时序运气。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from . import paging
from .wal import WAL


@dataclass
class PageStore:
    page_size: int
    pages: dict[int, bytes] = field(default_factory=dict)
    wal: WAL = field(default_factory=WAL)
    root_pid: int = -1
    next_pid: int = 0
    # 活跃事务的私有暂存：pid -> 磁盘上尚未被该事务覆盖的前镜像
    _active: dict[int, dict[int, bytes | None]] = field(default_factory=dict)

    # ---- 基本分配与导航 -------------------------------------------------
    def allocate(self, txn: int) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.wal.alloc(txn, pid)
        return pid

    def write(self, txn: int, pid: int, after: bytes) -> None:
        before = self.pages.get(pid)
        self._active.setdefault(txn, {}).setdefault(pid, before)
        self.wal.write_page(txn, pid, before, after)
        self.pages[pid] = after

    def raw_write_no_wal(self, pid: int, blob: bytes) -> None:
        self.pages[pid] = blob

    def set_root(self, txn: int, pid: int) -> None:
        self.wal.root_set(txn, pid)
        self.root_pid = pid

    def begin(self) -> int:
        txn = 1 + max([0] + list(self._active.keys()))
        self._active[txn] = {}
        self.wal.begin(txn)
        return txn

    def commit(self, txn: int) -> int:
        lsn = self.wal.commit(txn)
        self._active.pop(txn, None)
        return lsn

    def abort(self, txn: int) -> int:
        # 逆序用前镜像回滚
        for pid, before in reversed(list(self._active.get(txn, {}).items())):
            if before is None:
                self.pages.pop(pid, None)
            else:
                self.pages[pid] = before
        lsn = self.wal.abort(txn)
        self._active.pop(txn, None)
        return lsn

    def reachable_pids(self) -> set[int]:
        seen: set[int] = set()
        stack = [self.root_pid] if self.root_pid >= 0 else []
        while stack:
            pid = stack.pop()
            if pid in seen or pid not in self.pages:
                continue
            seen.add(pid)
            page = self.pages[pid]
            h = paging.header(page)
            if h["ptype"] == paging.INTERNAL:
                if h["first_child"] >= 0:
                    stack.append(h["first_child"])
                for cell in paging.cells(page):
                    if cell.first_child >= 0:
                        stack.append(cell.first_child)
            else:
                nxt = h["next_leaf"]
                if nxt >= 0:
                    stack.append(nxt)
        return seen

    def snapshot(self) -> tuple[dict[int, bytes], int, int, bytes]:
        return (
            dict(self.pages),
            self.root_pid,
            self.next_pid,
            bytes(self.wal.buf),
        )
