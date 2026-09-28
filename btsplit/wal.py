"""WAL 记录与页池。

* 每次页修改先记录 before-image（页前镜像）与 after 内容，赋予单调 LSN。
* 写页步骤可被 crash 计划中断：中断时指定页被写成半页。
* recovery 重放 WAL 到页池；重复重放必须幂等。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .pages import Entry, Page, ParsedPage, PAGE_HEADER_SIZE, parse_page

META_PAGE_ID = 0
FIRST_USER_PAGE = 1


@dataclass
class WalRecord:
    lsn: int
    page_id: int
    before_hex: str | None
    after_hex: str
    reason: str
    txn: str

    def to_dict(self) -> dict:
        return {
            "lsn": self.lsn,
            "page_id": self.page_id,
            "before_hex": self.before_hex,
            "after_hex": self.after_hex,
            "reason": self.reason,
            "txn": self.txn,
        }


@dataclass
class AllocationRecord:
    seq: int
    page_id: int
    page_type: str
    reason: str

    def to_dict(self) -> dict:
        return {"seq": self.seq, "page_id": self.page_id,
                "page_type": self.page_type, "reason": self.reason}


class WriteInterrupted(Exception):
    """模拟的写页中断（崩溃点）。"""


class Pool:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.pages: dict[int, bytes] = {}
        self.wal: list[WalRecord] = []
        self.allocations: list[AllocationRecord] = []
        self.next_page_id = FIRST_USER_PAGE
        self.torn_pages: set[int] = set()
        # crash_plan(tick, page_id, reason) -> None|'torn'：返回 'torn' 时中断
        self.crash_plan: Callable[[int, int, str], str | None] | None = None
        self._tick = 0
        self.crashed_at: dict | None = None

    # ---- 分配 ----
    def allocate(self, page_type: int, reason: str) -> Page:
        pid = self.next_page_id
        self.next_page_id += 1
        self.allocations.append(AllocationRecord(
            len(self.allocations), pid,
            "leaf" if page_type == 1 else "internal", reason))
        return Page(page_id=pid, page_type=page_type, page_size=self.page_size)

    # ---- WAL + 写 ----
    def write(self, page: Page, reason: str, txn: str = "T") -> None:
        before = self.pages.get(page.page_id)
        lsn = len(self.wal) + 1
        page.lsn = lsn & 0xFFFF
        after = page.to_bytes()
        self.wal.append(WalRecord(
            lsn=lsn, page_id=page.page_id,
            before_hex=before.hex() if before is not None else None,
            after_hex=after.hex(), reason=reason, txn=txn,
        ))
        self._tick += 1
        action = None
        if self.crash_plan is not None:
            action = self.crash_plan(self._tick, page.page_id, reason)
        if action == "torn":
            torn = page.to_bytes(torn_guard=True)
            self.pages[page.page_id] = torn
            self.torn_pages.add(page.page_id)
            self.crashed_at = {"tick": self._tick, "page_id": page.page_id,
                               "reason": reason, "lsn": lsn}
            raise WriteInterrupted(
                f"写页中断 tick={self._tick} page={page.page_id} ({reason})")
        self.pages[page.page_id] = after
        self.torn_pages.discard(page.page_id)

    # ---- 读 ----
    def read(self, page_id: int) -> ParsedPage:
        return parse_page(self.pages[page_id], self.page_size, page_id)

    def has(self, page_id: int) -> bool:
        return page_id in self.pages

    def snapshot_bytes(self) -> dict[int, bytes]:
        return dict(self.pages)

    # ---- 恢复：把 WAL 中所有 after 重放（redo-only）----
    def replay_wal(self, committed_lsns: set[int] | None = None) -> dict:
        applied = 0
        for rec in self.wal:
            if committed_lsns is not None and rec.lsn not in committed_lsns:
                continue
            self.pages[rec.page_id] = bytes.fromhex(rec.after_hex)
            applied += 1
        self.torn_pages = {pid for pid, raw in self.pages.items()
                           if parse_page(raw, self.page_size, pid).torn}
        return {"applied": applied, "wal_records": len(self.wal)}

    def state_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for pid in sorted(self.pages):
            h.update(str(pid).encode())
            h.update(self.pages[pid])
        return h.hexdigest()[:16]

    def reachable(self, root_page_id: int) -> dict[int, ParsedPage]:
        """从根 BFS 得到可达页面；半页也包含并标记 torn。"""
        out: dict[int, ParsedPage] = {}
        queue = [root_page_id]
        seen = set()
        while queue:
            pid = queue.pop(0)
            if pid in seen:
                continue
            seen.add(pid)
            parsed = self.read(pid)
            out[pid] = parsed
            if parsed.page is None:
                continue
            if parsed.page.page_type == 2:  # internal
                child = struct_child(parsed.page)
                if child is not None:
                    queue.append(child)
                for entry in parsed.page.entries[1 if child is not None else 0:]:
                    queue.append(struct_unpack_child(entry.value))
            else:  # leaf：沿 sibling link
                if parsed.page.next_page:
                    queue.append(parsed.page.next_page)
        return out


def struct_child(page: Page) -> int | None:
    if page.leftmost_child is not None:
        return page.leftmost_child
    if page.entries:
        return struct_unpack_child(page.entries[0].value)
    return None


def struct_unpack_child(value: bytes) -> int:
    import struct
    return struct.unpack(">H", value)[0]


def pack_child(pid: int) -> bytes:
    import struct
    return struct.pack(">H", pid)
