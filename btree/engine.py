"""固定页大小的 B+tree 引擎。

特性（全部被案例包覆盖）：
- 变长键，分裂按 *字节空间* 决定，而不是键数；所有合法分裂点作为候选保留，
  系统记录采纳与未采纳的结果。
- separator 规则有两种且复制/提升口径显式一致：
  ``copy``（经典 B+tree，叶子保留全部键，分隔键复制到上层）与
  ``promote``（提升键独占在内部节点，叶子范围为严格区间）。
- 根分裂提升树高；叶子分裂可向父节点传播成级联分裂。
- 重复键策略 ``last_wins`` / ``first_wins``。
- 删除后空页：``keep_empty_leaf``（保留）或 ``remove_empty_leaf``（合并摘除）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import paging
from .store import PageStore
from .trace import (
    Tracer,
    OP,
    PAGE_ALLOC,
    SPLIT_CANDIDATE,
    SPLIT_DECISION,
    SPLIT_PROPAGATE,
    ROOT_SPLIT,
    ROOT_CHANGE,
    DUP_KEY,
    PAGE_EMPTY,
    PAGE_VIEW,
    SIBLING_LINK,
    SEPARATOR_RULE,
)

COPY = "copy"
PROMOTE = "promote"
LAST_WINS = "last_wins"
FIRST_WINS = "first_wins"
KEEP_EMPTY = "keep_empty_leaf"
REMOVE_EMPTY = "remove_empty_leaf"


@dataclass
class EngineConfig:
    page_size: int = 256
    separator_rule: str = COPY
    duplicate_policy: str = LAST_WINS
    empty_leaf_policy: str = KEEP_EMPTY

    def version_key(self) -> dict:
        return {
            "page_size": self.page_size,
            "separator_rule": self.separator_rule,
            "duplicate_policy": self.duplicate_policy,
            "empty_leaf_policy": self.empty_leaf_policy,
        }


@dataclass
class Engine:
    config: EngineConfig = field(default_factory=EngineConfig)
    store: PageStore | None = None
    tracer: Tracer = field(default_factory=Tracer)

    def __post_init__(self):
        if self.store is None:
            self.store = PageStore(self.config.page_size)
            txn = self.store.begin()
            root = self.store.allocate(txn)
            self.tracer.emit(PAGE_ALLOC, pid=root, role="initial_root")
            self.store.write(
                txn, root, paging.empty_leaf(self.config.page_size)
            )
            self.store.set_root(txn, root)
            self.tracer.emit(ROOT_CHANGE, root_pid=root, reason="bootstrap")
            self.store.commit(txn)

    # ---- 派生信息 -------------------------------------------------------
    @property
    def page_size(self) -> int:
        return self.config.page_size

    def page_view(self, pid: int, note: str = "") -> dict:
        page = self.store.pages[pid]
        h = paging.header(page)
        cells = paging.cells(page)
        used = sum(c.byte_len for c in cells) + paging.HEADER_SIZE + 2 * len(cells)
        view = {
            "pid": pid,
            "unit": "byte",
            "ptype": ("leaf" if h["ptype"] == paging.LEAF else "internal"),
            "nslots": h["nslots"],
            "slots": paging.slots(page),
            "free_space_byte": h["free_space"],
            "used_byte": used,
            "capacity_byte": self.page_size,
            "cell_end_offset": h["cell_end"],
            "header_size_byte": paging.HEADER_SIZE,
            "next_leaf": h["next_leaf"],
            "first_child": h["first_child"],
            "lsn": h["lsn"],
            "keys": [paging.render_key(c.key) for c in cells],
            "key_byte_lengths": [len(c.key) for c in cells],
            "note": note,
        }
        self.tracer.emit(PAGE_VIEW, **view)
        return view

    # ---- 读路径 ---------------------------------------------------------
    def _descend(self, key: bytes):
        """下降到应持有 key 的叶子；途中若遇到被提升的数据键则返回其所在内部页。

        返回 ``(path, leaf_or_internal_pid, promoted_cell_or_None)``。
        - copy：内部单元格都是空 value 的分隔键，返回叶子。
        - promote：内部单元格中 value 非空的是被提升的数据键，命中即返回；
          空 value 的仍为路由分隔键（其 child 指向键 >= 该分隔键的子树）。
        """
        pid = self.store.root_pid
        path: list[int] = []
        while True:
            page = self.store.pages[pid]
            h = paging.header(page)
            path.append(pid)
            if h["ptype"] == paging.LEAF:
                return path, pid, None
            cells = paging.cells(page)
            keys = [c.key for c in cells]
            idx = int(
                np.searchsorted(np.array(keys, dtype=object), key, side="left")
            )
            exact = idx < len(keys) and keys[idx] == key
            if exact:
                hit = cells[idx]
                if hit.value:  # 提升的数据键（promote 规则）
                    return path, pid, hit
                # 纯分隔键：值不在此，走它指向的右子树
                pid = hit.first_child
                continue
            if idx == 0:
                pid = h["first_child"]
            else:
                pid = cells[idx - 1].first_child

    def get(self, key: bytes) -> bytes | None:
        _, pid, promoted = self._descend(key)
        if promoted is not None:
            return promoted.value
        for cell in paging.cells(self.store.pages[pid]):
            if cell.key == key:
                return cell.value
        return None

    # ---- 分裂候选（按字节空间）-----------------------------------------
    def _split_candidates(self, cells, ptype):
        """枚举合法分裂点（决定依据是字节空间而非键数）。

        ``sp``（1..n-1）：左=cells[:sp]，右=cells[sp:]。
        内部页分裂时，第 sp-1 个单元格（边界项）保留在左半页，其 child
        成为右半页的 first_child，并在父节点复制一个新分隔键指向右新页。
        """
        n = len(cells)
        ps = self.page_size
        blen = np.array([c.byte_len + 2 for c in cells], dtype=np.int64)
        pref = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(blen, out=pref[1:])
        pref += paging.HEADER_SIZE
        legal, rejected = [], []
        for sp in range(1, n):
            left_bytes = int(pref[sp])
            right_bytes = int(pref[n] - pref[sp])
            lkeys, rkeys = sp, n - sp
            if ptype == paging.INTERNAL:
                # 右半页 cells[sp:] 通过 first_child 承载孩子；边界孩子由
                # 左半页边界项继续指向，因此右半页键数即其显式 cell 数。
                pass
            rec = {
                "split_at": sp,
                "left_used_byte": left_bytes,
                "right_used_byte": right_bytes,
                "left_key_count": lkeys,
                "right_key_count": rkeys,
                "capacity_byte": ps,
                "decided_by": "byte_space",
            }
            ok = left_bytes <= ps and right_bytes <= ps
            if ptype == paging.LEAF and (lkeys == 0 or rkeys == 0):
                ok = False
            (legal if ok else rejected).append(rec)
        chosen = None
        if legal:
            chosen = min(
                legal, key=lambda r: abs(r["left_used_byte"] - r["right_used_byte"])
            )
        return legal, rejected, chosen

    # ---- 写路径：插入 ---------------------------------------------------
    def insert(self, op_index: int, raw_key, raw_value=b"") -> dict:
        from .units import coerce_bytes
        kq = coerce_bytes(raw_key, field_name="key", op_index=op_index)
        vq = coerce_bytes(raw_value, field_name="value", op_index=op_index)
        self.tracer.emit(
            OP,
            op_index=op_index,
            op="insert",
            key_raw=repr(raw_key),
            value_raw=repr(raw_value),
            key_diagnostic=kq.diagnostic,
            value_diagnostic=vq.diagnostic,
            key_unit="byte",
        )
        if not (kq.convertible and vq.convertible):
            return {"status": "rejected_unconvertible",
                    "key_diagnostic": kq.diagnostic,
                    "value_diagnostic": vq.diagnostic}
        return self._insert(op_index, kq.value, vq.value)

    def _insert(self, op_index: int, key: bytes, value: bytes) -> dict:
        txn = self.store.begin()
        try:
            path, target_pid, promoted = self._descend(key)
            target = self.store.pages[target_pid]
            existing = paging.cells(target)
            keys = [c.key for c in existing]
            pos = paging.bisect_left(keys, key)
            if promoted is not None or (
                pos < len(keys) and keys[pos] == key
            ):
                result = self._handle_duplicate(
                    txn, op_index, target_pid, existing, pos, key, value,
                    is_internal=(promoted is not None),
                )
            else:
                new_cells = existing[:pos] + [paging.Cell(key, value)] + existing[pos:]
                result = self._after_leaf_mut(
                    txn, op_index, path, target_pid, new_cells
                )
            if result["status"] == "ok":
                self.store.commit(txn)
            else:
                self.store.abort(txn)
            return result
        except Exception:
            self.store.abort(txn)
            raise

    def _handle_duplicate(self, txn, op_index, pid, cells, pos, key, value,
                          is_internal=False):
        policy = self.config.duplicate_policy
        self.tracer.emit(
            DUP_KEY,
            op_index=op_index,
            pid=pid,
            key=paging.render_key(key),
            policy=policy,
            location=("internal_promoted" if is_internal else "leaf"),
            candidates=["last_wins", "first_wins"],
        )
        if policy == FIRST_WINS:
            return {"status": "duplicate_ignored", "policy": policy}
        new_cells = list(cells)
        if is_internal:
            # 提升的数据键：保留其 child 指针，仅替换 value
            child = new_cells[pos].first_child
            new_cells[pos] = paging.Cell(key, value, first_child=child)
        else:
            new_cells[pos] = paging.Cell(key, value)
        after = paging.replace_cells(self.store.pages[pid], new_cells,
                                     lsn=self.store.wal.tell())
        self.store.write(txn, pid, after)
        return {"status": "ok", "pid": pid}

    def _after_leaf_mut(self, txn, op_index, path, pid, new_cells):
        ps = self.page_size
        try:
            after = paging.replace_cells(
                self.store.pages[pid], new_cells, lsn=self.store.wal.tell()
            )
            self.store.write(txn, pid, after)
            return {"status": "ok", "pid": pid}
        except paging.PageOverflow:
            return self._split_leaf(txn, op_index, path, pid, new_cells)

    def _split_leaf(self, txn, op_index, path, pid, cells):
        legal, rejected, chosen = self._split_candidates(cells, paging.LEAF)
        self.tracer.emit(
            SPLIT_CANDIDATE,
            op_index=op_index,
            pid=pid,
            level="leaf",
            legal=legal,
            rejected=rejected,
            chosen_split_at=(chosen["split_at"] if chosen else None),
        )
        if chosen is None:
            return {"status": "split_impossible", "pid": pid}
        sp = chosen["split_at"]
        ps = self.page_size
        h = paging.header(self.store.pages[pid])
        old_next = h["next_leaf"]
        new_pid = self.store.allocate(txn)
        self.tracer.emit(PAGE_ALLOC, pid=new_pid, role="split_sibling")

        if self.config.separator_rule == COPY:
            left_cells, right_cells = cells[:sp], cells[sp:]
            sep_key = right_cells[0].key
            promoted_entry = paging.Cell(sep_key, b"", first_child=new_pid)
        else:
            # promote：cells[sp-1] 提升为内部数据键（携带 value），其右子树
            # 是严格更大的键；左叶保留更小的键，右叶从 sp 开始。
            left_cells, promoted_cell, right_cells = (
                cells[: sp - 1], cells[sp - 1], cells[sp:]
            )
            sep_key = promoted_cell.key
            promoted_entry = paging.Cell(
                sep_key, promoted_cell.value, first_child=new_pid
            )

        left_page = paging.build_page(
            ps, paging.LEAF, left_cells,
            lsn=self.store.wal.tell(), next_leaf=new_pid,
        )
        right_page = paging.build_page(
            ps, paging.LEAF, right_cells,
            lsn=self.store.wal.tell(), next_leaf=old_next,
        )
        self.store.write(txn, pid, left_page)
        self.store.write(txn, new_pid, right_page)
        self.tracer.emit(
            SPLIT_DECISION,
            op_index=op_index,
            pid=pid,
            new_pid=new_pid,
            level="leaf",
            split_at=sp,
            separator_key=paging.render_key(sep_key),
            separator_rule=self.config.separator_rule,
            byte_evidence=chosen,
            adopted=True,
            untaken=[{"split_at": r["split_at"],
                      "left_used_byte": r["left_used_byte"],
                      "right_used_byte": r["right_used_byte"]}
                     for r in legal if r["split_at"] != sp],
            rejected=rejected,
        )
        self.tracer.emit(SIBLING_LINK,
                         left_pid=pid, right_pid=new_pid,
                         next_leaf_of_left=new_pid,
                         next_leaf_of_right=old_next)
        return self._insert_into_parent(
            txn, op_index, pid, new_pid, sep_key, depth=1,
            entry_override=promoted_entry,
        )


    # ---- 分裂向上传播（内部页）-----------------------------------------
    def _find_parent(self, child_pid: int):
        """按子指针（不依赖 key 比较规则）定位包含 child_pid 的内部页。"""
        s = self.store

        def walk(pid, path):
            page = s.pages[pid]
            h = paging.header(page)
            if h["ptype"] != paging.INTERNAL:
                return None
            cur = path + [pid]
            if h["first_child"] == child_pid:
                return cur, -1
            for i, cell in enumerate(paging.cells(page)):
                if cell.first_child == child_pid:
                    return cur, i
                found = walk(cell.first_child, cur)
                if found:
                    return found
            return None

        return walk(s.root_pid, [])

    def _insert_into_parent(self, txn, op_index, left_pid, right_pid,
                            sep_key, depth, entry_override=None):
        s = self.store
        entry = entry_override or paging.Cell(
            sep_key, b"", first_child=right_pid
        )
        found = self._find_parent(left_pid)
        if found is None:
            new_root = s.allocate(txn)
            self.tracer.emit(PAGE_ALLOC, pid=new_root, role="new_root")
            root_page = paging.build_page(
                self.page_size, paging.INTERNAL, [entry],
                lsn=s.wal.tell(), first_child=left_pid,
            )
            s.write(txn, new_root, root_page)
            s.set_root(txn, new_root)
            self.tracer.emit(
                ROOT_SPLIT, op_index=op_index, old_root=left_pid,
                new_root=new_root, right_pid=right_pid,
                separator_key=paging.render_key(sep_key),
                separator_rule=self.config.separator_rule, depth=depth,
            )
            self.tracer.emit(ROOT_CHANGE, root_pid=new_root, reason="root_split")
            return {"status": "ok", "root_split": True, "root_pid": new_root}

        path, idx = found
        parent_pid = path[-1]
        parent = s.pages[parent_pid]
        h = paging.header(parent)
        old_cells = paging.cells(parent)
        if idx == -1:
            new_cells = [entry] + old_cells
            new_first = left_pid
        else:
            new_cells = old_cells[: idx + 1] + [entry] + old_cells[idx + 1 :]
            new_first = h["first_child"]
        try:
            after = paging.replace_cells(
                parent, new_cells, lsn=s.wal.tell(), first_child=new_first
            )
            s.write(txn, parent_pid, after)
            self.tracer.emit(
                SPLIT_PROPAGATE, op_index=op_index, parent_pid=parent_pid,
                left_pid=left_pid, right_pid=right_pid,
                separator_key=paging.render_key(sep_key), depth=depth,
            )
            return {"status": "ok", "propagated": True, "pid": parent_pid}
        except paging.PageOverflow:
            return self._split_internal(
                txn, op_index, parent_pid, new_cells, new_first, depth
            )

    def _split_internal(self, txn, op_index, pid, cells, first_child,
                        depth):
        s = self.store
        legal, rejected, chosen = self._split_candidates(cells, paging.INTERNAL)
        self.tracer.emit(
            SPLIT_CANDIDATE, op_index=op_index, pid=pid, level="internal",
            legal=legal, rejected=rejected,
            chosen_split_at=(chosen["split_at"] if chosen else None),
        )
        if chosen is None:
            return {"status": "split_impossible", "pid": pid}
        sp = chosen["split_at"]
        new_pid = s.allocate(txn)
        self.tracer.emit(PAGE_ALLOC, pid=new_pid, role="internal_sibling")

        boundary = cells[sp - 1]
        left_cells = cells[:sp]
        right_cells = cells[sp:]
        right_first = boundary.first_child

        left_page = paging.build_page(
            self.page_size, paging.INTERNAL, left_cells,
            lsn=s.wal.tell(), first_child=first_child,
        )
        right_page = paging.build_page(
            self.page_size, paging.INTERNAL, right_cells,
            lsn=s.wal.tell(), first_child=right_first,
        )
        s.write(txn, pid, left_page)
        s.write(txn, new_pid, right_page)
        # 父节点新增的分隔键：键=边界键，孩子=右新页。promote 规则下若
        # 边界项是提升的数据键（value 非空），其值仍保留在左半页，父项
        # 只是新复制出的路由分隔键——复制/提升口径在这里保持一致。
        promoted_entry = paging.Cell(
            boundary.key, b"", first_child=new_pid
        )
        self.tracer.emit(
            SPLIT_DECISION, op_index=op_index, pid=pid, new_pid=new_pid,
            level="internal", split_at=sp,
            separator_key=paging.render_key(boundary.key),
            separator_rule=self.config.separator_rule,
            byte_evidence=chosen, adopted=True,
            untaken=[{"split_at": r["split_at"],
                      "left_used_byte": r["left_used_byte"],
                      "right_used_byte": r["right_used_byte"]}
                     for r in legal if r["split_at"] != sp],
            rejected=rejected,
        )
        return self._insert_into_parent(
            txn, op_index, pid, new_pid, boundary.key, depth + 1,
            entry_override=promoted_entry,
        )

    # ---- 删除与空页 -----------------------------------------------------
    def delete(self, op_index: int, raw_key) -> dict:
        from .units import coerce_bytes
        kq = coerce_bytes(raw_key, field_name="key", op_index=op_index)
        self.tracer.emit(
            OP, op_index=op_index, op="delete", key_raw=repr(raw_key),
            key_diagnostic=kq.diagnostic, key_unit="byte",
        )
        if not kq.convertible:
            return {"status": "rejected_unconvertible",
                    "key_diagnostic": kq.diagnostic}
        key = kq.value
        txn = self.store.begin()
        try:
            path, leaf_pid = self._descend(key)
            old = paging.cells(self.store.pages[leaf_pid])
            pos = paging.bisect_left([c.key for c in old], key)
            if pos >= len(old) or old[pos].key != key:
                self.store.abort(txn)
                return {"status": "not_found", "pid": leaf_pid}
            new_cells = old[:pos] + old[pos + 1 :]
            if new_cells:
                after = paging.replace_cells(
                    self.store.pages[leaf_pid], new_cells,
                    lsn=self.store.wal.tell(),
                )
                self.store.write(txn, leaf_pid, after)
                self.store.commit(txn)
                return {"status": "ok", "pid": leaf_pid}
            return self._empty_leaf(txn, op_index, path, leaf_pid)
        except Exception:
            self.store.abort(txn)
            raise

    def _empty_leaf(self, txn, op_index, path, leaf_pid):
        s = self.store
        h = paging.header(s.pages[leaf_pid])
        self.tracer.emit(
            PAGE_EMPTY, op_index=op_index, pid=leaf_pid,
            root_pid=s.root_pid,
            policy=self.config.empty_leaf_policy,
            next_leaf=h["next_leaf"], is_root=(leaf_pid == s.root_pid),
            candidates=[KEEP_EMPTY, REMOVE_EMPTY],
        )
        if self.config.empty_leaf_policy == KEEP_EMPTY or leaf_pid == s.root_pid:
            empty = paging.build_page(
                self.page_size, paging.LEAF, [],
                lsn=s.wal.tell(), next_leaf=h["next_leaf"],
            )
            s.write(txn, leaf_pid, empty)
            s.commit(txn)
            return {"status": "ok", "empty_leaf_kept": True, "pid": leaf_pid}
        return self._remove_empty_leaf(txn, op_index, path, leaf_pid)

    def _remove_empty_leaf(self, txn, op_index, path, leaf_pid):
        s = self.store
        h = paging.header(s.pages[leaf_pid])
        nxt = h["next_leaf"]
        # 修正前驱 sibling link：沿左孩子找到内部节点中指向 leaf_pid 的项
        if len(path) >= 2:
            parent_pid = path[-2]
            parent = s.pages[parent_pid]
            ph = paging.header(parent)
            pcells = paging.cells(parent)
            new_cells = []
            new_first = ph["first_child"]
            if ph["first_child"] == leaf_pid:
                # 空页是最左孩子：第一个 cell 成为最左孩子，其分隔键删除
                first = pcells[0]
                new_first = first.first_child
                new_cells = [
                    paging.Cell(c.key, c.value, first_child=c.first_child)
                    for c in pcells[1:]
                ]
            else:
                for cell in pcells:
                    if cell.first_child == leaf_pid:
                        continue  # 删除指向空页的分隔键
                    new_cells.append(cell)
            after = paging.replace_cells(
                parent, new_cells, lsn=s.wal.tell(), first_child=new_first
            )
            s.write(txn, parent_pid, after)
        # 修正左兄弟的 next_leaf
        for other_pid, other in s.pages.items():
            oh = paging.header(other)
            if oh["ptype"] == paging.LEAF and oh["next_leaf"] == leaf_pid:
                s.write(
                    txn, other_pid,
                    paging.set_meta(other, lsn=s.wal.tell(), next_leaf=nxt),
                )
        # 空页保留在磁盘上但不再可达（page id 不复用）
        s.commit(txn)
        return {"status": "ok", "empty_leaf_detached": True, "pid": leaf_pid,
                "reachable": leaf_pid in s.reachable_pids()}

    def emit_rule_basis(self) -> None:
        """把三条判断依据显式写成机器可读事件。"""
        self.tracer.emit(
            SEPARATOR_RULE,
            rule=self.config.separator_rule,
            split_metric="byte_space",
            split_metric_unit="byte",
            statement=(
                "变长键按字节空间而非键数决定分裂；separator 的复制(copy)"
                "或提升(promote)规则在叶子与内部节点必须一致"
            ),
            legal_options=[COPY, PROMOTE],
        )
