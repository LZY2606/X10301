"""页分裂策略。

核心规则（判定依据以机器可读事件落库）：
  * 变长键按 **字节空间** 而非键数决定分裂；
  * leaf 分裂时 separator **复制** 到父层（左右两页都保留边界键）；
  * internal 分裂时 separator **提升**（不复制，原页不再保留它）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pages import HEADER_SIZE, SLOT_SIZE, Cell, Page, INTERNAL

SPLIT_BYTE_BALANCE = "byte_balance"
SPLIT_LEFTMOST_FIT = "leftmost_fit"
TIE_LEFTMOST = "leftmost"
TIE_RIGHTMOST = "rightmost"
TIE_UNSET = "unset"   # 无法唯一决定：保留全部合法候选


@dataclass(frozen=True)
class SplitCandidate:
    index: int                 # 左页保留 cells[0:index]
    left_used: int
    right_used: int
    imbalance: int             # |left_cells - right_cells| 字节
    separator_key: bytes
    separator_rule: str        # "copy"（leaf）/ "promote"（internal）

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "left_used_bytes": self.left_used,
            "right_used_bytes": self.right_used,
            "imbalance_bytes": self.imbalance,
            "separator_key": self.separator_key.hex(),
            "separator_rule": self.separator_rule,
        }


def cell_bytes(cell: Cell) -> int:
    return cell.encoded_len()


def _used(cells: list[Cell], ptype: int) -> int:
    return HEADER_SIZE + SLOT_SIZE * len(cells) + sum(c.encoded_len() for c in cells)


def legal_split_points(page: Page, page_size: int) -> list[int]:
    """所有能让两页都装得下的切分点（按字节空间判定）。

    每个新页至少要装下自己那一侧最大的 cell；这里直接按真实 used 判定。
    返回 index 列表，1 <= index < n。
    """
    n = len(page.cells)
    legal: list[int] = []
    weights = np.array([cell_bytes(c) for c in page.cells], dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(weights)))
    total_cells = int(prefix[-1])
    for i in range(1, n):
        left = HEADER_SIZE + SLOT_SIZE * i + int(prefix[i])
        right = HEADER_SIZE + SLOT_SIZE * (n - i) + (total_cells - int(prefix[i]))
        if left <= page_size and right <= page_size:
            legal.append(i)
    return legal


def split_candidates(page: Page, page_size: int) -> list[SplitCandidate]:
    """枚举全部合法候选；无法唯一决定时一个都不丢弃。"""
    rule = "copy" if page.is_leaf else "promote"
    out: list[SplitCandidate] = []
    n = len(page.cells)
    weights = np.array([cell_bytes(c) for c in page.cells], dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(weights)))
    total = int(prefix[-1])
    for i in legal_split_points(page, page_size):
        left = HEADER_SIZE + SLOT_SIZE * i + int(prefix[i])
        right = HEADER_SIZE + SLOT_SIZE * (n - i) + (total - int(prefix[i]))
        # leaf: 右页首键被复制到父层；internal: 该键提升后右页不再保留
        sep_index = i if page.is_leaf else i
        out.append(SplitCandidate(
            index=i, left_used=left, right_used=right,
            imbalance=abs(left - right),
            separator_key=page.cells[sep_index].key,
            separator_rule=rule,
        ))
    return out


def choose_split(page: Page, page_size: int,
                 policy: str = SPLIT_BYTE_BALANCE,
                 tiebreak: str = TIE_LEFTMOST) -> tuple[SplitCandidate | None,
                                                        list[SplitCandidate]]:
    """选定分裂候选。

    返回 (选中候选 或 None（未唯一决定）, 全部合法候选)。
    tiebreak == "unset" 时不选，只保留全部候选，等待人工裁定或参数修订。
    """
    candidates = split_candidates(page, page_size)
    if not candidates:
        return None, candidates
    if policy == SPLIT_LEFTMOST_FIT:
        ordered = list(candidates)
    else:
        imba = np.array([c.imbalance for c in candidates], dtype=np.int64)
        min_i = int(np.min(imba))
        ordered = [c for c in candidates if c.imbalance == min_i]
    if tiebreak == TIE_UNSET:
        if len(ordered) == 1:
            return ordered[0], candidates
        return None, candidates
    if tiebreak == TIE_RIGHTMOST:
        return ordered[-1], candidates
    return ordered[0], candidates


def apply_split(page: Page, cand: SplitCandidate) -> tuple[Page, Page, bytes, str]:
    """按候选把 page 切成 (left, right, separator, rule)。

    leaf:     right 保留 cells[i:]，separator = right 首键，规则 copy；
    internal: right 保留 cells[i+1:]，separator = cells[i].key，规则 promote，
              children 相应重排。
    """
    i = cand.index
    if page.is_leaf:
        left_cells = page.cells[:i]
        right_cells = page.cells[i:]
        left = Page(pid=page.pid, ptype=page.ptype, cells=list(left_cells),
                    prev=page.prev, next=None, lsn=page.lsn)
        right = Page(pid=-1, ptype=page.ptype, cells=list(right_cells),
                     prev=None, next=page.next, lsn=page.lsn)
        return left, right, cand.separator_key, "copy"
    # internal
    children = [c.child for c in page.cells] + [page.high]
    sep = page.cells[i].key
    left_cells = [Cell(key=c.key, child=c.child) for c in page.cells[:i]]
    left = Page(pid=page.pid, ptype=INTERNAL, cells=left_cells,
                high=children[i], lsn=page.lsn)
    right_cells = [Cell(key=c.key, child=c.child) for c in page.cells[i + 1:]]
    right = Page(pid=-1, ptype=INTERNAL, cells=right_cells,
                 high=children[-1], lsn=page.lsn)
    return left, right, sep, "promote"
