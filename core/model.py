"""B-tree 内存模型：页、槽、分隔键候选与规则版本。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from . import encoding as enc

DUPLICATE_OVERWRITE = "overwrite"   # 重复键：覆盖
DUPLICATE_REJECT = "reject"         # 重复键：拒绝
EMPTY_LEAF_DELETE = "delete"        # 删除后空页：回收
EMPTY_LEAF_KEEP = "keep"            # 删除后空页：保留空页
SPLIT_BALANCE_BYTES = "balance_bytes"  # 候选裁定：按字节均衡
SPLIT_FIRST_FIT = "first_fit"         # 候选裁定：首个合法切分点


def digest(*parts: object) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.digest()[:12].hex()


def cell_id(pid: int, slot: int, kind: str, key: bytes, value: bytes, child: int | None = None) -> str:
    return f"cell:{digest(pid, slot, kind, key, value, child)}"


def leaf_bytes(key: bytes, value: bytes) -> int:
    return enc.leaf_cell_len(key, value)


def internal_bytes(key: bytes, value: bytes) -> int:
    return enc.internal_cell_len(key, value)


@dataclass
class Page:
    pid: int
    kind: int                    # enc.LEAF / enc.INTERNAL
    # leaf: key -> value(bytes); internal: key -> (child_pid, value)
    entries: "dict[bytes, object]"
    children: list[int] = field(default_factory=list)
    lsn: int = 0
    created_lsn: int = 0
    freed: bool = False

    # ---- leaf ----
    def leaf_get(self, key: bytes) -> bytes | None:
        return self.entries.get(key)

    def leaf_put(self, key: bytes, value: bytes) -> None:
        self.entries[key] = value

    def leaf_del(self, key: bytes) -> bool:
        if key in self.entries:
            del self.entries[key]
            return True
        return False

    def leaf_items_sorted(self) -> list[tuple[bytes, bytes]]:
        return [(k, self.entries[k]) for k in sorted(self.entries)]

    def leaf_used(self) -> int:
        return sum(leaf_bytes(k, v) for k, v in self.entries.items())

    # ---- internal ----
    def internal_view(self) -> tuple[list[int], list[bytes], list[bytes]]:
        """children = [c0, c1, ...]; separators=[s1<s2<...]; ci: c_{i-1}<si<=c_i。

        entries 保存 s -> (child_i, value)；children[0] 为最左孩子。
        """
        seps = sorted(self.entries)
        rights = [self.entries[s][0] for s in seps]
        vals = [self.entries[s][1] for s in seps]
        return list(self.children), seps, vals, rights

    def internal_used(self) -> int:
        _, seps, vals, _ = self.internal_view()
        return sum(internal_bytes(k, v) for k, v in zip(seps, vals))

    def encode(self, page_size: int) -> bytes:
        if self.kind == enc.LEAF:
            return enc.encode_page(self.pid, enc.LEAF, self.lsn,
                                   self.leaf_items_sorted(), page_size)
        children, seps, vals, rights = self.internal_view()
        cells = list(zip(rights, seps, vals))  # 最左孩子不落在任何 cell 内
        return enc.encode_page(self.pid, enc.INTERNAL, self.lsn, cells, page_size)


@dataclass(frozen=True)
class Rules:
    """规则版本。任何参数变化都产生新版本，不允许就地覆盖旧结论。"""

    version: str
    page_size: int
    reserved_header: int = enc.HEADER_SIZE
    duplicate: str = DUPLICATE_OVERWRITE
    empty_leaf: str = EMPTY_LEAF_DELETE
    split_choice: str = SPLIT_BALANCE_BYTES

    def usable(self) -> int:
        return self.page_size - self.reserved_header - 2  # 减去尾部 crc

    def leaf_fits(self, used: int, n: int) -> bool:
        return used + 2 * n <= self.usable()

    def internal_fits(self, used: int, n_sep: int) -> bool:
        # n 个分隔 cell 对应 n+1 个孩子；槽目录按 cell 数计
        return used + 2 * n_sep <= self.usable()


def leaf_split_candidates(keys: list[bytes], values: dict[bytes, bytes],
                          rules: Rules) -> list[int]:
    """返回全部合法切分点 k：左半含前 k 个键。按字节空间判定，两侧都必须放得下。

    “按字节空间而非键数”：即使两侧键数不均衡，只要各自字节放得下即为合法候选。
    """
    total = [leaf_bytes(k, values[k]) for k in keys]
    prefix = 0
    out: list[int] = []
    n = len(keys)
    for k in range(1, n):
        prefix += total[k - 1]
        left_ok = prefix + 2 * k <= rules.usable()
        right = sum(total[k:])
        right_ok = right + 2 * (n - k) <= rules.usable()
        if left_ok and right_ok:
            out.append(k)
    return out


def choose_split(candidates: list[int], keys: list[bytes], values: dict[bytes, bytes],
                 rules: Rules) -> int:
    if rules.split_choice == SPLIT_FIRST_FIT:
        return candidates[0]
    total = [leaf_bytes(k, values[k]) for k in keys]
    best, best_score = candidates[0], None
    for k in candidates:
        left = sum(total[:k])
        right = sum(total[k:])
        score = abs(left - right)
        if best_score is None or score < best_score:
            best, best_score = k, score
    return best


def internal_split_candidates(children: list[int], seps: list[bytes], vals: list[bytes],
                              rights: list[int], rules: Rules) -> list[int]:
    """内部页切分点 j：左半 children[:j] + seps[:j-1]；
    提升的分隔键 = seps[j-1]；右半 children[j:] + seps[j:]。
    合法：左侧 n_sep=j-1 放得下，右侧 n_sep=len(seps)-j 放得下。
    """
    n = len(seps)
    sizes = [internal_bytes(s, v) for s, v in zip(seps, vals)]
    out: list[int] = []
    for j in range(1, n + 1):
        left_used = sum(sizes[: j - 1])
        right_used = sum(sizes[j:])
        if (left_used + 2 * (j - 1) <= rules.usable()
                and right_used + 2 * (n - j) <= rules.usable()):
            out.append(j)
    return out
