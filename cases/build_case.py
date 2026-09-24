"""确定性离线案例包构建器：页裂档案固定叙事。

不使用任何通用表格占位；材料全部是领域量（page_size 带 byte 单位、
键值给 hex 原始表示、长度带 length_bytes 坐标基准、崩溃点带语义名）。

叙事阶段（不拆成无关按钮，分支时间线连续）：
  S1 常规小键插入，触发第一次根分裂；
  S2 持续插入，触发级联分裂；
  S3 变长键：单键占大半页，证明按字节而非键数决定分裂；
  S4 重复键策略（append）；
  S5 删除导致空页释放；
  S6 写页中断（torn_split）+ 恢复重放两次；
  S7 commit 前中断（commit_flush）+ 恢复重放两次；
  S8 无法转换的值（长度诊断，留在原位置）。
之后仓库另外携带：一次参数修订（split_tie left→right）、
一次只影响局部依赖的源数据改动、一对互相冲突的人工决定（由 app 层演示）。
"""
from __future__ import annotations

import json
import os
import random

PAGE_SIZE = 64  # byte
VALUE = {"hex": "76" * 8, "length_bytes": 8, "unit": "byte"}  # b"v"*8


def material(mid, op, key_hex, value=None, crash=None,
             length_bytes=None, note=None):
    m = {"id": mid, "op": op,
         "key_hex": key_hex,
         "coordinate_basis": "byte offset within fixed page",
         "key_unit": "byte",
         "stage_note": note}
    if value is not None:
        m["value"] = value
    if crash is not None:
        m["crash"] = crash
    return m


def build() -> dict:
    mats = []
    seq = 0

    def add(op, key, value=None, crash=None, note=None):
        nonlocal seq
        seq += 1
        mid = f"m{seq:03d}"
        kv = bytes.fromhex(key) if isinstance(key, str) else key
        mats.append(material(mid, op, kv.hex(), value, crash, note=note))
        return mid

    rng = random.Random(20260925)

    # ---- S1 + S2：80 个小键以固定洗牌顺序插入，得到根分裂与级联分裂 ----
    small_keys = [f"k{i:03d}".encode() for i in range(80)]
    rng.shuffle(small_keys)
    for i, k in enumerate(small_keys):
        note = "S1 regular inserts" if i < 30 else "S2 cascade build-up"
        add("insert", k, VALUE, note=note)

    # ---- S3：变长键，单条 cell 占掉近半页（32B key + 8B value）----
    add("insert", b"L" + b"a" * 31,
        {"hex": "78" * 8, "length_bytes": 8, "unit": "byte"},
        note="S3 variable-length key: 32 bytes dominate split by bytes")
    add("insert", b"M" + b"b" * 30,
        {"hex": "79" * 8, "length_bytes": 8, "unit": "byte"},
        note="S3 variable-length key: 31 bytes")

    # ---- S4：重复键（append 策略）----
    add("insert", small_keys[0], VALUE,
        note="S4 duplicate key, duplicate_policy=append")
    add("insert", small_keys[0], VALUE,
        note="S4 duplicate key again, append version suffix")

    # ---- S5：删除一批相邻小键，制造空页释放 ----
    for i in range(0, 12):
        add("delete", small_keys[i], note="S5 delete to empty a leaf")

    # ---- S6：torn split（新分配页写一半中断）----
    add("insert", b"Z" + b"z" * 20,
        {"hex": "7a" * 8, "length_bytes": 8, "unit": "byte"},
        crash="torn_split",
        note="S6 torn page during split; recover twice (idempotent)")
    # 崩溃后继续：验证恢复出的树可继续服务
    add("insert", b"Z" + b"y" * 10,
        {"hex": "7a" * 8, "length_bytes": 8, "unit": "byte"},
        note="S6 post-recovery insert must succeed")

    # ---- S7：commit_flush（页已落，commit 记录前中断 → undo）----
    add("insert", b"Q" + b"q" * 12,
        {"hex": "71" * 8, "length_bytes": 8, "unit": "byte"},
        crash="commit_flush",
        note="S7 crash before commit; loser txn undone; recover twice")

    # ---- S8：无法转换的值（声明长度与实际不符）----
    seq += 1
    bad = {
        "id": f"m{seq:03d}",
        "op": "insert",
        "key_hex": b"BAD".hex(),
        "value": {"hex": "616263", "length_bytes": 99, "unit": "byte"},
        "stage_note": "S8 unconvertible: declared 99 bytes != 3 bytes; kept in place",
        "coordinate_basis": "byte offset within fixed page",
        "key_unit": "byte",
    }
    mats.append(bad)

    case = {
        "case_id": "page_split_archive_v1",
        "title": "页裂档案：B-tree 页面布局、分裂传播与恢复检查",
        "page_size_bytes": PAGE_SIZE,
        "coordinate_basis": {
            "byte_offset": "0-based offset from page start",
            "slot_index": "0-based key order within slot directory",
            "lsn": "global monotonic log sequence number",
            "page_id": "allocator-issued, never reused within the story",
        },
        "initial_rule_version": {
            "rule_version_id": "rV1",
            "duplicate_policy": "append",
            "split_tie": "left",
            "separator_rule": {
                "leaf": "copy_rightmost_min_key_to_parent",
                "internal": "promote_middle_key_to_parent",
            },
        },
        "parameter_revision": {
            "rule_version_id": "rV2",
            "changes": {"split_tie": "right"},
            "rationale": "参数修订：tie-break 从偏左改为偏右，只重算依赖该参数的判定。",
        },
        "local_source_change": {
            "material_id": "m001",
            "field": "value",
            "from": VALUE,
            "to": {"hex": "76" * 8, "length_bytes": 8, "unit": "byte",
                   "annotation": "re-imported annotation only; bytes unchanged",
                   "source_edit_ref": "src-change-1"},
            "rationale": "只影响局部依赖：仅 m001 的解释与其下游视图失效，其余身份不变。",
        },
        "materials": mats,
        "invariant_judgments": [
            {"judgment_id": "J1", "rule": "split_decided_by_byte_space_not_key_count"},
            {"judgment_id": "J2", "rule": "separator_copy_for_leaf_promote_for_internal_consistent"},
            {"judgment_id": "J3", "rule": "no_reachable_half_page_after_crash_point"},
            {"judgment_id": "J4", "rule": "recovery_replay_twice_idempotent"},
            {"judgment_id": "J5", "rule": "sibling_chain_covers_all_reachable_leaves"},
            {"judgment_id": "J6", "rule": "page_ids_monotonic_no_reuse"},
            {"judgment_id": "J7", "rule": "unconvertible_values_remain_in_place_with_diagnostic"},
            {"judgment_id": "J8", "rule": "lsn_monotonic_and_wal_before_pages"},
        ],
    }
    return case


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "page_split_archive")
    os.makedirs(out_dir, exist_ok=True)
    case = build()
    with open(os.path.join(out_dir, "case.json"), "w", encoding="utf-8") as fh:
        json.dump(case, fh, ensure_ascii=False, indent=2)
    print(f"wrote {len(case['materials'])} materials")


if __name__ == "__main__":
    main()
