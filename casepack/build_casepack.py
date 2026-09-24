"""Build the offline case pack '页裂档案：B-tree 页面布局、分裂传播与恢复检查'.

Deterministic: fixed page size, variable-length keys, duplicates, deletes that
empty a leaf, a torn write, and a double-replayed recovery. The expected
committed hash at the recovery point is computed by actually running the
script without the crash.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from btree.engine import Engine, DEFAULT_RULES  # noqa: E402

PAGE_SIZE_RAW = "256B"


def build_script():
    ops = []
    # variable-length keys and values; byte-driven splits, root split, cascade
    for i in range(1, 111):
        key = (f"key{i:03d}" + "x" * (i * 7 % 19)).encode()
        value = (f"val{i}" + "y" * (i * 11 % 29)).encode()
        ops.append({"id": f"op-{i:03d}", "op": "insert",
                    "key": key.hex(), "value": value.hex()})
    # duplicate-key policy probes (same key as op-007)
    dup_key = bytes.fromhex(ops[6]["key"])
    ops.append({"id": "op-dup-1", "op": "insert", "key": dup_key.hex(),
                "value": b"duplicate-one".hex()})
    ops.append({"id": "op-dup-2", "op": "insert", "key": dup_key.hex(),
                "value": b"duplicate-two".hex()})
    # delete a contiguous key range so at least one leaf becomes empty
    for j, i in enumerate(range(40, 50)):
        ops.append({"id": f"op-del-{j:02d}", "op": "delete",
                    "key": (f"key{i:03d}" + "x" * (i * 7 % 19)).encode().hex()})
    # torn write: the next page write stops after 100 of 256 bytes
    ops.append({"id": "op-crash-1", "op": "crash", "bytes": 100})
    ops.append({"id": "op-post-1", "op": "insert",
                "key": b"post-crash-a".hex(), "value": b"after-crash".hex()})
    # recovery replayed twice; committed_hash filled in below
    ops.append({"id": "op-recover-1", "op": "recover", "passes": 2})
    for i, k in enumerate((b"post-recovery-1", b"post-recovery-2")):
        ops.append({"id": f"op-postrec-{i}", "op": "insert",
                    "key": k.hex(), "value": b"ok".hex()})
    return ops


def committed_hash_at_recovery(script):
    """Hash of the reachable durable state at the recovery point, had the
    interrupted write completed normally."""
    pre = [op for op in script if op["op"] != "crash"]
    cut = next(i for i, op in enumerate(pre) if op["id"] == "op-recover-1")
    eng = Engine(256, dict(DEFAULT_RULES))
    eng.run_script(pre[:cut])
    return eng.state_hash()


def main():
    script = build_script()
    digest = committed_hash_at_recovery(script)
    for op in script:
        if op["id"] == "op-recover-1":
            op["committed_hash"] = digest
    case = {
        "case_id": "page-split-archive-v1",
        "title": "页裂档案：B-tree 页面布局、分裂传播与恢复检查",
        "narrative": "存储引擎小组导入固定页大小、键值操作和崩溃事件，"
                     "逐步审阅 leaf/internal page 的槽位、分裂与根变化。",
        "quantities": {
            "page_size": {"raw": PAGE_SIZE_RAW},
            "cache_hint": {"raw": "7 furlongs"},  # unconvertible -> diagnostic
        },
        "ops": script,
        "chapters": [
            {"id": "param-revision",
             "kind": "rules_revision",
             "note": "参数修订：duplicate_policy 改为 overwrite，"
                     "失效范围只覆盖依赖 rules-v1 与重复键操作的事件",
             "params": {"duplicate_policy": "overwrite"},
             "affected_source_ids": ["op-dup-1", "op-dup-2"]},
            {"id": "local-source-change",
             "kind": "source_revision",
             "note": "局部源数据改动：崩溃点写入字节 100 -> 40，"
                     "只应使崩溃与恢复相关事件失效",
             "source_id": "op-crash-1",
             "new_raw": {"id": "op-crash-1", "op": "crash", "bytes": 40}},
            {"id": "conflicting-decision",
             "kind": "conflict_demo",
             "note": "互相冲突的人工决定：两个窗口基于同一版本提交裁定，"
                     "后提交者必须看到版本冲突与双方差异"},
        ],
    }
    out = os.path.join(os.path.dirname(__file__), "split_archive_case.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(case, fh, ensure_ascii=False, indent=1)
    print(f"wrote {out} ({len(script)} ops, committed hash {digest[:12]}...)")


if __name__ == "__main__":
    main()
