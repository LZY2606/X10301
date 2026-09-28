"""稳定身份与规范化。

所有身份都由内容决定（内容寻址），因此同一案例包导出后删除本地数据库、
再导入，仍得到相同 id、顺序与来源关系。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

ID_PREFIX_LEN = 12


def canonical_json(obj: Any) -> str:
    """生成排序键、无多余空白的 JSON 文本（跨平台稳定）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(kind: str, payload: Any) -> str:
    """对 ``(kind, payload)`` 生成稳定短 id。"""
    h = hashlib.sha256(f"{kind}|{canonical_json(payload)}".encode("utf-8")).hexdigest()
    return f"{kind}_{h[:ID_PREFIX_LEN]}"


def event_id(seq: int, branch_id: str, etype: str, payload: Any) -> str:
    return digest("evt", {"seq": seq, "branch": branch_id, "type": etype, "payload": payload})


def artifact_id(kind: str, payload: Any) -> str:
    return digest("art", {"kind": kind, "payload": payload})


def run_id(branch_id: str, head_seq: int, params_id: str) -> str:
    return digest("run", {"branch": branch_id, "head": head_seq, "params": params_id})


def finding_id(run: str, rule: str, target: str) -> str:
    return digest("fnd", {"run": run, "rule": rule, "target": target})


def candidate_id(dp_id: str, key: str) -> str:
    return digest("cand", {"dp": dp_id, "key": key})


def decision_point_id(branch_id: str, kind: str, target: str) -> str:
    return digest("dp", {"branch": branch_id, "kind": kind, "target": target})
