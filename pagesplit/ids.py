"""稳定身份：所有派生 ID 都由内容坐标确定性生成，导出/再导入保持不变。"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

_SAFE = re.compile(r"[^0-9A-Za-z_@.\-]+")


def sid(*parts: object) -> str:
    """用结构化坐标拼出稳定 ID，长字节片段退化为短哈希。"""
    out: list[str] = []
    for p in parts:
        if p is None:
            out.append("-")
        elif isinstance(p, (bytes, bytearray)):
            out.append("b" + hashlib.sha1(bytes(p)).hexdigest()[:10])
        else:
            s = str(p)
            if len(s) <= 48:
                out.append(_SAFE.sub("_", s))
            else:
                out.append("h" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:10])
    return ":".join(out)


def digest(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_order(rows: Iterable[dict], key: str = "id") -> list[dict]:
    return sorted(rows, key=lambda r: r[key])
