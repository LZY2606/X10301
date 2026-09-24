"""稳定身份工具。

所有显示值在导入、派生、导出/再导入之间都必须保持同一身份。
身份一律由稳定输入内容派生（不使用易变的行号/自增 id），
因此删除本地数据库后重新导入同一档案，身份保持不变。
"""
from __future__ import annotations

import hashlib
from typing import Any


def sid(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def stable_int_id(prefix: str, index: int, width: int = 3) -> str:
    return f"{prefix}-{index:0{width}d}"
