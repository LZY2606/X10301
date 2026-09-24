"""稳定身份（stable identity）工具。

导出 -> 删除本地数据库 -> 重新导入后，所有身份、顺序、来源关系必须保持。
因此身份一律由 *确定性内容坐标* 构成，不能依赖 SQLite rowid 或内存对象地址。
"""
from __future__ import annotations


def branch_id(name: str) -> str:
    return f"br/{name}"


def event_id(pack_id: str, branch: str, seq: int, slug: str) -> str:
    return f"ev/{pack_id}/{branch}/{seq:04d}-{slug}"


def fragment_id(pack_id: str, batch: str, seq: int) -> str:
    return f"fr/{pack_id}/{batch}/{seq:03d}"


def page_guid(pack_id: str, page_id: int) -> str:
    return f"pg/{pack_id}/p{page_id}"


def cell_guid(pack_id: str, op_seq: int | None, page_id: int, slot: int, key_hex: str) -> str:
    ctx = f"op{op_seq:03d}" if op_seq is not None else "snap"
    return f"ce/{pack_id}/{ctx}/p{page_id}/s{slot:02d}/{key_hex}"


def quantity_guid(pack_id: str, page_id: int, region: str, op_seq: int | None = None) -> str:
    ctx = f"op{op_seq:03d}" if op_seq is not None else "snap"
    return f"qn/{pack_id}/{ctx}/p{page_id}/{region}"


def verdict_id(pack_id: str, branch: str, rule: str, seq: int) -> str:
    return f"vd/{pack_id}/{branch}/{seq:04d}/{rule}"
