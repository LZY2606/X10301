"""机器可读事件类型与追踪。

每条事件都是结构化 dict，带稳定 ``type`` 与坐标基准，复核界面与失效分析
都消费同一套事件。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# 每次逻辑操作
OP = "op"
# 页前镜像 / 写页
PAGE_BEFORE = "page_before"
PAGE_AFTER = "page_after"
# slot directory / free space（作为页视图的坐标化读数，随页事件附带）
PAGE_VIEW = "page_view"
# sibling link
SIBLING_LINK = "sibling_link"
# page id 分配
PAGE_ALLOC = "page_alloc"
# 日志序号
WAL_APPEND = "wal_append"
# 分裂（候选、采纳/未采纳、字节依据、separator 规则）
SPLIT_CANDIDATE = "split_candidate"
SPLIT_DECISION = "split_decision"
SPLIT_PROPAGATE = "split_propagate"
# 根变化
ROOT_SPLIT = "root_split"
ROOT_CHANGE = "root_change"
# 重复键
DUP_KEY = "dup_key"
# 删除后空页
PAGE_EMPTY = "page_empty"
# 崩溃 / 写页中断 / 恢复
CRASH = "crash"
TORN_PAGE = "torn_page"
RECOVERY_STEP = "recovery_step"
RECOVERY_SUMMARY = "recovery_summary"
# 结构不变量
INVARIANT_CHECK = "invariant_check"
# separator 规则一致性
SEPARATOR_RULE = "separator_rule"


@dataclass
class Tracer:
    events: list[dict] = field(default_factory=list)

    def emit(self, etype: str, **payload) -> dict:
        ev = {
            "seq": len(self.events),
            "type": etype,
            "ts": round(time.time(), 6),
            "payload": payload,
        }
        self.events.append(ev)
        return ev

    def of_type(self, etype: str) -> list[dict]:
        return [e for e in self.events if e["type"] == etype]
