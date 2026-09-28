"""领域量：所有值携带单位或坐标基准。

导入层保留原始表示(raw)，派生层才做规范化。无法转换的值留在原位置并附诊断。
"""
from __future__ import annotations

from dataclasses import dataclass

BYTES = "B"          # 字节空间
KEYS = "keys"        # 键数（只用于诊断，不能决定分裂）
PAGE_ID = "page_id"  # 坐标基准：页标识
SLOT = "slot"        # 坐标基准：槽位序号
LSN = "lsn"          # 坐标基准：日志序号
NO_UNIT = ""


@dataclass(frozen=True)
class Q:
    """带单位的标量。value 规范化为 int；raw 保存导入时的原始表示。"""

    value: int
    unit: str
    raw: str | None = None
    diagnostic: str | None = None

    def __int__(self) -> int:
        return self.value

    @staticmethod
    def parse(text: object, allowed: tuple[str, ...], default_unit: str) -> "Q":
        if isinstance(text, bool):  # bool 是 int 的子类，先排除
            return Q(0, default_unit, str(text), diagnostic="expected scalar")
        if isinstance(text, int):
            return Q(text, default_unit)
        if isinstance(text, str):
            token = text.strip()
        else:
            return Q(0, default_unit, str(text), diagnostic="unconvertible value")
        for unit in allowed:
            if unit and token.endswith(unit):
                body = token[: -len(unit)].strip()
                if body.lstrip("-").isdigit():
                    return Q(int(body), unit, raw=token)
        if token.lstrip("-").isdigit():
            return Q(int(token), default_unit, raw=token)
        return Q(0, default_unit, raw=token, diagnostic=f"unconvertible value: {token!r}")

    def to_dict(self) -> dict:
        out = {"value": self.value, "unit": self.unit}
        if self.raw is not None:
            out["raw"] = self.raw
        if self.diagnostic is not None:
            out["diagnostic"] = self.diagnostic
        return out
