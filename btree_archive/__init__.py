"""页裂档案：B-tree 页面布局、分裂传播与恢复检查。

所有领域量都携带单位或坐标基准：
- 页面大小 page_size 单位为 byte；
- 字节偏移 coordinate 单位为 byte，基准为页起始（0-based）；
- slot index 单位为 1 个槽位，基准为 0；
- 键/值长度单位为 byte。
"""

__all__ = ["engine", "pages", "runner", "archive", "derived"]
