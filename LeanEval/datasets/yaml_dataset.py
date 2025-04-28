from __future__ import annotations

from typing import Iterable, List

import yaml

from .base import BaseDataset
from .schema import LeanItem


class YamlDataset(BaseDataset):
    """一次性加载整个 YAML 文件

    支持两种顶层结构：
    1) 直接是列表   - [{...}, {...}]
    2) dict 包含 items 字段
       items:
         - {...}
         - {...}
    """

    # ---------- 读取原始行 ----------
    def _read_raw(self) -> Iterable[dict]:
        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
            # 容错：若最外层是 dict 且含 items 字段
            if isinstance(data, dict) and "items" in data:
                data = data["items"] or []
            if not isinstance(data, list):
                raise ValueError("YAML 文件格式应为列表，或根节点包含 items 列表")
            return data

    # ---------- 构造 LeanItem ----------
    def _load(self) -> List[LeanItem]:
        return [LeanItem(**item) for item in self._read_raw()]
