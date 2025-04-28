from __future__ import annotations
import json
from typing import Iterable, List

from .base import BaseDataset
from .schema import LeanItem


class JsonDataset(BaseDataset):
    """一次性加载整个 JSON (list) 文件"""

    def _read_raw(self) -> Iterable:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)                 # 返回 list[dict]

    def _load(self) -> List[LeanItem]:
        raw = self._read_raw()
        return [LeanItem(**item) for item in raw]


class JsonlDataset(BaseDataset):
    """逐行 JSONL"""

    def _read_raw(self) -> Iterable[str]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def _load(self) -> List[LeanItem]:
        return [LeanItem(**item) for item in self._read_raw()]
