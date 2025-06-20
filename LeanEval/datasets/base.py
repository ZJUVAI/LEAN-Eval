from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Iterable, Sequence, Protocol, List


from .schema import LeanItem


class SupportsLenAndGetitem(Protocol):
    def __len__(self): ...
    def __getitem__(self, idx: int): ...


class BaseDataset(ABC, Sequence[LeanItem]):
    """
    抽象数据集：
      * 顺序索引 __getitem__/__len__
      * 支持迭代器 .iter(batch_size=…)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: List[LeanItem] | None = None

    # ------- 子类必须实现两个私有方法 ------- #
    @abstractmethod
    def _load(self) -> List[LeanItem]: ...
    @abstractmethod
    def _read_raw(self) -> Iterable: ...

    # ------- Sequence的协议 ------- #
    def __len__(self) -> int:
        if self._data is None:
            self._data = self._load()
        return len(self._data)

    def __getitem__(self, idx: int) -> LeanItem:
        if self._data is None:
            self._data = self._load()
        return self._data[idx]

    # 小批量迭代器
    def iter(self, batch_size: int | None = None) -> Iterator[List[LeanItem] | LeanItem]:
        if self._data is None:
            self._data = self._load()
        if batch_size is None or batch_size <= 1:
            yield from self._data
        else:
            for i in range(0, len(self._data), batch_size):
                yield self._data[i : i + batch_size]

    # 过滤器链：难度、关键词……
    def filter(self, **conds) -> "BaseDataset":
        if self._data is None:
            self._data = self._load()

        def _match(it: LeanItem) -> bool:
            return all(getattr(it, k) == v for k, v in conds.items())

        clone = self.__class__(self.path)
        clone._data = [it for it in self._data if _match(it)]
        return clone
