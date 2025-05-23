from __future__ import annotations
from typing import Optional, List
from enum import Enum
import itertools #获取自增id
import threading #多线程锁

class Status(Enum):
    Open = "Open"
    Proved = "Proved"
    Error = "Error"

class Node:
    _id_generator = itertools.count(start=1)

    leanCode: str
    src: Optional[Edge]
    dst: Optional[List[Edge]]
    height: int
    status: Status
    id: int

    def __init__(self, leanCode: str, src: Optional[Edge] = None, dst: Optional[List[Edge]] = [], height: int = 0, status: str = Status.Open):
        self.leanCode = leanCode
        self.src = src
        self.dst = dst
        self.height = height
        self.status = status
        self.id = next(self._id_generator)
        self._lock = threading.Lock()

    def Update(self):
        with self._lock:
            if self.status == Status.Open and len(self.dst) > 0:
                if any(self.dst[i].dst.status == Status.Proved for i in range(len(self.dst))):
                    self.status = Status.Proved
                elif all(self.dst[i].dst.status == Status.Error for i in range(len(self.dst))):
                    self.status = Status.Error
                else:
                    self.status = Status.Open

            if self.status != Status.Open and self.src is not None:
                shouldUpdate = True
            else:
                shouldUpdate = False
        if shouldUpdate and self.src is not None and self.src.src is not None:  #将递归函数判断语句放在锁外以避免线程问题
            self.src.src.Update()
    def __lt__(self, other: Node):
        return self.height < other.height


class Edge:

    def __init__(self, tactic: str, height: int, src: Optional[Node] = None, dst: Optional[Node] = None):
        self.tactic = tactic
        self.src = src
        self.dst = dst
        self.height = height

    tactic: str
    src: Node
    dst: Node
    height: int