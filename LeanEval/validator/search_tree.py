# LeanEval/validator/search_tree.py

from __future__ import annotations
from typing import Optional, List, Dict, Any
from enum import Enum
import itertools
import threading

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

    def __init__(self, leanCode: str, src: Optional[Edge] = None, dst: Optional[List[Edge]] = None, height: int = 0, status: Status = Status.Open):
        self.leanCode = leanCode
        self.src = src if src is not None else None
        self.dst = dst if dst is not None else []
        self.height = height
        self.status = status
        self.id = next(self._id_generator)
        self._lock = threading.Lock()

    def Update(self):
        with self._lock:
            if self.status == Status.Open and len(self.dst) > 0:
                if any(edge.dst.status == Status.Proved for edge in self.dst if edge.dst):
                    self.status = Status.Proved
                elif all(edge.dst.status == Status.Error for edge in self.dst if edge.dst):
                    self.status = Status.Error
                else:
                    self.status = Status.Open

            shouldUpdate = self.status != Status.Open and self.src is not None
        
        if shouldUpdate and self.src and self.src.src:
            self.src.src.Update()

    def __lt__(self, other: Node):
        return self.height < other.height

    # <<< --- 新增的方法 --- >>>
    def reconstruct_path(self) -> List[Dict[str, Any]]:
        """
        从当前节点回溯到根节点，重建证明路径上的所有策略和代码状态。

        Returns:
            List[Dict[str, Any]]: 一个包含每一步信息的列表，
                                 例如 [{'step': 0, 'tactic': 'intro h', 'code_after_tactic': '...'}, ...]
        """
        path = []
        current_node = self
        while current_node and current_node.src:
            step_info = {
                'step': current_node.height - 1,
                'tactic': current_node.src.tactic,
                'code_after_tactic': current_node.leanCode
            }
            path.append(step_info)
            current_node = current_node.src.src
        
        # 路径是倒序的，需要反转回来
        return sorted(path, key=lambda x: x['step'])
    # <<< --- 新增结束 --- >>>

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