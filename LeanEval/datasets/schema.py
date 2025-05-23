from __future__ import annotations # 推迟类型注解的求值时机
from pydantic import BaseModel, Field #BaseModel 作为数据我们定义的数据结构的基类
from typing import List


class LeanItem(BaseModel, frozen=True, extra="allow"):
    id: str
    imports: List[str] = Field(default_factory=list) #Field 为属性添加约束
    statement: str
    extra_ctx: str | None = ""
    difficulty: int = 1

    @property
    def prompt_ready_stmt(self) -> str:
        """预拼接好 imports + statement，方便 PromptBuilder 直接用"""
        parts = []
        if self.imports:
            imports_txt = "\n".join([f"import {i}" for i in self.imports])
            parts.append(imports_txt)
        if self.extra_ctx:
            parts.append(self.extra_ctx)
        parts.append(self.statement)
        return "\n\n".join(parts)
    
    @property
    def difficulty(self) -> int:
        """返回这道题的难度，用于评分"""
        return self.difficulty
    
