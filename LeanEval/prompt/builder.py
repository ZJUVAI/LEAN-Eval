from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Sequence

from datasets import LeanItem   

# ============================================================
# 抽象基类
# ============================================================

class PromptBuilder(ABC):
    """LeanItem ➜ Prompt 的统一接口"""

    # ---- 单字符串形式（instruct / completion 风格）---------
    @abstractmethod
    def build_str(self, item: LeanItem) -> str: ...

    # ---- Chat 形式（OpenAI / DeepSeek / Claude 等）----------
    def build_chat(self, item: LeanItem) -> List[dict]:
        # 默认：把 build_str() 塞进 1 条 user 消息
        return [{"role": "user", "content": self.build_str(item)}]
    
    def build_chat_for_tactic(self, leanCode: str, tips: List[str]) -> str: ...


# ============================================================
# 工具函数
# ============================================================

def _make_lean_block(code: str) -> str:
    """加 ```lean 包裹；避免提示模型错判"""
    return f"```lean\n{code.strip()}\n```"


# ============================================================
#  SimplePromptBuilder —— 零样例
# ============================================================

_SIMPLE_TEMPLATE = (
    "你是 Lean4 的专家，请为下列定理补全证明，只返回完整 Lean 代码。\n\n{code_block}"
)

class SimplePromptBuilder(PromptBuilder):
    """LeanItem.prompt_ready_stmt 直接塞到模板"""

    def __init__(self, template: str | None = None):
        self.template = template or _SIMPLE_TEMPLATE

    def build_str(self, item: LeanItem) -> str:
        code_block = _make_lean_block(
            f"{item.prompt_ready_stmt} := by\n  -- your proof here"
        )
        return self.template.format(code_block=code_block)


# ============================================================
#  FewShotPromptBuilder —— 带示例对话
# ============================================================

_FEWSHOT_SYSTEM = "你是 Lean4 的专家，请为下列定理补全证明，只返回完整 Lean 代码，风格见示例。"
_USER_PREFIX = "请补全以下 Lean 定理：\n\n"

class FewShotPromptBuilder(PromptBuilder):
    """
    shots: [("用户消息", "助理回复 Lean 代码"), ...]
    """

    def __init__(
        self,
        shots: Sequence[tuple[str, str]],
        system_msg: str = _FEWSHOT_SYSTEM,
        user_prefix: str = _USER_PREFIX,
    ):
        self.shots = list(shots)
        self.system_msg = system_msg
        self.user_prefix = user_prefix

    # -- Chat 风格：多轮 messages ------------------------------
    def build_chat(self, item: LeanItem) -> List[dict]:
        messages: List[dict] = [{"role": "system", "content": self.system_msg}]
        # few-shot 示例
        for u, a in self.shots:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})

        # 当前题目
        user_msg = self.user_prefix + _make_lean_block(
            f"{item.prompt_ready_stmt} := by\n  -- your proof here"
        )
        messages.append({"role": "user", "content": user_msg})
        return messages
    
    def build_chat_for_tactic(self, leanCode: str, tips: List[str]) -> str:
        system_msg: str = "你是 Lean4 的专家，请根据已有的定理证明过程和有关目标的提示，补充一行证明，使其更接近证明完成，只返回完整 Lean 代码，风格见示例。"
        messages: List[dict] = [{"role": "system", "content": system_msg}]
        # few-shot 示例
        for u, a in self.shots:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})

        # 当前题目
        for idx, tip in enumerate(tips):
            user_prefix = f"下面给出第{idx + 1}个提示:\n\n"
            user_msg = user_prefix + tip
            messages.append({"role": "user", "content": user_msg})

        user_prefix = "请对以下Lean定理证明补充一行:\n\n"
        user_msg = user_prefix + _make_lean_block(
            f"{leanCode}\n  -- the next line of the proof here"
        )
        messages.append({"role": "user", "content": user_msg})
        return messages

    def build_str(self, item: LeanItem) -> str:   # pragma: no cover
        raise NotImplementedError(
            "FewShotPromptBuilder 主要用于 chat 消息；请调用 build_chat()"
        )


# ============================================================
#  Builder Registry —— 方便配置化选择
# ============================================================

BUILDER_REGISTRY: dict[str, type[PromptBuilder]] = {
    "simple": SimplePromptBuilder,
    "fewshot": FewShotPromptBuilder,
}

def get_builder(name: str, **kwargs) -> PromptBuilder:
    """
    通过字符串创建 Builder，便于在 YAML/JSON 配置里写:
        prompt_builder:
          name: fewshot
          shots: [...]
    """
    if name not in BUILDER_REGISTRY:
        raise KeyError(f"Unknown PromptBuilder: {name}")
    return BUILDER_REGISTRY[name](**kwargs)

def register_builder(name: str, cls: type[PromptBuilder], *, override: bool = False) -> None:
    """
    把新的 PromptBuilder 子类注册到 BUILDER_REGISTRY。

    参数
    ----
    name        : 注册名（配置文件里用的字符串）
    cls         : PromptBuilder 的子类
    override    : 若为 False 且 name 已存在，则报错；为 True 时允许覆盖
    """
    if not issubclass(cls, PromptBuilder):
        raise TypeError(f"{cls} 必须继承 PromptBuilder")

    if name in BUILDER_REGISTRY and not override:
        raise KeyError(f"PromptBuilder '{name}' 已存在；"
                       f"如需覆盖请传 override=True")

    BUILDER_REGISTRY[name] = cls