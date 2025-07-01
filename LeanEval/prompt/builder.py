# LeanEval/prompt/builder.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Sequence, Dict, Any
import json
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

# 确保导入 LeanItem
from LeanEval.datasets import LeanItem

# ============================================================
# 新增：Prompt管理器，这是一个辅助类，只在使用模板时才被调用
# ============================================================
class PromptManager:
    _instance = None
    _initialized = False

    def __new__(cls, template_dir: str | Path = 'LeanEval/prompt_templates'):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, template_dir: str | Path = 'LeanEval/prompt_templates'):
        if self._initialized:
            return
        template_path = Path(template_dir)
        if not template_path.is_dir():
            raise FileNotFoundError(f"Prompt template directory not found: {template_path.resolve()}")
        self.env = Environment(loader=FileSystemLoader(template_path), trim_blocks=True, lstrip_blocks=True)
        prompts_file = template_path / 'prompts.json'
        if not prompts_file.is_file():
            raise FileNotFoundError(f"Prompts config file not found: {prompts_file.resolve()}")
        with prompts_file.open('r', encoding='utf-8') as f:
            self.prompts = json.load(f)
        self._initialized = True

    def render_chat_prompt(self, template_name: str, **context: Any) -> List[Dict[str, str]]:
        if template_name not in self.prompts:
            raise ValueError(f"Template '{template_name}' not found in prompts.json")
        prompt_structure = self.prompts[template_name]
        system_template = self.env.from_string(prompt_structure.get('system', ''))
        user_template = self.env.from_string(prompt_structure.get('user', ''))
        messages = []
        system_content = system_template.render(**context).strip()
        if system_content:
            messages.append({"role": "system", "content": system_content})
        user_content = user_template.render(**context).strip()
        if user_content:
            messages.append({"role": "user", "content": user_content})
        return messages

# ============================================================
# 抽象基类 (保留原有结构)
# ============================================================
class PromptBuilder(ABC):
    @abstractmethod
    def build_str(self, item: LeanItem) -> str: ...

    def build_chat(self, item: LeanItem) -> List[dict]:
        return [{"role": "user", "content": self.build_str(item)}]
    
    def build_chat_for_tactic(self, leanCode: str, tips: List[str]) -> List[dict]:
        raise NotImplementedError("This builder does not support tactic generation.")

# ============================================================
# 工具函数 (保留)
# ============================================================
def _make_lean_block(code: str) -> str:
    return f"```lean\n{code.strip()}\n```"

# ============================================================
#  SimplePromptBuilder (完全保留，确保向后兼容)
# ============================================================
_SIMPLE_TEMPLATE = (
    "你是 Lean4 的专家，请为下列定理补全证明，只返回完整 Lean 代码。\n\n{code_block}"
)

class SimplePromptBuilder(PromptBuilder):
    def __init__(self, template: str | None = None):
        self.template = template or _SIMPLE_TEMPLATE

    def build_str(self, item: LeanItem) -> str:
        code_block = _make_lean_block(
            f"{item.prompt_ready_stmt} := by\n  -- your proof here"
        )
        return self.template.format(code_block=code_block)

# ============================================================
#  FewShotPromptBuilder (保留原有逻辑，但重写策略生成方法)
# ============================================================
_FEWSHOT_SYSTEM = "你是 Lean4 的专家，请为下列定理补全证明，只返回完整 Lean 代码，风格见示例。"
_USER_PREFIX = "请补全以下 Lean 定理：\n\n"

class FewShotPromptBuilder(PromptBuilder):
    def __init__(
        self,
        shots: Sequence[tuple[str, str]],
        system_msg: str = _FEWSHOT_SYSTEM,
        user_prefix: str = _USER_PREFIX,
    ):
        self.shots = list(shots)
        self.system_msg = system_msg
        self.user_prefix = user_prefix

    # 原有的 build_chat 方法完全保留
    def build_chat(self, item: LeanItem) -> List[dict]:
        messages: List[dict] = [{"role": "system", "content": self.system_msg}]
        for u, a in self.shots:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        user_msg = self.user_prefix + _make_lean_block(
            f"{item.prompt_ready_stmt} := by\n  -- your proof here"
        )
        messages.append({"role": "user", "content": user_msg})
        return messages

    # <<< --- 核心修改：重写此方法以使用模板管理器 --- >>>
    def build_chat_for_tactic(self, lean_code: str, tips: List[str]) -> List[dict]:
        """
        为 BFSProver 构建一个优化的 prompt，用于生成下一步的 tactic。
        此方法现在使用外部JSON模板来构造请求。
        """
        # 在方法内部实例化管理器，以避免修改__init__签名
        manager = PromptManager()
        return manager.render_chat_prompt(
            'tactic_generation_multiple',
            lean_code=lean_code,
            tips=tips
        )
    # <<< --- 修改结束 --- >>>

    def build_str(self, item: LeanItem) -> str:
        raise NotImplementedError("FewShotPromptBuilder主要用于chat消息；请调用 build_chat()")

# ============================================================
# 注册表与工厂函数 (完全保留)
# ============================================================
BUILDER_REGISTRY: dict[str, type[PromptBuilder]] = {
    "simple": SimplePromptBuilder,
    "fewshot": FewShotPromptBuilder,
}

def get_builder(name: str, **kwargs) -> PromptBuilder:
    if name not in BUILDER_REGISTRY:
        raise KeyError(f"Unknown PromptBuilder: {name}")
    return BUILDER_REGISTRY[name](**kwargs)

def register_builder(name: str, cls: type[PromptBuilder], *, override: bool = False) -> None:
    if not issubclass(cls, PromptBuilder):
        raise TypeError(f"{cls} 必须继承 PromptBuilder")
    if name in BUILDER_REGISTRY and not override:
        raise KeyError(f"PromptBuilder '{name}' 已存在；如需覆盖请传 override=True")
    BUILDER_REGISTRY[name] = cls