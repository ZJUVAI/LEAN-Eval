# llm_framework/models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Generator, Type, Dict, Any
import logging, concurrent.futures as cf
from pydantic import BaseModel as _PydanticModel, Field
logger = logging.getLogger(__name__)

# ---------- Registry 注册表 ---------- #
class _ModelRegistry:
    """模型注册表，用于动态注册和创建模型类。"""
    
    _classes: Dict[str, Type["BaseModel"]] = {}

    @classmethod
    def register(cls, name: str):
        """注册一个模型类到注册表中。

        参数:
            name (str): 模型名称。

        返回:
            Callable: 用于装饰模型类的装饰器函数。
        """
        def decorator(model_cls: Type["BaseModel"]):
            if name in cls._classes:
                logger.warning("Model '%s' repeated, old=%s, new=%s",
                               name, cls._classes[name], model_cls)
            cls._classes[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **cfg) -> "BaseModel":
        """根据名称和配置创建一个模型实例。

        参数:
            name (str): 已注册的模型名称。
            cfg (dict): 模型初始化所需的配置项。

        返回:
            BaseModel: 创建好的模型实例。
        """
        if name not in cls._classes:
            raise ValueError(f"Model '{name}' not registered")
        return cls._classes[name](Config(**cfg))


# ---------- 模型配置 ---------- #
class Config(_PydanticModel):
    """模型配置类，包含基础字段及扩展字段支持。"""

    model_name: str = Field(default="unknown")
    device: str = Field(default="cpu")
    api_url: str | None = None
    api_key: str | None = None
    max_retry: int = Field(default=3, ge=0)
    timeout: float = Field(default=30.0, gt=0)
    temperature: float = Field(default=0.7)
    # 可扩展自定义字段

    class Config:
        extra = "allow"  # 允许传入额外字段


# ---------- 抽象基类 ---------- #
class BaseModel(ABC):
    """所有模型的统一接口基类，定义基本推理与生命周期管理方法。"""

    def __init__(self, cfg: Config):
        """初始化模型实例，并加载模型资源。

        参数:
            cfg (Config): 模型配置对象。
        """
        self.cfg = cfg
        self._loaded = False
        self.load()

    # —— 模型生命周期 —— #
    @abstractmethod
    def load(self) -> None:
        """加载模型资源，子类需实现。"""
        ...

    def release(self) -> None:
        """释放模型资源，可根据需要重写。"""
        ...

    def __enter__(self):
        """支持 with 语法的上下文管理，进入时返回自身。

        返回:
            BaseModel: 当前模型实例。
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """支持 with 语法的上下文管理，退出时释放资源。

        参数:
            exc_type (Type[BaseException] | None): 异常类型。
            exc (BaseException | None): 异常实例。
            tb (TracebackType | None): 异常回溯信息。
        """
        self.release()

    # —— 推理接口 —— #
    @abstractmethod
    def predict(self, inp: str, max_len: int = 1024) -> str:
        """对单条输入进行推理，返回结果。

        参数:
            inp (str): 输入文本。
            max_len (int, 可选): 最大生成长度，默认 1024。

        返回:
            str: 推理生成的文本结果。
        """
        pass

    def batch_predict(self, inputs: Iterable[str], max_len: int = 1024,
                      workers: int = 4) -> List[str]:
        """批量推理，逐条预测并返回结果列表。

        参数:
            inputs (Iterable[str]): 输入文本集合。
            max_len (int, 可选): 每条输入的最大生成长度，默认 1024。

        返回:
            List[str]: 预测结果列表。
        """
        with cf.ThreadPoolExecutor(workers) as pool:
            fut = [pool.submit(self.predict, x, max_len) for x in inputs]
            return [f.result() for f in fut]

    def stream_predict(self, inp: str, max_len: int = 128) -> Generator[str, None, None]:
        """流式推理，逐步生成预测结果（默认整体返回）。

        参数:
            inp (str): 输入文本。
            max_len (int, 可选): 最大生成长度，默认 128。

        返回:
            Generator[str, None, None]: 逐步生成的预测文本。
        """
        yield self.predict(inp, max_len)

    async def async_predict(self, inp: str, max_len: int = 128) -> str:
        """异步推理接口（默认同步实现）。

        参数:
            inp (str): 输入文本。
            max_len (int, 可选): 最大生成长度，默认 128。

        返回:
            str: 推理生成的文本结果。
        """
        return self.predict(inp, max_len)

    # —— 元信息 —— #
    def __repr__(self):
        """返回模型的字符串表示，用于打印和调试。

        返回:
            str: 模型字符串描述。
        """
        return f"{self.__class__.__name__}(model_name={self.cfg.model_name})"


# ----------------- 导出 ----------------- #
ModelRegistry = _ModelRegistry   # 给外部使用时的别名
__all__ = ["Config", "ModelRegistry"]
