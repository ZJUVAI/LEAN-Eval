from __future__ import annotations

from typing import Dict,List,Any

from .base_api import BaseAPIModel
from .base import ModelRegistry, Config


@ModelRegistry.register("deepseek_api")
class DeepSeekAPIModel(BaseAPIModel):
    """DeepSeek Chat（OpenAI 兼容接口）API 模型"""

    # 若用户未提供 api_url，则使用默认端点
    DEFAULT_API_URL = "https://api.deepseek.com/beta/chat/completions"

    # —— 构建请求体 —— #
    def _build_payload(self, prompt: str|List[str], max_len: int) -> Dict[str, Any]:
        """构建 DeepSeek / OpenAI Chat Completion 请求体。"""
        if isinstance(prompt,str):
            message = [{"role":"user","content":prompt}]
        elif isinstance(prompt,dict):
            message = [prompt]
        elif isinstance(prompt,list):
            message = [p for p in prompt]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}. Must be str or List[str].")
        return {
            "model": self.cfg.model_name,
            "messages":message,
            "max_tokens":max_len,
            "temperature":self.cfg.temperature
        }  
        

    # —— 解析响应 —— #
    def _parse_response(self, resp: Dict[str, Any]) -> str:
        """解析 Chat Completion 响应，提取 assistant 回复。"""
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return resp.get("error", "Unknown response format")

    # —— 覆写 load —— #
    def load(self) -> None:
        """如果未显式提供 api_url，则使用默认 DeepSeek 端点。"""
        if not self.cfg.api_url:
            self.cfg.api_url = self.DEFAULT_API_URL
        super().load()
