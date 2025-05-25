from __future__ import annotations

from typing import Dict, List, Any

from .base_api import BaseAPIModel
from .base import ModelRegistry, Config


@ModelRegistry.register("gemini_api")
class GeminiAPIModel(BaseAPIModel):
    """Gemini API 模型"""

    # 如果 Gemini 有默认的 API URL，在这里设置
    # DEFAULT_API_URL = "你的_GEMINI_默认_API_URL"

    def _build_payload(self, prompt: str | List[dict], max_len: int) -> Dict[str, Any]:
        """构建 Gemini API 的请求体。"""
        message = []
        if isinstance(prompt, str):
            # 如果只是一个字符串 prompt，假设使用默认的 system message
            message.append({"role": "user", "content": prompt})
        elif isinstance(prompt,dict):
            message = [prompt]
        elif isinstance(prompt, list):
            # 确保 prompt 是一个字典列表 (messages)
            if not all(isinstance(p, dict) for p in prompt):
                raise ValueError("如果 prompt 是列表，它必须是消息字典的列表。")
            message = prompt
        else:
            raise ValueError(
                f"Invalid prompt type: {type(prompt)}. Must be str,dict or List[str]."
            )
        
        payload = {
            "model": self.cfg.model_name,  # 例如："gemini-2.0-flash-exp"
            "messages": message,
            "temperature": self.cfg.temperature,
            # Gemini 可能使用像 'maxOutputTokens' 这样的参数而不是 'max_tokens'
            # 根据 Gemini 的特定 API 要求添加或修改参数。
            "max_tokens": max_len, # 这可能需要为 Gemini 进行调整
        }
        return payload

    def _parse_response(self, resp: Dict[str, Any]) -> str:
        """解析 Gemini API 响应以提取内容。"""
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            error_message = resp.get("error", {}).get("message", "未知的响应格式")
            print(f"解析 Gemini 响应时出错: {e}, API 错误: {error_message}")
            return f"API 返回错误: {error_message}"

    def load(self) -> None:
        if not self.cfg.api_url and hasattr(self.cfg, 'base_url') and self.cfg.base_url:
            self.cfg.api_url = self.cfg.base_url # BaseAPIModel 使用 self.cfg.api_url 进行请求
        super().load()