from __future__ import annotations

import json
import threading
import logging
from typing import Dict, Any

import requests
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    before_sleep_log,
    RetryError,
)

from .base import BaseModel, Config, ModelRegistry


# ---------- 日志设置 ---------- #
logger = logging.getLogger(__name__)


class BaseAPIModel(BaseModel):
    """REST / HTTP API 形式大模型的通用父类。"""

    _lock = threading.Lock()  # 线程安全锁（防止多线程同时写 socket）

    # —— 重试策略 —— #
    RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=20)  # 指数退避
    RETRY_STOP = stop_after_attempt(5)  # 最多重试 5 次，可在子类覆盖

    # —— 内部工具：带重试的 POST —— #
    @retry(
        wait=RETRY_WAIT,
        stop=RETRY_STOP,
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送 POST 请求，失败时自动重试并输出日志。

        参数:
            payload (Dict[str, Any]): 请求体 JSON。

        返回:
            Dict[str, Any]: 解析后的 JSON 响应体。
        """
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        with self._lock:
            resp = requests.post(
                self.cfg.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.cfg.timeout,
            )
        resp.raise_for_status()
        return resp.json()

    # —— 抽象方法 —— #
    def _build_payload(self, prompt: str, max_len: int) -> Dict[str, Any]:
        """构建请求体，子类必须实现。"""
        raise NotImplementedError

    def _parse_response(self, resp: Dict[str, Any]) -> str:
        """解析 API 响应，子类可按需覆写。"""
        return resp.get("text", "")

    # —— BaseModel 接口实现 —— #
    def load(self) -> None:
        """API 型模型通常无需重量级加载，直接标记为已加载。"""
        self._loaded = True

    def predict(self, inp: str, max_len: int = 2048) -> str:
        """调用远端 API 同步推理。"""
        try:
            payload = self._build_payload(inp, max_len)
            resp = self._post(payload)
            return self._parse_response(resp)
        except RetryError as e:
            raise RuntimeError(f"API call failed after {self.RETRY_STOP.max_attempt_number} attempts: {e}") from e
