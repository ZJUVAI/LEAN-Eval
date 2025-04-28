from __future__ import annotations

import json
import threading
import logging
from pathlib import Path
from typing import Dict, Optional, Union, Sequence, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

    # —— 带重试的 POST —— #
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
        """构建请求体，子类必须实现。返回的payload是一个列表，每一个元素是一个 {"role":"user", ...} 形式的对话或单独是字符串"""
        raise NotImplementedError

    def _parse_response(self, resp: Dict[str, Any]) -> str:
        """解析 API 响应，子类可按需覆写。"""
        return resp.get("text", "")

    # —— BaseModel 接口实现 —— #
    def load(self) -> None:
        """API 型模型通常无需重量级加载，直接标记为已加载。"""
        self._loaded = True

    def predict(
            self, 
            prompt: Union[str | dict | Sequence[dict]], 
            max_len: int = 2048, 
            num_workers: int = 8, 
            save_dir: Optional[Path] = None,
            save_paths: Optional[Sequence[Union[str | Path]]] = None
        ) -> Union[str | List[str]]:
        """
        - prompt: str，dict 或 List[str](具体类型取决于prompt的生成方式)；若为 List，自动并行推理
        - num_workers: 线程池大小，仅批量时生效
        - save_dir: 若提供，则把结果依次保存为 0.lean, 1.lean, ...
        - save_paths: 与 prompt 的数量相等；逐条指定保存文件名（优先生效）
        """
        # ------- 只传入了一个问题时（str）------------
        if isinstance(prompt, str):
            file_path = (Path(save_dir) / "0.lean") if save_dir else None
            if save_paths:
                file_path = Path(save_paths[0])
            return self._predict_one(prompt,file_path)

        prompts: Sequence[str] = list(prompt)

        # -------- 生成保存文件路径 -------------------
        if save_paths:
            if len(save_paths) != len(prompts):
                raise ValueError("save_path length mush match number of prompts")
            paths = [Path(p) for p in save_paths]
        elif save_dir:
            base = Path(save_dir)
            paths = [base / f"{i}.lean" for i in range(len(prompts))]
        else:
            paths = [None]*len(prompts)
        
        results: List[Optional[str]] = [None]*len(prompts)

        def job(i: int) -> None:
            try:
                results[i] = self._predict_one(prompt=prompts[i],max_len=max_len,save_path=paths[i])
            except Exception as e:
                results[i] = f"<error>:{e}"
        
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(job,i): i for i in range(len(prompts))}
            for fu in tqdm(as_completed(futures),total=len(futures)):
                pass
        return results
    
    def _predict_one(self, prompt: str , max_len: int = 2048, save_path: Optional[Path] = None) -> str:
        try:
            payload = self._build_payload(prompt,max_len)
            resp = self._post(payload)
            result = self._parse_response(resp)
            if save_path:
                save_path.parent.mkdir(parents=True,exist_ok=True)
                save_path.write_text(result,encoding="utf-8")
            return result
        except RetryError as e:
            raise RuntimeError(
                f"API call failed after {self.RETRY_STOP.max_attempt_number} "
                f"attempts: {e}"
            ) from e
