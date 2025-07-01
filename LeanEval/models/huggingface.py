# LeanEval/models/huggingface.py
from __future__ import annotations

import logging
from typing import Iterable, List, Generator, Union, Sequence, Optional, Dict, Any
from pathlib import Path
import concurrent.futures as cf

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from huggingface_hub import snapshot_download
from pydantic import Field

from .base import BaseModel, Config, ModelRegistry

logger = logging.getLogger(__name__)

class HuggingFaceModelConfig(Config):
    model_path: Optional[str] = None  
    tokenizer_name: Optional[str] = None 
    torch_dtype: Optional[str] = "auto"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    task: str = "text-generation"
    generation_kwargs: Dict[str, Any] = Field(default_factory=dict)

@ModelRegistry.register("huggingface")
class HuggingFaceModel(BaseModel):
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None

    def __init__(self, cfg: HuggingFaceModelConfig):
        super().__init__(cfg)

    def load(self) -> None:
        if self._loaded:
            logger.info(f"Model '{self.cfg.model_name}' is already loaded.")
            return

        logger.info(f"Loading Hugging Face model: {self.cfg.model_name}...")
        model_id_or_path = self.cfg.model_path if self.cfg.model_path else self.cfg.model_name
        tokenizer_id_or_path = self.cfg.tokenizer_name if self.cfg.tokenizer_name else model_id_or_path

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "auto": "auto"}
        selected_torch_dtype = dtype_map.get(self.cfg.torch_dtype.lower() if self.cfg.torch_dtype else "auto", "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id_or_path,
            trust_remote_code=self.cfg.trust_remote_code,
            use_fast=self.cfg.use_fast_tokenizer
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=selected_torch_dtype,
            trust_remote_code=self.cfg.trust_remote_code,
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(self.cfg.device)
        self.model.eval()
        self._loaded = True
        logger.info(f"Model '{self.cfg.model_name}' loaded successfully on device '{self.cfg.device}'.")

    def release(self) -> None:
        if self.model is not None: del self.model
        if self.tokenizer is not None: del self.tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self._loaded = False
        logger.info(f"Model '{self.cfg.model_name}' released.")

    def _get_underlying_model(self) -> PreTrainedModel:
        actual_model = self.model
        while hasattr(actual_model, "module"):
            actual_model = actual_model.module
        return actual_model

    # <<< --- 核心修改：重写 predict 方法 --- >>>
    def predict(self, prompt: Union[str, List[Dict[str, str]]], max_len: int = 1024, **kwargs) -> str:
        if not self._loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model '{self.cfg.model_name}' is not loaded. Call load() first.")

        # 如果输入是聊天记录 (List[dict])，则使用聊天模板转换
        if isinstance(prompt, list):
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise TypeError(
                    f"The tokenizer for {self.cfg.model_name} does not support chat templates. "
                    "Cannot process List[dict] input."
                )
            # 将聊天记录列表转换为单个字符串
            prompt_str = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True # 对话最后加上助手的起始提示
            )
        elif isinstance(prompt, str):
            prompt_str = prompt
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}. Must be str or List[Dict[str, str]].")

        underlying_model = self._get_underlying_model()
        device = next(underlying_model.parameters()).device

        inputs = self.tokenizer(prompt_str, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = self.cfg.generation_kwargs.copy()
        gen_kwargs.update(kwargs)
        if "max_new_tokens" not in gen_kwargs and "max_length" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = max_len

        with torch.no_grad():
            output_ids = underlying_model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs
            )
        
        # 解码时跳过输入部分
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return result_text.strip()
    # <<< --- 修改结束 --- >>>

    # <<< --- 相应地，也更新 batch_predict 方法 --- >>>
    def batch_predict(
        self,
        prompts: Iterable[Union[str, List[Dict[str, str]]]],
        max_len: int = 1024,
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        if not self._loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model '{self.cfg.model_name}' is not loaded.")

        # 将所有输入统一转换为字符串
        prompt_strings = []
        for p in prompts:
            if isinstance(p, list):
                prompt_strings.append(self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                ))
            elif isinstance(p, str):
                prompt_strings.append(p)
            else:
                raise TypeError(f"Unsupported prompt type in batch: {type(p)}")

        underlying_model = self._get_underlying_model()
        device = next(underlying_model.parameters()).device
        
        results = []
        for i in range(0, len(prompt_strings), batch_size):
            batch_prompts = prompt_strings[i:i+batch_size]
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            gen_kwargs = self.cfg.generation_kwargs.copy()
            gen_kwargs.update(kwargs)
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = max_len

            with torch.no_grad():
                output_ids = underlying_model.generate(**inputs, **gen_kwargs)

            # 解码每个结果
            for idx in range(output_ids.shape[0]):
                input_len = inputs['input_ids'][idx].ne(self.tokenizer.pad_token_id).sum().item()
                generated_ids = output_ids[idx][input_len:]
                result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                results.append(result_text.strip())
        return results

    # stream_predict 和 async_predict 保持不变
    def stream_predict(self, inp: str, max_len: int = 1024, **kwargs) -> Generator[str, None, None]:
        yield self.predict(inp, max_len, **kwargs)

    async def async_predict(self, inp: str, max_len: int = 1024, **kwargs) -> str:
        return self.predict(inp, max_len, **kwargs)