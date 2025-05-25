# LeanEval/models/huggingface.py
from __future__ import annotations

import logging
from typing import Iterable, List, Generator, Union, Sequence, Optional, Dict, Any
from pathlib import Path
import concurrent.futures as cf # 用于 batch_predict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from huggingface_hub import snapshot_download # 用于下载整个模型仓库
from pydantic import Field

from .base import BaseModel, Config, ModelRegistry

logger = logging.getLogger(__name__)

# --- HuggingFace 模型特定配置 (在基础 Config 上扩展) ---
class HuggingFaceModelConfig(Config):
    model_path: Optional[str] = None  
    tokenizer_name: Optional[str] = None 
    torch_dtype: Optional[str] = "auto" # "float16", "bfloat16", "float32", "auto"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    task: str = "text-generation"
    # 传递给 model.generate() 的参数
    generation_kwargs: Dict[str, Any] = Field(default_factory=dict)


@ModelRegistry.register("huggingface")
class HuggingFaceModel(BaseModel):
    """
    本地运行的 Hugging Face Transformer 模型。
    主要用于文本生成任务 (如代码补全、定理证明)。
    """
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None

    def __init__(self, cfg: HuggingFaceModelConfig):
        super().__init__(cfg)

    def load(self) -> None:
        """
        从 Hugging Face Hub 或本地路径加载模型和分词器。
        """
        if self._loaded:
            logger.info(f"Model '{self.cfg.model_name}' is already loaded.")
            return

        logger.info(f"Loading Hugging Face model: {self.cfg.model_name}...")
        model_id_or_path = self.cfg.model_path if self.cfg.model_path else self.cfg.model_name
        tokenizer_id_or_path = self.cfg.tokenizer_name if self.cfg.tokenizer_name else model_id_or_path

        # 确定 torch_dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto"
        }
        selected_torch_dtype = dtype_map.get(self.cfg.torch_dtype.lower() if self.cfg.torch_dtype else "auto", "auto")

        try:
            logger.info(f"Loading tokenizer from: {tokenizer_id_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id_or_path,
                trust_remote_code=self.cfg.trust_remote_code,
                use_fast=self.cfg.use_fast_tokenizer
            )
            logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")

            logger.info(f"Loading model from: {model_id_or_path} with dtype: {selected_torch_dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                torch_dtype=selected_torch_dtype,
                trust_remote_code=self.cfg.trust_remote_code,
            )
            logger.info(f"Model loaded: {self.model.__class__.__name__}")

            # 确保pad_token_id
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is not None:
                    logger.info(f"Tokenizer does not have pad_token_id. Setting it to eos_token_id: {self.tokenizer.eos_token_id}")
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
                else:
                    logger.warning(f"Tokenizer has no pad_token_id and no eos_token_id. Adding a new pad_token '[PAD]'.")
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id


            # 将模型移动到指定设备
            self.model.to(self.cfg.device)
            self.model.eval()

            self._loaded = True
            logger.info(f"Model '{self.cfg.model_name}' loaded successfully on device '{self.cfg.device}'.")

        except Exception as e:
            logger.error(f"Failed to load model '{self.cfg.model_name}': {e}", exc_info=True)
            self._loaded = False
            raise

    def release(self) -> None:
        """
        释放模型和分词器占用的资源。
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info(f"Model '{self.cfg.model_name}' released.")

    def _get_underlying_model(self) -> PreTrainedModel:
        """
        辅助函数，用于获取被 Accelerator (DDP/DP) 包装的底层原始模型。
        """
        if not self.model:
            raise RuntimeError("Model is not loaded.")
        
        # accelerate.prepare 可能会返回一个 AcceleratedOptimizer, AcceleratedScheduler,
        # 对于模型，它通常会返回一个 DistributedDataParallel 包装的模型，
        # 我们需要访问 .module 属性来获取原始的 nn.Module (Hugging Face PreTrainedModel)
        
        actual_model = self.model
        # 循环解包，以防有多层包装 (例如 DDP + AMP包装)
        while hasattr(actual_model, "module"):
            actual_model = actual_model.module
        
        if not isinstance(actual_model, PreTrainedModel):
            # 如果解包后仍然不是 PreTrainedModel，可能出了问题或者模型类型不同
            # 或者模型就是原始的，没有被包装（例如单CPU或单GPU未使用DDP时）
            # 但我们期望在调用 generate 之前，它应该是 PreTrainedModel 或其子类
            logger.warning(f"Underlying model type is {type(actual_model)}, not PreTrainedModel. "
                           "Ensure it has a 'generate' method if called.")
            # 如果上面没有 if isinstance(...) 的检查，也可以直接假设它有 generate 方法，
            # 但如果类型不对，调用时仍会出错。

        return actual_model

    def predict(self, prompt: str, max_len: int = 1024, **kwargs) -> str:
        """
        对单条输入进行推理，返回生成的文本。

        参数:
            prompt (str): 输入文本/Prompt。
            max_len (int): 指的是最大生成 token 的数量 (max_new_tokens)。
                           HuggingFace `generate` 中的 `max_length` 指的是 prompt+new_tokens 的总长度。
                           为了与 API 模型行为更一致，这里 `max_len` 代表 `max_new_tokens`。
            **kwargs: 可以覆盖 self.cfg.generation_kwargs 中的参数。

        返回:
            str: 模型生成的文本。
        """
        if not self._loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model '{self.cfg.model_name}' is not loaded. Call load() first.")

        underlying_model = self._get_underlying_model() # 获取原始模型
        device = next(underlying_model.parameters()).device # 获取原始模型实际所在的设备

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids_length = inputs['input_ids'].shape[1]

        gen_kwargs = self.cfg.generation_kwargs.copy()
        gen_kwargs.update(kwargs)
        if "max_new_tokens" not in gen_kwargs and "max_length" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = max_len
        elif "max_length" in gen_kwargs and "max_new_tokens" in gen_kwargs:
             logger.warning("Both max_length and max_new_tokens provided in generation_kwargs. Preferring max_new_tokens.")
             del gen_kwargs["max_length"] # 优先使用 max_new_tokens 如果两者都存在
        elif "max_length" in gen_kwargs:
            # 如果只提供了 max_length，需要确保它比输入长
            if gen_kwargs["max_length"] <= input_ids_length:
                logger.warning(f"max_length ({gen_kwargs['max_length']}) <= input length ({input_ids_length}). Forcing max_length to input_length + max_len_from_predict_param ({max_len}).")
                gen_kwargs["max_length"] = input_ids_length + max_len


        with torch.no_grad(): # 推理时不需要梯度
            output_ids = underlying_model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id, # 确保传入pad_token_id
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs
            )
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return result_text.strip()
    def batch_predict(
        self,
        prompts: Iterable[str],
        max_len: int = 1024, # max_new_tokens
        batch_size: int = 8, # 用于 tokenizer padding 和模型批处理
        **kwargs
    ) -> List[str]:
        if not self._loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model '{self.cfg.model_name}' is not loaded. Call load() first.")

        underlying_model = self._get_underlying_model() # 获取原始模型
        device = next(underlying_model.parameters()).device

        logger.info(f"Starting batch predict for {len(list(prompts))} prompts with batch_size {batch_size}...")
        results = []
        prompts_list = list(prompts) # 确保可以索引和切片

        # 合并和覆盖生成参数
        gen_kwargs = self.cfg.generation_kwargs.copy()
        gen_kwargs.update(kwargs)
        if "max_new_tokens" not in gen_kwargs and "max_length" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = max_len

        for i in range(0, len(prompts_list), batch_size):
            batch_prompts = prompts_list[i:i+batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids_lengths = [inputs['input_ids'][j].ne(self.tokenizer.pad_token_id).sum().item() for j in range(inputs['input_ids'].shape[0])]


            with torch.no_grad():
                final_gen_kwargs = gen_kwargs.copy()
                if "eos_token_id" not in final_gen_kwargs and self.tokenizer.eos_token_id is not None:
                    final_gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
                if "pad_token_id" not in final_gen_kwargs and self.tokenizer.pad_token_id is not None:
                    final_gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_type_id
                elif "pad_token_id" not in final_gen_kwargs and self.tokenizer.eos_token_id is not None:
                    final_gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

                output_ids = underlying_model.generate(
                    **inputs,
                    **final_gen_kwargs
                )

            # Decode
            for idx in range(output_ids.shape[0]):
                # output_ids[idx] 是完整的序列
                # input_ids_lengths[idx] 是原始prompt的长度 (不含padding)
                generated_ids = output_ids[idx][input_ids_lengths[idx]:]
                result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                results.append(result_text.strip())
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(prompts_list) -1 )//batch_size + 1}")

        return results

    # stream_predict 和 async_predict 可以暂时 NotImplemented 或依赖基类默认
    def stream_predict(self, inp: str, max_len: int = 1024, **kwargs) -> Generator[str, None, None]:
        logger.warning("Stream predict for HuggingFaceModel is not truly streaming yet, yields full result once.")
        yield self.predict(inp, max_len, **kwargs)

    async def async_predict(self, inp: str, max_len: int = 1024, **kwargs) -> str:
        logger.warning("Async predict for HuggingFaceModel is not truly async yet, runs synchronously.")
        return self.predict(inp, max_len, **kwargs)