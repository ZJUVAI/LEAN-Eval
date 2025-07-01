# LeanEval/runner/local_runner_with_search.py
import os
import torch
import sys
from pathlib import Path
from time import time, strftime, time_ns
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json

import accelerate
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from torch.utils.data import Dataset as TorchDataset, DataLoader

# 导入项目模块
from LeanEval.datasets import JsonDataset, LeanItem
from LeanEval.models import ModelRegistry, HuggingFaceModelConfig
from LeanEval.validator.proof_validator import ProofValidator
# 核心：导入 BFSProver
from LeanEval.validator.proof_search import BFSProver
# 导入用于策略生成的 PromptBuilder
from LeanEval.prompt import get_builder, FewShotPromptBuilder
from LeanEval.utils import extract_lean_block

class LocalSearchRunner:
    """
    一个使用Hugging Face模型和BFS搜索式证明在本地运行LeanEval评估的Runner。
    """
    def __init__(
        self,
        model_id: str,
        dataset_path: str,
        output_dir_base: str = "./outputs_local_search_runner",
        # --- BFSProver 相关配置 ---
        tactic_shots: List[Tuple[str, str]] = None,
        bfs_degree: int = 4,
        bfs_timeout: int = 1800,
        bfs_prover_num_workers: int = 4,
        # --- 模型与环境配置 ---
        per_device_batch_size: int = 1, # 搜索任务通常一次只处理一个问题
        dataloader_num_workers: int = 2,
        # 为生成单步策略优化的模型参数
        max_new_tokens: int = 4096,
        temperature: float = 0.4,
        mixed_precision: str = 'fp16', # 'no', 'fp16', 'bf16'
        hf_config_overrides: Dict[str, Any] = None,
    ):
        # 初始化 Accelerator
        self.accelerator = Accelerator(mixed_precision=mixed_precision if torch.cuda.is_available() else 'no')
        self.device = self.accelerator.device

        # --- 保存与 Runner 自身相关的配置 ---
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.output_dir_base = output_dir_base
        self.per_device_batch_size = per_device_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.hf_config_overrides = hf_config_overrides or {}

        # --- 保存 BFSProver 的特定配置 ---
        self.tactic_shots = tactic_shots or []
        self.bfs_degree = bfs_degree
        self.bfs_timeout = bfs_timeout
        self.bfs_prover_num_workers = bfs_prover_num_workers
        self.model_gen_max_tokens = max_new_tokens
        self.model_gen_temperature = temperature

        # --- 设置输出路径 (仅主进程创建) ---
        model_short_name = self.model_id.split('/')[-1]
        timestamp = strftime('%Y%m%d-%H%M%S')
        self.output_dir = Path(self.output_dir_base) / f"{model_short_name}_{timestamp}"
        self.proof_save_dir = self.output_dir / "proofs"
        self.tmp_dir = self.output_dir / "tmp" # 为Prover提供临时目录
        self.results_log_file = self.output_dir / "results.json"

        if self.accelerator.is_main_process:
            self.proof_save_dir.mkdir(parents=True, exist_ok=True)
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            self.accelerator.print(f"BFS Search Runner initialized. Outputs will be saved to: {self.output_dir.resolve()}")
            self.accelerator.print(f"Accelerator: num_processes={self.accelerator.num_processes}, device='{str(self.device)}', mixed_precision='{self.accelerator.mixed_precision}'")

    def _setup_hf_config(self) -> HuggingFaceModelConfig:
        """配置 Hugging Face 模型"""
        base_config_dict = {
            "model_name": self.model_id,
            "device": str(self.device),
            "torch_dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else \
                           ("float16" if torch.cuda.is_available() else "auto"),
            "trust_remote_code": True, "use_fast_tokenizer": True,
            "generation_kwargs": {
                "max_new_tokens": self.model_gen_max_tokens,
                "temperature": self.model_gen_temperature,
                "top_p": 0.95, "do_sample": True,
            }
        }
        if self.hf_config_overrides:
            override_gen_kwargs = self.hf_config_overrides.get("generation_kwargs", {})
            base_config_dict["generation_kwargs"].update(override_gen_kwargs)
            other_overrides = {k: v for k, v in self.hf_config_overrides.items() if k != "generation_kwargs"}
            base_config_dict.update(other_overrides)
        return HuggingFaceModelConfig(**base_config_dict)

    class _InMemoryListDataset(TorchDataset):
        """一个简单的用于LeanItem的内存数据集"""
        def __init__(self, items: List[Any]):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx) -> Any:
            return self.items[idx]

    def _setup_dataloader(self, items_for_round: List[LeanItem]) -> DataLoader:
        """为给定的LeanItem列表创建DataLoader"""
        dataset = self._InMemoryListDataset(items_for_round)
        # collate_fn 直接返回 LeanItem 对象列表
        def custom_collate_fn(batch_items: List[LeanItem]) -> List[LeanItem]:
            return batch_items
        return DataLoader(dataset, batch_size=self.per_device_batch_size, shuffle=False,
                          num_workers=self.dataloader_num_workers, collate_fn=custom_collate_fn, pin_memory=True)

    def run(self):
        """执行整个评估流程"""
        overall_start_time = time()
        hf_config = self._setup_hf_config()

        # 加载完整数据集
        if self.accelerator.is_main_process:
             self.accelerator.print(f"Loading dataset from: {self.dataset_path}")
        full_dataset_items: List[LeanItem] = list(JsonDataset(self.dataset_path))
        
        # 为所有数据创建 DataLoader，Accelerator 会自动分发
        dataloader = self._setup_dataloader(full_dataset_items)

        with ModelRegistry.create("huggingface", **hf_config.model_dump()) as hf_model_wrapper:
            # 关键步骤: 使用 accelerator.prepare 包装模型和数据加载器
            prepared_model, prepared_dataloader = self.accelerator.prepare(
                hf_model_wrapper.model, dataloader
            )
            hf_model_wrapper.model = prepared_model

            if hf_model_wrapper.tokenizer.pad_token_id is None and hf_model_wrapper.tokenizer.eos_token_id is not None:
                hf_model_wrapper.tokenizer.pad_token_id = hf_model_wrapper.tokenizer.eos_token_id
            
            # --- 核心修改：初始化 BFSProver ---
            prompt_builder = FewShotPromptBuilder(shots=self.tactic_shots)
            prover_tmp_dir = self.tmp_dir / f"process_{self.accelerator.process_index}"
            bfs_prover = BFSProver(
                model=hf_model_wrapper,
                prompt_builder=prompt_builder,
                tmp_dir=prover_tmp_dir,
                timeout=self.bfs_timeout,
                degree=self.bfs_degree
            )

            # --- 核心修改：执行证明搜索循环 ---
            current_process_outputs = []
            progress_bar = tqdm(
                prepared_dataloader, 
                desc=f"BFS Proof Search (Process {self.accelerator.process_index})",
                disable=not self.accelerator.is_local_main_process,
            )
            
            for batch_items in progress_bar:
                for item in batch_items:
                    # 构造初始目标
                    goal_to_prove = f"{item.prompt_ready_stmt}\n"
                    
                    self.accelerator.print(f"[Proc {self.accelerator.process_index}] 开始搜索证明: {item.id}")
                    search_start_time = time()

                    # 调用 BFSProver 进行搜索
                    root, final_proof_code = bfs_prover.thread_prove(
                        goal_to_prove, 
                        num_workers=self.bfs_prover_num_workers
                    )

                    search_duration = time() - search_start_time
                    is_proved = root is not None

                    self.accelerator.print(f"[Proc {self.accelerator.process_index}] 完成搜索: {item.id}, 证明成功: {is_proved}, 用时: {search_duration:.2f}s")
                    
                    # 保存当前进程的输出结果
                    current_process_outputs.append({
                        "id": item.id,
                        "statement": item.statement,
                        "proved": is_proved,
                        "proof_code": final_proof_code,
                        "search_time": search_duration,
                        "process_index": self.accelerator.process_index,
                    })

                    # 保存证明文件，无论成功与否
                    safe_id = item.id.replace('/', '_').replace('\\', '_')
                    status_tag = 'ok' if is_proved else 'fail'
                    proof_file = self.proof_save_dir / f"{safe_id}_{status_tag}.lean"
                    with proof_file.open("w", encoding="utf-8") as f:
                        f.write(final_proof_code or (goal_to_prove + "  -- Proof Search Failed --"))

            # --- 循环结束，等待所有进程 ---
            self.accelerator.wait_for_everyone()

            # --- 结果聚合 (与您提供的代码逻辑保持一致) ---
            gathered_outputs_list = gather_object(current_process_outputs)

            if self.accelerator.is_main_process:
                self.accelerator.print("\n--- 所有搜索完成，开始聚合结果 ---")
                final_results = [item for item in gathered_outputs_list ]
                self.accelerator.print(f"从所有进程共收集到 {len(final_results)} 条结果。")

                with self.results_log_file.open("w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)
                
                self.accelerator.print(f"最终结果已保存至: {self.results_log_file.resolve()}")
                self.accelerator.print(f"\n整体运行结束. 总用时: {time() - overall_start_time:.2f}s")

        self.accelerator.wait_for_everyone()