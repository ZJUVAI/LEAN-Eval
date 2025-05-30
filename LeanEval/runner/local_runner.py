# LeanEval/runner/local_runner.py
import os
import torch
from pathlib import Path
from time import time, strftime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json
import concurrent.futures

import accelerate
from accelerate import Accelerator
from torch.utils.data import DataLoader

from LeanEval.datasets import JsonDataset, LeanItem
from LeanEval.prompt import get_builder, PromptBuilder
from LeanEval.models import ModelRegistry, HuggingFaceModelConfig, HuggingFaceModel
from LeanEval.validator.proof_validator import ProofValidator
from LeanEval.utils import extract_lean_block

class LocalHuggingFaceRunner:
    """
    封装使用 Accelerate 进行本地 Hugging Face 模型多卡推理的逻辑。
    """
    def __init__(
        self,
        model_id: str,
        dataset_path: str,
        output_dir_base: str = "./outputs_local_runner",
        prompt_template: str = "Complete the Lean 4 proof below. Only output the Lean code for the complete proof\n```lean\n{code_block_statement} := by\n```",
        per_device_batch_size: int = 1,
        dataloader_num_workers: int = 2,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        mixed_precision: str = 'fp16', # 'no', 'fp16', 'bf16'
        validation_timeout: int = 60,
        hf_config_overrides: Dict[str, Any] = None # 允许覆盖 HF 配置
    ):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.output_dir_base = output_dir_base
        self.prompt_template = prompt_template
        self.per_device_batch_size = per_device_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.mixed_precision = mixed_precision if torch.cuda.is_available() else 'no'
        self.validation_timeout = validation_timeout
        self.hf_config_overrides = hf_config_overrides or {}

        self.accelerator = Accelerator(mixed_precision=self.mixed_precision)
        self.device = self.accelerator.device

        # 设置输出目录
        model_short_name = self.model_id.split('/')[-1]
        self.output_dir = Path(self.output_dir_base) / f"{model_short_name}_{strftime('%Y%m%d-%H%M%S')}"
        self.proof_save_dir = self.output_dir / "proofs"
        self.results_log_file = self.output_dir / "results.json" # 保存为 JSON

        if self.accelerator.is_main_process:
            self.proof_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Runner initialized. Outputs will be saved to: {self.output_dir.resolve()}")

    def _setup_hf_config(self) -> HuggingFaceModelConfig:
        """设置 HuggingFace 模型配置。"""
        base_config = {
            "model_name": self.model_id,
            "device": str(self.device), # 虽然 accelerator 会处理，但初始加载可能需要
            "torch_dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else \
                           ("float16" if torch.cuda.is_available() else "auto"),
            "trust_remote_code": True,
            "use_fast_tokenizer": True,
            "generation_kwargs": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": 0.95,
                "do_sample": True,
            }
        }
        # 应用覆盖
        base_config.update(self.hf_config_overrides)
        # 确保 generation_kwargs 被正确更新
        if "generation_kwargs" in self.hf_config_overrides:
             base_config["generation_kwargs"].update(self.hf_config_overrides["generation_kwargs"])

        return HuggingFaceModelConfig(**base_config)

    def _setup_dataloader(self) -> DataLoader:
        """设置数据集和 DataLoader。"""
        dataset = JsonDataset(self.dataset_path)
        prompt_builder = get_builder("simple", template=self.prompt_template)

        def custom_collate_fn(batch_items: List[LeanItem]) -> Dict[str, Any]:
            prompts_for_model = []
            original_items_metadata = []
            for item in batch_items:
                code_block_stmt = item.prompt_ready_stmt
                prompt_str = prompt_builder.template.format(code_block_statement=code_block_stmt)
                prompts_for_model.append(prompt_str)
                original_items_metadata.append({
                    "id": item.id,
                    "prompt_ready_stmt": item.prompt_ready_stmt,
                    "imports_txt":item.imports_txt
                })
            return {"prompts_for_model": prompts_for_model, "original_items_metadata": original_items_metadata}

        return DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )

    def _run_validation(self, saved_files: List[Path]) -> List[Dict]:
        """运行验证并返回结果。"""
        if not saved_files:
            return []

        self.accelerator.print(f"Starting validation for {len(saved_files)} proofs...")
        validator = ProofValidator(timeout=self.validation_timeout)
        
        passed_files, failed_files_with_msg = [], []

        def validate_task(filepath):
            success, msg = validator.validate_file(filepath)
            return filepath, success, msg

        num_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_filepath = {
                executor.submit(validate_task, filepath): filepath
                for filepath in saved_files
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_filepath),
                total=len(saved_files),
                desc="Validating",
                disable=not self.accelerator.is_main_process
            ):
                filepath, success, msg = future.result()
                if success:
                    passed_files.append(filepath)
                else:
                    failed_files_with_msg.append((filepath, msg))

        self.accelerator.print(f"Validation: Passed={len(passed_files)}, Failed={len(failed_files_with_msg)}")
        
        results_map = {f: {"status": "Passed", "log": ""} for f in passed_files}
        results_map.update({f: {"status": "Failed", "log": msg} for f, msg in failed_files_with_msg})
        
        return results_map


    def run(self):
        """执行完整的推理和验证流程。"""
        start_time = time()

        # 1. 设置
        hf_config = self._setup_hf_config()
        eval_dataloader = self._setup_dataloader()

        # 2. 加载模型
        with ModelRegistry.create("huggingface", **hf_config.model_dump()) as hf_model:
            if hf_model.tokenizer.pad_token_id is None:
                hf_model.tokenizer.pad_token_id = hf_model.tokenizer.eos_token_id
                hf_model.model.config.pad_token_id = hf_model.tokenizer.eos_token_id

            # 3. Accelerator Prepare
            prepared_model, prepared_dataloader = self.accelerator.prepare(
                hf_model.model, eval_dataloader
            )
            hf_model.model = prepared_model # 更新包装器中的模型引用

            # 4. 推理
            current_process_outputs = []
            if hasattr(hf_model.model, 'eval'):
                hf_model.model.eval()

            progress_bar = tqdm(
                prepared_dataloader,
                desc=f"Inference (Proc {self.accelerator.process_index})",
                disable=not self.accelerator.is_local_main_process
            )

            with torch.no_grad():
                for batch_data in progress_bar:
                    prompts = batch_data["prompts_for_model"]
                    metadata = batch_data["original_items_metadata"]

                    generated_batch_texts = hf_model.batch_predict(
                        prompts,
                        batch_size=len(prompts)
                    )

                    for i,gen_text in enumerate(generated_batch_texts):
                        current_process_outputs.append({
                            "id":metadata[i]["id"],
                            "generated_proof_part":gen_text.strip(),
                            "prompt_ready_stmt":metadata[i]["prompt_ready_stmt"],
                            "imports_txt":metadata[i]["imports_txt"],
                            "process_index":self.accelerator.process_index
                        })

            self.accelerator.wait_for_everyone()

            # 5. 收集与保存
            gathered_outputs = accelerate.utils.gather_object(current_process_outputs)
            saved_files_map = {} # id -> Path
            if self.accelerator.is_main_process:
                print(f"Gathered {len(gathered_outputs)} results. Saving proofs...")
                for output_data in tqdm(gathered_outputs, desc="Saving proofs"):
                    item_id = output_data["id"]
                    proof_part = output_data["generated_proof_part"]
                    stmt = output_data["prompt_ready_stmt"]
                    imports = output_data["imports_txt"]
                    # 尝试提取 ```lean ... ```, 否则用原始输出
                    extracted_code = extract_lean_block(proof_part) or proof_part
                    full_code = f"{imports}\n{extracted_code}"

                    safe_id = str(item_id).replace("/", "_").replace("\\", "_")
                    proof_file = self.proof_save_dir / f"{safe_id}.lean"
                    proof_file.write_text(full_code, encoding="utf-8")
                    saved_files_map[item_id] = proof_file
                
                # 6. 验证 (仅主进程)
                validation_results = self._run_validation(list(saved_files_map.values()))

                # 7. 组合最终结果
                final_results = []
                for res in gathered_outputs:
                    item_id = res['id']
                    path = saved_files_map.get(item_id)
                    val_res = validation_results.get(path, {"status": "Unknown", "log": "File not found in validation map"})
                    res.update({
                        "proof_file": str(path),
                        "validation_status": val_res["status"],
                        "validation_log": val_res["log"]
                    })
                    final_results.append(res)
                
                # 8. 保存 JSON 结果
                with open(self.results_log_file, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)
                
                print(f"Runner finished. Total time: {time() - start_time:.2f}s. Results saved to {self.results_log_file}")

        self.accelerator.wait_for_everyone()