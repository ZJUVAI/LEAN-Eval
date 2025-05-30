# LeanEval/runner/local_runner.py
import os
import torch
from pathlib import Path
from time import time, strftime, time_ns # time_ns for unique filenames
from typing import List, Dict, Any, Tuple, Iterable # Added Iterable
from tqdm import tqdm
import json
import concurrent.futures

import accelerate
from accelerate import Accelerator
from torch.utils.data import Dataset as TorchDataset, DataLoader # Explicitly import TorchDataset

from LeanEval.datasets import JsonDataset, LeanItem, BaseDataset # Added BaseDataset
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
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        mixed_precision: str = 'fp16', # 'no', 'fp16', 'bf16'
        validation_timeout: int = 60,
        hf_config_overrides: Dict[str, Any] = None, # 允许覆盖 HF 配置
        num_proof_rounds: int = 1, # 新增：最大证明轮数
        num_validation_workers:int = 4
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
        self.num_proof_rounds = num_proof_rounds # 存储最大轮数
        self.num_validation_workers = num_validation_workers

        self.accelerator = Accelerator(mixed_precision=self.mixed_precision)
        self.device = self.accelerator.device

        model_short_name = self.model_id.split('/')[-1]
        self.output_dir = Path(self.output_dir_base) / f"{model_short_name}_{strftime('%Y%m%d-%H%M%S')}"
        self.proof_save_dir = self.output_dir / "proofs"
        self.results_log_file = self.output_dir / "results.json"

        if self.accelerator.is_main_process:
            self.proof_save_dir.mkdir(parents=True, exist_ok=True)
            self.accelerator.print(f"Runner initialized. Max proof rounds: {self.num_proof_rounds}. Outputs will be saved to: {self.output_dir.resolve()}")
            self.accelerator.print(f"Accelerator: num_processes={self.accelerator.num_processes}, device='{str(self.device)}', mixed_precision='{self.accelerator.mixed_precision}'")


    def _setup_hf_config(self) -> HuggingFaceModelConfig:
        """设置 HuggingFace 模型配置。"""
        base_config_dict = {
            "model_name": self.model_id,
            "device": str(self.device), 
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
        
        if self.hf_config_overrides:
            override_gen_kwargs = self.hf_config_overrides.get("generation_kwargs")
            if override_gen_kwargs and isinstance(base_config_dict["generation_kwargs"], dict) and isinstance(override_gen_kwargs, dict):
                base_config_dict["generation_kwargs"].update(override_gen_kwargs)
            
            other_overrides = {k: v for k, v in self.hf_config_overrides.items() if k != "generation_kwargs"}
            base_config_dict.update(other_overrides)
            
        return HuggingFaceModelConfig(**base_config_dict)

    # 辅助 Dataset 类，用于从内存中的 LeanItem 列表创建 Dataset
    class _InMemoryListDataset(TorchDataset): # Inherit from torch.utils.data.Dataset
        def __init__(self, items: List[LeanItem]):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx) -> LeanItem: # Ensure it returns LeanItem
            return self.items[idx]

    def _setup_dataloader_for_items(self, items_for_round: List[LeanItem]) -> DataLoader:
        """为当前轮次的 LeanItem 列表创建 DataLoader。"""
        dataset = self._InMemoryListDataset(items_for_round)
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
                    "imports": item.imports, 
                    "statement": item.statement,
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

    def _run_validation(self, files_to_validate: List[Path]) -> Dict[Path, Dict[str, str]]:
        """
        运行验证并返回一个从文件路径到验证结果的映射。
        仅在主进程中执行。
        """
        if not files_to_validate or not self.accelerator.is_main_process:
            return {}

        self.accelerator.print(f"Starting validation for {len(files_to_validate)} proofs on main process...")
        validator = ProofValidator(timeout=self.validation_timeout)
        
        validation_map_results: Dict[Path, Dict[str, str]] = {}

        def validate_task(filepath: Path) -> Tuple[Path, bool, str]:
            success, msg = validator.validate_file(filepath)
            return filepath, success, msg

        # 使用较少的 worker 数以避免过多占用 CPU 核心，特别是在 GPU 推理也在进行时
        num_workers = self.num_validation_workers
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_filepath = {
                executor.submit(validate_task, filepath): filepath
                for filepath in files_to_validate
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_filepath),
                total=len(files_to_validate),
                desc="Validating proofs"
            ):
                try:
                    filepath, success, msg = future.result()
                    status_str = "Passed" if success else "Failed"
                    validation_map_results[filepath] = {"status": status_str, "log": msg}
                except Exception as e:
                    filepath_from_future = future_to_filepath[future]
                    self.accelerator.print(f"Error validating file {filepath_from_future}: {e}")
                    validation_map_results[filepath_from_future] = {"status": "ErrorInValidation", "log": str(e)}

        passed_count = sum(1 for res in validation_map_results.values() if res["status"] == "Passed")
        self.accelerator.print(f"Validation complete: Passed={passed_count}, TotalAttemptedThisRound={len(files_to_validate)}")
        return validation_map_results

    def run(self):
        overall_start_time = time()
        hf_config = self._setup_hf_config()

        full_dataset_items: List[LeanItem] = list(JsonDataset(self.dataset_path))
        
        problem_results_map: Dict[str, Dict[str, Any]] = {
            item.id: {
                "status": "unsolved", "solved_in_round": None, "final_proof_path": None,
                "final_proof_code": None, "final_validation_log": None,
                "original_statement": item.statement, "imports": item.imports,
                "all_attempts": []
            } for item in full_dataset_items
        }

        items_to_process_this_round: List[LeanItem] = list(full_dataset_items)

        with ModelRegistry.create("huggingface", **hf_config.model_dump()) as hf_model_wrapper:
            if hf_model_wrapper.tokenizer.pad_token_id is None:
                if hf_model_wrapper.tokenizer.eos_token_id is not None:
                    hf_model_wrapper.tokenizer.pad_token_id = hf_model_wrapper.tokenizer.eos_token_id
                    if hf_model_wrapper.model and hasattr(hf_model_wrapper.model, 'config'):
                         hf_model_wrapper.model.config.pad_token_id = hf_model_wrapper.tokenizer.eos_token_id
                else:
                    self.accelerator.print("Warning: pad_token_id and eos_token_id are None.")
            
            prepared_model = self.accelerator.prepare(hf_model_wrapper.model)
            hf_model_wrapper.model = prepared_model

            for round_num in range(1, self.num_proof_rounds + 1):
                if not items_to_process_this_round:
                    self.accelerator.print(f"Round {round_num}: No items left to process. Finishing.")
                    break
                
                self.accelerator.print(f"\n--- Starting Proof Round {round_num}/{self.num_proof_rounds} ---")
                self.accelerator.print(f"Items to process in this round: {len(items_to_process_this_round)}")

                current_round_dataloader = self._setup_dataloader_for_items(items_to_process_this_round)
                # DataLoader needs to be prepared in each round if its underlying dataset changes
                prepared_round_dataloader = self.accelerator.prepare(current_round_dataloader)

                current_process_outputs_this_round = []
                actual_model_to_eval = self.accelerator.unwrap_model(hf_model_wrapper.model)
                if hasattr(actual_model_to_eval, 'eval'):
                    actual_model_to_eval.eval()

                progress_bar = tqdm(
                    prepared_round_dataloader,
                    desc=f"Round {round_num} Inference (Proc {self.accelerator.process_index})",
                    disable=not self.accelerator.is_local_main_process,
                )
                
                round_inference_start_time = time()
                with torch.no_grad():
                    for batch_data in progress_bar:
                        prompts = batch_data["prompts_for_model"]
                        metadata_batch = batch_data["original_items_metadata"]

                        generated_batch_texts = hf_model_wrapper.batch_predict(
                            prompts, batch_size=len(prompts)
                        )

                        for i, gen_text in enumerate(generated_batch_texts):
                            item_meta = metadata_batch[i]
                            current_process_outputs_this_round.append({
                                "id": item_meta["id"],
                                "imports": item_meta["imports"],
                                "statement": item_meta["statement"], # Original statement
                                "generated_proof_part": gen_text.strip(),
                                "process_index": self.accelerator.process_index,
                                "round_num": round_num
                            })
                
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"Round {round_num} Inference finished in {time() - round_inference_start_time:.2f}s.")

                gathered_outputs_this_round_list_of_lists = accelerate.utils.gather_object(current_process_outputs_this_round)
                
                round_saved_proof_files = [] # Paths of files saved in this round for validation
                processed_outputs_with_paths_this_round = [] # List of {'data': output_data, 'path': Path}

                if self.accelerator.is_main_process:
                    all_gathered_data_this_round = []
                    for sublist in gathered_outputs_this_round_list_of_lists:
                        if isinstance(sublist, list): all_gathered_data_this_round.extend(sublist)
                        elif isinstance(sublist, dict): all_gathered_data_this_round.append(sublist) # Should be list
                    
                    self.accelerator.print(f"Round {round_num}: Gathered {len(all_gathered_data_this_round)} results. Saving and validating...")
                    
                    for output_data in tqdm(all_gathered_data_this_round, desc=f"Round {round_num} Saving"):
                        item_id = output_data["id"]
                        imports_list = output_data["imports"]
                        model_gen_code = output_data["generated_proof_part"]

                        imports_block = "\n".join([f"import {i}" for i in imports_list]) if imports_list else ""
                        cleaned_model_output = extract_lean_block(model_gen_code) or model_gen_code
                        
                        full_code = f"{imports_block}\n\n{cleaned_model_output}" if imports_block else cleaned_model_output

                        safe_id = str(item_id).replace("/", "_").replace("\\", "_")
                        proof_file = self.proof_save_dir / f"proof_{safe_id}_round{round_num}_proc{output_data.get('process_index','X')}_{time_ns()}.lean"
                        
                        try:
                            proof_file.write_text(full_code, encoding="utf-8")
                            round_saved_proof_files.append(proof_file)
                            processed_outputs_with_paths_this_round.append({
                                'data': output_data, 
                                'path': proof_file, 
                                'full_code': full_code
                            })
                        except Exception as e:
                            self.accelerator.print(f"Error writing file {proof_file}: {e}")

                    validation_results_map_this_round = self._run_validation(round_saved_proof_files)

                    newly_solved_in_this_round_count = 0
                    for entry in processed_outputs_with_paths_this_round:
                        output_data = entry['data']
                        proof_path = entry['path']
                        item_id = output_data["id"]

                        val_res = validation_results_map_this_round.get(proof_path, {"status": "Not Validated", "log": "File not found."})
                        
                        problem_results_map[item_id]["all_attempts"].append({
                            "round": round_num,
                            "saved_file_path": str(proof_path),
                            "validation_status": val_res["status"],
                            "validation_log": val_res["log"]
                        })

                        if val_res["status"] == "Passed" and problem_results_map[item_id]["status"] == "unsolved":
                            problem_results_map[item_id].update({
                                "status": "solved",
                                "solved_in_round": round_num,
                                "final_proof_path": str(proof_path),
                                "final_proof_code": entry["full_code"],
                                "final_validation_log": val_res["log"]
                            })
                            newly_solved_in_this_round_count += 1
                    
                    self.accelerator.print(f"Round {round_num}: Solved {newly_solved_in_this_round_count} new items in this round.")

                    # Prepare items for the next round
                    next_round_items = []
                    for item in full_dataset_items: # Iterate through original full list to maintain order if desired
                        if problem_results_map[item.id]["status"] == "unsolved":
                            next_round_items.append(item)
                    items_to_process_this_round = next_round_items
                
                # Synchronize the list of items for the next round to all processes
                # We use gather_object as a way to broadcast from main process
                # Main process sends the list, others send an empty list or None
                # Then all processes pick the first non-empty/non-None list
                object_to_broadcast = items_to_process_this_round if self.accelerator.is_main_process else [] # Send empty list from non-main
                
                # gathered_next_round_item_lists contains a list of lists (one per process)
                # The main process's list is the one we want.
                gathered_next_round_item_lists = accelerate.utils.gather_object(object_to_broadcast)
                
                if not self.accelerator.is_main_process:
                    # Find the list from the main process (it's typically the first one if gather preserves order,
                    # or the only non-empty one if others sent empty)
                    for lst in gathered_next_round_item_lists:
                        if lst: # Assuming main process sent a potentially non-empty list
                            items_to_process_this_round = lst
                            break
                # Ensure all processes have the same list for the next round

                payload_for_gather = items_to_process_this_round if self.accelerator.is_main_process else []
                # print(f'payload for gather is {payload_for_gather}')

                gathered_payloads = accelerate.utils.gather_object(payload_for_gather)
                # print(f'gathered_payloads is {gathered_payloads}')
                # gathered_payloads 现在是一个列表的列表，例如：
                # [ L_main, [], [], [] ]  (如果 L_main 是主进程的列表)
                # 或者 [ [], L_main, [], [] ] 等，取决于主进程的 rank

                # 所有进程现在从 gathered_payloads 中提取权威的下一轮任务列表
                # 权威列表是那个非空的列表（即来自主进程的列表）
                # 如果主进程的列表也是空的（所有任务完成），那么最终结果将是空列表
                
                authoritative_next_round_items = gathered_payloads # 默认为空列表
                
                items_to_process_this_round = authoritative_next_round_items
                self.accelerator.wait_for_everyone()


            # All rounds finished
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                final_output_json_list = []
                for item_id, result_data in problem_results_map.items():
                    if result_data["status"] == "unsolved": # Still unsolved after all rounds
                        result_data["status"] = "failed_max_rounds"
                    
                    # Add original statement and imports from result_data itself
                    entry = {
                        "id": item_id,
                        "original_statement": result_data["original_statement"],
                        "imports": result_data["imports"],
                        "final_status": result_data["status"],
                        "solved_in_round": result_data["solved_in_round"],
                        "final_proof_path": result_data["final_proof_path"],
                        # "final_proof_code": result_data["final_proof_code"], # Can be large, consider if needed
                        "final_validation_log": result_data["final_validation_log"],
                        "all_attempts_details": result_data["all_attempts"]
                    }
                    final_output_json_list.append(entry)

                with open(self.results_log_file, "w", encoding="utf-8") as f:
                    json.dump(final_output_json_list, f, indent=2, ensure_ascii=False)
                
                self.accelerator.print(f"All rounds finished. Total script time: {time() - overall_start_time:.2f}s. Results saved to {self.results_log_file}")

        self.accelerator.wait_for_everyone()