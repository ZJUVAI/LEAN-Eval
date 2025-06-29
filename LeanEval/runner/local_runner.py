import os
import torch
import sys
from pathlib import Path
from time import time, strftime, time_ns
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json
import concurrent.futures

import accelerate
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from torch.utils.data import Dataset as TorchDataset, DataLoader

# 假设这些模块在你项目的正确位置
from LeanEval.datasets import JsonDataset, LeanItem
from LeanEval.models import ModelRegistry, HuggingFaceModelConfig
from LeanEval.validator.proof_validator import ProofValidator
from LeanEval.utils import extract_lean_block
import pdb

class LocalHuggingFaceRunner:
    """
    一个使用 Hugging Face 模型在本地运行 LeanEval 评估的 Runner。
    它利用 accelerate 库来支持多GPU/分布式推理和验证。
    """
    def __init__(
        self,
        model_id: str,
        dataset_path: str,
        output_dir_base: str = "./outputs_local_runner",
        prompt_template: str = "Complete the Lean 4 proof below. Only output the Lean code for the complete proof \n```lean\n{code_block_statement} := by\n```",
        per_device_batch_size: int = 1,
        dataloader_num_workers: int = 2,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        mixed_precision: str = 'fp16', # 'no', 'fp16', 'bf16'
        validation_timeout: int = 120,
        hf_config_overrides: Dict[str, Any] = None,
        num_proof_rounds: int = 1,
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
        self.num_proof_rounds = num_proof_rounds

        self.accelerator = Accelerator(mixed_precision=self.mixed_precision)
        self.device = self.accelerator.device

        model_short_name = self.model_id.split('/')[-1]
        timestamp = strftime('%Y%m%d-%H%M%S')
        self.output_dir = Path(self.output_dir_base) / f"{model_short_name}_{timestamp}"
        self.proof_save_dir = self.output_dir / "proofs"
        self.results_log_file = self.output_dir / "results.json"

        if self.accelerator.is_main_process:
            self.proof_save_dir.mkdir(parents=True, exist_ok=True)
            self.accelerator.print(f"Runner initialized. Max proof rounds: {self.num_proof_rounds}. Outputs will be saved to: {self.output_dir.resolve()}")
            self.accelerator.print(f"Accelerator: num_processes={self.accelerator.num_processes}, device='{str(self.device)}', mixed_precision='{self.accelerator.mixed_precision}'")

    def _setup_hf_config(self) -> HuggingFaceModelConfig:
        base_config_dict = {
            "model_name": self.model_id,
            "device": str(self.device),
            "torch_dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else \
                           ("float16" if torch.cuda.is_available() else "auto"),
            "trust_remote_code": True, "use_fast_tokenizer": True,
            "generation_kwargs": {
                "max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
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
        def __init__(self, items: List[Any]):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx) -> Any:
            return self.items[idx]

    def _setup_dataloader_for_items(self, items_for_round: List[LeanItem]) -> DataLoader:
        dataset = self._InMemoryListDataset(items_for_round)
        # 假设 LeanEval.prompt.get_builder 存在
        from LeanEval.prompt import get_builder
        prompt_builder = get_builder("simple", template=self.prompt_template)
        def custom_collate_fn(batch_items: List[LeanItem]) -> Dict[str, Any]:
            prompts_for_model, original_items_metadata = [], []
            for item in batch_items:
                prompt_str = prompt_builder.template.format(code_block_statement=item.prompt_ready_stmt)
                prompts_for_model.append(prompt_str)
                original_items_metadata.append({"id": item.id, "imports": item.imports, "statement": item.statement})
            return {"prompts_for_model": prompts_for_model, "original_items_metadata": original_items_metadata}
        return DataLoader(dataset, batch_size=self.per_device_batch_size, shuffle=False,
                          num_workers=self.dataloader_num_workers, collate_fn=custom_collate_fn, pin_memory=True)

    def _run_validation_distributed(self, all_files_to_validate: List[str]) -> List[Tuple[str, Dict[str, str]]]:
        """
        手动将验证任务分配给所有进程，并返回一个 (key, value) 元组的列表，以避免 gather_object 的问题。
        """
        files_for_this_process = all_files_to_validate[self.accelerator.process_index::self.accelerator.num_processes]
        if not files_for_this_process:
            return []
        # self.accelerator.print(f"Proc {self.accelerator.process_index}: Assigned {len(files_for_this_process)} files to validate.")
        
        validator = ProofValidator(timeout=self.validation_timeout)
        local_results_map: Dict[str, Dict[str, str]] = {}
        progress_bar = tqdm(files_for_this_process, 
                            desc=f"Validating (Proc {self.accelerator.process_index})",
                            position=self.accelerator.process_index)
                            
        for filepath_str in progress_bar:
            try:
                self.accelerator.print(f"Proc {self.accelerator.process_index}: Validating file {filepath_str}")
                success, msg = validator.validate_file(filepath_str)
                local_results_map[filepath_str] = {"status": "Passed" if success else "Failed", "log": msg}
                self.accelerator.print(f"Proc {self.accelerator.process_index}: Validation result for {filepath_str}: {local_results_map[filepath_str]}")
            except Exception as e:
                self.accelerator.print(f"Error validating file {filepath_str} on process {self.accelerator.process_index}: {e}", file=sys.stderr)
                local_results_map[filepath_str] = {"status": "ErrorInValidation", "log": str(e)}

        # self.accelerator.print(f"Proc {self.accelerator.process_index}: Finished validation for {len(local_results_map)} files.")
        
        # 返回一个元组列表，而不是字典，以确保 gather_object 的可靠性。
        return list(local_results_map.items())

    def run(self):
        overall_start_time = time()
        hf_config = self._setup_hf_config()
        full_dataset_items: List[LeanItem] = list(JsonDataset(self.dataset_path))
        problem_results_map: Dict[str, Dict[str, Any]] = {
            item.id: {"status": "unsolved", "solved_in_round": None, "final_proof_path": None, "final_proof_code": None,
                      "final_validation_log": None, "original_statement": item.statement, "imports": item.imports, "all_attempts": []}
            for item in full_dataset_items
        }
        items_to_process_this_round: List[LeanItem] = list(full_dataset_items)

        with ModelRegistry.create("huggingface", **hf_config.model_dump()) as hf_model_wrapper:
            if hf_model_wrapper.tokenizer.pad_token_id is None and hf_model_wrapper.tokenizer.eos_token_id is not None:
                hf_model_wrapper.tokenizer.pad_token_id = hf_model_wrapper.tokenizer.eos_token_id
            hf_model_wrapper.model = self.accelerator.prepare(hf_model_wrapper.model)

            for round_num in range(1, self.num_proof_rounds + 1):
                if not items_to_process_this_round:
                    if self.accelerator.is_main_process:
                        self.accelerator.print(f"Round {round_num}: No items left to process. Finishing.")
                    break
                
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"\n--- Starting Proof Round {round_num}/{self.num_proof_rounds} ---")
                    self.accelerator.print(f"Items to process in this round: {len(items_to_process_this_round)}")

                current_round_dataloader = self._setup_dataloader_for_items(items_to_process_this_round)
                prepared_round_dataloader = self.accelerator.prepare(current_round_dataloader)
                
                current_process_outputs_this_round = []
                hf_model_wrapper.model.eval()
                progress_bar = tqdm(
                    prepared_round_dataloader, 
                    desc=f"Round {round_num} Inference (Proc {self.accelerator.process_index})", 
                    position=self.accelerator.process_index
                    )
                
                with torch.no_grad():
                    for batch_data in progress_bar:
                        prompts, metadata_batch = batch_data["prompts_for_model"], batch_data["original_items_metadata"]
                        generated_batch_texts = hf_model_wrapper.batch_predict(prompts, batch_size=len(prompts))
                        for i, gen_text in enumerate(generated_batch_texts):
                            item_meta = metadata_batch[i]
                            current_process_outputs_this_round.append({
                                "id": item_meta["id"], "imports": item_meta["imports"], "statement": item_meta["statement"],
                                "generated_proof_part": gen_text.strip(), "process_index": self.accelerator.process_index, "round_num": round_num
                            })
                
                self.accelerator.wait_for_everyone()

                gathered_outputs_list = gather_object(current_process_outputs_this_round)
                processed_outputs_with_paths_this_round = []

                if self.accelerator.is_main_process:
                    all_gathered_data = [item for sublist in gathered_outputs_list for item in sublist] if gathered_outputs_list and isinstance(gathered_outputs_list[0], list) else gathered_outputs_list
                    
                    # self.accelerator.print(f"Round {round_num}: Main process gathered {len(all_gathered_data)} total results. Saving files...")
                    for output_data in tqdm(all_gathered_data, desc=f"Round {round_num} Saving"):
                        item_id, imports_list, model_gen_code = output_data.get("id"), output_data.get("imports", []), output_data.get("generated_proof_part", "")
                        if not isinstance(imports_list, list): imports_list = []
                        imports_block = "\n".join(imports_list)
                        statement = output_data.get("statement", "")
                        proof_body = extract_lean_block(model_gen_code) or model_gen_code
                        # full_code = f"{imports_block}\n\n{statement} := by\n  {proof_body}"
                        full_code = f"{imports_block}\n  {proof_body}"

                        safe_id = str(item_id).replace("/", "_").replace("\\", "_")
                        proof_file = self.proof_save_dir / f"proof_{safe_id}_round{round_num}_proc{output_data.get('process_index','X')}_{time_ns()}.lean"
                        proof_file.write_text(full_code, encoding="utf-8")
                        
                        canonical_path_str = proof_file.resolve().as_posix()
                        
                        processed_outputs_with_paths_this_round.append({
                            'data': output_data, 
                            'path': proof_file,
                            'canonical_path': canonical_path_str,
                            'full_code': full_code
                        })
                
                self.accelerator.wait_for_everyone()
                
                # --- 分布式验证 ---
                path_strings_to_validate = []
                if self.accelerator.is_main_process:
                    path_strings_to_validate = [entry['canonical_path'] for entry in processed_outputs_with_paths_this_round]

                broadcasted_list = broadcast_object_list([path_strings_to_validate])
                files_to_validate_on_all_procs = broadcasted_list[0]
                self.accelerator.wait_for_everyone()

                validation_results_map_this_round = {}
                if files_to_validate_on_all_procs:
                    local_validation_results = self._run_validation_distributed(files_to_validate_on_all_procs)
                    
                    self.accelerator.print("Gathering validation results...")
                    gathered_results = gather_object(local_validation_results)
                    
                    if self.accelerator.is_main_process:
                        
                        # **FINAL FIX**: The log shows `gather_object` returns a flat list of items 
                        # from all workers when each worker returns a list. We can directly convert this
                        # flat list of (key, value) tuples into a dictionary.
                        try:
                            # Flatten the list first, as gather_object returns a list of lists.
                            all_items = [item for item in gathered_results]
                            validation_results_map_this_round = dict(all_items)
                        except (TypeError, ValueError) as e:
                             self.accelerator.print(f"CRITICAL: Could not convert gathered results to dictionary. Error: {e}")

                        self.accelerator.print(f"Total {len(validation_results_map_this_round)} validation results collected.")

                else:
                    if self.accelerator.is_main_process:
                        self.accelerator.print("No new proofs to validate in this round. Skipping validation.")
                
                self.accelerator.wait_for_everyone()

                # --- 结果处理与下一轮准备 (主进程) ---
                if self.accelerator.is_main_process:
                    newly_solved_count = 0
                    
                    for entry in processed_outputs_with_paths_this_round:
                        item_id = entry['data']['id']
                        proof_path = entry['path']
                        lookup_key = entry['canonical_path']
                        
                        val_res = validation_results_map_this_round.get(lookup_key, {"status": "ValidationNotFound", "log": "File path not found in validation results."})
                        
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
                            newly_solved_count += 1

                    self.accelerator.print(f"Round {round_num}: Solved {newly_solved_count} new items.")
                    
                    items_to_process_this_round = [item for item in full_dataset_items if problem_results_map[item.id]["status"] == "unsolved"]
                else:
                    items_to_process_this_round = []
                
                broadcast_payload = [items_to_process_this_round]
                received_payload = broadcast_object_list(broadcast_payload)
                items_to_process_this_round = received_payload[0]
                
                self.accelerator.wait_for_everyone()

            # --- 所有轮次结束 ---
            if self.accelerator.is_main_process:
                final_output_list = []
                for item_id, result_data in problem_results_map.items():
                    final_entry = result_data.copy()
                    final_entry["id"] = item_id
                    if final_entry["status"] == "unsolved": final_entry["status"] = "failed_max_rounds"
                    final_output_list.append(final_entry)

                with open(self.results_log_file, "w", encoding="utf-8") as f:
                    json.dump(final_output_list, f, indent=2, ensure_ascii=False)
                self.accelerator.print(f"\nAll rounds finished. Total time: {time() - overall_start_time:.2f}s")
                self.accelerator.print(f"Results saved to {self.results_log_file.resolve()}")

        self.accelerator.wait_for_everyone()
