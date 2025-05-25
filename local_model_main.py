# LeanEval/run_eval_hf_local_accelerate.py
import concurrent.futures
import os
import torch
from pathlib import Path
from time import time, strftime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from LeanEval.utils import extract_lean_block
import concurrent
import accelerate
from accelerate import Accelerator
from torch.utils.data import DataLoader
from LeanEval.datasets import JsonDataset, LeanItem 
from LeanEval.prompt import get_builder, PromptBuilder
from LeanEval.models import ModelRegistry, HuggingFaceModelConfig
from LeanEval.validator.proof_validator import ProofValidator


def main():
    # 1. 初始化 Accelerator
    accelerator = Accelerator(mixed_precision='fp16' if torch.cuda.is_available() else 'no')
    device = accelerator.device

    
    #------------------自主配置--------------------
    dataset_path = "./data/json/data_1.json"  # 数据集路径
    model_id = "deepseek-ai/DeepSeek-Prover-V2-7B" # 模型配置
    #------------------自主配置--------------------


    # 输出目录配置
    model_short_name_for_path = model_id.split('/')[-1]
    base_output_dir = Path(f"./outputs_local_{model_short_name_for_path}")
    current_run_output_dir = base_output_dir / strftime("%Y%m%d-%H%M%S")
    proof_save_dir = current_run_output_dir / "proofs"
    results_log_file = current_run_output_dir / "results.log"

    hf_model_cfg_dict = {
        "model_name": model_id,
        "device": str(device), 
        "torch_dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else \
                       ("float16" if torch.cuda.is_available() else "auto"),
        "trust_remote_code": True, 
        "use_fast_tokenizer": True,
        "generation_kwargs": {
            "max_new_tokens": 512*2*2,
            "temperature": 0.1,
            "top_p": 0.95,
            "do_sample": True,
        }
    }
    hf_config = HuggingFaceModelConfig(**hf_model_cfg_dict)

    per_device_batch_size = 1
    dataloader_num_workers = min(4, os.cpu_count() // accelerator.num_processes if accelerator.num_processes > 0 and os.cpu_count() is not None else 1)
    _PROVER_TEMPLATE = (
        "Complete the Lean 4 proof below. Only output the Lean code for the complete proof"
        "```lean\n{code_block_statement} := by\n```"
    )
    prompt_builder: PromptBuilder = get_builder("simple",template=_PROVER_TEMPLATE)

    if accelerator.is_main_process:
        print("--- LeanEval Local HuggingFace Model Run ---")
        print(f"Time: {strftime('%Y-%m-%d %H:%M:%S')}")
        print("--- Configuration ---")
        print(f"Accelerator: mixed_precision='{accelerator.mixed_precision}', num_processes={accelerator.num_processes}, device='{str(device)}'")
        print(f"Dataset path: {Path(dataset_path).resolve()}")
        print(f"Output directory: {current_run_output_dir.resolve()}")
        print(f"Model Config: \n{hf_config.model_dump_json(indent=2)}")
        print(f"Per-device inference batch size: {per_device_batch_size}")
        print(f"Dataloader num_workers: {dataloader_num_workers}")
        print(f"Prompt Builder: {prompt_builder.__class__.__name__}")
        print("--------------------")

    # 2. 加载数据集和 DataLoader
    if accelerator.is_main_process:
        print("Loading dataset...")
    dataset = JsonDataset(dataset_path)

    def custom_collate_fn(batch_items: List[LeanItem]) -> Dict[str, Any]:
        prompts_for_model = []
        original_items_metadata = []

        for item in batch_items:
            if hasattr(prompt_builder, 'template') and "{code_block_statement}" in prompt_builder.template:
                 code_block_stmt_for_template = item.prompt_ready_stmt 
                 prompt_str = prompt_builder.template.format(code_block_statement=code_block_stmt_for_template)
            else: 
                 prompt_str = prompt_builder.build_str(item)


            prompts_for_model.append(prompt_str)
            original_items_metadata.append({
                "id": item.id,
                "prompt_ready_stmt": item.prompt_ready_stmt,
            })
        return {"prompts_for_model": prompts_for_model, "original_items_metadata": original_items_metadata}

    eval_dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    # 3. 初始化本地 HuggingFace 模型
    if accelerator.is_main_process:
        print(f"Initializing HuggingFace model wrapper for: {hf_config.model_name}")

    with ModelRegistry.create("huggingface", **hf_config.model_dump()) as hf_model_wrapper:
        if hf_model_wrapper.tokenizer and hasattr(hf_model_wrapper.tokenizer, "eos_token_id"):
            if "eos_token_id" not in hf_model_wrapper.cfg.generation_kwargs: 
                 hf_model_wrapper.cfg.generation_kwargs["eos_token_id"] = hf_model_wrapper.tokenizer.eos_token_id

        # 4. 使用 accelerator.prepare()
        if accelerator.is_main_process:
            print("Preparing model and dataloader with Accelerator...")
        
        prepared_nn_model, prepared_dataloader = accelerator.prepare(
            hf_model_wrapper.model, eval_dataloader 
        )
        hf_model_wrapper.model = prepared_nn_model 

        if accelerator.is_main_process:
            proof_save_dir.mkdir(parents=True, exist_ok=True)
            accelerator.print(f"Proofs will be saved in: {proof_save_dir.resolve()}")
            accelerator.print(f"Results log will be saved in: {results_log_file.resolve()}")
        accelerator.wait_for_everyone() # 确保所有进程都看到目录已创建 (如果后续非主进程也需要访问)

        # 5. 推理循环
        current_process_outputs: List[Dict[str, str]] = []

        if hasattr(hf_model_wrapper.model, 'eval'):
            hf_model_wrapper.model.eval()
        
        if accelerator.is_main_process:
            print("Starting inference loop...")
        inference_start_time = time()

        with torch.no_grad():
            progress_bar = tqdm(
            prepared_dataloader,
            desc="Inference",
            disable=not accelerator.is_local_main_process
        )
            for batch_data in progress_bar:
                prompts_to_model_input = batch_data["prompts_for_model"]
                original_items_metadata_batch = batch_data["original_items_metadata"]

                generated_batch_texts: List[str] = hf_model_wrapper.batch_predict(
                    prompts_to_model_input, 
                    max_len=hf_config.generation_kwargs.get("max_new_tokens", 100), 
                    batch_size=len(prompts_to_model_input) 
                )

                for i, gen_text in enumerate(generated_batch_texts):
                    item_metadata = original_items_metadata_batch[i]
                    current_process_outputs.append({
                        "id": item_metadata["id"],
                        "generated_proof_part": gen_text.strip(),
                        "prompt_ready_stmt": item_metadata["prompt_ready_stmt"]
                    })
                
        
        accelerator.wait_for_everyone()
        inference_end_time = time()
        if accelerator.is_main_process:
            print(f"Inference finished in {inference_end_time - inference_start_time:.2f} seconds across {accelerator.num_processes} processes.")

        # 6. 收集所有进程的结果到主进程
        if accelerator.is_main_process:
            print("Gathering results from all processes...")
        gathered_outputs_list_of_lists = accelerate.utils.gather_object(current_process_outputs)

        # 7. 在主进程中处理、保存和验证结果
        if accelerator.is_main_process:
            final_outputs_to_verify: List[Dict[str, str]] = []
            for sublist in gathered_outputs_list_of_lists:
                final_outputs_to_verify.append(sublist)
            saved_files_for_verification = []
            if not final_outputs_to_verify:
                print("No outputs were generated or gathered.")
            
            for idx, output_data in enumerate(tqdm(final_outputs_to_verify, desc="Saving proofs on main process")):
                item_id = output_data["id"]
                generated_proof_part = output_data["generated_proof_part"]
                prompt_ready_stmt = output_data["prompt_ready_stmt"]
                full_lean_code = f"{prompt_ready_stmt}\n{extract_lean_block(generated_proof_part)}"
                if idx == 0:
                    print("\n--- Example of Lean code to be saved ---")
                    print(f"ID: {item_id}")
                    print(f"Prompt Ready Stmt (from LeanItem):\n{prompt_ready_stmt}")
                    print(f"Generated Proof Part (from model):\n{generated_proof_part}")
                    if full_lean_code:
                        print(f"Concatenated Full Lean Code:\n{full_lean_code[:500]}...") 
                    print("--------------------------------------\n")


                safe_item_id = str(item_id).replace("/", "_").replace("\\", "_")
                proof_file_path = proof_save_dir / f"proof_{idx:04d}_{safe_item_id}.lean"
                
                try:
                    with open(proof_file_path, "w", encoding="utf-8") as f:
                        f.write(full_lean_code)
                    saved_files_for_verification.append(proof_file_path)
                except Exception as e:
                    print(f"Error writing file {proof_file_path}: {e}")


            # 8. 验证 (在主进程中)
            validation_results_log_entries = []
            if saved_files_for_verification:
                print(f"Starting validation for {len(saved_files_for_verification)} saved proofs...")
                lean_project_root = Path(".").resolve()
                validator = ProofValidator(
                    lean_cmd=["lake", "env", "lean"], 
                    work_dir=lean_project_root, 
                    timeout=20
                    )

                passed_count = 0
                validation_start_time = time()
                num_validation_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
                validation_map_results: List[Tuple[Path,bool,str]] = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_validation_workers) as executor:
                    future_to_filepath = {
                        executor.submit(validator.validate_file,filepath): filepath for filepath in saved_files_for_verification
                    }
                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_filepath),
                        total=len(saved_files_for_verification),
                        desc="Validating proofs (ThreadPool)"
                    ):
                        filepath = future_to_filepath[future]
                        try:
                            success,msg = future.result()
                            validation_map_results.append((filepath,success,msg))
                        except Exception as e:
                            accelerator.print(f"{filepath} generated an exception: {e}")
                            validation_map_results.append((filepath,False,str(e)))
                validation_end_time = time()
                accelerator.print(f"Validation (processing part) finished in {validation_end_time - validation_start_time:.2f} seconds.")
                results_dict = {fp: (s, m) for fp, s, m in validation_map_results}
                ordered_results = []
                for fp_original in saved_files_for_verification:
                    if fp_original in results_dict:
                        s, m = results_dict[fp_original]
                        ordered_results.append((fp_original, s, m))
                    else:
                        ordered_results.append((fp_original, False, "Validation task did not complete or was not found in results_dict"))
                total_validated = len(saved_files_for_verification)
                pass_rate = (passed_count / total_validated) * 100 if total_validated > 0 else 0.0
                summary = (f"\n--- Validation Summary ---\n"
                           f"Total Proofs Attempted: {total_validated}\n"
                           f"Successfully Validated: {passed_count}\n"
                           f"Pass Rate: {pass_rate:.2f}%\n")
                accelerator.print(summary)
                validation_results_log_entries.append(summary)

                try:
                    with open(results_log_file, "w", encoding="utf-8") as f:
                        f.write(f"Model: {model_id}\n")
                        f.write(f"Dataset: {dataset_path}\n")
                        f.write(f"Timestamp: {strftime('%Y%m%d-%H%M%S')}\n")
                        f.write(f"Configuration:\n{hf_config.model_dump_json(indent=2)}\n")
                        f.write("--- Validation Details ---\n")
                        for entry in validation_results_log_entries:
                            f.write(entry + "\n")
                    accelerator.print(f"Validation log saved to: {results_log_file.resolve()}")
                except Exception as e:
                    accelerator.print(f"Error writing results log file {results_log_file}: {e}")
            else:
                accelerator.print("No proofs were saved, skipping validation.")

    accelerator.wait_for_everyone() 
    if accelerator.is_main_process:
        total_run_time = time() - accelerator.process_index 
        print(f"Local HuggingFace model evaluation pipeline finished. runtime: {total_run_time}")

if __name__ == '__main__':
    overall_start_time = time()
    main()
    if Accelerator().is_main_process:
        print(f"Total script execution time: {time() - overall_start_time:.2f} seconds.")