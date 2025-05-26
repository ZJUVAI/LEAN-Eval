# hf_search_main.py
import os
import torch
from pathlib import Path
from time import time, strftime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json
import sys
import accelerate
from accelerate import Accelerator
from torch.utils.data import DataLoader

# 确保 LeanEval 可以被导入
LEAN_EVAL_ROOT = Path(__file__).resolve().parent
sys.path.append(str(LEAN_EVAL_ROOT))

from LeanEval.datasets import JsonDataset, LeanItem
from LeanEval.prompt import get_builder, PromptBuilder, FewShotPromptBuilder
from LeanEval.models import ModelRegistry, HuggingFaceModelConfig
from LeanEval.validator.proof_validator import ProofValidator
from LeanEval.validator.proof_search import BFSProver

# --- 配置 ---
DATASET_PATH = "./data/json/minif2f_processed.json"  # 指向您处理后的数据集
MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B" # 或您选择的 HF 模型
OUTPUT_DIR_BASE = "./outputs_hf_search"
PER_DEVICE_BATCH_SIZE = 1 # 证明搜索时，每个设备一次处理一个定理
DATALOADER_NUM_WORKERS = 2 # 每个进程的数据加载器工作线程数
BFS_DEGREE = 5 # 每一步尝试多少个策略
BFS_TIMEOUT = 1800 # 每个定理的超时时间（秒）
BFS_NUM_WORKERS = 4 # 每个 Prover 实例的线程数

# --- 策略示例 (必须优化!) ---
TACTIC_SHOTS = [
    (
        ("You are a Lean 4 expert... \nUser:\nHint 1: ⊢ n + 0 = n\nComplete the next line for:\n"
         "```lean\nimport Mathlib\ntheorem add_zero (n : Nat) : n + 0 = n := by\n  -- the next line of the proof here\n```\n\nAssistant:\n"),
        "```lean\nrw [Nat.add_zero]\n```"
    ),
    (
       ("You are a Lean 4 expert... \nUser:\nHint 1: x : α\nh : x ∈ s ∪ t\n⊢ x ∈ t ∪ s\nComplete the next line for:\n"
        "```lean\nimport Mathlib.Data.Set.Basic\ntheorem union_comm {α : Type*} (s t : Set α) : s ∪ t = t ∪ s := by\n  ext x\n  simp only [Set.mem_union]\n  intro h\n  cases h\n  -- the next line of the proof here\n```\n\nAssistant:\n"),
       "```lean\n· exact Or.inr h.left\n```"
    )
]
# ---------------------

def main():
    accelerator = Accelerator(mixed_precision='fp16' if torch.cuda.is_available() else 'no')
    device = accelerator.device

    # 输出目录设置
    model_short_name = MODEL_ID.split('/')[-1]
    current_run_output_dir = Path(OUTPUT_DIR_BASE) / f"{model_short_name}_{strftime('%Y%m%d-%H%M%S')}"
    proof_save_dir = current_run_output_dir / "proofs"
    results_file = current_run_output_dir / "results.json"

    # HF 模型配置
    hf_config = HuggingFaceModelConfig(
        model_name=MODEL_ID,
        device=str(device),
        torch_dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
        trust_remote_code=True,
        generation_kwargs={
            "max_new_tokens": 64,
            "temperature": 0.4,
            "top_p": 0.95,
            "do_sample": True,
            # 检查您的模型！DeepSeek Prover V2 可能使用 32021 或其他 ID
            "pad_token_id": 32021
        }
    )

    # Prompt Builder
    prompt_builder = FewShotPromptBuilder(shots=[], tactic_shots=TACTIC_SHOTS)

    # 主进程打印配置
    if accelerator.is_main_process:
        print("--- LeanEval HuggingFace 证明搜索 ---")
        print(f"时间: {strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据集: {DATASET_PATH}")
        print(f"输出目录: {current_run_output_dir}")
        print(f"模型: {MODEL_ID}")
        print(f"Accelerator 进程数: {accelerator.num_processes}")
        print(f"BFS 搜索度: {BFS_DEGREE}, 超时: {BFS_TIMEOUT}s")
        proof_save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    try:
        dataset = JsonDataset(DATASET_PATH)
    except FileNotFoundError:
        if accelerator.is_main_process:
            print(f"错误: 数据集未找到于 {DATASET_PATH}.")
            print("请先运行 LeanEval/utils/process_dataset.py 或提供正确路径。")
        accelerator.wait_for_everyone()
        return

    # 定义 Collate 函数
    def lean_item_collate_fn(batch):
        # 如果 batch 已经是 LeanItem 列表，直接返回
        # 如果是字典列表，需要转换
        return [LeanItem(**item) if isinstance(item, dict) else item for item in batch]

    dataloader = DataLoader(
        dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        collate_fn=lean_item_collate_fn
    )

    # 初始化模型和 Prover
    with ModelRegistry.create("huggingface", **hf_config.model_dump()) as hf_model:
        # 使用 Accelerate 准备模型和数据加载器
        prepared_model, prepared_dataloader = accelerator.prepare(
            hf_model.model, dataloader
        )
        hf_model.model = prepared_model # 将准备好的模型放回包装器

        # 确保 pad_token_id 设置正确
        if hf_model.tokenizer.pad_token_id is None:
            if hf_model.tokenizer.eos_token_id is not None:
                hf_model.tokenizer.pad_token_id = hf_model.tokenizer.eos_token_id
                hf_model.model.config.pad_token_id = hf_model.tokenizer.eos_token_id
                accelerator.print("已将 pad_token_id 设置为 eos_token_id")
            else:
                 # 手动设置一个ID (e.g., 0) 或加载模型时使用的ID
                 hf_model.tokenizer.pad_token_id = hf_config.generation_kwargs.get("pad_token_id", 0)
                 hf_model.model.config.pad_token_id = hf_config.generation_kwargs.get("pad_token_id", 0)
                 accelerator.print(f"警告: pad_token_id 未设置, 已设为 {hf_model.tokenizer.pad_token_id}")


        if "eos_token_id" not in hf_model.cfg.generation_kwargs and hf_model.tokenizer.eos_token_id:
            hf_model.cfg.generation_kwargs["eos_token_id"] = hf_model.tokenizer.eos_token_id

        # 在每个进程中初始化 Prover
        bfs_prover = BFSProver(
            hf_model,
            prompt_builder,
            tmp_dir=current_run_output_dir / f"tmp_proofs_proc_{accelerator.process_index}",
            timeout=BFS_TIMEOUT,
            degree=BFS_DEGREE,
        )

        # 运行搜索循环
        process_results = []
        progress_bar = tqdm(
            prepared_dataloader,
            desc=f"证明搜索 (进程 {accelerator.process_index})",
            disable=not accelerator.is_local_main_process
        )

        for batch in progress_bar:
            for item in batch:
                goal = f"{item.prompt_ready_stmt} := by\n"
                accelerator.print(f"进程 {accelerator.process_index}: 开始证明 {item.id}...")
                search_start_time = time()

                root, final_code = bfs_prover.thread_prove(goal, num_workers=BFS_NUM_WORKERS)

                search_time = time() - search_start_time
                proved = root is not None and final_code is not None

                accelerator.print(f"进程 {accelerator.process_index}: 完成 {item.id}. 证明成功: {proved}, 用时: {search_time:.2f}s")

                result_data = {
                    "id": item.id,
                    "statement": item.statement,
                    "proved": proved,
                    "proof": final_code,
                    "time": search_time,
                    "process_index": accelerator.process_index,
                }
                process_results.append(result_data)

                # 立即保存证明文件
                safe_id = item.id.replace('/', '_').replace('\\', '_')
                proof_file = proof_save_dir / f"{safe_id}_{'ok' if proved else 'fail'}.lean"
                with proof_file.open("w", encoding="utf-8") as f:
                    f.write(final_code or goal + "  -- Proof Search Failed --")

        accelerator.wait_for_everyone()
        gathered_results = accelerator.gather_object(process_results)

        # 在主进程中进行验证和保存
        if accelerator.is_main_process:
            print("\n--- 聚合并保存结果 ---")
            print(f"收集到 {len(gathered_results)} 条结果。")

            with results_file.open("w", encoding="utf-8") as f:
                json.dump(gathered_results, f, indent=2, ensure_ascii=False)
            print(f"原始结果已保存到 {results_file}")

            print("\n--- 开始最终验证 ---")
            validator = ProofValidator(timeout=120)
            files_to_validate = list(proof_save_dir.glob("*_ok.lean"))
            print(f"找到 {len(files_to_validate)} 个可能成功的证明进行验证。")

            if files_to_validate:
                passed_files, failed_files = validator.validate_batch(files_to_validate)

                print(f"\n--- 验证总结 ---")
                print(f"总尝试数: {len(gathered_results)}")
                proved_count = sum(1 for r in gathered_results if r['proved'])
                print(f"搜索报告成功数: {proved_count}")
                print(f"验证通过数: {len(passed_files)}")
                print(f"验证失败数: {len(failed_files)}")

                validated_map = {Path(f).stem.replace('_ok',''): 'Validated' for f in passed_files}
                validated_map.update({Path(f).stem.replace('_ok',''): 'Failed Validation' for f in failed_files})

                for result in gathered_results:
                    stem = result['id'].replace('/', '_').replace('\\', '_')
                    result['validation_status'] = validated_map.get(stem, 'Not Validated' if not result['proved'] else 'Validation Error')

                final_results_file = current_run_output_dir / "results_final.json"
                with final_results_file.open("w", encoding="utf-8") as f:
                    json.dump(gathered_results, f, indent=2, ensure_ascii=False)
                print(f"带验证的最终结果已保存到 {final_results_file}")
            else:
                print("搜索未报告成功证明，跳过验证。")

    print(f"--- 进程 {accelerator.process_index} 完成 ---")

if __name__ == "__main__":
    main()