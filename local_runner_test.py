# 在您的主脚本中 (例如 run_local_eval.py)
from LeanEval.runner.local_runner import LocalHuggingFaceRunner
from LeanEval.runner.local_runner_search import LocalSearchRunner
from pathlib import Path
from LeanEval.utils import process_dataset
import accelerate

def main():
    # download_path = "./data/downloaded/DeepSeek-Prover-V1.5/datasets/minif2f.jsonl"
    # output_json_path = "./data/json/minilean.json"
    # process_dataset.process_jsonl_dataset(download_path=download_path,ouput_json_path=output_json_path)
    runner = LocalHuggingFaceRunner(
        model_id="deepseek-ai/DeepSeek-Prover-V2-7B",
        dataset_path="./data/json/minif2f.json",
        output_dir_base="./outputs_runner_test",
        per_device_batch_size=1,
        max_new_tokens=512,
        mixed_precision='bf16', # 或 fp16
        num_proof_rounds=2
    )
    runner.run()

# --- 如何运行这个脚本 ---
def run_search_evaluation():
    """实例化并运行 LocalSearchRunner 的示例函数。"""
    
    # 为策略生成任务优化的 few-shot 示例
    tactic_shots = [
        (
            "Given the following partial proof and the current goals, what is the next single tactic to apply?\n\n"
            "### Current Proof State:\n"
            "```lean\nimport Mathlib.Tactic\n\ntheorem Nat.add_comm (n m : Nat) : n + m = m + n := by\n```\n\n"
            "### Current Goals from Lean InfoView:\n"
            "```\n- n m : ℕ ⊢ n + m = m + n\n```\n\n"
            "Your response must be the next single tactic.",
            "```lean\ninduction m\n```"
        ),
    ]
    
    runner = LocalSearchRunner(
        model_id="deepseek-ai/DeepSeek-Prover-V2-7B",
        dataset_path="./data/json/ez_lean.json",
        output_dir_base="./outputs_runner_test",
        tactic_shots=tactic_shots,
        bfs_degree=5,
        bfs_timeout=300,
        mixed_precision='bf16' # 或 'fp16'
    )
    runner.run()


if __name__ == "__main__":
    # 使用 `accelerate launch <your_script_name>.py` 来运行
    run_search_evaluation()