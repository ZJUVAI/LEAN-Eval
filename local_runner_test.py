# 在您的主脚本中 (例如 run_local_eval.py)
from LeanEval.runner.local_runner import LocalHuggingFaceRunner
from pathlib import Path
from LeanEval.utils import process_dataset
import accelerate

def main():
    dataset_root_directory = Path("./data/test").resolve()
    output_file = Path("./data/json/minilean.json").resolve()
    process_dataset.process_dataset(dataset_root_directory, output_file)
    runner = LocalHuggingFaceRunner(
        model_id="deepseek-ai/DeepSeek-Prover-V2-7B",
        dataset_path="./data/json/minilean.json", # 使用您处理好的数据集
        output_dir_base="./outputs_runner_test",
        per_device_batch_size=1, # 根据您的 GPU 显存调整
        max_new_tokens=512,
        mixed_precision='bf16', # 或 fp16
        num_proof_rounds=2
    )
    runner.run()

if __name__ == "__main__":
    # 使用 accelerate launch 运行此脚本
    # accelerate launch run_local_eval.py
    main()