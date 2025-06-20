# 在您的主脚本中 (例如 run_local_eval.py)
from LeanEval.runner.local_runner import LocalHuggingFaceRunner
from pathlib import Path
from LeanEval.utils import process_dataset
import accelerate

def main():
    # download_path = "./data/downloaded/DeepSeek-Prover-V1.5/datasets/minif2f.jsonl"
    # output_json_path = "./data/json/minilean.json"
    # process_dataset.process_jsonl_dataset(download_path=download_path,ouput_json_path=output_json_path)
    runner = LocalHuggingFaceRunner(
        model_id="deepseek-ai/DeepSeek-Prover-V2-7B",
        dataset_path="./data/json/minilean.json",
        output_dir_base="./outputs_runner_test",
        per_device_batch_size=1,
        max_new_tokens=512,
        mixed_precision='bf16', # 或 fp16
        num_proof_rounds=2
    )
    runner.run()

if __name__ == "__main__":
    main()