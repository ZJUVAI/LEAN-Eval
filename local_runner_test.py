# 在您的主脚本中 (例如 run_local_eval.py)
from LeanEval.runner.local_runner import LocalHuggingFaceRunner
from pathlib import Path
from LeanEval.utils import process_dataset
from LeanEval.datasets import downloader
import accelerate
import os
import json

def main():
    # github_downloader = downloader.GitHubDownloader(
    # url="https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git",
    # output_dir="./data/downloaded"
    # )

    # github_downloader.download()
    # print('github download success')

    # download_path = "./data/downloaded/DeepSeek-Prover-V1.5/datasets/minif2f.jsonl"
    output_json_path = "./data/json/minif2f.json"
    # process_dataset.process_jsonl_dataset(download_path=download_path,ouput_json_path=output_json_path)
    # with open(output_json_path,"r",encoding="utf-8") as f:
    #     data = json.load(f)
    #     with open("./data/json/minilean.json","w",encoding="utf-8") as wf:
    #         json.dump(data[:4],wf,indent=2,ensure_ascii=False)

    runner = LocalHuggingFaceRunner(
        model_id="gpt2",
        dataset_path="./data/json/minilean.json", 
        output_dir_base="./outputs_runner_test",
        per_device_batch_size=1, 
        max_new_tokens=512,
        mixed_precision='bf16', # 或 fp16
        num_proof_rounds=2
    )
    runner.run()

if __name__ == "__main__":
    # 使用 accelerate launch 运行此脚本
    # accelerate launch run_local_eval.py
    main()