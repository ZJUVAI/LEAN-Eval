from prompt import get_builder
import os
from datasets import LeanItem
from datasets import JsonDataset
from datasets import JsonlDataset
from models import ModelRegistry
from models import DeepSeekAPIModel
from validator.proof_validator import ProofValidator
from time import time

json_path = "./data/json/data_1.json"
jsonl_path = "./data/jsonl/data_1.jsonl"
ds = JsonDataset(json_path)


prompt_builder_str = get_builder("simple")
shots = [
    ("```lean\nlemma two_mul (n : Nat) : 2 * n = n + n := by\n  -- proof?\n```",
     "```lean\nlemma two_mul (n : Nat) : 2 * n = n + n := by\n  simp [two_mul]\n```")
]
prompt_builder_chat = get_builder("fewshot", shots=shots)

proof_dir = "./proof/output"
prompts = []
if not os.path.exists(proof_dir):
    os.makedirs(proof_dir)

with ModelRegistry.create(
    "deepseek_api",
    model_name="deepseek-chat",
    api_url="https://api.deepseek.com/beta/chat/completions",
    api_key="sk-c5172f87dfe4418899fefd6cb6ee7309",
    timeout=60,               # 可以自定义传入Config内未定义的字段
    temperature=0.8,
) as model:
    for idx,item in enumerate(ds):
        start = time()
        prompt_str = prompt_builder_str.build_chat(item)
        # print(prompt_str)
        prompts.append(prompt_str)
        # print("model block starts ...")
        model.predict(
            prompts,
            num_workers=20,
            save_dir=proof_dir
        )
        end = time()
        print(f"run {len(prompts)} questions using {end - start}s")
        

validator = ProofValidator(work_dir=proof_dir)
results = validator.validate_dir()
print(results)
end = time()
print(end - start)