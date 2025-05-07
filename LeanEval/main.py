from prompt import get_builder
import os
from datasets import LeanItem
from datasets import JsonDataset
from datasets import JsonlDataset
from models import ModelRegistry
from models import DeepSeekAPIModel
from models import GeminiAPIModel
from validator.proof_validator import ProofValidator
from time import time
import logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - Thread %(thread)d (%(threadName)s) - %(message)s',
#     # stream=sys.stdout # 可选，默认是 sys.stderr，如果想输出到 stdout 可以指定
# )
# logging.getLogger("urllib3").setLevel(logging.INFO) # 或 logging.WARNING
# logging.getLogger("requests").setLevel(logging.INFO)


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
    "gemini_api",
    model_name="gemini-2.0-flash-exp",
    api_url="https://api-zjuvai.newnan.city/v1/chat/completions",
    api_key="sk-Knq6X2wZut1jYaOXBc1120C2474141E2885a14F3A0D11fF8",
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
            num_workers=1,
            save_dir=proof_dir
        )
        end = time()
        print(f"run {len(prompts)} questions using {end - start}s")
        

validator = ProofValidator(work_dir=proof_dir)
results = validator.validate_dir()
print(results)
end = time()
print(end - start)