from LeanEval.prompt import get_builder
import os
from LeanEval.datasets import LeanItem
from LeanEval.datasets import JsonDataset
from LeanEval.datasets import JsonlDataset
from LeanEval.models import ModelRegistry
from LeanEval.models import DeepSeekAPIModel
from LeanEval.models import GeminiAPIModel
from LeanEval.validator.proof_validator import ProofValidator
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
API_URL = "https://api-zjuvai.newnan.city/v1/chat/completions"
BOT_NAME = "gemini_api"
MODEL_NAME = "gemini-2.0-flash-exp"
API_KEY = "sk-Knq6X2wZut1jYaOXBc1120C2474141E2885a14F3A0D11fF8"
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
    BOT_NAME,
    model_name=MODEL_NAME,
    api_url=API_URL,
    api_key=API_KEY,
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
        num_workers=4,
        save_dir=proof_dir
    )
    end = time()
    print(f"run {len(prompts)} questions using {end - start}s")
        

validator = ProofValidator(work_dir=proof_dir)
results = validator.validate_dir()
print(results)
end = time()
print(end - start)