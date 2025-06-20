from LeanEval.prompt import get_builder
from LeanEval.datasets import LeanItem
from LeanEval.datasets import JsonDataset
from LeanEval.models import ModelRegistry
from LeanEval.models import DeepSeekAPIModel
json_path = "./data/json/data_1.json"

ds = JsonDataset(json_path)
item = ds[0]


# 1. 零样例
pb = get_builder("simple")
prompt_str  = pb.build_str(item)   # 纯字符串
# print(prompt_str)
chat_prompt = pb.build_chat(item)  # [{"role":"user", ...}]
# print(chat_prompt)
# # 2. few-shot
shots = [
    ("```lean\nlemma two_mul (n : Nat) : 2 * n = n + n := by\n  -- proof?\n```",
     "```lean\nlemma two_mul (n : Nat) : 2 * n = n + n := by\n  simp [two_mul]\n```")
]
# pb_fs = get_builder("fewshot", shots=shots)
# messages = pb_fs.build_chat(item)
# for _ in messages:
#     print(_)

with ModelRegistry.create(
    "deepseek_api",
    model_name="deepseek-chat",
    api_url="https://api.deepseek.com/beta/chat/completions",
    api_key="sk-c5172f87dfe4418899fefd6cb6ee7309",
    timeout=60,               # 可以自定义传入Config内未定义的字段
    temperature=0.8,
) as model:
    for item in ds:
        prompt_str = pb.build_str(item)
        print(prompt_str)
        # print(len(prompt_str))
        # print(type(prompt_str))
        # print(model.predict(prompt_str))