# LeanEval/hf_model_test.py
import torch
from LeanEval.models import ModelRegistry, HuggingFaceModelConfig # 导入 HuggingFaceModelConfig

# --- 配置你的本地 Hugging Face 模型 ---
# 选择一个较小的模型进行快速测试，例如 'gpt2' 或 'deepseek-ai/DeepSeek-Prover-V2-7B'
# 确保你的机器可以运行它，特别是显存
cfg_dict = {
    "model_name": "gpt2",  # Hugging Face Model ID
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_dtype": "float16" if torch.cuda.is_available() else "float32", # FP16 on GPU if available
    "trust_remote_code": False,
    # 可以添加 generation_kwargs 用于控制生成行为
    "generation_kwargs": {
        "temperature": 0.7,
        "top_k": 50,
        "do_sample": True, # 如果想要采样而不是贪婪解码
    }
}

# 使用 HuggingFaceModelConfig 来创建配置实例
hf_config = HuggingFaceModelConfig(**cfg_dict)

print(f"Attempting to load model '{hf_config.model_name}' on device '{hf_config.device}'...")

try:
    # 使用 ModelRegistry.create 来实例化
    # 注意：注册名是 "huggingface"
    with ModelRegistry.create("huggingface", **hf_config.model_dump()) as model:
        print(f"Model {model.cfg.model_name} loaded successfully.")

        prompt1 = "theorem simple_add (a b : Nat) : a + b ="
        print(f"\n--- Testing single predict for: '{prompt1}' ---")
        # max_len 在这里代表 max_new_tokens
        generated_text1 = model.predict(prompt1, max_len=20)
        print(f"Prompt: {prompt1}")
        print(f"Generated: {generated_text1}")

        prompt2 = "def hello_world : String :="
        print(f"\n--- Testing single predict for: '{prompt2}' ---")
        generated_text2 = model.predict(prompt2, max_len=10, temperature=0.9, top_p=0.9) # 覆盖kwargs
        print(f"Prompt: {prompt2}")
        print(f"Generated: {generated_text2}")


        prompts_batch = [
            "import Mathlib\ntheorem ex1 (n : Nat) : n + 0 =",
            "lemma ex2 (p q : Prop) (hp : p) (h_imp : p → q) :",
            "def factorial_aux (n acc : Nat) : Nat := match n with | 0 => acc | Nat.succ k => factorial_aux k ((Nat.succ k) * acc)"
        ]
        print(f"\n--- Testing batch predict for {len(prompts_batch)} prompts ---")
        # batch_predict 的 max_len 也是 max_new_tokens
        batch_results = model.batch_predict(prompts_batch, max_len=30, batch_size=2)
        for p, r in zip(prompts_batch, batch_results):
            print(f"Prompt: {p}")
            print(f"Generated: {r}\n")

        print("Testing release...")
    print("Model released.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()