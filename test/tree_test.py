from LeanEval.prompt import get_builder
import os
from LeanEval.models import ModelRegistry
from LeanEval.models import DeepSeekAPIModel
from LeanEval.validator.proof_validator import ProofValidator
from time import time
from LeanEval.validator.proof_search import BFSProver

shots = [
    ("```lean\n"
     "import Mathlib.Data.Set.Basic\n"
     "theorem union_comm {α : Type*} (s t : Set α) : s ∪ t = t ∪ s := by\n"
     "ext x\n"
     "simp only [Set.mem_union]\n"
     "constructor\n"
     "· intro h\n"
     "  cases h with\n"
     "  | inl hs => exact Or.inr hs\n"
     "-- the next line of the proof here\n"
     "```"
     ,
     "```lean\n"
     "  | inr ht => exact Or.inl ht\n"
     "```")
]
prompt_builder_chat = get_builder("fewshot", shots=shots)

with ModelRegistry.create(
    "deepseek_api",
    model_name="deepseek-chat",
    api_url="https://api.deepseek.com/beta/chat/completions",
    api_key="sk-c5172f87dfe4418899fefd6cb6ee7309",
    timeout=60,               # 可以自定义传入Config内未定义的字段
    temperature=0.8,
) as model:
    bfsProver = BFSProver(model, prompt_builder_chat, degree = 3)
    goal: str = ("import Mathlib.Data.Set.Basic\n\n"
                "theorem inter_assoc {α : Type*} (r s t : Set α) : r ∩ s ∩ t = r ∩ (s ∩ t) := by\n")
    Root, result = bfsProver.thread_prove(goal)
    print("result:\n",result)