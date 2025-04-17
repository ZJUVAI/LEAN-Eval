# 标准库：正则表达式，方便字符串分割
import re

#第三方库：openai，用于调用大模型API
from openai import OpenAI


# 初始化大模型的客户端，采用beta来实行FIM补全功能
client = OpenAI(
    api_key="sk-c5172f87dfe4418899fefd6cb6ee7309",  # 替换为你的API密钥
    base_url="https://api.deepseek.com/beta"
)

"""
在这里模仿deepseek-Prover的操作，我们把定理证明视为Lean4代码的补全过程，
给出prompt（inst）和代码开头（前缀code_prefix），使用deepseek的FIM补全功能，来完成Lean4代码
的证明过程，然后输出为Lean4文件，保存到output文件夹中，调用verifier验证。
"""

inst = r'''Complete the following Lean 4 code with no explanation:

```lean4
'''
code_prefix = r'''import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
Show that it is $\frac{2\sqrt{3}}{3}$.-/
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
'''
model_input = inst + code_prefix
suffix = r'''
  <;> simp_all
  <;> nlinarith
  <;> linarith
'''

try:
    response = client.completions.create(
        model="deepseek-chat",
        prompt=model_input,  
        max_tokens=2048,  
        temperature=1.0,
        top_p = 0.95,
        n = 1,
    )
    
    generated_part = response.choices[0].text.strip()
    result = inst + code_prefix + generated_part
    print(result)
    code = code_prefix + generated_part.rstrip('```') #删去generated_part右端的'```'字符
    path = "../output/proof.lean";
    file = open(path, "w")
    file.write(code)
    file.close()

except Exception as e:
    print(f"API调用失败：{str(e)}")

    # TODO: @yangxingtao 需要和@dongchenxuan 再同步一下mathlib环境等，现在的问题是输出的Lean4文件无法运行，大概是环境配置问题