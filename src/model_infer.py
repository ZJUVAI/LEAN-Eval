# # 标准库：正则表达式，方便字符串分割
# import re

# #第三方库：openai，用于调用大模型API
# from openai import OpenAI


# # 初始化大模型的客户端，采用beta来实行FIM补全功能
# client = OpenAI(
#     api_key="sk-c5172f87dfe4418899fefd6cb6ee7309",  # 替换为你的API密钥
#     base_url="https://api.deepseek.com/beta"
# )

# """
# 在这里模仿deepseek-Prover的操作，我们把定理证明视为Lean4代码的补全过程，
# 给出prompt（inst）和代码开头（前缀code_prefix），使用deepseek的FIM补全功能，来完成Lean4代码
# 的证明过程，然后输出为Lean4文件，保存到output文件夹中，调用verifier验证。
# """

# inst = r'''Complete the following Lean 4 code with no explanation:

# ```lean4
# '''
# code_prefix = r'''import Mathlib
# import Aesop

# set_option maxHeartbeats 0

# open BigOperators Real Nat Topology Rat

# /-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
# Show that it is $\frac{2\sqrt{3}}{3}$.-/
# theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
#   (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
# '''
# model_input = inst + code_prefix
# suffix = r'''
#   <;> simp_all
#   <;> nlinarith
#   <;> linarith
# '''

# try:
#     response = client.completions.create(
#         model="deepseek-chat",
#         prompt=model_input,  
#         max_tokens=2048,  
#         temperature=1.0,
#         top_p = 0.95,
#         n = 1,
#     )
    
#     generated_part = response.choices[0].text.strip()
#     result = inst + code_prefix + generated_part
#     print(result)
#     code = code_prefix + generated_part.rstrip('```') #删去generated_part右端的'```'字符
#     path = "../output/proof.lean"
#     file = open(path, "w")
#     file.write(code)
#     file.close()

# except Exception as e:
#     print(f"API调用失败：{str(e)}")

    # TODO: @yangxingtao 需要和@dongchenxuan 再同步一下mathlib环境等，现在的问题是输出的Lean4文件无法运行，大概是环境配置问题

import os
import re
from openai import OpenAI
from utils import extract_lean_code_after_marker

class DeepSeekLeanProver:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/beta"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = "deepseek-chat"
        self.output_dir = "../output"
        self.counter = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def lean_prove(self, code_prefix: str, suffix: str = "") -> str:
        """
        使用 DeepSeek 模型补全 Lean4 定理证明

        Returns:
            str: 生成的完整 Lean 代码
        """
        inst = r'''Complete the following Lean 4 code with no explanation:

    ```lean4
    '''
        suffix = r'''
      <;> simp_all
      <;> nlinarith
      <;> linarith
    '''
        model_input = inst + code_prefix
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=model_input,
                max_tokens=2048,
                temperature=1.0,
                top_p=0.95,
                n=1,
            )
            generated_part = response.choices[0].text.strip()
            generated_part = generated_part.rstrip('```')  # 去掉尾部 markdown 语法
            lean_code = code_prefix + generated_part + suffix
            save_name = f"proof_{self.counter}.lean"
            output_path = os.path.join(self.output_dir, save_name)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(lean_code)
            self.counter += 1

            return lean_code
        except Exception as e:
            print(f"模型调用失败: {e}")
            return ""
        
    def nl_prove(self, nature_language: str, suffix: str = "") -> str:
        self.counter += 1
        filename = f"proof_{self.counter}.lean"
        output_path = os.path.join(self.output_dir, filename)

        code_prefix = f"""import Mathlib
        /--
        You are a formal proof assistant. Given a math problem, translate it into Lean 4 using Mathlib.
        Prove the statement in a human-readable, idiomatic way using tactics.
        Make sure the code includes both the problem statement (as a docstring or comment) and its full formal proof.

        Problem: {nature_language}
        -/
        theorem auto_theorem_{self.counter} :
        """
        inst = "Complete the following Lean 4 code with no explanation:"
        prompt = inst + "\n```lean\n" + code_prefix

        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=2048,
                temperature=1.0,
                top_p=0.95,
                n=1,
            )

            generated_part = response.choices[0].text.strip().rstrip("```")
            generated_part = extract_lean_code_after_marker(generated_part)
            lean_code = generated_part + suffix

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(lean_code)

            return lean_code

        except Exception as e:
            print(f"模型调用失败: {e}")
            return ""

    # TODO 针对自然语言的数据集的问题输入，调用该方法的prompt目前似乎还比较简陋，产生的结果很多质量比较低

if __name__ == "__main__":
    # prover = DeepSeekLeanProver(api_key="sk-c5172f87dfe4418899fefd6cb6ee7309")
    # code_prefix = r'''import Mathlib
    # import Aesop

    # set_option maxHeartbeats 0

    # open BigOperators Real Nat Topology Rat

    # /-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
    # Show that it is $\frac{2\sqrt{3}}{3}$.-/
    # theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
    #   (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
    # '''
    # result = prover.prove(code_prefix)
    # print("=== 模型返回内容（完整 prompt + 生成） ===")
    # print(result)
    prover = DeepSeekLeanProver(api_key="sk-c5172f87dfe4418899fefd6cb6ee7309")

    nl_problem = """Let x, y, and z be positive real numbers such that:
    log₂(x / (yz)) = 1/2
    log₂(y / (xz)) = 1/3
    log₂(z / (xy)) = 1/4
    Find |log₂(x⁴ y³ z²)| as a reduced fraction m/n. Output m + n."""

    lean_code = prover.nl_prove(nl_problem)
    print(lean_code)


