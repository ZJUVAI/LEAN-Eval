import re

def extract_lean_code_after_marker(text: str) -> str:
    """
    提取从 ```lean 开始之后的所有代码内容，直到文本结尾或下一个块。

    Args:
        text (str): 模型返回的完整输出文本

    Returns:
        str: 去除 ```lean 前缀的代码部分
    """
    marker = "```lean"
    end_marker = "```"
    if marker in text:
        # 分割开始标记后的内容
        parts = text.split(marker, 1)
        # 再分割结束标记
        code_part = parts[1].split(end_marker, 1)[0]
        return code_part.strip()
    return text.strip()

if __name__ == "__main__":
    raw_output = """
    import Mathlib

    /--
    You are a formal proof assistant. Given a math problem, translate it into Lean 4 using Mathlib.
    Prove the statement in a human-readable, idiomatic way using tactics.
    Make sure the code includes both the problem statement (as a docstring or comment) and its full formal proof.

    Problem: Let x, y, and z be positive real numbers such that:
        log₂(x / (yz)) = 1/2
        log₂(y / (xz)) = 1/3
        log₂(z / (xy)) = 1/4
        Find |log₂(x⁴ y³ z²)| as a reduced fraction m/n. Output m + n.
    -/
    theorem auto_theorem_1 :
    ∃ m n : ℕ, m.Coprime n ∧
        ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
        Real.logb 2 (x / (y * z)) = 1 / 2 →
        Real.logb 2 (y / (x * z)) = 1 / 3 →
        Real.logb 2 (z / (x * y)) = 1 / 4 →
        |Real.logb 2 (x^4 * y^3 * z^2)| = (m : ℝ) / n := by
    sorry
    ```Here's the completed Lean 4 code with the proof:

    ```lean
    import Mathlib

    /--
    You are a formal proof assistant. Given a math problem, translate it into Lean 4 using Mathlib.
    Prove the statement in a human-readable, idiomatic way using tactics.
    Make sure the code includes both the problem statement (as a docstring or comment) and its full formal proof.

    Problem: Let x, y, and z be positive real numbers such that:
        log₂(x / (yz)) = 1/2
        log₂(y / (xz)) = 1/3
        log₂(z / (xy)) = 1/4
        Find |log₂(x⁴ y³ z²)| as a reduced fraction m/n. Output m + n.
    -/
    theorem auto_theorem_1 :
        ∃ m n : ℕ, m.Coprime n ∧
        ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
        Real.logb 2 (x / (y * z)) = 1 / 2 →
        Real.logb 2 (y / (x * z)) = 1 / 3 →
        Real.logb 2 (z / (x * y)) = 1 / 4 →
        |Real.logb 2 (x^4 * y^3 * z^2)| = (m : ℝ) / n := by
    use 5, 6
    constructor
    · decide
    intro x y z hx hy hz h1 h2 h3
    have hx' : Real.logb 2 x - (Real.logb 2 y + Real.logb 2 z) = 1/2 := by
        rw [← Real.logb_div, Real.logb_mul hy hz] at h1
        exact h1
    have hy' : Real.logb 2 y - (Real.logb 2 x + Real.logb 2 z) = 1/3 := by
        rw [← Real.logb_div, Real.logb_mul hx hz] at h2
        exact h2
    have hz' : Real.logb 2 z - (Real.logb 2 x + Real.logb 2 y) = 1/4 := by
        rw [← Real.logb_div, Real.logb_mul hx hy] at h3
        exact h3
    set a := Real.logb 2 x
    set b := Real.logb 2 y
    set c := Real.logb 2 z
    have eq1 : a - b - c = 1/2 := by linear_combination hx'
    have eq2 : b - a - c = 1/3 := by linear_combination hy'
    have eq3 : c - a - b = 1/4 := by linear_combination hz'
    have ha : a = 17/24 := by
        have : 4*a = 17/6 := by linear_combination eq1 + eq2 - 2*eq3
        linarith
    have hb : b = 5/24 := by
        have : 4*b = 5/6 := by linear_combination -eq1 + eq2 + 2*eq3
        linarith
    have hc : c = -1/24 := by
        have : 4*c = -1/6 := by linear_combination eq1 + eq2 + 2*eq3
        linarith
    rw [Real.logb_mul (pow_pos hx 4) (mul_pos (pow_pos hy 3) (pow_pos hz 2)),
        Real.logb_pow, Real.logb_mul (pow_pos hy 3) (pow_pos hz 2), Real.logb_pow,
        Real.logb_pow, ha, hb, hc]
    norm_num
    rw [abs_of_pos]
    norm_num
    linarith
    """
    print(extract_lean_code_after_marker(raw_output))