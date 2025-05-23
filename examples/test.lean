-- 导入 Mathlib 库
import Mathlib.Data.Nat.Basic

-- 定义定理：自然数加法的交换律
theorem add_comm (a b : ℕ) : a + b = b + a := by
  -- 使用归纳法证明
  induction a with
  | zero =>
    -- 基本情况：当 a =  时
    simp
  | succ a ih =>
    -- 归纳步骤：假设 a + b = b + a，证明 (a + 1) + b = b + (a + 1)
    simp [Nat.succ_add, ih]
