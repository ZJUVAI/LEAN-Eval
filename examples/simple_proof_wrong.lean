import Mathlib.Data.Nat.Basic
-- 删去最后一步
-- 定义定理：对任意自然数 n，n + 0 = n 成立
theorem add_zero (n : ℕ) : n + 0 = n := by
  -- 对 n 进行归纳
  induction n with
  | zero =>
    -- 基本情况 (n=0): 0 + 0 = 0 显然成立
    simp
  | succ n ih =>
    -- 归纳步骤：假设 n + 0 = n 成立，证明 (n + 1) + 0 = n + 1

