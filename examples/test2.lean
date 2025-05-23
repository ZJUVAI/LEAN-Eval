import Mathlib.Algebra.Group.Defs

theorem add_comm1 (a b : ℕ) : a + b = b + a := by
  -- 使用归纳法证明
  induction a with
  | zero =>
    -- 基本情况：当 a =  时
    simp
  | succ a ih =>
    -- 归纳步骤：假设 a + b = b + a，证明 (a + 1) + b = b + (a + 1)
    simp [Nat.succ_add, ih]
