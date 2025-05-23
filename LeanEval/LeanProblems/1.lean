import Mathlib.Data.Nat.Basic   -- 只需最基础的自然数库
/--
证明目标：对任意自然数 `a b`，都有 `a + b = b + a`。
这是加法在 `ℕ` 上的交换律，库中已有引理 `Nat.add_comm`。
-/

theorem add_comm_example (a b : ℕ) : a + b = b + a := by
  -- `Nat.add_comm` 恰好就是我们要的结论，直接 `exact` 即可
  exact Nat.add_comm a b
