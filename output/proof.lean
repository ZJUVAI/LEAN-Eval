import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
Show that it is $\frac{2\sqrt{3}}{3}$.-/
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
have h₃ := h₀ 1
  have h₄ := h₀ 3
  rw [h₁] at h₃
  rw [h₂] at h₄
  have h₅ : r ^ 2 = 3 := by
    rw [← h₄, ← h₃]
    field_simp
    ring
  have h₆ : r = Real.sqrt 3 ∨ r = -Real.sqrt 3 := by
    rw [← pow_two_eq_pow_two_iff_eq_or_eq_neg, h₅]
    exact Real.sq_sqrt 3
  have h₇ : a = 2 / r := by
    rw [h₃]
    field_simp
  rw [h₇, h₀]
  simp
  rcases h₆ with h | h
  · left
    rw [h]
    field_simp
    rw [Real.sqrt_mul_self_eq_abs, abs_of_pos]
    · field_simp
      rw [mul_comm]
    · linarith
  · right
    rw [h]
    field_simp
    rw [Real.sqrt_mul_self_eq_abs, abs_of_neg]
    · field_simp
      rw [mul_comm]
    · linarith
