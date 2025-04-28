```lean
import Mathlib.Data.Set.Basic

theorem subset_trans {α : Type*} {s t u : Set α} (h₁ : s ⊆ t) (h₂ : t ⊆ u) : s ⊆ u := by
  intro x hx
  apply h₂
  apply h₁
  exact hx
```