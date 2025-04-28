```lean
import Mathlib.Data.Set.Basic

theorem inter_subset_right {α : Type*} (s t : Set α) : s ∩ t ⊆ t := by
  intro x h
  exact h.right
```