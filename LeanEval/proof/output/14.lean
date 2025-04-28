```lean
import Mathlib.Data.Set.Basic

theorem diff_subset {α : Type*} (s t : Set α) : s \ t ⊆ s := by
  intro x hx
  exact hx.1
```