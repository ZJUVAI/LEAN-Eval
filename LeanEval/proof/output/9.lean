```lean
import Mathlib.Data.Set.Basic

theorem inter_empty_right {α : Type*} (s : Set α) : s ∩ ∅ = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, and_false]
```