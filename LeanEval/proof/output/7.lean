```lean
import Mathlib.Data.Set.Basic

theorem inter_self {α : Type*} (s : Set α) : s ∩ s = s := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_setOf_eq, and_self_iff]
```