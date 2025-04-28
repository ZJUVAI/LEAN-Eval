```lean
import Mathlib.Data.Set.Basic

theorem union_self {α : Type*} (s : Set α) : s ∪ s = s := by
  ext x
  simp only [Set.mem_union, Set.mem_setOf_eq, or_self_iff]
```