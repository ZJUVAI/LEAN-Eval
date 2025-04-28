```lean
import Mathlib.Data.Set.Basic

theorem compl_compl {α : Type*} (s : Set α) : Set.compl (Set.compl s) = s := by
  ext x
  simp only [Set.mem_compl_iff, not_not]
  rfl
```