import Mathlib.Data.Set.Basic

theorem compl_univ {α : Type*} : Set.compl (Set.univ : Set α) = ∅ := by
  ext x
  simp only [Set.mem_compl_iff, Set.mem_univ, Set.mem_empty_iff_false, not_true]