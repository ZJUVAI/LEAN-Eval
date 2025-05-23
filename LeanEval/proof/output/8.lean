import Mathlib.Data.Set.Basic

theorem union_empty_left {α : Type*} (s : Set α) : ∅ ∪ s = s := by
  ext x
  simp