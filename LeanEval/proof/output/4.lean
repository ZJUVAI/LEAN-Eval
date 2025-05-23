import Mathlib.Data.Set.Basic

theorem union_assoc {α : Type*} (r s t : Set α) : r ∪ s ∪ t = r ∪ (s ∪ t) := by
  ext x
  simp only [Set.mem_union]
  tauto