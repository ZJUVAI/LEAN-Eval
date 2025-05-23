import Mathlib.Data.Set.Basic

theorem inter_assoc {α : Type*} (r s t : Set α) : r ∩ s ∩ t = r ∩ (s ∩ t) := by
  ext x
  simp only [Set.mem_inter_iff]
  constructor
  · intro h
    exact ⟨h.left.left, h.left.right, h.right⟩
  · intro h
    exact ⟨⟨h.left, h.right.left⟩, h.right.right⟩