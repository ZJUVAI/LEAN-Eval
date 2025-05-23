import Mathlib.Data.Set.Basic

theorem inter_subset_left {α : Type*} (s t : Set α) : s ∩ t ⊆ s := by
  intro x h
  exact h.left