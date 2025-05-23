import Mathlib.Data.Set.Basic

theorem subset_refl {α : Type*} (s : Set α) : s ⊆ s := by
  intro x h
  exact h