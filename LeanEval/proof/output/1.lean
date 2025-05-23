import Mathlib.Data.Set.Basic

theorem subset_univ {α : Type*} (s : Set α) : s ⊆ Set.univ := by
  intro x hx
  exact Set.mem_univ x