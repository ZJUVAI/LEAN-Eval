import Mathlib.Data.Set.Basic

theorem subset_insert {α : Type*} (a : α) (s : Set α) : s ⊆ Insert a s := by
  intro x hx
  rw [Set.mem_insert_iff]
  right
  exact hx