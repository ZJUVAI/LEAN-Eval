```lean
import Mathlib.Data.Set.Basic

theorem union_comm {α : Type*} (s t : Set α) : s ∪ t = t ∪ s := by
  ext x
  simp only [Set.mem_union]
  constructor
  · intro h
    cases h with
    | inl h => exact Or.inr h
    | inr h => exact Or.inl h
  · intro h
    cases h with
    | inl h => exact Or.inr h
    | inr h => exact Or.inl h
```