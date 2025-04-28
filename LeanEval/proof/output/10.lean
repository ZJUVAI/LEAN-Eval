```lean
import Mathlib.Data.Set.Basic

theorem subset_union_left {α : Type*} (s t : Set α) : s ⊆ s ∪ t := by
  intro x hx
  exact Or.inl hx
```