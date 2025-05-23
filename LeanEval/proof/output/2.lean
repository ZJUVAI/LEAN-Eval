import Mathlib.Data.Set.Basic

theorem union_comm {α : Type*} (s t : Set α) : s ∪ t = t ∪ s := by
  ext x
  simp only [Set.mem_union]
  constructor
  · intro h
    cases h with
    | inl hs => exact Or.inr hs
    | inr ht => exact Or.inl ht
  · intro h
    cases h with
    | inl ht => exact Or.inr ht
    | inr hs => exact Or.inl hs