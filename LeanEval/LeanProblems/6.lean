import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Intervals.Basic

theorem Icc_subset_Ico {α : Type*} [Preorder α] {a b : α} : Set.Icc a b ⊆ Set.Ico a b