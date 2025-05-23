import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Function

theorem image_union {α β : Type*} (f : α → β) (s t : Set α) : f '' (s ∪ t) = f '' s ∪ f '' t