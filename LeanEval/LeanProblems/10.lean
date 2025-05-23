import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Function

theorem preimage_union {α β : Type*} (f : α → β) (s t : Set β) : f ⁻¹' (s ∪ t) = f ⁻¹' s ∪ f ⁻¹' t