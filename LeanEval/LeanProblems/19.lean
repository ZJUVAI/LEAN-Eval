import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Lattice

theorem inter_assoc {α : Type*} (r s t : Set α) : r ∩ s ∩ t = r ∩ (s ∩ t)