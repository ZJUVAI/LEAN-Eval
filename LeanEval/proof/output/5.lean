```lean
import Mathlib.Data.Set.Basic

theorem inter_assoc {α : Type*} (r s t : Set α) : r ∩ s ∩ t = r ∩ (s ∩ t) := by
  ext x
  simp only [Set.mem_inter_iff]
  constructor
  · intro ⟨⟨h₁, h₂⟩, h₃⟩
    exact ⟨h₁, h₂, h₃⟩
  · intro ⟨h₁, h₂, h₃⟩
    exact ⟨⟨h₁, h₂⟩, h₃⟩
```