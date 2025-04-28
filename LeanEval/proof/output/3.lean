```lean
import Mathlib.Data.Set.Basic

theorem inter_comm {α : Type*} (s t : Set α) : s ∩ t = t ∩ s := by
  ext x
  simp only [Set.mem_inter_iff]
  constructor
  · intro ⟨hxs, hxt⟩
    exact ⟨hxt, hxs⟩
  · intro ⟨hxt, hxs⟩
    exact ⟨hxs, hxt⟩
```