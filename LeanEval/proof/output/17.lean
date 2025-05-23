import Mathlib.Data.Set.Basic

theorem image_empty {α β : Type*} (f : α → β) : f '' (∅ : Set α) = ∅ := by
  ext y
  simp only [Set.mem_image, Set.mem_empty_iff_false, iff_false]
  intro ⟨_, h, _⟩
  exact h