import Mathlib

theorem auto_theorem_10 :
 ∀ (A : Finset ℕ) (hA : ∀ a ∈ A, 0 < a),
 (Fintype.card { B : Finset ℕ | B.Nonempty ∧ (max' B (Nonempty.to_subtype B.Nonempty) ∈ A) } = 2024) →
 ∑ a in A, a = 10 := by
  intro A hA hcard
  have hA_nonempty : A.Nonempty := by
    by_contra h
    simp only [Finset.not_nonempty_iff_eq_empty.mp h] at hcard
    simp only [Set.setOf_false, Set.finite_empty, Fintype.card_ofFinset, Finset.card_empty] at hcard
    linarith
  let max_A := A.max' hA_nonempty
  have hmax_A : max_A ∈ A := Finset.max'_mem _ _
  have hmax_pos : 0 < max_A := hA _ hmax_A
  have hA_eq : A = {max_A} := by
    by_contra h
    have h' : ∃ a ∈ A, a ≠ max_A := by
      rw [Finset.eq_singleton_iff_nonempty_unique_mem]
      push_neg
      exact ⟨hA_nonempty, h⟩
    rcases h' with ⟨a, ha, hane⟩
    have hlt : a < max_A := lt_of_le_of_ne (Finset.le_max' A a ha) hane
    let B1 := {max_A}
    let B2 := {a, max_A}
    have hB1 : B1 ∈ {B : Finset ℕ | B.Nonempty ∧ max' B _ ∈ A} := by
      simp [B1, max'_singleton, hmax_A]
    have hB2 : B2 ∈ {B : Finset ℕ | B.Nonempty ∧ max' B _ ∈ A} := by
      simp [B2, max'_cons, hmax_A]
    have hB_ne : B1 ≠ B2 := by
      intro heq
      have : a ∈ B2 := by simp [B2]
      rw [heq] at this
      simp [B1] at this
      exact hane this
    have hcard_ge : Fintype.card {B : Finset ℕ | B.Nonempty ∧ max' B _ ∈ A} ≥ 2 := by
      apply Fintype.card_le_of_injective (fun b => if b = 0 then B1 else B2)
      intro x y h
      fin_cases x <;> fin_cases y <;> simp at h ⊢
      · rfl
      · exact (hB_ne h.symm).elim
      · exact (hB_ne h).elim
      · rfl
    linarith [hcard]
  rw [hA_eq, Finset.sum_singleton] at *
  have hcard_eq : Fintype.card {B : Finset ℕ | B.Nonempty ∧ max' B _ ∈ {max_A}} = 2 ^ max_A - 1 := by
    have : {B : Finset ℕ | B.Nonempty ∧ max' B _ ∈ {max_A}} = {B : Finset ℕ | B.Nonempty ∧ max' B _ = max_A} := by
      ext B
      simp
    rw [this]
    have : {B : Finset ℕ | B.Nonempty ∧ max' B _ = max_A} = {B : Finset ℕ | B.Nonempty ∧ max' B _ ≤ max_A ∧ max_A ∈ B} := by
      ext B
      constructor
      · intro h
        exact ⟨h.1, le_of_eq h.2, by rw [h.2]; exact max'_mem _ _⟩
      · intro h
        exact ⟨h.1, (max'_eq_of_le _ h.2.2 h.2.1).symm⟩
    rw [this]
    have : {B : Finset ℕ | B.Nonempty ∧ max' B _ ≤ max_A ∧ max_A ∈ B} = 
          {B : Finset ℕ | max_A ∈ B ∧ B ⊆ (Icc 1 max_A : Finset ℕ)} := by
      ext B
      simp only [Set.mem_setOf_eq, Finset.mem_Icc, and_assoc]
      constructor
      · intro ⟨hne, hmax_le, hmem⟩
        refine ⟨hmem, fun x hx => ?_⟩
        have hx_le := le_max' B x hx
        rw [hmax_le] at hx_le
        exact ⟨Nat.pos_of_ne_zero (ne_of_gt (hA x (by rwa [hA_eq, Finset.mem_singleton] at hx_le)), hx_le⟩
      · intro ⟨hmem, hsub⟩
        refine ⟨Finset.nonempty_of_mem hmem, ?_, hmem⟩
        apply max'_le
        intro x hx
        exact (hsub hx).2
    rw [this]
    have : Fintype.card {B : Finset ℕ | max_A ∈ B ∧ B ⊆ Icc 1 max_A} = 
          2 ^ (max_A - 1) := by
      simp [Finset.card_powerset, Nat.sub_add_cancel hmax_pos]
    rw [this]
    have : 2 ^ (max_A - 1) = 2 ^ max_A - 1 ↔ max_A = 4 := by
      constructor
      · intro h
        have h' : 2 ^ (max_A - 1) + 1 = 2 ^ max_A := by linarith
        have : max_A ≤ 4 := by
          by_contra h''
          push_neg at h''
          have : 2 ^ (max_A - 1) + 1 < 2 ^ max_A := by
            apply lt_of_lt_of_le _ (pow_le_pow_right (by norm_num) (Nat.sub_le_sub_right h'' 1))
            norm_num
          linarith
        interval_cases max_A
        · simp at hmax_pos
        · simp at h'
        · simp at h'
        · simp at h'
      · intro h
        rw [h]
        norm_num
    rw [← this]
    exact hcard
  rw [hcard_eq] at hcard
  norm_num at hcard
  rw [hcard]
  norm_num