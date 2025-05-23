import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.ModEq

/--
Alice and Bob play the following game. A stack of `n` tokens lies before them. The players take turns with Alice going first. 
On each turn, the player removes either 1 token or 4 tokens from the stack. Whoever removes the last token wins. 
Find the number of positive integers `n` less than or equal to 2024 for which there exists a strategy for Bob that guarantees 
that Bob will win the game regardless of Alice's play.
-/
theorem bob_wins_count : {n : ℕ | 0 < n ∧ n ≤ 2024 ∧ n % 5 = 0 ∨ n % 5 = 2}.toFinset.card = 810 := by
  have h_period : ∀ n, n % 5 = 0 ∨ n % 5 = 2 ↔ ∃ k, n = 5 * k ∨ n = 5 * k + 2 := by
    intro n
    constructor
    · intro h
      cases' h with h0 h2
      · exact ⟨n / 5, Or.inl (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h0)).symm⟩
      · exact ⟨n / 5, Or.inr (by rw [← h2, Nat.add_comm, Nat.mod_add_div])⟩
    · rintro ⟨k, rfl | rfl⟩
      · left; rw [Nat.mul_mod_right]
      · right; rw [Nat.add_mod, Nat.mul_mod_right, zero_add, mod_mod]
  
  have h_range : (Finset.range 2025).filter (fun n => n % 5 = 0 ∨ n % 5 = 2) = 
      ((Finset.range 405).map ⟨(· * 5), fun _ _ => by simp⟩) ∪ 
      ((Finset.range 405).map ⟨(· * 5 + 2), fun _ _ => by simp⟩) := by
    ext n
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_union, Finset.mem_map, Function.Embedding.coeFn_mk]
    rw [h_period]
    constructor
    · rintro ⟨hn, ⟨k, rfl | rfl⟩⟩
      · refine Or.inl ⟨k, ?_, rfl⟩
        rw [Nat.mul_le_left_iff_pos_left (by norm_num), Nat.lt_succ_iff] at hn
        exact hn
      · refine Or.inr ⟨k, ?_, rfl⟩
        rw [Nat.add_le_iff_nonneg_right, Nat.mul_le_left_iff_pos_left (by norm_num), Nat.lt_succ_iff] at hn
        exact hn
    · rintro (⟨k, hk, rfl⟩ | ⟨k, hk, rfl⟩)
      · refine ⟨?_, ⟨k, Or.inl rfl⟩⟩
        rw [Nat.mul_le_left_iff_pos_left (by norm_num), Nat.lt_succ_iff] at hk
        exact hk.trans (by norm_num)
      · refine ⟨?_, ⟨k, Or.inr rfl⟩⟩
        rw [Nat.add_le_iff_nonneg_right, Nat.mul_le_left_iff_pos_left (by norm_num), Nat.lt_succ_iff] at hk
        exact (Nat.add_le_add_right hk 2).trans (by norm_num)
  
  have h_disj : Disjoint ((Finset.range 405).map ⟨(· * 5), fun _ _ => by simp⟩) 
      ((Finset.range 405).map ⟨(· * 5 + 2), fun _ _ => by simp⟩) := by
    simp [Finset.disjoint_left, Function.Embedding.coeFn_mk]
    intros x hx y hy h
    rw [← Nat.add_left_cancel_iff] at h
    exact Nat.ne_of_lt (Nat.mod_lt _ (by norm_num)) h.symm
  
  rw [Set.toFinset_card, Nat.card_eq_fintype_card, Fintype.card_ofFinset, h_range, 
      Finset.card_disjoint_union h_disj, Finset.card_map, Finset.card_map]
  simp only [Finset.card_range]
  norm_num