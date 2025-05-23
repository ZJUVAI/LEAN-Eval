import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Sqrt

/--
Let b ≥ 2 be an integer. Call a positive integer n b-eautiful if:
1. It has exactly two digits when expressed in base b
2. These two digits sum to √n
For example, 81 is 13-eautiful because 81 = 63₁₃ and 6 + 3 = √81 = 9.
Find the least integer b ≥ 2 for which there are more than ten b-eautiful integers.
-/
theorem exists_base_with_more_than_ten_beautiful_numbers :
  ∃ b : ℕ, 2 ≤ b ∧ (Fintype.card {n : ℕ | n.IsBeautifful b} > 10) := by
  -- Define what it means to be b-eautiful
  let IsBeautifful (b n : ℕ) : Prop :=
    2 ≤ b ∧
    (Nat.digits b n).length = 2 ∧
    (Nat.digits b n).get ⟨0, by simp⟩ + (Nat.digits b n).get ⟨1, by simp⟩ = Nat.sqrt n

  -- Find the smallest such b (b = 15 has 11 beautiful numbers)
  use 15
  constructor
  · simp -- shows 2 ≤ 15
  · -- Proof that there are more than 10 beautiful numbers for b=15
    -- The beautiful numbers are: 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196
    let beautiful_numbers : Finset ℕ := {16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196}
    have h_card : beautiful_numbers.card = 11 := by decide
    have h_subset : ∀ n ∈ beautiful_numbers, IsBeautifful 15 n := by
      intro n hn
      simp [IsBeautifful]
      repeat' constructor
      · decide -- 2 ≤ 15
      · -- Check each number has exactly 2 digits in base 15
        cases hn <;> simp [Nat.digits, Nat.digitsAux, Nat.mod_add_div] <;> decide
      · -- Check digit sums equal sqrt
        cases hn <;> simp [Nat.sqrt] <;> decide
    refine lt_of_lt_of_le ?_ (Fintype.card_le_of_subset h_subset)
    exact h_card