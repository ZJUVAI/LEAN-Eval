import Mathlib.Combinatorics.Enumerative.Combination
import Mathlib.Data.Fintype.Basic
import Mathlib.Probability.Conditional

open Finset Nat

/- Problem: Jen enters a lottery by picking 4 distinct numbers from S={1,2,3,...,9,10}. 4 numbers are randomly chosen from S. She wins a prize if at least two of her numbers were 2 of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is m/n where m and n are relatively prime positive integers. Find m+n. -/

theorem lottery_probability :
    let S : Finset ℕ := range 1 11
    let totalOutcomes := (S.card).choose 4
    let prizeOutcomes := (4.choose 2) * ((S.card - 4).choose 2) + (4.choose 3) * ((S.card - 4).choose 1) + (4.choose 4) * ((S.card - 4).choose 0))
    let grandPrizeOutcomes := 1
    let conditionalProbability := grandPrizeOutcomes / prizeOutcomes
    conditionalProbability = 1 / 14 := by
  let S : Finset ℕ := range 1 11
  have hS : S.card = 10 := by simp
  simp [hS]
  norm_num