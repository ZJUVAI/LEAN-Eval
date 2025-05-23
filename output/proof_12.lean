let digits := Finset.Icc 0 9
         let grid := Fin 2 → Fin 3 → ℕ
         let rows (g : grid) := ![g 0 0 * 100 + g 0 1 * 10 + g 0 2, g 1 0 * 100 + g 1 1 * 10 + g 1 2]
         let cols (g : grid) := ![g 0 0 * 10 + g 1 0, g 0 1 * 10 + g 1 1, g 0 2 * 10 + g 1 2]
         let valid (g : grid) := (∀ i j, g i j ∈ digits) ∧ (List.sum (rows g).data = 999) ∧ (List.sum (cols g).data = 99)
         Fintype.card { g : grid // valid g } = 4 := by
          sorry