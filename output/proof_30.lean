let ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)
         let P : ℂ := ∏ k in Finset.range 13, (2 - 2 * ω ^ k + ω ^ (2 * k))
         (P.re.round : ℤ) % 1000 = 273 := by
          sorry