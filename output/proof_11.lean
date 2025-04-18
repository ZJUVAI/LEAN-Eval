import Mathlib
        /--
        You are a formal proof assistant. Given a math problem, translate it into Lean 4 using Mathlib.
        Prove the statement in a human-readable, idiomatic way using tactics.
        Make sure the code includes both the problem statement (as a docstring or comment) and its full formal proof.

        Problem: Find the largest possible real part of \[(75+117i)z + \frac{96+144i}{z}\] where $z$ is a complex number with $|z|=4$.
        -/
        theorem auto_theorem_11 :
         let f : ℂ → ℂ := fun z ↦ (75 + 117 * I) * z + (96 + 144 * I) / z
         let S : Set ℂ := { z : ℂ | Complex.abs z = 4 }
         IsGreatest (Set.image (fun z ↦ (f z).re) S) 876 := by
          intro f S
          simp only [Set.mem_setOf_eq]
          constructor
          · use 4
            simp [f, S]
            ring_nf
            simp [Complex.abs_ofReal]
            norm_num
          · intro x hx
            obtain ⟨z, hz, rfl⟩ := hx
            have hz' : Complex.abs z = 4 := hz
            let a := (75 + 117 * I) * z
            let b := (96 + 144 * I) / z
            have hab : (f z).re = a.re + b.re := by
              simp [f, Complex.add_re]
            rw [hab]
            have ha : a.re = 75 * z.re - 117 * z.im := by
              simp [Complex.mul_re]
              ring
            have hb : b.re = (96 * z.re + 144 * z.im) / (z.re^2 + z.im^2) := by
              simp [Complex.div_re]
              ring
            rw [ha, hb]
            have hznorm : z.re^2 + z.im^2 = 16 := by
              rw [← Complex.normSq_eq_abs, hz']
              norm_num
            rw [hznorm]
            simp
            have := abs_le'.1 (abs_im_le_abs z)
            have h₁ : z.re ≤ 4 := by
              apply le_trans (abs_le'.1 (abs_re_le_abs z)).2
              rw [hz']
              norm_num
            have h₂ : z.im ≤ 4 := by
              apply le_trans (abs_le'.1 (abs_im_le_abs z)).2
              rw [hz']
              norm_num
            have h₃ : -4 ≤ z.re := by
              apply (abs_le'.1 (abs_re_le_abs z)).1
              rw [hz']
              norm_num
            have h₄ : -4 ≤ z.im := by
              apply (abs_le'.1 (abs_im_le_abs z)).1
              rw [hz']
              norm_num
            have : 75 * z.re + 6 * z.im ≤ 75 * 4 + 6 * 4 := by
              gcongr
              exact h₁
              exact h₂
            linarith