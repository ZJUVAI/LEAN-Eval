import Mathlib
        /--
        You are a formal proof assistant. Given a math problem, translate it into Lean 4 using Mathlib.
        Prove the statement in a human-readable, idiomatic way using tactics.
        Make sure the code includes both the problem statement (as a docstring or comment) and its full formal proof.

        Problem: Consider the paths of length $16$ that follow the lines from the lower left corner to the upper right corner on an $8\times 8$ grid. Find the number of such paths that change direction exactly four times, as in the examples shown below.
        -/
        theorem auto_theorem_15 :
         -- The number of paths is 364.
         364 = 2 * (Nat.choose 7 1 * Nat.choose 7 2 + Nat.choose 7 3 * Nat.choose 7 2)
        := by
        simp [Nat.choose]
        norm_num