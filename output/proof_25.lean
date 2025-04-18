import Mathlib

/--
Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.
-/
theorem auto_theorem_25 :
  900 = 195 + 367 + 562 + (900 - 195 - 367 - 562 + 2 * 437 + 3 * 234) - 437 - 234 + (900 - (195 + 367 + 562) + (437 + 234) - (2 * 437 + 3 * 234)) := by
  have h_total : 900 = 900 := by rfl
  have h_ring := 195
  have h_golf := 367
  have h_spade := 562
  have h_candy := 900
  have h_two := 437
  have h_three := 234
  have h_all := 900 - (h_ring + h_golf + h_spade) + (h_two + h_three) - (2 * h_two + 3 * h_three)
  rw [h_ring, h_golf, h_spade, h_two, h_three]
  ring