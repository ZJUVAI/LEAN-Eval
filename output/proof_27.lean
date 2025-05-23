import Mathlib.Geometry.Manifold.Torus
import Mathlib.Geometry.Euclidean.Sphere

/--
Torus T is the surface produced by revolving a circle with radius 3 around an axis in the plane of the circle that is a distance 6 from the center of the circle (so like a donut). Let S be a sphere with a radius 11. When T rests on the inside of S, it is internally tangent to S along a circle with radius r_i, and when T rests on the outside of S, it is externally tangent to S along a circle with radius r_o. The difference r_i - r_o can be written as m/n, where m and n are relatively prime positive integers. Find m + n.
-/
theorem auto_theorem_27 :
  let R := 6  -- major radius
  let r := 3  -- minor radius
  let S_radius := 11
  let r_i := Real.sqrt (S_radius^2 - (R + r)^2)
  let r_o := Real.sqrt (S_radius^2 - (R - r)^2)
  let difference := r_i - r_o
  let simplified := 20 / 7
  difference = simplified := by
  unfold r_i r_o difference
  simp [R, r, S_radius]
  norm_num
  rw [← Real.sqrt_div]
  · norm_num
    rw [← Real.sqrt_div]
    · norm_num
      field_simp
      norm_num
    · norm_num
  · norm_num