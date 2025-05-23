import Mathlib.Geometry.Euclidean.Sphere.Basic
import Mathlib.Geometry.Euclidean.Tetrahedron

open Real

noncomputable section

def A : EuclideanSpace ℝ (Fin 3) := ![0, 0, 0]
def B : EuclideanSpace ℝ (Fin 3) := ![5, 0, 0]
def C : EuclideanSpace ℝ (Fin 3) := ![4, 8, 0]
def D : EuclideanSpace ℝ (Fin 3) := ![-4, 4, 7]

def tetrahedron : Affine.Tetrahedron ℝ (EuclideanSpace ℝ (Fin 3)) :=
  Affine.Tetrahedron.mk A B C D

theorem AB_eq : dist A B = sqrt 41 := by
  simp [A, B, dist_eq_norm, norm_eq_sqrt_inner, inner, Fin.sum_univ_succ]
  norm_num

theorem AC_eq : dist A C = sqrt 80 := by
  simp [A, C, dist_eq_norm, norm_eq_sqrt_inner, inner, Fin.sum_univ_succ]
  norm_num

theorem AD_eq : dist A D = sqrt 89 := by
  simp [A, D, dist_eq_norm, norm_eq_sqrt_inner, inner, Fin.sum_univ_succ]
  norm_num

theorem BC_eq : dist B C = sqrt 89 := by
  simp [B, C, dist_eq_norm, norm_eq_sqrt_inner, inner, Fin.sum_univ_succ]
  norm_num

theorem BD_eq : dist B D = sqrt 80 := by
  simp [B, D, dist_eq_norm, norm_eq_sqrt_inner, inner, Fin.sum_univ_succ]
  norm_num

theorem CD_eq : dist C D = sqrt 41 := by
  simp [C, D, dist_eq_norm, norm_eq_sqrt_inner, inner, Fin.sum_univ_succ]
  norm_num

def I : EuclideanSpace ℝ (Fin 3) := ![-1, 2, 2]

theorem I_inside : Affine.Simplex.interiorMember I tetrahedron := by
  rw [Affine.Simplex.interiorMember_iff_barycentric_coords_pos]
  have : Affine.Tetrahedron.barycentricCoords tetrahedron I = ![1/4, 1/4, 1/4, 1/4] := by
    apply Affine.Tetrahedron.barycentricCoords_eq_of_point_eq_centroid
    simp [I, A, B, C, D, Affine.Tetrahedron.centroid]
    fin_cases i <;> norm_num
  rw [this]
  fin_cases i <;> norm_num

theorem distance_to_ABC : Affine.Tetrahedron.planeDistance I tetrahedron 0 = sqrt 5 / 3 := by
  rw [Affine.Tetrahedron.planeDistance_eq]
  · simp [A, B, C, I, Affine.Tetrahedron.facePlane, Affine.Subspace.direction,
          Affine.Simplex.orthogonalProjection_eq_orthogonalProjection_of_affineSpan_eq,
          orthogonalProjection, inner, Fin.sum_univ_succ]
    norm_num
  · simp [Affine.Tetrahedron.facePlane_mem]

theorem distance_to_ABD : Affine.Tetrahedron.planeDistance I tetrahedron 1 = sqrt 5 / 3 := by
  rw [Affine.Tetrahedron.planeDistance_eq]
  · simp [A, B, D, I, Affine.Tetrahedron.facePlane, Affine.Subspace.direction,
          Affine.Simplex.orthogonalProjection_eq_orthogonalProjection_of_affineSpan_eq,
          orthogonalProjection, inner, Fin.sum_univ_succ]
    norm_num
  · simp [Affine.Tetrahedron.facePlane_mem]

theorem distance_to_ACD : Affine.Tetrahedron.planeDistance I tetrahedron 2 = sqrt 5 / 3 := by
  rw [Affine.Tetrahedron.planeDistance_eq]
  · simp [A, C, D, I, Affine.Tetrahedron.facePlane, Affine.Subspace.direction,
          Affine.Simplex.orthogonalProjection_eq_orthogonalProjection_of_affineSpan_eq,
          orthogonalProjection, inner, Fin.sum_univ_succ]
    norm_num
  · simp [Affine.Tetrahedron.facePlane_mem]

theorem distance_to_BCD : Affine.Tetrahedron.planeDistance I tetrahedron 3 = sqrt 5 / 3 := by
  rw [Affine.Tetrahedron.planeDistance_eq]
  · simp [B, C, D, I, Affine.Tetrahedron.facePlane, Affine.Subspace.direction,
          Affine.Simplex.orthogonalProjection_eq_orthogonalProjection_of_affineSpan_eq,
          orthogonalProjection, inner, Fin.sum_univ_succ]
    norm_num
  · simp [Affine.Tetrahedron.facePlane_mem]

theorem auto_theorem_28 :
  ∃ m n p : ℕ, Nat.Coprime m p ∧ Squarefree n ∧
  (∃ I : EuclideanSpace ℝ (Fin 3), Affine.Simplex.interiorMember I tetrahedron ∧
    ∀ i : Fin 4, Affine.Tetrahedron.planeDistance I tetrahedron i = m * sqrt n / p) ∧
  m + n + p = 12 := by
  use 1, 5, 3
  constructor
  · norm_num
  constructor
  · exact Nat.squarefree_iff_nodup_factors.mpr (by decide)
  constructor
  · use I
    constructor
    · exact I_inside
    · intro i
      fin_cases i
      · exact distance_to_ABC
      · exact distance_to_ABD
      · exact distance_to_ACD
      · exact distance_to_BCD
  · norm_num