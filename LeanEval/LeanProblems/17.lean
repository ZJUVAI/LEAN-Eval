import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Pointwise

theorem singleton_mul {α : Type*} [Monoid α] (a b : α) : ({a} * {b} : Set α) = {a * b}