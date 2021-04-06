export problem_types_handled, can_solve_type

"""
    problem_types_handled(solver)

List the problem types handled by the `solver`.
"""
problem_types_handled(::Type{<:AbstractSolver}) = []


"""
    can_solve_type(solver, type)

Check if the `solver` can solve problems of `type`.
Call `problem_types_handled` for a list of problem types that the `solver` can solve.
"""
function can_solve_type(::Type{S}, t :: Symbol) where S <: AbstractSolver
  return t ∈ problem_types_handled(S)
end

# Optimization
const opt_problem_type = [:unc, :bnd, :equ, :bndequ, :ineq, :genopt]

# I was hoping to add these to the metaprogramming code below, but failed
can_solve_unc(::Type{S}) where S <: AbstractSolver = can_solve_type(S, :unc)
can_solve_bnd(::Type{S}) where S <: AbstractSolver = can_solve_type(S, :bnd)
can_solve_equ(::Type{S}) where S <: AbstractSolver = can_solve_type(S, :equ)
can_solve_bndequ(::Type{S}) where S <: AbstractSolver = can_solve_type(S, :bndequ)
can_solve_ineq(::Type{S}) where S <: AbstractSolver = can_solve_type(S, :ineq)
can_solve_optgen(::Type{S}) where S <: AbstractSolver = can_solve_type(S, :optgen)

for ptype in opt_problem_type
  fname = Symbol("can_solve_$ptype")
  help = """
        $fname(solver)

  Check if the `solve` can solve optimization problems of type `$ptype`.
  Call `problem_types_handled` for a list of problem types that the `solver` can solve.
  """
  @eval begin
    @doc $help $fname
    export $fname
  end
end

# Linear System
const linear_problem_type = [:sym, :sqd, :gen, :lst_sqr, :lst_norm]

# Same...