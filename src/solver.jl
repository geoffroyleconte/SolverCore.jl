export AbstractSolver, solve!

"""
    AbstractSolver

Base type for JSO-compliant solvers.
A solver must have three members:
- `initialized :: Bool`, indicating whether the solver was initialized
- `params :: Dict`, a dictionary of solvers
- `workspace`, a named tuple with arrays used by the solver.
"""
abstract type AbstractSolver{T} end

function Base.show(io :: IO, solver :: AbstractSolver)
  println(io, "Solver $(typeof(solver))")
end

"""
    output = solve!(solver, problem)

Solve `problem` with `solver`.
"""
function solve! end
