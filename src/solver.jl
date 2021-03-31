export AbstractSolver, solve!, parameters

"""
    AbstractSolver

Base type for JSO-compliant solvers.
"""
abstract type AbstractSolver{T} end

function Base.show(io :: IO, solver :: AbstractSolver)
    show(io, "Solver $(typeof(solver))")
end

"""
    output = solve!(solver, problem)

Solve `problem` with `solver`.
This modifies internal
"""
function solve!(::AbstractSolver, ::AbstractNLPModel) end

"""
    named_tuple = parameters(solver)
    named_tuple = parameters(SolverType)
    named_tuple = parameters(SolverType{T})

Return the parameters of a `solver`, or of the type `SolverType`.
You can specify the type `T` of the `SolverType`.
The returned structure is a nested NamedTuple.
Each key of `named_tuple` is the name of a parameter, and its value is a NamedTuple containing
- `default`: The default value of the parameter.
- `type`: The type of the parameter, which can any of:
  - `:real`: A continuous value within a range
  - `:log`: A positive continuous value that should be explored logarithmically (like 10⁻², 10⁻¹, 1, 10).
  - `:int`: Integer value.
  - `:bool`: Boolean value.
- `min`: Minimum value (may not be included for some parameter types).
- `max`: Maximum value.
"""
function parameters(::Type{AbstractSolver{T}}) where T end

parameters(::Type{S}) where S <: AbstractSolver = parameters(S{Float64})
parameters(solver :: AbstractSolver) = parameters(typeof(solver))