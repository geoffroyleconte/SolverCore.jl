export AbstractSolver, solve!, parameters

"""
    AbstractSolver

Base type for JSO-compliant solvers.
"""
abstract type AbstractSolver{T} end

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
  - `:log`: A continuous value that should be explorer logarithmically around it's lower value (usually 0) to avoid the bound itself.
  - `:int`: Integer value.
  - `:bool`: Boolean value.
- `min`: Minimum value (may not be included for some parameter types).
- `max`: Maximum value.
"""
function parameters(::Type{AbstractSolver{T}}) where T end

parameters(::Type{S}) where S <: AbstractSolver = parameters(S{Float64})
parameters(solver :: AbstractSolver) = parameters(typeof(solver))

"""
    nlp = parameter_problem(solver)

Return the problem associated with the tuning of the parameters of `solver`.
"""
function parameter_problem(::AbstractSolver) end

# parameter_problem(
#   solver::DummySolver,
#   problems,
#   cost,
#   cost_bad
# ) = ADNLPModel(
#   x -> begin
#     total_cost = 0.0
#     for nlp in problems
#       try
#         output = with_logger(NullLogger()) do
#           output, _ = DummySolver(nlp)
#         end
#         total_cost += cost(output)
#       catch
#         total_cost +=
#       end
#     end
#   end
# )