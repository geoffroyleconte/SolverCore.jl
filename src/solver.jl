export AbstractSolver

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

Return the parameters of `solver`.
"""
function parameters(::AbstractSolver) end

"""
    nlp = parameter_problem(solver)

Return the problem associated with the tuning of the parameters of `solver`.
"""
function parameter_problem(::AbstractSolver) end