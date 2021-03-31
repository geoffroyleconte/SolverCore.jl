module SolverCore

# stdlib
using Logging, Printf

# our packages
using NLPModels

include("solver.jl")
include("grid-search-tuning.jl")
include("logger.jl")
include("stats.jl")

end
