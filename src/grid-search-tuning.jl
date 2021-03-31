export grid_search_tune

"""
    solver, results = grid_search_tune(SolverType, problems; kwargs...)

Simple tuning of solver `SolverType` by grid search, on `problems`, which should be iterable.
The following keyword arguments are available:
- `success`: A function to be applied on a solver output that returns whether the problem has terminated succesfully. Defaults to `o -> o.status == :first_order`.
- `costs`: A vector of cost functions and penalties. Each element is a tuple of two elements. The first is a function to be applied to the output of the solver, and the second is the cost when the solver fails (see `success` above) or throws an error. Defaults to
```
[
  (o -> o.elapsed_time, 100.0),
  (o -> o.counters.neval_obj + o.counters.neval_cons, 1000),
  (o -> !success(o), 1),
]
```
which represent the total elapsed_time (with a penalty of 100.0 for failures); the number of objective and constraints functions evaluations (with a penalty of 1000 for failures); and the number of failures.
- `grid_length`: The number of points in the ranges of the grid for continuous points.
- `solver_kwargs`: Arguments to be passed to the solver. Note: use this to set the stopping parameters, but not the other parameters being optimize.
- Any parameters accepted by the `Solver`: a range to be used instead of the default range.

The default ranges are based on the parameters types, and are as follows:
- `:real`: linear range from `:min` to `:max` with `grid_length` points.
- `:log`: logarithmic range from `:min` to `:max` with `grid_length` points. Computed by exp of linear range of `log(:min)` to `log(:max)`.
- `:bool`: either `false` or `true`.
- `:int`: integer range from `:min` to `:max`.
"""
function grid_search_tune(
  ::Type{Solver},
  problems;
  success = o -> o.status == :first_order,
  costs = [
    (o -> o.elapsed_time, 100.0),
    (o -> o.counters.neval_obj + o.counters.neval_cons, 1000),
    (o -> !success(o), 1),
  ],
  grid_length = 10,
  solver_kwargs = Dict(),
  kwargs...
) where Solver <: AbstractSolver

  solver_params = parameters(Solver)
  params = Dict()
  for (k,v) in pairs(solver_params)
    if v[:type] == :real
      params[k] = LinRange(v[:min], v[:max], grid_length)
    elseif v[:type] == :log
      params[k] = exp.(LinRange(log(v[:min]), log(v[:max]), grid_length))
    elseif v[:type] == :bool
      params[k] = (false, true)
    elseif v[:type] == :int
      params[k] = v[:min]:v[:max]
    end
  end
  for (k,v) in kwargs
    params[k] = v
  end

  # Precompiling
  nlp = first(problems)
  try
    solver = Solver(Val(:nosolve), nlp)
    output = with_logger(NullLogger()) do
      solve!(solver, nlp)
    end
  finally
    finalize(nlp)
  end

  cost(θ) = begin
    total_cost = [zero(x[2]) for x in costs]
    for nlp in problems
      reset!(nlp)
      try
        solver = Solver(Val(:nosolve), nlp)
        output = with_logger(NullLogger()) do
          solve!(solver, nlp; (k => θi for (k,θi) in zip(keys(solver_params), θ))...)
        end
        for (i, c) in enumerate(costs)
          if success(output)
            total_cost[i] += (c[1])(output)
          else
            total_cost[i] += c[2]
          end
        end
      catch ex
        for (i, c) in enumerate(costs)
          total_cost[i] += c[2]
        end
        @show ex
      finally
        finalize(nlp)
      end
    end
    total_cost
  end

  [θ => cost(θ) for θ in Iterators.product(values(params)...)]
end