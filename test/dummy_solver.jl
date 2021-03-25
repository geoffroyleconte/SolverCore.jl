mutable struct DummySolver <: AbstractSolver
  initialized :: Bool
  x :: Vector
  xt :: Vector
  gx :: Vector
  dual :: Vector
  y :: Vector
  cx :: Vector
  ct :: Vector
end

function DummySolver(nvar :: Integer, ncon :: Integer = 0)
  DummySolver(true, zeros(nvar), zeros(nvar), zeros(nvar), zeros(nvar), zeros(ncon), zeros(ncon), zeros(ncon))
end

function DummySolver(nlp :: AbstractNLPModel)
  solver = DummySolver(nlp.meta.nvar, nlp.meta.ncon)
  output = solve!(solver, nlp)
  return output, solver
end

function solve!(solver::DummySolver, nlp :: AbstractNLPModel;
  x :: AbstractVector = nlp.meta.x0,
  atol :: Real = sqrt(eps(eltype(x))),
  rtol :: Real = sqrt(eps(eltype(x))),
  max_eval :: Int = 1000,
  max_time :: Float64 = 30.0,
  α :: Float64 = 1e-2,
  δ :: Float64 = 1e-8,
)
  solver.initialized || error("Solver not initialized.")
  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon

  start_time = time()
  elapsed_time = 0.0
  solver.x .= x # Copy values
  x = solver.x  # Change reference

  T = eltype(x)

  cx = solver.cx .= ncon > 0 ? cons(nlp, x) : zeros(T, 0)
  ct = solver.ct = zeros(T, ncon)
  grad!(nlp, x, solver.gx)
  gx = solver.gx
  Jx = ncon > 0 ? jac(nlp, x) : zeros(T, 0, nvar)
  y = solver.y .= -Jx' \ gx
  Hxy = ncon > 0 ? hess(nlp, x, y) : hess(nlp, x)

  dual = solver.dual .= gx .+ Jx' * y

  iter = 0

  ϵd = atol + rtol * norm(dual)
  ϵp = atol

  ϕ(fx, cx, y) = fx + norm(cx)^2 / 2δ + dot(y, cx)
  fx = obj(nlp, x)
  @info log_header([:iter, :f, :c, :dual, :t], [Int, T, T, Float64])
  @info log_row(Any[iter, fx, norm(cx), norm(dual), elapsed_time])
  solved = norm(dual) < ϵd && norm(cx) < ϵp
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time

  while !(solved || tired)
    Hxy = ncon > 0 ? hess(nlp, x, y) : hess(nlp, x)
    W = Symmetric([Hxy  zeros(T, nvar, ncon); Jx  -δ * I], :L)
    Δxy = -W \ [dual; cx]
    Δx = Δxy[1:nvar]
    Δy = Δxy[nvar+1:end]

    AΔx = Jx * Δx
    ϕx = ϕ(fx, cx, y)
    xt = solver.xt .= x + Δx
    if ncon > 0
      cons!(nlp, xt, ct)
    end
    ft = obj(nlp, xt)
    slope = -dot(Δx, Hxy, Δx) - norm(AΔx)^2 / δ
    t = 1.0
    while !(ϕ(ft, ct, y) ≤ ϕx + α * t * slope)
      t /= 2
      xt .= x + t * Δx
      if ncon > 0
        cons!(nlp, xt, ct)
      end
      ft = obj(nlp, xt)
    end

    x .= xt
    y .+= t * Δy

    fx = ft
    grad!(nlp, x, gx)
    Jx = ncon > 0 ? jac(nlp, x) : zeros(T, 0, nvar)
    cx .= ct
    dual .= gx .+ Jx' * y
    elapsed_time = time() - start_time
    solved = norm(dual) < ϵd && norm(cx) < ϵp
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time

    iter += 1
    @info log_row(Any[iter, fx, norm(cx), norm(dual), elapsed_time])
  end

  status = if solved
    :first_order
  elseif elapsed_time > max_time
    :max_time
  else
    :max_eval
  end

  return GenericExecutionStats(
    status,
    nlp,
    objective=fx,
    dual_feas=norm(dual),
    primal_feas=norm(cx),
    multipliers=y,
    multipliers_L=zeros(T, nvar),
    multipliers_U=zeros(T, nvar),
    elapsed_time=elapsed_time,
    solution=x,
    iter=iter
  )
end

parameters(::DummySolver) = NamedTuple(α = 1e-2, δ = 1e-8)

parameter_problem(solver::DummySolver, problems, cost) = ADNLPModel(
  x -> begin
    for nlp in problems
      try
        output = with_logger(NullLogger()) do
          output, solver = DummySolver()
        end
    end
  end
)