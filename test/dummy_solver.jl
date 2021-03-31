mutable struct DummySolver{T} <: AbstractSolver{T}
  initialized :: Bool
  params :: Dict
  x :: Vector{T}
  xt :: Vector{T}
  gx :: Vector{T}
  dual :: Vector{T}
  y :: Vector{T}
  cx :: Vector{T}
  ct :: Vector{T}
end

function SolverCore.parameters(::Type{DummySolver{T}}) where T
  (
    α = (default=T(1e-2), type=:log, min=√√eps(T), max=one(T) / 2),
    δ = (default=√eps(T), type=:log, min=√eps(T), max=√√√eps(T)),
    reboot_y = (default=false, type=:bool)
  )
end

function DummySolver(::Type{T}, meta :: AbstractNLPModelMeta; kwargs...) where T
  nvar, ncon = meta.nvar, meta.ncon
  params = parameters(DummySolver{T})
  solver = DummySolver{T}(true,
    Dict(k => v[:default] for (k,v) in pairs(params)),
    zeros(T, nvar),
    zeros(T, nvar),
    zeros(T, nvar),
    zeros(T, nvar),
    zeros(T, ncon),
    zeros(T, ncon),
    zeros(T, ncon),
  )
  for (k,v) in kwargs
    solver.params[k] = v
  end
  solver
end

function DummySolver(::Type{T}, ::Val{:nosolve}, nlp :: AbstractNLPModel) where T
  solver = DummySolver(T, nlp.meta)
  return solver
end

function DummySolver(::Type{T}, nlp :: AbstractNLPModel) where T
  solver = DummySolver(T, nlp.meta)
  output = solve!(solver, nlp)
  return output, solver
end

DummySolver(meta :: AbstractNLPModelMeta) = DummySolver(Float64, meta :: AbstractNLPModelMeta)
DummySolver(::Val{:nosolve}, nlp :: AbstractNLPModel) = DummySolver(Float64, Val(:nosolve), nlp :: AbstractNLPModel)
DummySolver(nlp :: AbstractNLPModel) = DummySolver(Float64, nlp :: AbstractNLPModel)


function SolverCore.solve!(solver::DummySolver{T}, nlp :: AbstractNLPModel;
  x :: AbstractVector{T} = T.(nlp.meta.x0),
  atol :: Real = sqrt(eps(T)),
  rtol :: Real = sqrt(eps(T)),
  max_eval :: Int = 1000,
  max_time :: Float64 = 30.0,
  kwargs...
) where T
  solver.initialized || error("Solver not initialized.")
  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
  for (k,v) in kwargs
    solver.params[k] = v
  end
  α = solver.params[:α]
  δ = solver.params[:δ]
  reboot_y = solver.params[:reboot_y]

  start_time = time()
  elapsed_time = 0.0
  solver.x .= x # Copy values
  x = solver.x  # Change reference

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


    fx = ft
    grad!(nlp, x, gx)
    Jx = ncon > 0 ? jac(nlp, x) : zeros(T, 0, nvar)
    if reboot_y
      y .= -Jx' \ gx
    else
      y .+= t * Δy
    end
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
