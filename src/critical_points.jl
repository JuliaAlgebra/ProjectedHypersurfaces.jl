
export critical_points

export RoutingPointsResult,
    routing_points,
    complex_critical_points,
    result,
    monodromy_result

import HomotopyContinuation: MonodromyOptions, UniquePoints, EndgameTracker

@doc raw"""
    critical_points(r, S0, rhs0; kwargs...)

Find critical points of the routing function using monodromy and gradient flow.
Returns a [`RoutingPointsResult`](@ref).

Start solutions are first collected by applying Newton's method to ``∇r = rhs0`` from `newton_start_points` and gradient flow to ``∇r = 0`` from `flow_start_points`. The two lists are independent, and their default lengths do not depend on the number of projection variables ``k``.

# Keyword arguments
- `verbose = true`: print progress information.
- `flow_start_points`: list of real vectors of length `k` for gradient flow (default: 100 independent standard normal points). Pass `[]` to skip gradient flow.
- `newton_start_points`: list of vectors of length `k` for Newton's method (default: 10 independent standard normal points). Pass `[]` to skip Newton's method.
- `monodromy_at_zero = false`: run monodromy at the right-hand side ``0`` instead of a random one.
- `options`: `MonodromyOptions` for the monodromy solver.

# Example
```julia
flow_points = [collect(p) for p in Iterators.product(fill(-2:1:2, nvariables(r))...)]
critical_points(r; flow_start_points = flow_points, newton_start_points = [])
```
"""
function critical_points(
    r::RoutingFunction,
    S0::Union{AbstractVector{<:AbstractVector{<:Number}},Nothing} = nothing,
    rhs0::Union{AbstractVector{<:Number},Nothing} = nothing;
    verbose = true,
    flow_start_points = [randn(nvariables(r)) for _ in 1:100],
    newton_start_points = [randn(nvariables(r)) for _ in 1:10],
    monodromy_at_zero = false,
    options = MonodromyOptions(
        parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)),
        max_loops_no_progress = 15
    ),
)

    ∇r = RoutingGradient(r)

    # Step 1: Setup monodromy solver
    MS, H, S0, rhs0 = _setup_monodromy_solver(
        ∇r, S0, rhs0;
        monodromy_at_zero = monodromy_at_zero,
        options = options,
    )

    # Step 2: Expand start solutions via Newton's method and gradient flow
    S0, new_pts = _expand_start_solutions(
        ∇r, H, S0, rhs0;
        verbose = verbose,
        flow_start_points = flow_start_points,
        newton_start_points = newton_start_points,
        monodromy_at_zero = monodromy_at_zero,
    )

    # Step 3: Solve and trace to critical points
    return _solve_and_trace(
        ∇r, MS, H, S0, rhs0, new_pts;
        monodromy_at_zero = monodromy_at_zero,
    )
end

"""
    _setup_monodromy_solver(∇r, S0, rhs0; monodromy_at_zero, options)

Set up the monodromy solver and initial start pair.
Returns (MS, H, S0, rhs0) where MS is the MonodromySolver, H is the homotopy,
S0 are the start solutions, and rhs0 is the target parameters.
"""
function _setup_monodromy_solver(
    ∇r::RoutingGradient,
    S0::Union{AbstractVector{<:AbstractVector{<:Number}},Nothing} = nothing,
    rhs0::Union{AbstractVector{<:Number},Nothing} = nothing;
    monodromy_at_zero = false,
    options = MonodromyOptions(
        parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)),
        max_loops_no_progress = 15
    ),
)
    k = size(∇r, 2) # number of variables
    p1 = zeros(k)
    q1 = randn(k)
    H = RoutingPointsHomotopy(∇r, p1, q1)

    ### Use monodromy to the system ∇r = rhs0 where we view the right-hand side are the parameters of the system
    trackers = [EndgameTracker(H)]
    x₀ = zeros(ComplexF64, size(H, k))

    unique_points = UniquePoints(x₀, 1;)

    trace = zeros(ComplexF64, length(x₀) + 1, 3)
    P = Vector{ComplexF64}
    MS = HomotopyContinuation.MonodromySolver(
        trackers,
        HomotopyContinuation.MonodromyLoop{P}[],
        unique_points,
        ReentrantLock(),
        nothing,
        options,
        HomotopyContinuation.MonodromyStatistics(),
        trace,
        ReentrantLock(),
    )

    #### set up start pair
    if !monodromy_at_zero
        if isnothing(rhs0) || isnothing(S0)
            s0 = randn(ComplexF64, k)
            rhs0 = evaluate(∇r, s0)
            S0 = [s0]
        end
    else
        rhs0 = zeros(k)
        if isnothing(S0)
            S0 = Vector{ComplexF64}[]
        end
    end

    return MS, H, S0, rhs0
end

"""
    _expand_start_solutions(∇r, H, S0, rhs0; verbose, flow_start_points, newton_start_points, monodromy_at_zero)

Expand S0 using Newton's method on ∇r=rhs0 and gradient flow followed by tracing from ∇r=0 to ∇r=rhs0.
The start-point lists are required; their defaults are chosen by `critical_points`.
Returns (S0, new_pts) where S0 is the expanded set of start solutions and new_pts are the points found at ∇r=0.
"""
function _expand_start_solutions(
    ∇r::RoutingGradient,
    H::RoutingPointsHomotopy,
    S0::AbstractVector{<:AbstractVector{<:Number}},
    rhs0::AbstractVector{<:Number};
    verbose = true,
    flow_start_points,
    newton_start_points,
    monodromy_at_zero = false,
)
    k = size(∇r, 2)
    new_pts = Vector{ComplexF64}[]
    for points in (flow_start_points, newton_start_points)
        all(p -> length(p) == k, points) || throw(DimensionMismatch("Start points must have length $k."))
    end
    if isempty(flow_start_points) && isempty(newton_start_points)
        return S0, new_pts
    end

    # First we try to find start solutions via blindly applying Newton's method to ∇r=rhs0.
    verbose && println("Expanding start solutions via Newton's method...")
    newton_success_count = 0
    start_pt = zeros(ComplexF64, k)
    ProgressMeter.@showprogress for start_point in newton_start_points
            start_pt .= start_point # reuse the buffer without modifying the supplied point
            # Newton's method on each initial guess
            try
                pt = newton(∇r, start_pt, rhs0; max_iters = 200) |> solution
                if norm(evaluate(∇r, pt, rhs0)) < 1e-10
                    newton_success_count += 1
                    push!(new_pts, pt)
                end
            catch e
                continue
            end

    end

    num_newton_pts = 0
    if length(new_pts) > 0
        new_pts = HC.unique_points(new_pts)
        num_newton_pts += length(new_pts)
    end

    # since the points obtained via Newton's method are already solutions to ∇r=rhs0, we can directly add them to S0 if monodromy_at_zero is false
    if !monodromy_at_zero && !isempty(new_pts)
        S0 = HC.unique_points([S0; new_pts])
        # to avoid redundant work later, we remove the points found via Newton's method from new_pts so that we aren't tracing them using monodromy
        empty!(new_pts)
    end

    verbose && !isempty(newton_start_points) && println("Successful Newton's method attempts: $(newton_success_count) out of $(length(newton_start_points)) ($(round(newton_success_count / length(newton_start_points) * 100, digits=2))%)")
    verbose && println("Found $num_newton_pts solutions to ∇r(z)=rhs0.")
    
    # Now we try gradient flow
    g(x, param, t) = real(evaluate(∇r, x))
    tspan = (0.0, 1e4)
    
    verbose && println("Expanding the set of start solutions via gradient flow...")

    gradient_success_count = 0
    start_pt = zeros(k)
    ProgressMeter.@showprogress for start_point in flow_start_points
        try
            start_pt .= start_point # reuse the buffer without modifying the supplied point
            prob = SciMLBase.ODEProblem(g, start_pt, tspan)
            sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
            convergence_point = last(sol.u)
            refinement = newton(∇r, convergence_point)
            HC.is_success(refinement) || continue # skip points for which Newton didn't converge
            improved_point = solution(refinement)
            push!(new_pts, improved_point)
            gradient_success_count += 1                
        catch e
            continue
        end
    end
    if length(new_pts) > 0
        new_pts = HC.unique_points(new_pts)
    end
    verbose && !isempty(flow_start_points) && println("Successful gradient flow attempts: $(gradient_success_count) out of $(length(flow_start_points)) ($(round(gradient_success_count / length(flow_start_points) * 100, digits=2))%)")
    verbose && println("Found $(length(new_pts)) routing points via gradient flow.")

    isempty(new_pts) && return S0, new_pts

    if !monodromy_at_zero
        start_parameters!(H, zeros(ComplexF64, length(rhs0)))
        target_parameters!(H, rhs0)
        S0_new_sols = HC.solve(H, new_pts) |> solutions
        number_of_old_sols = length(S0)
        S0 = HC.unique_points([S0; S0_new_sols])
        verbose && println(
            "Traced to $(length(S0)-number_of_old_sols) additional start solutions for the monodromy.",
        )
    else
        S0 = [S0; new_pts]
    end

    return S0, new_pts
end

"""
    _solve_and_trace(∇r, MS, H, S0, rhs0, new_pts; monodromy_at_zero)

Perform monodromy solving and trace solutions to ∇r=0.
Returns a [`RoutingPointsResult`](@ref).
"""
function _solve_and_trace(
    ∇r::RoutingGradient,
    MS::HomotopyContinuation.MonodromySolver,
    H::RoutingPointsHomotopy,
    S0::AbstractVector{<:AbstractVector{<:Number}},
    rhs0::AbstractVector{<:Number},
    new_pts::AbstractVector{<:AbstractVector{<:Number}};
    monodromy_at_zero = false,
)
    ### Monodromy
    mon_result = monodromy_solve(MS, S0, rhs0, rand(UInt32))

    ### Trace to ∇r=0
    if !monodromy_at_zero
        intermediate_rhs = randn(ComplexF64, length(rhs0))
        start_parameters!(H, rhs0)
        target_parameters!(H, intermediate_rhs)
        result_intermediate = HomotopyContinuation.solve(H, solutions(mon_result))
        start_parameters!(H, intermediate_rhs)
        target_parameters!(H, zeros(ComplexF64, length(rhs0)))
        result = HomotopyContinuation.solve(H, result_intermediate)
        routing_points = real_solutions(result)

        # Make sure none of the routing points found via gradient flow are lost
        if !isempty(new_pts)
            routing_points = HC.unique_points([routing_points; real.(new_pts)])
        end
        return RoutingPointsResult(routing_points, result, mon_result, ∇r.r)
    else
        routing_points = real_solutions(results(mon_result))
        return RoutingPointsResult(routing_points, mon_result, mon_result, ∇r.r)
    end
end



import HomotopyContinuation:
    solutions, real_solutions, nsolutions, results, nresults

@doc raw"""
    RoutingPointsResult

Result returned by [`critical_points`](@ref). Use [`routing_points`](@ref) for the
real routing points, [`result`](@ref) for the final result from tracking to to `∇r = 0`,
[`monodromy_result`](@ref) for the underlying monodromy computation.
"""
struct RoutingPointsResult{P,T,M}
    routing_points::P
    result::T
    monodromy_result::M
    r::RoutingFunction
end

@doc raw"""
    routing_points(result::RoutingPointsResult)

Return the real critical points used for routing.
"""
routing_points(R::RoutingPointsResult) = R.routing_points

@doc raw"""
    result(result::RoutingPointsResult)

Return the final HomotopyContinuation result obtained by tracing to ∇r = 0.
"""
result(R::RoutingPointsResult) = R.result

@doc raw"""
    complex_critical_points(result::RoutingPointsResult)

Return the complex critical points of the routing function (the real points are routing points).
"""
complex_critical_points(R::RoutingPointsResult) = solutions(R.result)

@doc raw"""
    monodromy_result(result::RoutingPointsResult)

Return the underlying monodromy computation result.
"""
monodromy_result(R::RoutingPointsResult) = R.monodromy_result


function Base.show(io::IO, R::RoutingPointsResult)
    npts = length(routing_points(R))
    println(io, "Routing points result with $npts routing point(s)")
end
