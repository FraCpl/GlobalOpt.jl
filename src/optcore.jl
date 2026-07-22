abstract type AbstractOptimizer end

mutable struct Population{F,B}
    Npop::Int                       # Number of members of the population
    fun::F                          # Cost function handler
    applyBounds!::B                 # lb, ub bounds function

    x::Vector{Vector{Float64}}      # Population members
    lb::Vector{Float64}
    ub::Vector{Float64}
    iBest::Int                      # Index of the fittest member of population in x
    cost::Vector{Float64}           # Cost of x
    constr::Vector{Float64}         # Constraint violation of x
    fit::Vector{Float64}            # Fitness of x

    # Allocations
    xTmp::Vector{Float64}
    isFeasible::BitVector
    idx::Vector{Int}
end

function Population(Npop, f, x0, lb, ub, eqTol, bounds)
    Nx = length(lb)

    # Choose bounds function
    applyBounds! = if bounds === :clip
        (x -> clipBounds!(x, lb, ub))
    elseif bounds === :rand
        (x -> randBounds!(x, lb, ub))
    else
        (x -> nothing)
    end

    pop = Population(
        Npop,
        x -> evalFunction(f, x, eqTol),   # fun
        applyBounds!,                     # bounds
        [zeros(Nx) for _ in 1:Npop],      # x
        lb,
        ub,
        1,                                # iBest
        zeros(Npop),                      # cost
        zeros(Npop),                      # constr
        zeros(Npop),                      # fit
        zeros(Nx),                        # xTmp
        falses(Npop),                     # isFeasible
        collect(1:Npop),
    )

    # Random initialization of all members of the population
    @inbounds for i in eachindex(pop.x), j in 1:Nx
        pop.x[i][j] = lb[j] + (ub[j] - lb[j])*rand()
    end

    # Add initial guesses as provided by the user to the population
    # TODO: if the user provides more than Npop, then select the Npop fittest out of x0
    if !isnan(x0[1][1])
        for i in 1:min(lastindex(x0), Npop)
            pop.x[i] = copy(x0[i])
            pop.applyBounds!(pop.x[i])
        end

        # Shuffle order of population
        shuffle!(pop.x)
    end

    # Evaludate fitness of initial population
    @inbounds for i in eachindex(pop.x)
        pop.cost[i], pop.constr[i] = pop.fun(pop.x[i])
    end
    evalFitness!(pop)

    return pop
end

@inline function clipBounds!(x, lb, ub)
    @inbounds for i in eachindex(x)
        x[i] = clamp(x[i], lb[i], ub[i])
    end
    return nothing
end

@inline function randBounds!(x, lb, ub)
    @inbounds for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            x[i] = lb[i] + (ub[i] - lb[i])*rand()
        end
    end
    return nothing
end

@inline function evalFunction(f, x, eqTol)
    cost, g, h = f(x)
    if isnan(cost)
        cost = Inf
    end
    constr = 0.0
    @inbounds for gj in g
        if gj > 0.0
            constr += gj
        end
    end
    @inbounds for hj in h
        if hj < -eqTol || hj > eqTol
            constr += abs(hj)
        end
    end
    return cost, constr
end

function evalFitness!(pop::Population)
    # Constraints accounting http://repository.ias.ac.in/9407/1/310.pdf
    # Deb, An Efficient Constraint Handling Method for Genetic Algorithms

    # Determine feasibility of each element in the population
    maxFeasibleCost = -Inf
    anyFeasible = false
    @inbounds for i in eachindex(pop.isFeasible)
        isFeas = pop.constr[i] == 0.0
        pop.isFeasible[i] = isFeas
        pop.fit[i] = pop.cost[i]
        if isFeas
            anyFeasible = true
            if pop.cost[i] > maxFeasibleCost
                maxFeasibleCost = pop.cost[i]
            end
        end
    end

    if anyFeasible
        # At least one solution is feasible
        @inbounds for j in eachindex(pop.fit)
            if !pop.isFeasible[j]
                pop.fit[j] = maxFeasibleCost + pop.constr[j]
            end
        end
    else
        # All solutions are unfeasible
        @inbounds for j in eachindex(pop.fit)
            pop.fit[j] += pop.constr[j]
        end
    end

    # Identify best element in the population
    pop.iBest = argmin(pop.fit)
    return nothing
end

@inline function compare1vs1!(i::Int, pop::Population, xNew::Vector{Float64}, costNew::Float64, constrNew::Float64)
    costOld = pop.cost[i]
    constrOld = pop.constr[i]

    # Deb's rules in a single if
    if (constrNew == 0.0 && constrOld > 0.0) ||                     # new feasible, old infeasible
        (constrNew > 0.0 && constrNew ≤ constrOld) ||                # both infeasible, new less violation
        (constrNew == 0.0 && constrOld == 0.0 && costNew ≤ costOld)  # both feasible, cheaper
        pop.x[i] .= xNew
        pop.cost[i] = costNew
        pop.constr[i] = constrNew
    end
end

"""
Solve  min f(x)
s.t.,  lb ≤ x ≤ ub
       g(x) ≤ 0
       h(x) = 0
The function 'f' must return
    f, g, h = f(x)
"""
function optimize(
    f::Function,
    lb::Vector{Float64},
    ub::Vector{Float64};
    optimizer::T=DE(),
    x0::Vector{Vector{Float64}}=[NaN*ones(length(lb))],
    minFit::Float64=(-Inf),
    maxIter::Int=200,
    stallIter::Int=20,
    eqTol::Float64=1e-6,
    Npop::Int=min(50, 10*length(ub)),
    bounds=:clip,                       # clip, rand, or none
    verbose::Bool=true,
    iterCallback::F=(iter, pop)->false,
) where {T<:AbstractOptimizer, F}

    # Check NelderMead properties
    if T == NelderMead
        Nx = length(lb)
        Npop = Nx + 1
        @assert optimizer.Nx == Nx "Wrong number of parameters set for NelderMead, use: optimizer=NelderMead($Nx)"
    end

    # Initialize population
    pop = Population(Npop, f, x0, lb, ub, eqTol, bounds)

    # Init iterations
    msg = "Maximum number of iterations reached."
    costHist = fill(NaN, maxIter)

    # Start optimizing
    @inbounds for iter in 1:maxIter
        # Perform one iteration
        evolve!(pop, optimizer)

        # Post-process iteration
        costHist[iter] = pop.fit[pop.iBest]
        stop = iterCallback(iter, pop)          # User-defined callback
        verbose && println("Iter $iter, cost: $(pop.cost[pop.iBest]), constraint: $(pop.constr[pop.iBest])")

        # Check exit conditions
        if stop
            msg = "Iterations stopped (iterCallback)"
            break
        end
        if costHist[iter] < minFit
            msg = "Solution found."
            break
        end
        if iter > stallIter && costHist[iter] == costHist[iter - stallIter]
            msg = "Solution stalled."
            break
        end
    end

    # Print exit message and return solution
    verbose && println(msg)
    iLast = findlast(isnan, costHist)
    if !isnothing(iLast)
        costHist = costHist[1:iLast]
    end
    return pop.x[pop.iBest], pop, costHist
end
