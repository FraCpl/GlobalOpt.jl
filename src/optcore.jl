abstract type AbstractOptimizer end

mutable struct Population
    Npop::Int                       # Number of members of the population
    fun::Function                   # Cost function handler
    applyBounds::Function           # lb, ub bounds function

    x::Vector{Vector{Float64}}      # Population members
    lb::Vector{Float64}
    ub::Vector{Float64}
    iBest::Int                      # Index of the fittest member of population in x
    cost::Vector{Float64}           # Cost of x
    constr::Vector{Float64}         # Constraint violation of x
    fit::Vector{Float64}            # Fitness of x
end

function Population(Npop, f, x0, lb, ub, eqTol, bounds)
    Nx = length(lb)

    if bounds == :clip
        pop = Population(Npop, x -> evalFunction(f, x, eqTol), x -> clipBounds(x, lb, ub),
            [zeros(Nx) for _ in 1:Npop], lb, ub, 1, zeros(Npop), zeros(Npop), zeros(Npop))
    elseif bounds == :rand
        pop = Population(Npop, x -> evalFunction(f, x, eqTol), x -> randBounds(x, lb, ub),
            [zeros(Nx) for _ in 1:Npop], lb, ub, 1, zeros(Npop), zeros(Npop), zeros(Npop))
    else
        pop = Population(Npop, x -> evalFunction(f, x, eqTol), x -> x,
            [zeros(Nx) for _ in 1:Npop], lb, ub, 1, zeros(Npop), zeros(Npop), zeros(Npop))
    end

    # Random initialization of all members of the population
    for i in eachindex(pop.x)
        pop.x[i] .= lb + (ub - lb).*rand(Nx)
    end

    # Add initial guesses as provided by the user to the population
    # TODO: if the user provides more than Npop, then select the Npop fittest out of x0
    if !isnan(x0[1][1])
        for i in 1:min(lastindex(x0), Npop)
            pop.x[i] .= pop.applyBounds(x0[i])
        end

        # Shuffle order of population
        shuffle!(pop.x)
    end

    # Evaludate fitness of initial population
    for i in eachindex(pop.x)
        pop.cost[i], pop.constr[i] = pop.fun(pop.x[i])
    end
    evalFitness!(pop)

    return pop
end

function clipBounds(x, lb, ub)
    return min.(max.(x, lb), ub)
end

function randBounds(x, lb, ub)
    xOut = copy(x)
    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            xOut[i] = lb[i] + (ub[i] - lb[i])*rand()
        end
    end
    return xOut
end

function evalFunction(f, x, eqTol)
    cost, g, h = f(x)
    gh = vcat(g, abs.(h) .- eqTol)
    if isnan(cost)
        cost = Inf
    end
    return cost, sum(gh[gh .> 0.0])
end

function evalFitness!(pop::Population)
    # Constraints accounting http://repository.ias.ac.in/9407/1/310.pdf
    isFeasible = pop.constr .== 0.0
    pop.fit .= copy(pop.cost)
    if any(isFeasible)
        # At least one solution is feasible
        pop.fit[.!isFeasible] .= maximum(pop.cost[isFeasible]) .+ pop.constr[.!isFeasible]
    else
        # All solutions are unfeasible
        pop.fit .+= pop.constr
    end
    pop.iBest = argmin(pop.fit)
end

function compare1vs1!(i::Int, pop::Population, xNew, costNew, constrNew)

    function updatePop!()
        pop.x[i] .= copy(xNew)
        pop.cost[i] = costNew
        pop.constr[i] = constrNew
    end

    # New is feasible, old is not
    if constrNew == 0.0 && pop.constr[i] > 0.0
        updatePop!()
        return
    end

    # Both are unfeasible, but new is less unfeasible than old
    if constrNew > 0.0 && constrNew ≤ pop.constr[i]
        updatePop!()
        return
    end

    # Both are feasible and new is cheaper than old
    if constrNew == 0.0 && pop.constr[i] == 0.0 && costNew ≤ pop.cost[i]
        updatePop!()
        return
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
        optimizer::AbstractOptimizer=DifferentialEvolution(),
        x0::Vector{Vector{Float64}}=[fill(NaN, length(lb))],
        minFit::Float64=-Inf,
        maxIter::Int=200,
        stallIter::Int=20,
        eqTol::Float64=1e-6,
        Npop::Int=10*length(ub),
        bounds=:clip,                       # clip, rand, or none
        verbose::Bool=true,
    )

    # Initialize population
    pop = Population(Npop, f, x0, lb, ub, eqTol, bounds)

    # Init iterations
    msg = "Maximum number of iterations reached."
    costHist = fill(NaN, maxIter)

    # Start optimizing
    for iter in 1:maxIter
        # Perform one iteration
        evolve!(pop, optimizer)

        # Post-process iteration
        costHist[iter] = pop.fit[pop.iBest]
        if verbose
            println("Iter $iter, fitness: $(costHist[iter])")
        end

        # Check exit conditions
        if costHist[iter] < minFit
            msg = "Solution found."
            break
        end
        if iter > stallIter
            if costHist[iter] == costHist[iter-stallIter]
                msg = "Solution stalled."
                break
            end
        end
    end

    # Print exit message and return solution
    if verbose
        println(msg)
    end
    return copy(pop.x[pop.iBest]), pop, costHist[.!isnan.(costHist)]
end
