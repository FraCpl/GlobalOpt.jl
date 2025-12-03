mutable struct GA <: AbstractOptimizer
    selection::Function
    crossover::Function
    mutation::Function
    EP::Float64
    CP::Float64
    MP::Float64
    mutateParents::Bool

    # Internal data
    initDone::Bool
    Nelite::Int
    Ncross::Int
    Nmutate::Int
end
function GA(;
    selection::Function=tournament,  # tournament or roulette
    crossover::Function=crossoverBlend,
    mutation::Function=(mutateUniform!),
    EP::Float64=0.1,            # Elite percentage
    CP::Float64=0.8,            # Crossover percentage
    MP::Float64=0.07,           # Mutation probability
    mutateParents::Bool=true,
)
    return GA(selection, crossover, mutation, EP, CP, MP, mutateParents, false, 0, 0, 0)
end

function initGenetic!(optimizer::GA, pop::Population)
    optimizer.Nelite = ceil(Int, pop.Npop*optimizer.EP)
    if optimizer.mutateParents
        CP = min(optimizer.CP, 1.0 - optimizer.EP)
        optimizer.Ncross = round(Int, CP*pop.Npop/2)
        optimizer.Nmutate = pop.Npop - 2optimizer.Ncross - optimizer.Nelite
    else
        optimizer.Ncross = round(Int, (pop.Npop - optimizer.Nelite)/2)
        optimizer.Nelite = pop.Npop - 2optimizer.Ncross
        optimizer.Nmutate = 0
    end
    optimizer.initDone = true
end

function evolve!(pop::Population, optimizer::GA)
    # Init parameters if not done already
    if !optimizer.initDone
        initGenetic!(optimizer, pop)
    end

    # Allocate stuff
    iParents = zeros(Int, 2)
    xNew = copy(pop.x)
    k = optimizer.Nelite + 1

    # Run evolution cycle
    iSort = sortperm(pop.fit)
    for _ in 1:optimizer.Ncross
        # Select parents
        iParents[1] = optimizer.selection(iSort, pop)
        iParents[2] = optimizer.selection(iSort, pop)

        # Perform crossover of parents
        xNew[iSort[k]], xNew[iSort[k + 1]] = optimizer.crossover(pop.x[iParents[1]], pop.x[iParents[2]])

        # Mutate children
        if optimizer.Nmutate == 0
            optimizer.mutation(xNew[iSort[k]], pop, optimizer)
            optimizer.mutation(xNew[iSort[k + 1]], pop, optimizer)
        end
        k += 2
    end

    for _ in 1:optimizer.Nmutate
        # Select parents
        iParents[1] = optimizer.selection(iSort, pop)
        xNew[iSort[k]] = copy(pop.x[iParents[1]])

        # Mutate parents
        optimizer.mutation(xNew[iSort[k]], pop, optimizer)
        k += 1
    end

    # Update population
    for q in (optimizer.Nelite + 1):pop.Npop
        pop.x[iSort[q]] .= xNew[iSort[q]]
        pop.applyBounds!(pop.x[iSort[q]])
        pop.cost[iSort[q]], pop.constr[iSort[q]] = pop.fun(pop.x[iSort[q]])
    end

    # Evaluate fitness of new population
    evalFitness!(pop)
end

function tournament(iSort, pop::Population)
    return iSort[minimum(rand(iSort, 4))]  # best of 4 contestant
end

function roulette(iSort, pop::Population)
    fitRw = 1.0 ./ (1 .+ pop.fit)
    fitRw ./= sum(fitRw)
    # https://stackoverflow.com/questions/27559958/how-do-i-select-a-random-item-from-a-weighted-array-in-julia
    return iSort[findfirst(cumsum(fitRw) .> rand())]
end

function mutateUniform!(x, pop::Population, optimizer::GA)
    for j in eachindex(x)
        if rand() ≤ optimizer.MP
            x[j] = pop.lb[j] + (pop.ub[j] - pop.lb[j])*rand()
        end
    end
end

function crossoverBlend(x1, x2)
    α = rand(length(x1))
    Δx = α .* (x2 - x1)
    return x1 - Δx, x2 + Δx
end
