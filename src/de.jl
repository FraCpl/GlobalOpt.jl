# Differential Evolution Optimization
#
# References:
# [1] Qiang and Mitchell, Unified Differential Evolution Algorithm for
#     Global Optimization.
#     https://www.osti.gov/servlets/purl/1163659#:~:text=The%20DE%2Frand%2D%20to%2D,the%20current%20target%20parent%20vector.
# [2] Brest et al., Self-Adapting Control Parameters in Differential
#     Evolution: A Comparative Study on Numerical Benchmark Problems, IEEE
#     Transactions on Evolutionary Computation, Vol. 10, No. 6, Dec. 2006.
#     https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4016057
#
# Author: F. Capolupo
# European Space Agency, 2022
mutable struct DE <: AbstractOptimizer
    strategy::Int
    F::Float64
    CR::Float64
end
function DE(; strategy=2, F=0.8, CR=0.9)
    if abs(strategy) == 11
        CR = 0.8
    end
    F = max(0.0, min(2.0, F))
    CR = max(0.0, min(1.0, CR))
    return DE(strategy, F, CR)
end

function evolve!(pop::Population, optimizer::DE)
    # Pre-allocate stuff
    iParents = zeros(Int, 5)
    xOffspring = similar(pop.x[1])

    # Start DE cycle
    for i in eachindex(pop.x)
        # Select 5 different random parents
        iParents .= shuffle(vcat(1:i-1, i+1:pop.Npop))[1:5]

        # Perform Mutation
        xOffspring .= mutation(i, optimizer, pop, iParents)

        # Perform Crossover - BIN (if strategy > 0) or EXP (if strategy < 0)
        xOffspring .= crossover(i, optimizer, pop, xOffspring)

        # Evaluate cost and constraints of offspring
        xOffspring .= pop.applyBounds(xOffspring)
        costOffspring, constrOffspring = pop.fun(xOffspring)

        # Selection - one by one comparison
        compare1vs1!(i, pop, xOffspring, costOffspring, constrOffspring)
    end

    # Evaluate fitness of new population
    evalFitness!(pop)
end

# [1] Qiang and Mitchell, Unified Differential Evolution Algorithm for
#     Global Optimization.
#     https://www.osti.gov/servlets/purl/1163659#:~:text=The#20DE#2Frand#2D#20to#2D,the#20current#20target#20parent#20vector.
#
# Author: F. Capolupo
# European Space Agency, 2022
@views function mutation(i::Int, optimizer::DE, pop::Population, iParents::Vector{Int})

    strategy = abs(optimizer.strategy)
    if strategy > 99
        # Random strategy
        strategy = rand(1:11)
    end

    # Perform Mutation [1]
    # X: bin or exp, depending on 'useBin' flag
    # The general convention used above is DE/x/y/z, where DE stands for
    # “differential evolution,” x represents a string denoting the base vector
    # to be perturbed, y is the number of difference vectors considered for
    # perturbation of x, and z stands for the type of crossover being used
    # (exp: exponential; bin: binomial).
    if strategy == 1 # DE/rand/1/X
        return pop.x[iParents[1]] + optimizer.F.*(pop.x[iParents[2]] - pop.x[iParents[3]])
    end

    if strategy == 2 # DE/best/1/X
        return pop.x[pop.iBest] + optimizer.F.*(pop.x[iParents[1]] - pop.x[iParents[2]])
    end

    if strategy == 3 # DE/current-to-best/1/X
       return pop.x[i] + optimizer.F.*(pop.x[pop.iBest] - pop.x[i] + pop.x[iParents[1]] - pop.x[iParents[2]])
    end

    if strategy == 4 # DE/current-to-rand/1/X
        return pop.x[i] + optimizer.F.*(pop.x[iParents[1]] - pop.x[i] + pop.x[iParents[2]] - pop.x[iParents[3]])
    end

    if strategy == 5 # DE/rand-to-best/1/X
        return pop.x[iParents[1]] + optimizer.F.*(pop.x[pop.iBest] - pop.x[i] + pop.x[iParents[2]] - pop.x[iParents[3]])
    end

    if strategy == 6 # DE/rand/2/X
        return pop.x[iParents[1]] + optimizer.F.*(pop.x[iParents[2]] - pop.x[iParents[3]] + pop.x[iParents[4]] - pop.x[iParents[5]])
    end

    if strategy == 7 # DE/best/2/X
        return pop.x[pop.iBest] + optimizer.F.*(pop.x[iParents[2]] - pop.x[iParents[3]] + pop.x[iParents[4]] - pop.x[iParents[5]])
    end

    if strategy == 8 # DE/current-to-best/2/X
        return pop.x[i] + optimizer.F.*(pop.x[pop.iBest] - pop.x[i] + pop.x[iParents[1]] - pop.x[iParents[2]] + pop.x[iParents[3]] - pop.x[iParents[4]])
    end

    if strategy == 9 # DE/current-to-rand/2/X
        return pop.x[i] + optimizer.F.*(pop.x[iParents[1]] - pop.x[i] + pop.x[iParents[2]] - pop.x[iParents[3]] + pop.x[iParents[4]] - pop.x[iParents[5]])
    end

    if strategy == 10 # DE/rand-to-best/2/X
        return pop.x[iParents[1]] + optimizer.F.*(pop.x[pop.iBest] - pop.x[i] + pop.x[iParents[2]] - pop.x[iParents[3]] + pop.x[iParents[4]] - pop.x[iParents[5]])
    end

    if strategy == 11 # uDE from Qiang and Mitchell [This uses CR = 0.8]
        return 0.5*pop.x[i] + 0.25*pop.x[pop.iBest] + 0.25*pop.x[iParents[1]] + 0.2*(pop.x[iParents[2]] + pop.x[iParents[4]] - pop.x[iParents[3]] - pop.x[iParents[5]])
    end
end

function crossover(i::Int, optimizer::DE, pop::Population, xOffspring::Vector{Float64})
    x = copy(pop.x[i])
    Nx = length(x)

    if optimizer.strategy > 0
        # BIN - Binomial crossover
        jRand = rand(1:Nx)
        x[jRand] = xOffspring[1]    # Mutate at least one component
        for j in eachindex(x)
            if rand() < optimizer.CR
                x[j] = xOffspring[j]
            end
        end
    else
        # EXP - Exponential crossover
        L = 0
        while true
            L += 1
            if (rand() > optimizer.CR || L > Nx)
                break
            end
        end
        n = rand(1:Nx)
        j = mod.(n:n+L-1, Nx)
        j[j .== 0] .= Nx
        for jj in j
            x[jj] = xOffspring[jj]
        end
    end
    return x
end
