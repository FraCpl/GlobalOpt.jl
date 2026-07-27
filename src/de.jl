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
include("deStrategies.jl")

abstract type AbstractCrossoverDE end
struct BinomialCxDE <: AbstractCrossoverDE end
struct ExponentialCxDE <: AbstractCrossoverDE end

struct DE{S<:AbstractStrategyDE, X<:AbstractCrossoverDE} <: AbstractOptimizer
    strategy::S
    cross::X
    F::Float64
    CR::Float64
end

function DE(; strategy::S=StrategyDE2(), F=0.8, CR=0.9, cx::X=BinomialCxDE()) where {S<:AbstractStrategyDE, X<:AbstractCrossoverDE}
    return DE(strategy, cx, clamp(F, 0.0, 2.0), clamp(CR, 0.0, 1.0))
end

function evolve!(pop::Population, opt::DE{S, X}, rng) where {S<:AbstractStrategyDE, X<:AbstractCrossoverDE}
    # Pre-allocate stuff
    # xOffspring = pop.xTmp                 # [SINGLE-THREAD]
    # idx = pop.idx                         # [SINGLE-THREAD]
    N = length(pop.x)                       # [MULTITHREAD]

    # Start DE cycle
    # for i in eachindex(pop.x)             # [SINGLE-THREAD]
        # shuffle!(idx)                     # [SINGLE-THREAD]
    @threads for i in 1:N                   # [MULTITHREAD]
        xOffspring = similar(pop.x[1])      # [MULTITHREAD] local buffer
        idx = randperm(rng, N)                   # [MULTITHREAD] local shuffled indices

        # Perform Mutation
        mutation!(xOffspring, i, opt, pop, idx)

        # Perform Crossover - BIN (if strategy > 0) or EXP (if strategy < 0)
        crossover!(xOffspring, i, opt, pop, rng)

        # Evaluate cost and constraints of offspring
        pop.applyBounds!(xOffspring)
        costOffspring, constrOffspring = pop.fun(xOffspring)

        # Selection - one by one comparison
        compare1vs1!(i, pop, xOffspring, costOffspring, constrOffspring)
    end

    # Evaluate fitness of new population
    evalFitness!(pop)
end

function randomParentIndex(i, k, idx)
    k += 1
    iOut = idx[k]
    while iOut == i
        k += 1
        iOut = idx[k]
    end
    return iOut, k
end

# [1] Qiang and Mitchell, Unified Differential Evolution Algorithm for
#     Global Optimization.
#     https://www.osti.gov/servlets/purl/1163659#:~:text=The#20DE#2Frand#2D#20to#2D,the#20current#20target#20parent#20vector.
#
# Author: F. Capolupo
# European Space Agency, 2022
function mutation!(xOffspring::Vector{Float64}, i::Int, opt::DE, pop::Population, idx::Vector{Int})

    # Select 5 different random parents
    k = 0
    ip1, k = randomParentIndex(i, k, idx)
    ip2, k = randomParentIndex(i, k, idx)
    ip3, k = randomParentIndex(i, k, idx)
    ip4, k = randomParentIndex(i, k, idx)
    ip5, k = randomParentIndex(i, k, idx)

    # Perform Mutation [1]
    # X: bin or exp, depending on 'useBin' flag
    # The general convention used above is DE/x/y/z, where DE stands for
    # “differential evolution,” x represents a string denoting the base vector
    # to be perturbed, y is the number of difference vectors considered for
    # perturbation of x, and z stands for the type of crossover being used
    # (exp: exponential; bin: binomial).
    xp1 = pop.x[ip1]
    xp2 = pop.x[ip2]
    xp3 = pop.x[ip3]
    xp4 = pop.x[ip4]
    xp5 = pop.x[ip5]
    xi = pop.x[i]
    xBest = pop.x[pop.iBest]
    mutate!(opt.strategy, opt.F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)

    return nothing
end

# BIN - Binomial crossover
function crossover!(xOffspring::Vector{Float64}, i::Int, opt::DE{S, BinomialCxDE}, pop::Population, rng) where {S<:AbstractStrategyDE}
    xi = pop.x[i]
    Nx = length(xi)
    jRand = rand(rng, 1:Nx)              # Keep at least one component mutated
    @inbounds for j in eachindex(xi)
        if rand(rng) > opt.CR && j != jRand
            xOffspring[j] = xi[j]
        end
    end
    return nothing
end

# EXP - Exponential crossover
function crossover!(xOffspring::Vector{Float64}, i::Int, opt::DE{S, ExponentialCxDE}, pop::Population, rng) where {S<:AbstractStrategyDE}
    xi = pop.x[i]
    Nx = length(xi)
    L = 1
    while rand(rng) ≤ opt.CR && L < Nx
        L += 1
    end
    j0 = rand(rng, 1:Nx)
    @inbounds for k in 0:(L - 1)
        j = mod1(j0 + k, Nx)
        xOffspring[j] = xi[j]
    end
    return nothing
end
