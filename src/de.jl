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

function DE(; strategy = 2, F = 0.8, CR = 0.9)
    if abs(strategy) == 11
        CR = 0.8
    end
    return DE(strategy, clamp(F, 0.0, 2.0), clamp(CR, 0.0, 1.0))
end

function evolve!(pop::Population, optimizer::DE)
    # Pre-allocate stuff
    xOffspring = pop.xTmp

    # Start DE cycle
    for i in eachindex(pop.x)
        # Perform Mutation
        mutation!(xOffspring, i, optimizer, pop)

        # Perform Crossover - BIN (if strategy > 0) or EXP (if strategy < 0)
        crossover!(xOffspring, i, optimizer, pop)

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
@views function mutation!(
    xOffspring::Vector{Float64},
    i::Int,
    optimizer::DE,
    pop::Population,
)

    strategy = abs(optimizer.strategy)
    if strategy > 99
        ;
        strategy = rand(1:11);
    end         # Random strategy

    # Select 5 different random parents
    shuffle!(pop.idx);
    k = 0
    ip1, k = randomParentIndex(i, k, pop.idx)
    ip2, k = randomParentIndex(i, k, pop.idx)
    ip3, k = randomParentIndex(i, k, pop.idx)
    ip4, k = randomParentIndex(i, k, pop.idx)
    ip5, k = randomParentIndex(i, k, pop.idx)

    # Perform Mutation [1]
    # X: bin or exp, depending on 'useBin' flag
    # The general convention used above is DE/x/y/z, where DE stands for
    # “differential evolution,” x represents a string denoting the base vector
    # to be perturbed, y is the number of difference vectors considered for
    # perturbation of x, and z stands for the type of crossover being used
    # (exp: exponential; bin: binomial).
    xp1 = pop.x[ip1];
    xp2 = pop.x[ip2];
    xp3 = pop.x[ip3]
    xp4 = pop.x[ip4];
    xp5 = pop.x[ip5];
    xi = pop.x[i]
    xBest = pop.x[pop.iBest];
    F = optimizer.F

    if strategy == 1 # DE/rand/1/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xp1[j] + F*(xp2[j] - xp3[j])
        end
    elseif strategy == 2 # DE/best/1/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xBest[j] + F*(xp1[j] - xp2[j])
        end
    elseif strategy == 3 # DE/current-to-best/1/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xi[j] + F*(xBest[j] - xi[j] + xp1[j] - xp3[j])
        end
    elseif strategy == 4 # DE/current-to-rand/1/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xi[j] + F*(xp1[j] - xi[j] + xp2[j] - xp3[j])
        end
    elseif strategy == 5 # DE/rand-to-best/1/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xp1[j] + F*(xBest[j] - xi[j] + xp2[j] - xp3[j])
        end
    elseif strategy == 6 # DE/rand/2/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xp1[j] + F*(xp2[j] - xp3[j] + xp4[j] - xp5[j])
        end
    elseif strategy == 7 # DE/best/2/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xBest[j] + F*(xp1[j] - xp2[j] + xp3[j] - xp4[j])
        end
    elseif strategy == 8 # DE/current-to-best/2/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xi[j] + F*(xBest[j] - xi[j] + xp1[j] - xp2[j] + xp3[j] - xp4[j])
        end
    elseif strategy == 9 # DE/current-to-rand/2/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xi[j] + F*(xp1[j] - xi[j] + xp2[j] - xp3[j] + xp4[j] - xp5[j])
        end
    elseif strategy == 10 # DE/rand-to-best/2/X
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] =
                xp1[j] + F*(xBest[j] - xi[j] + xp2[j] - xp3[j] + xp4[j] - xp5[j])
        end
    else # uDE from Qiang and Mitchell [This uses CR = 0.8]
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] =
                0.5*xi[j] +
                0.25*xBest[j] +
                0.25*xp1[j] +
                0.2*(xp2[j] + xp4[j] - xp3[j] - xp5[j])
        end
    end
    return
end

function crossover!(xOffspring::Vector{Float64}, i::Int, optimizer::DE, pop::Population)
    xi = pop.x[i]
    Nx = length(xi)

    if optimizer.strategy > 0
        # BIN - Binomial crossover
        jRand = rand(1:Nx)              # Keep at least one component mutated
        @inbounds for j in eachindex(xi)
            if rand() > optimizer.CR && j != jRand
                xOffspring[j] = xi[j]
            end
        end
    else
        # EXP - Exponential crossover
        L = 1
        while rand() ≤ optimizer.CR && L < Nx
            L += 1
        end

        j0 = rand(1:Nx)
        @inbounds for k = 0:(L-1)
            j = mod1(j0 + k, Nx)
            xOffspring[j] = xi[j]
        end
    end
    return
end
