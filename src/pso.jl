# https://www.researchgate.net/publication/255756848_Standard_Particle_Swarm_Optimisation_2011_at_CEC-2013_A_baseline_for_future_PSO_improvements
mutable struct PSO <: AbstractOptimizer
    w::Float64
    c1::Float64
    c2::Float64
    useStandard::Bool

    # Internal data
    initDone::Bool
    Nx::Int
    xk::Vector{Vector{Float64}}
    vk::Vector{Vector{Float64}}
    vb::Vector{Float64}
end
function PSO(;
        w::Float64=0.4,
        c1::Float64=2.1,
        c2::Float64=1.05,
        useStandard=false,
        )
    return PSO(w, c1, c2, useStandard, false, 0, [], [], [])
end

function initPSO!(optimizer::PSO, pop::Population)
    optimizer.vb = abs.(pop.ub - pop.lb)
    optimizer.Nx = length(pop.x[1])
    optimizer.vk = [-optimizer.vb + 2optimizer.vb.*rand(optimizer.Nx) for _ in eachindex(pop.x)]
    optimizer.xk = copy(pop.x)
    optimizer.initDone = true
end

function evolve!(pop::Population, optimizer::PSO)
    # Init parameters if not done already
    if !optimizer.initDone
        initPSO!(optimizer, pop)
    end

    # Evolution cycle
    G = similar(pop.x[1])
    for i in eachindex(pop.x)

        # Update particles velocity
        if optimizer.useStandard
            # Option 1: Standard PSO 2011
            G .= (optimizer.c1*rand(optimizer.Nx).*(pop.x[i] - optimizer.xk[i]) +
                optimizer.c2*rand(optimizer.Nx).*(pop.x[pop.iBest] - optimizer.xk[i]))/3
            R = rand()*norm(G)
            optimizer.vk[i] .= optimizer.w*optimizer.vk[i] + G + normalize(randn(optimizer.Nx))*R
        else
            # Option 2: Classic PSO
            optimizer.vk[i] .= optimizer.w*optimizer.vk[i] +
                optimizer.c1*rand(optimizer.Nx).*(pop.x[i] - optimizer.xk[i]) +
                optimizer.c2*rand(optimizer.Nx).*(pop.x[pop.iBest] - optimizer.xk[i])
        end

        # Bound velocity
        optimizer.vk[i] .= clipBounds(optimizer.vk[i], -optimizer.vb, optimizer.vb)

        # Update particles position
        optimizer.xk[i] .+= optimizer.vk[i]

        # Update particles
        clipAndStop!(optimizer.xk[i], optimizer.vk[i], pop)
        costk, constrk = pop.fun(optimizer.xk[i])
        compare1vs1!(i, pop, optimizer.xk[i], costk, constrk)
    end

    # Evaluate fitness of new population
    evalFitness!(pop)
end

function clipAndStop!(x::Vector{Float64}, v::Vector{Float64}, pop::Population)
    for j in eachindex(x)
        if x[j] < pop.lb[j]
            x[j] = pop.lb[j]
            v[j] = 0.0
        elseif x[j] > pop.ub[j]
            x[j] = pop.lb[j]
            v[j] = 0.0
        end
    end
end
