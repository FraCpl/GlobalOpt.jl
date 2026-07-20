struct NelderMead <: AbstractOptimizer
    α::Float64   # reflection
    γ::Float64   # expansion
    ρ::Float64   # contraction
    σ::Float64   # shrink

    # workspace
    Nx::Int
    xc::Vector{Float64}   # centroid
    xr::Vector{Float64}
    xe::Vector{Float64}
    xc2::Vector{Float64}
end

NelderMead(Nx::Int; α::Float64=1.0, γ::Float64=2.0, ρ::Float64=0.5, σ::Float64=0.5) =
    NelderMead(α, γ, ρ, σ, Nx, zeros(Nx), zeros(Nx), zeros(Nx), zeros(Nx))

@inline function evalPoint!(pop::Population, x::Vector{Float64})
    pop.applyBounds!(x)
    return pop.fun(x)
end

# CHATGPT CODE
function evolve!(pop::Population, opt::NelderMead)
    Nx = length(pop.lb)
    Np = pop.Npop

    idx = pop.idx

    # Sort simplex by fitness
    sort!(idx, by = i -> pop.fit[i])

    best = idx[1]
    worst = idx[end]
    secondWorst = idx[end-1]

    fit_best = pop.fit[best]
    fit_sw = pop.fit[secondWorst]
    fit_worst = pop.fit[worst]

    xc = opt.xc
    xr = opt.xr
    xe = opt.xe
    xc2 = opt.xc2

    # --- Compute centroid (excluding worst)
    fill!(xc, 0.0)
    @inbounds for k in 1:(Np-1)
        xk = pop.x[idx[k]]
        for j in 1:Nx
            xc[j] += xk[j]
        end
    end
    @inbounds for j in 1:Nx
        xc[j] /= (Np - 1)
    end

    # --- Reflection: xr = xc + α*(xc - x_worst)
    xw = pop.x[worst]
    @inbounds for j in 1:Nx
        xr[j] = xc[j] + opt.α * (xc[j] - xw[j])
    end
    cost_r, constr_r = evalPoint!(pop, xr)

    # Case 1: reflection is better than the best point → try expansion
    if cost_r + constr_r < fit_best
        @inbounds for j in 1:Nx
            xe[j] = xc[j] + opt.γ * (xr[j] - xc[j])
        end
        cost_e, constr_e = evalPoint!(pop, xe)

        if cost_e + constr_e < cost_r + constr_r
            # Case 1a: Expanded point is better than reflected
            # The expaded point replaces the worst point
            pop.x[worst] .= xe
            pop.cost[worst] = cost_e
            pop.constr[worst] = constr_e
        else
            # Case 1b: Reflection is better than the expanded point.
            # The reflected point replaces the worst point
            pop.x[worst] .= xr
            pop.cost[worst] = cost_r
            pop.constr[worst] = constr_r
        end

    # Case 2: reflection is acceptable (i.e., is not going to be the new worst point).
    # The reflected point replaces the worst point
    elseif cost_r + constr_r < fit_sw
        pop.x[worst] .= xr
        pop.cost[worst] = cost_r
        pop.constr[worst] = constr_r

    # Case 3: contraction
    # The reflected point is better than the worst, but not better than the second worst
    else
        # Outside contraction
        if cost_r + constr_r < fit_worst
            @inbounds for j in 1:Nx
                xc2[j] = xc[j] + opt.ρ * (xr[j] - xc[j])
            end
        else
            # Inside contraction
            @inbounds for j in 1:Nx
                xc2[j] = xc[j] - opt.ρ * (xc[j] - xw[j])
            end
        end

        cost_c, constr_c = evalPoint!(pop, xc2)

        if cost_c + constr_c < fit_worst
            pop.x[worst] .= xc2
            pop.cost[worst] = cost_c
            pop.constr[worst] = constr_c
        else
            # --- Shrink
            # Shrink all vertices towards the best one
            xbest = pop.x[best]
            @inbounds for k in 2:Np
                xi = pop.x[idx[k]]
                for j in 1:Nx
                    xi[j] = xbest[j] + opt.σ * (xi[j] - xbest[j])
                end
                pop.applyBounds!(xi)
                pop.cost[idx[k]], pop.constr[idx[k]] = pop.fun(xi)
            end
        end
    end

    # Update fitness
    evalFitness!(pop)

    return nothing
end
