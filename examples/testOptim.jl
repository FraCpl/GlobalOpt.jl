using GlobalOpt
using GLMakie
include("TestFunctions.jl")

function main()

    f = ackley; lb, ub = ackleyBounds()
    # f = beale; lb, ub = bealeBounds()
    # f = himmelblau; lb, ub = himmelblauBounds()

    xBest, pop, costHistGA = optimize(f, lb, ub; Npop=50, optimizer=GA(), verbose=false, minFit=1e-10)
    @show pop.cost[pop.iBest], xBest

    xBest, pop, costHistDE = optimize(f, lb, ub; Npop=50, optimizer=DE(), verbose=false, minFit=1e-10)
    @show pop.cost[pop.iBest], xBest

    xBest, pop, costHistNM = optimize(f, lb, ub; optimizer=NelderMead(length(lb)), verbose=false, minFit=1e-10)
    @show pop.cost[pop.iBest], xBest

    # TODO: Not very good performance
    # xBest, pop, costHist = optimize(f, lb, ub; Npop=156, optimizer=PSO(), verbose=false, stallIter=100000, maxIter=1000)
    # @show pop.cost[pop.iBest], xBest
    # plot!(costHist; label="PSO", lw=2, color=3)

    fig = Figure(); display(fig)
    ax = GLMakie.Axis(fig[1, 1], xlabel="Iterations", ylabel="Fitness", yscale=log10)
    scatterlines!(ax, costHistGA; label="GA")
    scatterlines!(ax, costHistDE; label="DE")
    scatterlines!(ax, costHistNM; label="NelderMead")
    axislegend(ax)
end
main();
