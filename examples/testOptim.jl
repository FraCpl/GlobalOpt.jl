# using Plots
using GlobalOpt
include("TestFunctions.jl")

function main()
    # f = ackley; lb, ub = ackleyBounds()
    f = beale;
    lb, ub = bealeBounds()
    # plot()

    xBest, pop, costHist = optimize(f, lb, ub; Npop = 50, optimizer = GA(), verbose = false)
    @show pop.cost[pop.iBest], xBest
    # plot!(costHist; label="GA", lw=2, color=1)

    xBest, pop, costHist = optimize(f, lb, ub; Npop = 50, optimizer = DE(), verbose = false)
    @show pop.cost[pop.iBest], xBest
    # plot!(costHist; label="DE", lw=2, color=2)

    # TODO: Not very good performance
    # xBest, pop, costHist = optimize(f, lb, ub; Npop=156, optimizer=PSO(), verbose=false, stallIter=100000, maxIter=1000)
    # @show pop.cost[pop.iBest], xBest
    # plot!(costHist; label="PSO", lw=2, color=3)

    # display(plot!(xlabel="Iterations", ylabel="Fitness", yaxis=:log, ticks=:native, framestyle=:box, ylim=(1e-25, 10)))
end
main();
