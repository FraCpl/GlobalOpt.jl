abstract type AbstractStrategyDE end

# DE/rand/1/X
struct StrategyDE1 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE1, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = xi[j] + F*(xBest[j] - xi[j] + xp1[j] - xp2[j])
    end
end

# DE/best/1/X
struct StrategyDE2 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE2, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = xBest[j] + F*(xp1[j] - xp2[j])
    end
end

# DE/current-to-best/1/X
struct StrategyDE3 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE3, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = xi[j] + F*(xBest[j] - xi[j] + xp1[j] - xp2[j])
    end
end

# DE/current-to-rand/1/X
struct StrategyDE4 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE4, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
        @inbounds for j in eachindex(xOffspring)
            xOffspring[j] = xi[j] + F*(xp1[j] - xi[j] + xp2[j] - xp3[j])
        end
    end

# DE/rand-to-best/1/X
struct StrategyDE5 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE5, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = xp1[j] + F*(xBest[j] - xi[j] + xp2[j] - xp3[j])
    end
end

# DE/rand/2/X
struct StrategyDE6 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE6, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = xp1[j] + F*(xp2[j] - xp3[j] + xp4[j] - xp5[j])
    end
end

# DE/best/2/X
struct StrategyDE7 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE7, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = xBest[j] + F*(xp1[j] - xp2[j] + xp3[j] - xp4[j])
    end
end

# DE/current-to-best/2/X
struct StrategyDE8 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE8, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = xi[j] + F*(xBest[j] - xi[j] + xp1[j] - xp2[j] + xp3[j] - xp4[j])
    end
end

# DE/current-to-rand/2/X
struct StrategyDE9 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE9, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = xi[j] + F*(xp1[j] - xi[j] + xp2[j] - xp3[j] + xp4[j] - xp5[j])
    end
end

# DE/rand-to-best/2/X
struct StrategyDE10 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE10, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = xp1[j] + F*(xBest[j] - xi[j] + xp2[j] - xp3[j] + xp4[j] - xp5[j])
    end
end

# uDE from Qiang and Mitchell [This uses CR = 0.8]
struct StrategyDE11 <: AbstractStrategyDE end
@inline function mutate!(s::StrategyDE11, F, xOffspring, xi, xBest, xp1, xp2, xp3, xp4, xp5)
    @inbounds for j in eachindex(xOffspring)
        xOffspring[j] = 0.5*xi[j] + 0.25*xBest[j] + 0.25*xp1[j] + 0.2*(xp2[j] + xp4[j] - xp3[j] - xp5[j])
    end
end
