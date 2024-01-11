function ackley(x)
    d = length(x)
    c = 2π; b = 0.2; a = 20;
    term1 = -a*exp(-b*sqrt(sum(x.^2)/d))
    term2 = -exp(sum(cos.(c*x))/d)

    return term1 + term2 + a + exp(1.0), 0.0, 0.0
end

function ackleyBounds()
    return -32.768.*[1; 1], 32.768.*[1; 1]
end

function beale(x)
    return (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1] + x[1]*x[2]^2)^2 + (2.625 - x[1] + x[1]*x[2]^3)^2, 0.0, 0.0
end

function bealeBounds()
    return -4.5.*[1; 1], 4.5.*[1; 1]
end
