# x ∈ [-1, 1]
@inline scaleLin(x, lb, ub) = (ub + lb + (ub - lb)*clamp(x, -1.0, 1.0)) / 2
@inline scaleLog(x, log10lb, log10ub) = exp10(scaleLin(x, log10lb, log10ub))
