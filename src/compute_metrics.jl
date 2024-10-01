using Interpolations, Integrals, Distributions

function RMSE(zspec, zphot; normalised=false)
    if normalised == true
        norm_factor=1 .+zspec
    else
        norm_factor=1
    end
        
    return sqrt(sum(((zspec.-zphot)./norm_factor).^2)/length(zspec))
end

function FR15(zspec, zphot)
    return sum((100/length(zspec))*isless.(abs.((zspec.-zphot)./(1 .+zspec)), 0.15))
end

function FR05(zspec, zphot)
    return sum((100/length(zspec))*isless.(abs.((zspec.-zphot)./(1 .+zspec)), 0.05))
end

function bias(zspec, zphot; normalised=false)
    if normalised == true
        norm_factor=1 .+zspec
    else
        norm_factor=1
    end
    return sum((zspec.-zphot)./norm_factor)/length(zspec)
end

function find_medians(x, z)
    ii = findmin(abs.(cumsum(x) .- sum(x)/2))[2]
    return z[ii]
end

function find_modes(x, z)
    ii = findmax(x)[2]
    return z[ii]
end

function PIT(_x, _y, zspec)
    x = deepcopy(_x)
    y = deepcopy(_y)
    itp = linear_interpolation(x, y,extrapolation_bc=Line())
    push!(x, zspec)
    push!(y, itp(zspec))
    i = sortperm(x)
    x=x[i]
    y=y[i]
    i = findfirst(x .== zspec)
    if i==1
        return 0
    end
    x = x[1:i]
    y = y[1:i]
    return solve(SampledIntegralProblem(y, x), TrapezoidalRule()).u
end

function PIT_gaussian(mean, sigma, zspec)
    return cdf(Normal(mean, sigma), zspec)
end
