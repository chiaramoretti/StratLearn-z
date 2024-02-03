using LinearAlgebra
using Plots
using StatsBase

"""
    radialKernelDistance(distances, extra_kernel)

Given the distances and the bandwidth `eps.val`, computes the matrix of radial kernel.

# Arguments
- `distances`: A matrix of distances.
- `extra_kernel`: A dictionary containing the bandwidth value `eps.val`.

# Returns
- A matrix of the computed radial kernel values.

"""
function radialKernelDistance(distances, extra_kernel)
    # Given the distances and the bandwidth eps.val, computes the matrix of radial kernel
    return exp.(-distances .^ 2 / (4 * extra_kernel["eps.val"]))
end

"""
    symmetricNormalization(mymatrix)

Normalize `mymatrix` so that it becomes a Markov Matrix. `mymatrix` is a kernel matrix, therefore symmetric.

# Arguments
- `mymatrix`: A symmetric kernel matrix.

# Returns
- A dictionary containing the normalized kernel matrix (`KernelMatrixN`), the square root of the row means (`sqrtRowMeans`), and the square root of the column means (`sqrtColMeans`).

# Examples
```julia
julia> mymatrix = [0.5 0.2; 0.2 0.3]
2×2 Array{Float64,2}:
 0.5  0.2
 0.2  0.3

julia> symmetricNormalization(mymatrix)
Dict{String,Array{Float64,2}} with 3 entries:
  "KernelMatrixN" => [1.42857 0.676123; 0.676123 1.2]
  "sqrtRowMeans"  => [0.591608 0.5]
  "sqrtColMeans"  => [0.591608; 0.5;;]
  ```
  """
function symmetricNormalization(mymatrix)
    if !issymmetric(mymatrix) || any(mymatrix .< 0) || (size(mymatrix, 1) != size(mymatrix, 2))
        error("wrong argument, Matrix can't be Normalized")
    end

    @inbounds sqrtRowMeans = sqrt.(mean(mymatrix, dims=1))
    @inbounds sqrtColMeans = sqrt.(mean(mymatrix, dims=2))
    @inbounds KernelMatrixN = mymatrix ./ (sqrtRowMeans .* sqrtColMeans)

    return Dict("KernelMatrixN" => KernelMatrixN,
                "sqrtRowMeans" => sqrtRowMeans,
                "sqrtColMeans" => sqrtColMeans)
end


"""
    calculateBasis(z, nZ, system)

Calculate the basis matrix for a given system.

# Arguments
- `z`: A vector of values.
- `nZ`: The number of basis functions to generate.
- `system`: The system of basis functions to use. Must be either "cosine" or "Fourier".

# Returns
- The basis matrix.

# Examples
```julia
julia> z = [0.1, 0.2, 0.3]
julia> nZ = 4
julia> system = "cosine"
julia> calculateBasis(z, nZ, system)
3×4 Array{Float64,2}:
1.0  1.345      1.14412    0.831254
1.0  1.14412    0.437016  -0.437016
1.0  0.831254  -0.437016  -1.345
```
 """
function calculateBasis(z, nZ, system)
    if system == "cosine"
        basisZ = [sqrt(2) * cos.(xx * π * z) for xx in 1:(nZ - 1)]
        basisZ = hcat(ones(length(z)), basisZ...)
        return basisZ
    elseif system == "Fourier"
        n_half = round(Int, nZ / 2)
        sinBasisZ = [sqrt(2) * sin.(2 * xx * π * z) for xx in 1:n_half]
        cosBasisZ = [sqrt(2) * cos.(2 * xx * π * z) for xx in 1:n_half]
        basisZ = Matrix{Float64}(undef, length(z), 2 * n_half)
        basisZ[:, 1:2:end] = hcat(sinBasisZ...)
        basisZ[:, 2:2:end] = hcat(cosBasisZ...)
        basisZ = hcat(ones(length(z)), basisZ)
        basisZ = basisZ[:, 1:nZ]
        return basisZ
    else
        error("System of Basis not known")
    end
end

"""
    normalizeDensity(binSize, estimates, delta=0)

Normalize the density estimates using the given bin size.

# Arguments
- `binSize`: The size of each bin.
- `estimates`: The density estimates.
- `delta`: The threshold value for removing low-density regions. Default is 0.

# Returns
- The normalized density estimates.

# Examples
```julia
julia> binSize = 0.1
julia> estimates = [0.2, 0.3, 0.1, 0.4]
julia> normalizeDensity(binSize, estimates)
4-element Array{Float64,1}:
2.0
2.9999999999999996
1.0
4.0
 ```
 """
function normalizeDensity(binSize, estimates, delta=0)
    estimates = reshape(estimates, 1, length(estimates))
    if all(estimates .<= 0)
        estimates = ones(1, length(estimates))
    end
    estimatesThresh = estimates
    estimatesThresh[estimatesThresh .< 0] .= 0

    if sum(binSize * estimatesThresh) > 1
        maxDensity = maximum(estimates)
        minDensity = 0
        newXi = (maxDensity + minDensity) / 2
        eps = 1
        ii = 1
        while ii <= 1000
            estimatesNew = max.(0, estimates .- newXi)
            area = sum(binSize * estimatesNew)
            eps = abs(1 - area)
            if eps < 1e-7
                break
            end
            if area < 1
                maxDensity = newXi
            elseif area > 1
                minDensity = newXi
            end
            newXi = (maxDensity + minDensity) / 2
            ii += 1
        end
        estimatesNew = max.(0, estimates .- newXi)
    else
        estimatesNew = vec(estimatesThresh / (binSize * sum(estimatesThresh)))
    end

    runs = rle(reshape(estimatesNew,:) .> 0)
    nRuns = length(runs[1])
    if nRuns > 2
        lower = []
        upper = []
        area = []
        for ii in 1:nRuns
            if !runs[1][ii]               
                continue
            end
            whichMin = ii > 1 ? sum(runs[2][1:(ii-1)]) : 1
            whichMax = whichMin + runs[2][ii]
            push!(lower, whichMin)
            push!(upper, whichMax)
            push!(area, sum(binSize * estimatesNew[whichMin:whichMax]))
        end

        delta = min(delta, maximum(area))
        for ii in eachindex(area)
            if area[ii] < delta
                estimatesNew[lower[ii]:upper[ii]] .= 0
            end
        end
        estimatesNew = vec(estimatesNew / (binSize * sum(estimatesNew)))
    end

    return estimatesNew
end

"""
    findThreshold(binSize, estimates, confidence)

Find the threshold value that corresponds to a given confidence level in a set of density estimates.

# Arguments
- `binSize`: The size of each bin.
- `estimates`: An array of density estimates.
- `confidence`: The desired confidence level.

# Returns
- The threshold value that corresponds to the desired confidence level.

# Examples
```julia
julia> binSize = 0.1
julia> estimates = [0.2, 0.3, 0.1, 0.4]
julia> confidence = 0.5
julia> findThreshold(binSize, estimates, confidence)
0.1
```
"""
function findThreshold(binSize, estimates, confidence)
    densityRange = collect(extrema(estimates))
    newCut = sum(densityRange) / 2
    tolerance = 1e-7
    binEstimates = binSize * estimates

    for i in 1:1000
        @inbounds prob = sum(binEstimates[estimates .> newCut])
        eps = abs(confidence - prob)
        if eps < tolerance
            break # level found
        end
        if confidence > prob
            densityRange[2] = newCut
        else
            densityRange[1] = newCut
        end
        newCut = sum(densityRange) / 2
    end

    return newCut
end


function estimateErrorFinalEstimatorGeneric(predictedCompleteLTest,
                                            predictedCompleteUTest,
                                            zTestL,
                                            nBins,
                                            zMin, zMax;
                                            weightsZTestL=Float64[],
                                            boot=0)
    zGrid = range(zMin, zMax, length=nBins)

    colmeansComplete = mean.(eachcol(predictedCompleteUTest .^ 2))
    sSquare = mean(colmeansComplete) # sSquare is the first part of formula 9 (integral over unlabeled z)

    nU = size(predictedCompleteUTest, 1)
    nL = length(zTestL)
    predictedObserved = [predictedCompleteLTest[xx, argmin(abs.(zTestL[xx] .- zGrid))]
                         for xx in 1:nL]

    if isempty(weightsZTestL)
        likeli = mean(predictedObserved)
        println("Formula 9 (loss) WITHOUT beta weights computed")
    else
        println("Formula 9 (loss) WITH beta weights computed")
        likeli = mean(predictedObserved .* weightsZTestL)
    end

    output = Dict()
    output["mean"] = 0.5 * sSquare - likeli

    if boot != 0
    # Bootstrap
        meanBoot = [begin
                    sampleBootL = sample(1:nL, nL, replace=true)
                    sampleBootU = sample(1:nU, nU, replace=true)
                    
                    predictedCompleteBootL = predictedCompleteLTest[sampleBootL, :]
                    predictedCompleteBootU = predictedCompleteUTest[sampleBootU, :]
                    zTestBootL = zTestL[sampleBootL]
                    weightsZTestBootL = isempty(weightsZTestL) ? weightsZTestL : weightsZTestL[sampleBootL]

                    colmeansComplete = mean.(eachcol(predictedCompleteBootU .^ 2))
                    sSquare = mean(colmeansComplete)

                    predictedObserved_boot = [
                        predictedCompleteBootL[i, argmin(
                            abs.(zTestBootL[i] .- zGrid))] for i in 1:nL]
                    likeli = isempty(weightsZTestBootL) ? mean(predictedObserved_boot) : mean(predictedObserved_boot .* weightsZTestBootL)
                    0.5 * sSquare - likeli
                    end for _ in 1:boot]
    output["seBoot"] = std(meanBoot)
    end
    return output
end
