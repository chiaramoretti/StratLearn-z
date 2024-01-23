using Statistics
using LinearAlgebra

"""
    chooseDeltaStatio(object, zValidation, distanceXValidationTrain, deltaGrid, nBins)

Estimates the optimal value of delta for the final estimator in the Statio framework.

# Arguments
- `object`: The object containing the trained model.
- `zValidation`: The validation set of z values.
- `distanceXValidationTrain`: The distance matrix between the validation set and the training set.
- `deltaGrid`: A grid of delta values to consider.
- `nBins`: The number of bins for the density estimation.

# Returns
A dictionary containing the following keys:
- `"bestDelta"`: The optimal value of delta.
- `"bestDeltaoneSD"`: The maximum delta value within one standard deviation of the optimal delta.
"""
function chooseDeltaStatio(object, zValidation, distanceXValidationTrain,
                           deltaGrid, nBins)
    errors = []
    errorSE = []

    for delta in deltaGrid
        estimateErrors = estimateErrorFinalEstimatorStatio(object,
                                                           zValidation,
                                                           distanceXValidationTrain,
                                                           nBins,
                                                           boot=100,
                                                           zMin=0, zMax=1,
                                                           delta=delta)
        push!(errors, estimateErrors["mean"])
        push!(errorSE, estimateErrors["seBoot"])
    end

    minError = minimum(errors)
    whichMin = findfirst(isequal(minError), errors)
    bestDelta = deltaGrid[whichMin]
    bestDeltaoneSD = maximum(deltaGrid[errors .<= (errors[whichMin] +
                                                   errorSE[whichMin])])

    return Dict("bestDelta" => bestDelta,
                "bestDeltaoneSD" => bestDeltaoneSD)
end

"""
    condDensityStatio(distancesX, z;
                      nZMax=20, kernel_function=radialKernelDistance,
                      extra_kernel=Dict("eps.val" => 1),
                      nXMax=nothing, normalization=nothing, system="Fourier")

Computes the conditional density estimation using the Statio framework.

# Arguments
- `distancesX`: The distance matrix between the training set and the test set.
- `z`: The test set of z values.
- `nZMax`: The maximum number of basis functions for Z.
- `kernel_function`: The kernel function to compute the kernel matrix.
- `extra_kernel`: Extra parameters for the kernel function.
- `nXMax`: The maximum number of basis functions for X. If not provided, it is set to `length(z) - 10`.
- `normalization`: The normalization rule for the kernel matrix. If not provided, no normalization is applied.
- `system`: The system used for basis calculation.

# Returns
A dictionary containing the following keys:
- `"coefficients"`: The coefficients for the conditional density estimation.
- `"system"`: The system used for basis calculation.
- `"nX"`: The maximum number of basis functions for X.
- `"nZ"`: The maximum number of basis functions for Z.
- `"normalizationParameters"`: The parameters used for kernel matrix normalization.
- `"normalization"`: The normalization rule used for the kernel matrix.
- `"eigenX"`: The basis functions for X.
- `"eigenValuesX"`: The eigenvalues for X.
- `"kernelFunction"`: The kernel function used to compute the kernel matrix.
- `"extraKernel"`: The extra parameters for the kernel function.
"""
function condDensityStatio(distancesX, z;
                           nZMax=20, kernel_function=radialKernelDistance,
                           extra_kernel=Dict("eps.val" => 1),
                           nXMax=nothing, normalization=nothing, system="Fourier")
    n = length(distancesX[:,1])

    # Kernel Matrix Calculation
    kernelMatrix = kernel_function(distancesX, extra_kernel)
    # Normalization
    if normalization !== nothing
        if normalization == "symmetric"
            normalizationParameters = symmetricNormalization(kernelMatrix)
            kernelMatrixN = normalizationParameters["KernelMatrixN"]
        else
            error("Normalization rule not implemented!")
        end
    else
        normalizationParameters = Dict("KernelMatrixN" => kernelMatrix)
        kernelMatrixN = kernelMatrix
    end

    # Eigenvalue Decomposition
    if nXMax === nothing
        nXMax = length(z) - 10
    end
    p = 10
    Omega = reshape(randn(n*(nXMax+p)), (n, nXMax + p))
    Z = kernelMatrixN * Omega
    Y = kernelMatrixN * Z
    Q = qr(Y).Q
    B = transpose(Q) * Z * inv(transpose(Q) * Omega)
    eigenB = eigen(B, sortby = x -> (-real(x), -imag(x)))
    lambda = eigenB.values
    U = Q * eigenB.vectors
    basisX = real(sqrt(n) * U[:, 1:nXMax])
    eigenValues = real(lambda[1:nXMax] / n)

    # Basis Calculation
    basisZ = calculateBasis(z, nZMax, system)

    # Coefficient Calculation
    coefficients = transpose(transpose(basisZ) * basisX) / n

    # Output Construction
    object = Dict(
        "coefficients" => coefficients,
        "system" => system,
        "nX" => nXMax,
        "nZ" => nZMax,
        "normalizationParameters" => normalizationParameters,
        "normalization" => normalization,
        "eigenX" => basisX,
        "eigenValuesX" => eigenValues,
        "kernelFunction" => kernel_function,
        "extraKernel" => extra_kernel
    )

    return object
end

"""
    estimateErrorFinalEstimatorStatio(object, zTest, distanceXTestTrain, nBins;
                                      zMin=0, zMax=1, delta=0, boot=0, add_pred=false,
                                      predictedObserved=[], predictedComplete=[])

Estimates the error of the final estimator for a given object and test data.

# Arguments
- `object`: The object containing the necessary information for density estimation.
- `zTest`: The test points at which the densities are estimated.
- `distanceXTestTrain`: The distance between the test points and the training data.
- `nBins`: The number of bins used for density estimation.
- `zMin`: The minimum value of the test points range. Default is 0.
- `zMax`: The maximum value of the test points range. Default is 1.
- `delta`: The delta value used for density normalization. Default is 0.
- `boot`: The number of bootstrap samples to calculate the standard error. Default is 0.
- `add_pred`: A boolean indicating whether to include the predicted densities and observed densities in the output. Default is false.
- `predictedObserved`: The observed densities at the test points. If not provided, they will be calculated based on the predicted densities.
- `predictedComplete`: The predicted densities at all test points. If not provided, they will be calculated using the `predictDensityStatio` function.

# Returns
- If `add_pred` is false, returns a dictionary with the following keys:
    - `"mean"`: The estimated error of the final estimator.
    - `"seBoot"`: The standard error of the estimated error, calculated using bootstrap samples if `boot` is greater than 0.
- If `add_pred` is true, returns a dictionary with the following keys:
    - `"output"`: The same as above.
    - `"predictedComplete"`: The predicted densities at all test points.
    - `"predictedObserved"`: The observed densities at the test points.
""" 
function estimateErrorFinalEstimatorStatio(object, zTest, distanceXTestTrain,
                                           nBins; zMin=0, zMax=1, delta=0, 
                                           boot=0, add_pred=false,
                                           predictedObserved=[], predictedComplete=[])


    zGrid = range(zMin, zMax, length = nBins)

    if isempty(predictedComplete)
        predictedComplete = predictDensityStatio(object, distanceXTestTrain,
                                                 zTestMin=0, zTestMax=1, B=length(zGrid),
                                                 probabilityInterval=false, delta=delta)
    end

    colmeansComplete = mean.(eachcol(predictedComplete .^ 2))
    sSquare = mean(colmeansComplete)

    if isempty(predictedObserved)
        predictedObserved = [predictedComplete[i, argmin(abs.(zTest[i] .- zGrid))]
                             for i in 1:length(zTest)]
    end

    likeli = mean(predictedObserved)
    output = Dict("mean" => 0.5 * sSquare - likeli)

    if boot != 0
        # Bootstrap
        bootMeans = [begin
                     sampleBoot = sample(1:length(zTest), length(zTest), replace=true)
                     predictedCompleteBoot = predictedComplete[sampleBoot, :]
                     colmeansCompleteBoot = mean.(eachcol(predictedCompleteBoot .^ 2))
                     sSquareBoot = mean(colmeansCompleteBoot)
                     predictedObservedBoot = [
                         predictedCompleteBoot[i, argmin(
                             abs.(zTest[sampleBoot[i]] .- zGrid))]
                         for i in 1:length(zTest)]
                     likeliBoot = mean(predictedObservedBoot)
                     0.5 * sSquareBoot - likeliBoot
                     end for _ in 1:boot]  # Assuming 100 bootstrap samples
        output["seBoot"] = sqrt(var(bootMeans))
    end

    if add_pred
        return Dict("output" => output,
                    "predictedComplete" => predictedComplete,
                    "predictedObserved" => predictedObserved)
    else
        return output
    end
end

"""
    estimateErrorEstimatorStatio(object, zTest, distanceXTestTrain)

Estimates the error of the estimator in the Statio framework for a given object and test data.

# Arguments
- `object`: The object containing the necessary information for density estimation.
- `zTest`: The test points at which the densities are estimated.
- `distanceXTestTrain`: The distance between the test points and the training data.

# Returns
A modified object dictionary containing the following keys:
- `"nXBest"`: The optimal number of basis functions for X.
- `"nZBest"`: The optimal number of basis functions for Z.
- `"errors"`: The errors for each combination of basis functions.
- `"bestError"`: The minimum error among all combinations.
"""
function estimateErrorEstimatorStatio(object, zTest, distanceXTestTrain)
    kernelNewOld = object["kernelFunction"](distanceXTestTrain, object["extraKernel"])

    if object["normalization"] != nothing
        if object["normalization"] == "symmetric"
            sqrtColMeans = object["normalizationParameters"]["sqrtColMeans"]
            sqrtRowMeans = sqrt.(mean(kernelNewOld, dims=1))
            kernelNewOld = kernelNewOld ./ (sqrtRowMeans' * sqrtColMeans)
        end
    end

    if any(isnan.(kernelNewOld))
        error("Kernel with NA")
    end

    nX = object["nX"]
    nZ = object["nZ"]
    m, n = size(kernelNewOld)

    basisZ = calculateBasis(zTest, nZ, object["system"])
    eigenVectors = object["eigenX"]
    eigenValues = object["eigenValuesX"]

    basisX = kernelNewOld * eigenVectors
    basisX = (1 / n) .* basisX * Diagonal(1 ./ eigenValues)

    basisPsiMean = (1 / m) .* transpose(transpose(basisZ) * basisX)
    W = (1 / m) .* (transpose(basisX) * basisX)

    function compute_prod_matrix(xx)
        m = object["coefficients"][:,xx]
        auxMatrix = W .* (m * transpose(m))
        returnValue = diag(cumsum(transpose(cumsum(auxMatrix, dims=1)), dims=1))
        returnValue[1:nX]
    end
    prodMatrix = [compute_prod_matrix(xx) for xx in 1:nZ]

    D = cumsum(hcat(prodMatrix...), dims=2)

    # Step 1: Create the grid
    mygrid = Iterators.product(1:nX, 1:nZ)

    # Step 2: Apply a function to each element of the grid and reshape the result
    function compute_error(xx)
        sBeta = 1/2 * D[xx[1], xx[2]]
        sLikeli = sum(
            object["coefficients"][1:xx[1],1:xx[2]] .* basisPsiMean[1:xx[1],1:xx[2]])
        return sBeta - sLikeli
    end
    errors = [compute_error(xx) for xx in mygrid]

    nXBest, nZBest = Tuple(argmin(errors))

    object["nXBest"] = nXBest
    object["nZBest"] = nZBest
    object["errors"] = errors
    object["bestError"] = minimum(errors)

    return object
end



"""
    predictDensityStatio(object, zTestMin=0, zTestMax=1, B=1000,
                         distanceXTestTrain, probabilityInterval=false,
                         confidence=0.95, delta=0)

Estimates the density for each bin interval using the K-nearest neighbors method.

# Arguments
- `zTrainNearest`: An array of nearest neighbor values.
- `nBins`: The number of bins to divide the range of values into.
- `zMin`: The minimum value of the range.
- `zMax`: The maximum value of the range.

# Returns
A dictionary containing the estimated means for each bin interval and the bin intervals.

"""
# OK
function predictDensityStatio(object, distanceXTestTrain; zTestMin=0, zTestMax=1,
                              B=1000, probabilityInterval=false, confidence=0.95,
                              delta=0)
    println("Densities normalized to integrate 1 in the range of z given.")

    zGrid = range(zTestMin, zTestMax, length=B)

    kernelNewOld = object["kernelFunction"](distanceXTestTrain, object["extraKernel"])
    if object["normalization"] != nothing
        if object["normalization"] == "symmetric"
            sqrtColMeans = object["normalizationParameters"]["sqrtColMeans"]
            sqrtRowMeans = sqrt.(mean(kernelNewOld, dims=1))
            kernelNewOld = kernelNewOld ./ (sqrtRowMeans' * sqrtColMeans)
        end
    end

    nXBest = object["nXBest"] !== nothing ? object["nXBest"] : object["nX"]
    nZBest = object["nZBest"] !== nothing ? object["nZBest"] : object["nZ"]

    m, n = size(kernelNewOld)
    basisZ = calculateBasis(zGrid, nZBest, object["system"])

    eigenVectors = object["eigenX"][:, 1:nXBest]
    eigenValues = object["eigenValuesX"][1:nXBest]

    basisX = (1 / n) .* (kernelNewOld * eigenVectors) * Diagonal(1 ./ eigenValues)

    function compute_sum(yy, xx, coefficients)
        return sum(yy * xx' .* coefficients)
    end
    estimates = [compute_sum(basisX[i, :], basisZ[j, :],
                            object["coefficients"][1:nXBest, 1:nZBest])
                 for i in 1:size(basisX, 1), j in 1:size(basisZ, 1)]
    # estimates = transpose(estimates)

    binSize = (zTestMax - zTestMin) / (B - 1)
    normalizedEstimates = [normalizeDensity(binSize, estimates[i,:], delta)
                           for i in 1:size(estimates, 1)]
    normalizedEstimates = transpose(hcat(normalizedEstimates...))

    if !probabilityInterval
        return normalizedEstimates
    end

    # Gives threshold on density corresponding to probability interval
    thresholds = [findThreshold(binSize, normalizedEstimates[i, :], confidence)
                  for i in 1:size(normalizedEstimates, 1)]
    objectReturn = Dict("estimates" => normalizedEstimates,
                        "thresholdsIntervals" => thresholds)

    return objectReturn
end

"""
estimate_stratifiedpredictions_Statio

Estimates the stratified predictions based on the given input parameters.

Parameters
----------
object : Any
    The object containing the necessary information for prediction.
zTest : Array
    The array of test values for z.
distanceXTestTrain : Array
    The array of distances between test and train data.
nBins : Int
    The number of bins for the zGrid.
zMin : Float, optional
    The minimum value of z. Default is 0.
zMax : Float, optional
    The maximum value of z. Default is 1.
predictedComplete : Array, optional
    The array of predicted conditional densities. Default is an empty array.
delta : Float, optional
    The delta value for normalization. Default is 0.

Returns
-------
Dict
    A dictionary containing the predicted complete and observed values.

"""
function estimate_stratifiedpredictions_Statio(object, zTest, distanceXTestTrain,
                                               nBins; zMin=0, zMax=1,
                                               predictedComplete=[], delta=0 )
    zGrid = range(zMin, zMax, length = nBins)

    if isempty(predictedComplete)
        # If predicted conditional densities are not given, compute them
        predictedComplete = predictDensityStatio(object, distanceXTestTrain,
                                                 zTestMin=zMin, zTestMax=zMax,
                                                 B=length(zGrid),
                                                 probabilityInterval=false,
                                                 delta=delta)
    end

    # Compute the observed predictions
    predictedObserved = [predictedComplete[i, argmin(abs.(zTest[i] .- zGrid))]
                         for i in 1:length(zTest)]

    return Dict("predictedComplete" => predictedComplete,
                "predictedObserved" => predictedObserved)
end

"""
    condDensityKNNContinuousStatio(zTrainNearest, nBins, bandwidth, zMin, zMax)

Computes the conditional density estimation using the KNN Continuous Statio framework.

# Arguments
- `zTrainNearest`: The nearest neighbors of the test points in the training set.
- `nBins`: The number of bins used for density estimation.
- `bandwidth`: The bandwidth parameter for the kernel function.
- `zMin`: The minimum value of the test points range.
- `zMax`: The maximum value of the test points range.

# Returns
A dictionary containing the following keys:
- `"means"`: The estimated conditional density values at each bin interval.
- `"binsIntervals"`: The bin intervals used for density estimation.
"""
function condDensityKNNContinuousStatio(zTrainNearest, nBins, bandwidth, zMin, zMax)
    zGrid = range(zMin, zMax, length = nBins)

    binsMedium = range(zMin, stop=zMax, length=nBins)
    estimates = [sum(exp.(-abs.(xx .- zTrainNearest).^2 /
                          (4 * bandwidth)) / sqrt(pi * 4 * bandwidth)) /
                 length(zTrainNearest) for xx in binsMedium]

    output = Dict("means" => estimates, "binsIntervals" => binsMedium)
    return output
end

"""
    predictDensityKNN(distanceXTestTrain, zTrain, KNNneighbors, KNNbandwidth, zMin, zMax, nBins; normalization=false, delta=0)

Compute the density estimates for test data using the K-nearest neighbors (KNN) method.

# Arguments
- `distanceXTestTrain`: A matrix of distances between test data and training data. Each row corresponds to a test data point, and each column corresponds to a training data point.
- `zTrain`: An array of training data points.
- `KNNneighbors`: The number of nearest neighbors to consider for density estimation.
- `KNNbandwidth`: The bandwidth parameter for the KNN density estimation.
- `zMin`: The minimum value of the density estimation range.
- `zMax`: The maximum value of the density estimation range.
- `nBins`: The number of bins to divide the density estimation range into.
- `normalization`: (optional) A boolean indicating whether to normalize the density estimates. Default is `false`.
- `delta`: (optional) A small positive value used for density normalization. Default is `0`.

# Returns
- `estimates`: An array of density estimates for each test data point. Each row corresponds to a test data point, and each column corresponds to a bin in the density estimation range.

# Example
```julia
distanceXTestTrain = [0.1 0.2 0.3; 0.4 0.5 0.6]
zTrain = [1.0, 2.0, 3.0]
KNNneighbors = 2
KNNbandwidth = 0.1
zMin = 0.0
zMax = 1.0
nBins = 10

estimates = predictDensityKNN(distanceXTestTrain, zTrain, KNNneighbors, KNNbandwidth, zMin, zMax, nBins)
"""
function predictDensityKNN(distanceXTestTrain, zTrain, KNNneighbors, KNNbandwidth,
                           zMin, zMax, nBins; normalization=false, delta=0)
    zGrid = range(zMin, zMax, length = nBins)
    estimates = Array{Float64}(undef, size(distanceXTestTrain, 1), nBins)

    for i in 1:size(distanceXTestTrain, 1)
        nearest = sortperm(distanceXTestTrain[i, :], alg=QuickSort)[1:KNNneighbors]
        densityObject = condDensityKNNContinuousStatio(zTrain[nearest], nBins,
                                                       KNNbandwidth, zMin, zMax)
        estimates[i, :] = densityObject["means"]
    end

    if normalization
        binSize = (zMax - zMin) / (nBins - 1)
        for i in 1:size(estimates, 1)
            estimates[i, :] = normalizeDensity(binSize, estimates[i, :], delta)
        end
    end

    return estimates
end

"""
    estimateErrorFinalEstimatorKNNContinuousStatio(nNeigh, nBins, bandwidthBinsOpt,
                                                   zMin, zMax, zTrainL,
                                                   distanceXTestTrainL, zTestU;
                                                   boot=0,
                                                   add_pred=false,
                                                   predictedComplete=[],
                                                   predictedObserved=[],
                                                   normalization=false)

Estimates the error of the final estimator using the KNN method for continuous stationary data.

# Arguments
- `nNeigh::Int`: The number of nearest neighbors to consider.
- `nBins::Int`: The number of bins for density estimation.
- `bandwidthBinsOpt`: The bandwidth for density estimation.
- `zMin::Real`: The minimum value of the target variable.
- `zMax::Real`: The maximum value of the target variable.
- `zTrainL::AbstractVector`: The target variable values for the training set.
- `distanceXTestTrainL::AbstractMatrix`: The distance matrix between the test and training set.
- `zTestU::AbstractVector`: The target variable values for the test set.
- `boot::Int`: The number of bootstrap iterations for error estimation. Default is 0.
- `add_pred::Bool`: Whether to include the predicted complete and observed values in the output. Default is false.
- `predictedComplete::AbstractMatrix`: The predicted complete density values. Default is an empty matrix.
- `predictedObserved::AbstractVector`: The predicted observed density values. Default is an empty vector.
- `normalization::Bool`: Whether to normalize the density estimates. Default is false.

# Returns
- If `add_pred` is true, returns a dictionary with the following keys:
    - `"output"`: The estimated error of the final estimator.
    - `"predictedComplete"`: The predicted complete density values.
    - `"predictedObserved"`: The predicted observed density values.
- If `add_pred` is false, returns a dictionary with the following key:
    - `"mean"`: The estimated error of the final estimator.

# Examples
```julia
nNeigh = 5
nBins = 100
bandwidthBinsOpt = 0.1
zMin = 0.0
zMax = 1.0
zTrainL = [0.1, 0.2, 0.3, 0.4, 0.5]
distanceXTestTrainL = [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6]
zTestU = [0.6, 0.7, 0.8, 0.9, 1.0]

result = estimateErrorFinalEstimatorKNNContinuousStatio(nNeigh, nBins, bandwidthBinsOpt,
                                                        zMin, zMax, zTrainL,
                                                        distanceXTestTrainL, zTestU;
                                                        boot=100,
                                                        add_pred=true,
                                                        normalization=true)

println(result["output"])
println(result["seBoot"])
println(result["predictedComplete"])
println(result["predictedObserved"])
"""
function estimateErrorFinalEstimatorKNNContinuousStatio(nNeigh, nBins, bandwidthBinsOpt,
                                                        zMin, zMax, zTrainL,
                                                        distanceXTestTrainL, zTestU;
                                                        boot=0,
                                                        add_pred = false,
                                                        predictedComplete = [],
                                                        predictedObserved = [],
                                                        normalization = false)
    # Predict density using KNN
    if isempty(predictedComplete)
        predictedComplete = predictDensityKNN(distanceXTestTrainL, zTrainL, nNeigh,
                                              bandwidthBinsOpt, zMin, zMax, nBins,
                                              normalization=normalization)
    end

    # Error calculation
    colmeansComplete = mean.(eachcol(predictedComplete .^ 2))
    sSquare = mean(colmeansComplete)

    if isempty(predictedObserved)
        predictedObserved = [predictedComplete[i, argmin(
            abs.(zTestU[i] .- range(zMin, zMax, length = nBins)))]
                             for i in 1:length(zTestU)]
    end

    likeli = mean(predictedObserved)
    output = Dict("mean" => 0.5 * sSquare - likeli)

    if boot != 0
        # Bootstrap for error estimation
        bootMeans = [begin
                      sampleBoot = sample(1:length(zTestU), length(zTestU))
                      predictedCompleteBoot = predictedComplete[sampleBoot, :]
                      colmeansCompleteBoot = mean.(eachcol(predictedCompleteBoot .^ 2))
                      sSquareBoot = mean(colmeansCompleteBoot)
                      predictedObservedBoot = [
                          predictedCompleteBoot[i, argmin(
                              abs.(zTestU[sampleBoot[i]] .-
                                   range(zMin, zMax, length = nBins)))]
                          for i in 1:length(zTestU)]
                      likeliBoot = mean(predictedObservedBoot)
                      0.5 * sSquareBoot - likeliBoot
                      end for _ in 1:boot]
        output["seBoot"] = sqrt(var(bootMeans))
    end
    if add_pred
        return Dict("output" => output,
                    "predictedComplete" => predictedComplete,
                    "predictedObserved" => predictedObserved)
    else
        return output
    end
end

"""
    estimate_stratifiedpredictions_Statio_KNN(nNeigh, nBins, bandwidthBinsOpt, zMin, zMax, zTrain, distanceXTestTrainL, zTest; predictedComplete=nothing, normalization=false)

Estimates stratified predictions using the K-nearest neighbors (KNN) method.

# Arguments
- `nNeigh::Int`: The number of nearest neighbors to consider for each test point.
- `nBins::Int`: The number of bins to divide the range of the target variable.
- `bandwidthBinsOpt::Float64`: The bandwidth parameter for the KNN method.
- `zMin::Float64`: The minimum value of the target variable.
- `zMax::Float64`: The maximum value of the target variable.
- `zTrain::Vector{Float64}`: The target variable values of the training data.
- `distanceXTestTrainL::Matrix{Float64}`: The distances between each test point and the training data.
- `zTest::Vector{Float64}`: The target variable values of the test data.
- `predictedComplete::Matrix{Float64}=nothing`: The predicted densities for each test point and bin. If not provided, the densities will be computed using the KNN method.
- `normalization::Bool=false`: Whether to normalize the predicted densities.

# Returns
- `Dict{String, Any}`: A dictionary containing the following keys:
    - `"predictedComplete"`: The predicted densities for each test point and bin.
    - `"predictedObserved"`: The predicted densities for each test point and the nearest bin.
    - `"zTest"`: The target variable values of the test data.

# Example
```julia
nNeigh = 5
nBins = 10
bandwidthBinsOpt = 0.1
zMin = 0.0
zMax = 1.0
zTrain = [0.1, 0.2, 0.3, 0.4, 0.5]
distanceXTestTrainL = [0.1 0.2 0.3 0.4 0.5; 0.2 0.3 0.4 0.5 0.6]
zTest = [0.15, 0.25]
predictedComplete = estimate_stratifiedpredictions_Statio_KNN(nNeigh, nBins, bandwidthBinsOpt, zMin, zMax, zTrain, distanceXTestTrainL, zTest)
"""
function estimate_stratifiedpredictions_Statio_KNN(nNeigh, nBins, bandwidthBinsOpt,
                                                   zMin, zMax, zTrain,
                                                   distanceXTestTrainL, zTest;
                                                   predictedComplete=nothing,
                                                   normalization=false)
    # Check if predicted densities are provided
    if predictedComplete === nothing
        predictedComplete = predictDensityKNN(distanceXTestTrainL, zTrain, nNeigh,
                                              bandwidthBinsOpt, zMin, zMax, nBins,
                                              normalization=normalization)
    end

    # Stratified prediction
    zGrid = range(zMin, zMax, length = nBins)
    predictedObserved = [predictedComplete[i, argmin(abs.(zTest[i] .- zGrid))]
                         for i in 1:length(zTest)]

    # Output construction
    return Dict("predictedComplete" => predictedComplete,
                "predictedObserved" => predictedObserved,
                "zTest" => zTest)
end


function estimate_combined_stratified_risk_Statio_KNN(predictedComplete,
                                                      predictedObserved,
                                                      zTestU_ordered,
                                                      zMin, zMax, nBins;
                                                      boot=0)
    # Risk calculation
    colmeansComplete = mean.(eachcol(predictedComplete .^ 2))
    sSquare = mean(colmeansComplete)
    likeli = mean(predictedObserved)
    output = Dict("meanRisk" => 0.5 * sSquare - likeli)

    if boot != 0
        # Bootstrap for risk estimation
        bootRisks = [begin
                     sampleBoot = sample(1:length(zTestU_ordered),
                                         length(zTestU_ordered), replace=true)
                     predictedCompleteBoot = predictedComplete[sampleBoot, :]
                     colmeansCompleteBoot = mean.(eachcol(predictedCompleteBoot .^ 2))
                     sSquareBoot = mean(colmeansCompleteBoot)
                     predictedObservedBoot = predictedObserved[sampleBoot]
                     likeliBoot = mean(predictedObservedBoot)
                     0.5 * sSquareBoot - likeliBoot
                     end for _ in 1:boot]
        output["seBoot"] = std(bootRisks)
    end
    return output
end


function comb_test_loss_fct(predictedComplete, zTestU, zMin, zMax, nBins; boot=0)
    zGrid = range(zMin, zMax, length = nBins)

    colmeansComplete = mean.(eachcol(predictedComplete .^ 2))
    sSquare = mean(colmeansComplete)

    # Compute predictedObserved
    n = length(zTestU)
    predictedObserved = [predictedComplete[i, argmin(abs.(zTestU[i] .- zGrid))]
                         for i in 1:n]

    likeli = mean(predictedObserved)
    output = Dict("mean" => 0.5 * sSquare - likeli)

    if boot == 0
        return output
    else
        println("Bootstrap error is computed")
        bootMeans = [begin
                     sampleBoot = sample(1:n, n, replace=true)
                     predictedCompleteBoot = predictedComplete[sampleBoot, :]
                     colmeansCompleteBoot = mean.(eachcol(predictedCompleteBoot .^ 2))
                     sSquareBoot = mean(colmeansCompleteBoot)
                     predictedObservedBoot = [
                         predictedCompleteBoot[j, argmin(
                             abs.(zTestU[sampleBoot[j]] .- zGrid))]
                         for j in 1:n]
                     likeliBoot = mean(predictedObservedBoot)
                     0.5 * sSquareBoot - likeliBoot
                     end for _ in 1:boot]  # Assuming 100 bootstrap samples
        output["seBoot"] = sqrt(var(bootMeans))
    end

    return output
end
