# Import necessary packages
using CSV
using DataFrames
using Dates
using Random

using Statistics
using Distances

using GLM
using StatsBase
using Plots
using PrettyTables

using Serialization

# Include external functions defined in other files:
include("../src/balance_check.jl")
include("../src/conditional_density.jl")
include("../src/conditional_density_CS.jl")
# include("estimate_weights.jl")

# Set working directory (if needed)
# cd("./") # Uncomment and set the path if you need to change the working directory

function main()
    # Delete old results and create new result folder
    rm("summary_results", recursive=true, force=true) # Remove the directory if it exists
    mkdir("summary_results")

    # Load data: first nr_covariates columns are the covariates (r-band + colors),
    # last columns is spectroscopic redshift
    data_spectro_raw = CSV.read("../data/train.csv", DataFrame)
    data_photo_raw = CSV.read("../data/test.csv", DataFrame)

    # Script parameters
    nL = size(data_spectro_raw, 1) # nr of source samples used
    nU = size(data_photo_raw, 1) # nr of target samples used

    nr_groups = 5
    first_test_group = 1

    selected_seed = 2
    add_comment = ""

    # Random sampling (if needed)
    # data_spectro_raw = data_spectro_raw[rand(1:size(data_spectro_raw, 1), nL), :]
    # data_photo_raw = data_photo_raw[rand(1:size(data_photo_raw, 1), nU), :]

    # REPLACE covariate_names with the names of your covariates (colors and r-band)
    nr_covariates = 6
    covariate_names = names(data_photo_raw)[2:(nr_covariates + 1)]

    # Only keep the covariates and the spectroscopic redshift
    uniqueID_spectro = data_spectro_raw[:, 1]
    data_spectro = data_spectro_raw[:, [covariate_names; "Z"]]

    uniqueID_photo = data_photo_raw[:, 1]
    data_photo = data_photo_raw[:, [covariate_names; "Z"]]

    # Additional file parameters (mostly default used)
    nr_fzxbins = 201 # careful, this is the nr of boundaries including start and end
    z_scaling_type = "min_max_spectro_z" # "min_max_z", "min_max_spectro_z", "fixed"

    hyperparam_selection = "grid_search"
    time_stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS_")
    Random.seed!(selected_seed)
    out_comment = time_stamp * add_comment

    result_folder_path = "summary_results/"

    # Load beforehand optimized hyperparameters
    # (this is only needed if hyperparam_selection == "fixed")
    # "grid_search", "fixed" (when fixed, then the values in eps/delta_per_strata_fixed
    # are used as hyperparameters)
    hyperparam_file_name_tmp = ""
    time_stamp_hyperparams_file = ""
    hyperparam_file_name = hyperparam_file_name_tmp # Adjust as needed

    # These values used if hyperparam_selection = "fixed".
    # Define values e.g. via previous hyperparameter optimization on smaller data
    eps_per_strata_fixed = [NaN, NaN, NaN, NaN, NaN]
    nXBest_J_per_strata_fixed = [NaN, NaN, NaN, NaN, NaN]
    nZBest_I_per_strata_fixed = [NaN, NaN, NaN, NaN, NaN]
    delta_per_strata_fixed = [NaN, NaN, NaN, NaN, NaN]
    bandwidth_per_strata_fixed = [NaN, NaN, NaN, NaN, NaN]
    nearestneighbors_per_strata_fixed = [NaN, NaN, NaN, NaN, NaN]
    alpha_per_strata_fixed = [NaN, NaN, NaN, NaN, NaN]

    hyperparam_names = ["epsilon", "delta", "nXbest_J", "nZbest_I", "bandwidth",
                        "nearestNeighbors", "alpha"]
    stored_hyperparams_per_strata = DataFrame(
        epsilon = eps_per_strata_fixed,
        delta = delta_per_strata_fixed,
        nXbest_J = nXBest_J_per_strata_fixed,
        nZbest_I = nZBest_I_per_strata_fixed,
        bandwidth = bandwidth_per_strata_fixed,
        nearestNeighbors = nearestneighbors_per_strata_fixed,
        alpha = alpha_per_strata_fixed
    )

    if hyperparam_selection == "fixed"
        # Load the hyperparameters from a file (adjust the file loading as needed)
        stored_hyperparams_per_strata = load(result_folder_path *
                                             time_stamp_hyperparams_file * "/" *
                                             hyperparam_file_name * ".rds", convert=true)
        println("Hyperparameter file: $hyperparam_file_name is loaded!")
    end

    # Adjust column names if needed
    # rename!(stored_hyperparams_per_strata, hyperparam_names)

    # Obtain train and validation subsets
    nTrainL = round(Int, 0.5 * nL)
    nValidationL = nL - nTrainL

    nValidationU = round(Int, 0.5 * nU)
    nTestU = nU - nValidationU

    # Add train indicator
    data_spectro[!, :train] = ones(Int64, nrow(data_spectro))
    data_photo[!, :train] = zeros(Int64, nrow(data_photo))

    # Combine data
    data_full = vcat(data_spectro, data_photo)

    # Scale z to be between 0 and 1
    zBeforeScale = data_full[!, "Z"];
    if z_scaling_type == "min_max_z"
        zMin, zMax = extrema(zBeforeScale)
    elseif z_scaling_type == "min_max_spectro_z"
        zMin, zMax = extrema(zBeforeScale[1:nL])
    elseif z_scaling_type == "fixed"
        zMin, zMax = (0.06, 1.5)
    end
    z = (zBeforeScale .- zMin) ./ (zMax - zMin)
    println(zMax)

    # Grid to evaluate the conditional densities fzx on
    zGrid = range(0, 1, length=nr_fzxbins)
    rescaled_zGrid = zGrid .* (zMax - zMin) .+ zMin

    # Covariates used for computation, selected from full data frame
    # and rescaled to have mean 0 and std 1
    covariates = combine(transform(
        data_full, covariate_names .=> (col -> (col .- mean(col)) ./ std(col)) .=> covariate_names), covariate_names)

    covariatesL = covariates[1:nL, :]
    covariatesU = covariates[(nL+1):end, :]
zL = z[1:nL]
zU = z[(nL+1):end]

rescaled_zU = zU .* (zMax - zMin) .+ zMin;
rescaled_zL = zL .* (zMax - zMin) .+ zMin;

# Split L Sample
randomPermL = collect(1:nL) # shuffle(1:nL) !!!!
idx_TrainL = randomPermL[1:nTrainL]
idx_ValidationL = randomPermL[(nTrainL+1):end]
covariatesTrainL = Matrix(covariatesL[idx_TrainL, :])
covariatesValidationL = Matrix(covariatesL[idx_ValidationL, :])
zTrainL = zL[idx_TrainL]
zValidationL = zL[idx_ValidationL]

# Split U Sample
randomPermU = collect(1:nU) #shuffle(1:nU) !!!
idx_ValidationU = randomPermU[1:nValidationU]
idx_TestU = randomPermU[(nValidationU+1):end]
covariatesValidationU = Matrix(covariatesU[idx_ValidationU, :])
covariatesTestU = Matrix(covariatesU[idx_TestU, :])
zValidationU = zU[idx_ValidationU]
zTestU = zU[idx_TestU]

# Distances L
distanceXTrainL_TrainL = pairwise(Euclidean(),
                                  covariatesTrainL,
                                  covariatesTrainL, dims=1)
distanceXValidationL_TrainL = pairwise(Euclidean(),
                                       covariatesValidationL,
                                       covariatesTrainL, dims=1)
distanceXValidationU_TrainL = pairwise(Euclidean(),
                                       covariatesValidationU,
                                       covariatesTrainL, dims=1)
distanceXValidationL_ValidationL = pairwise(Euclidean(),
                                            covariatesValidationL,
                                            covariatesValidationL, dims=1)

# Distances U to L
distanceXValidationU_TrainL = pairwise(Euclidean(),
                                       covariatesValidationU,
                                       covariatesTrainL, dims=1)
distanceXTestU_TrainL = pairwise(Euclidean(),
                                 covariatesTestU,
                                 covariatesTrainL, dims=1)

################################################################################
# Beginning of the StratLearn part

# Estimate the propensity scores

all_data = covariates;
all_data[!, :Z] = data_full.Z;
all_data[!, :source_ind] = data_full.train;

# Logistic regression
formula = term(:source_ind) ~ sum(term.(Tuple(covariate_names)))
lreg_PS = glm(formula, select(all_data, Not(:Z)), Binomial(), LogitLink())

# Extract fitted values (propensity scores)
PS = predict(lreg_PS)

# Assign samples to strata
temp = invperm(sortperm(PS))
grp_size = length(all_data[:, 1]) / nr_groups
all_data[!, "group"] = fill(NaN, length(temp))
for k in 1:nr_groups
    all_data[temp .<= grp_size * (nr_groups - k + 1), "group"] .= k
end

# Plotting histograms of propensity scores
p1 = histogram(PS[1:nL], bins=50, normed=true, alpha=0.5,
               label="Source propensity score")
p2 = histogram(PS[(nL + 1):(nL + nU)], bins=50, normed=true, alpha=0.5,
               label="Target propensity score", legend=:topright)
plot(p1, p2, xlim=(0, 1), title="Hist: Estimated propensity scores (source vs. target) resampled data", xlabel="Propensity score")

# Save the plot
savefig("$(result_folder_path)/PS_distribution$(out_comment)_scaled_seed$(selected_seed)_$(time_stamp).pdf")

# Check proportions in each strata

# !!! could be improved by using `grouped_df = groupby(all_data, :group)`?
group_proportions = DataFrame("group" => Int64[], "# samples" => Int64[],
                              "mean redshift" => Float64[], "subset" => String[])
for s in 1:nr_groups
    push!(group_proportions,
          (s,length(all_data[(all_data.source_ind .== 1) .& (all_data.group .== s), :Z]),
           mean(all_data[(all_data.source_ind .== 1) .& (all_data.group .== s), :Z]),
           "source"))
    push!(group_proportions,
          (s,length(all_data[(all_data.source_ind .== 0) .& (all_data.group .== s), :Z]),
           mean(all_data[(all_data.source_ind .== 0) .& (all_data.group .== s), :Z]),
           "target"))
end
push!(group_proportions,
      (0, length(all_data[all_data.source_ind .== 1, :Z]),
       mean(all_data[all_data.source_ind .== 1, :Z]), "source"))
push!(group_proportions,
      (0, length(all_data[all_data.source_ind .== 0, :Z]),
       mean(all_data[all_data.source_ind .== 0, :Z]), "target"))

# Save group proportions to CSV, also export in LaTex format
CSV.write("$(result_folder_path)/group_proportions_K_$(nr_groups)_$(out_comment)$(time_stamp).csv", group_proportions)
f = open("$(result_folder_path)/group_proportions_K_$(nr_groups)_$(out_comment)$(time_stamp).tex", "w")
pretty_table(f, group_proportions, backend=Val(:latex))
close(f)
###

# Compute the covariate balance within strata

# Compute the standardized mean differences for raw data and within all strata
smd_raw = balance_ingroups_fct(all_data, covariate_names, 
                               use_group = [true, true, true, true, true],
                               indicator_var = "source_ind", measure = "smd")

smd_all = []
push!(smd_all, balance_ingroups_fct(all_data, covariate_names, 
                                    use_group = [true, false, false, false, false],
                                    indicator_var = "source_ind", measure = "smd"))
push!(smd_all, balance_ingroups_fct(all_data, covariate_names, 
                                    use_group = [false, true, false, false, false],
                                    indicator_var = "source_ind", measure = "smd"))
push!(smd_all, balance_ingroups_fct(all_data, covariate_names, 
                                    use_group = [false, false, true, false, false],
                                    indicator_var = "source_ind", measure = "smd"))
push!(smd_all, balance_ingroups_fct(all_data, covariate_names, 
                                    use_group = [false, false, false, true, false],
                                    indicator_var = "source_ind", measure = "smd"))
push!(smd_all, balance_ingroups_fct(all_data, covariate_names, 
                                    use_group = [false, false, false, false, true],
                                    indicator_var = "source_ind", measure = "smd"))

# Compute Kolmogorov-Smirnov test statistics for raw data and within all strata
ks_raw = balance_ingroups_fct(all_data, covariate_names, 
                              use_group = [true, true, true, true, true], 
                              indicator_var = "source_ind",
                              measure = "ks-statistics")

ks_all = []
push!(ks_all, balance_ingroups_fct(all_data, covariate_names, 
                                   use_group = [true, false, false, false, false], 
                                   indicator_var = "source_ind",
                                   measure = "ks-statistics"))
push!(ks_all, balance_ingroups_fct(all_data, covariate_names, 
                                   use_group = [false, true, false, false, false], 
                                   indicator_var = "source_ind",
                                   measure = "ks-statistics"))
push!(ks_all, balance_ingroups_fct(all_data, covariate_names, 
                                   use_group = [false, false, true, false, false], 
                                   indicator_var = "source_ind",
                                   measure = "ks-statistics"))
push!(ks_all, balance_ingroups_fct(all_data, covariate_names, 
                                   use_group = [false, false, false, true, false], 
                                   indicator_var = "source_ind",
                                   measure = "ks-statistics"))
push!(ks_all, balance_ingroups_fct(all_data, covariate_names, 
                                   use_group = [false, false, false, false, true], 
                                   indicator_var = "source_ind",
                                   measure = "ks-statistics"))

#-------------------------------------------------------------------------------
# !!! This is not working properly, needs to be fixed 
# Plot smd
# plot_balance_evaluation(smd_all, smd_raw,
#                             balance_measure = "smd", 
#                             result_folder = result_folder_path * "/")

# # Plot ks-statistics
# plot_balance_evaluation(ks_all,ks_raw,
#                         balance_measure = "ks-statistics",
#                         result_folder = result_folder_path * "/")

# Save results
serialize(result_folder_path * "/balance_results_" * out_comment * "seed" *
          string(selected_seed) * "_" * time_stamp * ".jls", [smd_all, ks_all])

#-------------------------------------------------------------------------------
# PS estimation is done. Now compute conditional densities on each stratum

# Initialize lists for train and validation strata
train_strata = [[1], [2], [3], [4], [5]]

# Note that for now the validation set is always only from the same strata as
# the test set (not combined)
val_strata = [[1], [2], [3], [4], [5]]

# Initialize variables for storing results
sta_finallossAdaptive = []
sta_predictedComplete = []
sta_predictedObserved = []
sta_ordered_zTestU = Float64[] # only to track testU strata sizes
sta_ordered_zU = Float64[]
sta_ordered_uniqueID_photo = Float64[]
sta_ordered_photo_indices = Float64[]

sta_zTestU_size = Float64[]
sta_zU_size = Float64[]

# Initialize lists for validation results
sta_validationL_finallossAdaptive = []
sta_validationL_predictedComplete_Adaptive = []
sta_validationL_ordered_z = []

sta_validationU_finallossAdaptive = []
sta_validationU_predictedComplete_Adaptive = []
sta_validationU_ordered_z = []

# Series hyperparameters
optimal_hyperparams = fill(NaN, (nr_groups, length(hyperparam_names)))
optimal_hyperparams = DataFrame(optimal_hyperparams,
                                ["epsilon", "delta", "nXBest_J", "nZbest_I",
                                 "bandwidth", "nearestNeighbors", "alpha"])

for stratum in first_test_group:nr_groups
    # Compute distance matrices for each stratum
    idx_train_stratum = findall(x -> x in train_strata[stratum],
                                all_data.group[idx_TrainL]);
    idx_val_stratum = findall(x -> x in val_strata[stratum],
                              all_data.group[idx_ValidationL]);
    idx_test_stratum = findall(x -> x == stratum,
                               all_data.group[(nL + 1):(nL + nU)][idx_TestU]);
    idx_valU_stratum = findall(x -> x == stratum,
                               all_data.group[(nL + 1):(nL + nU)][idx_ValidationU]);

    sta_distanceXTrainL_TrainL = distanceXTrainL_TrainL[idx_train_stratum,
                                                        idx_train_stratum];
    sta_distanceXValidationL_TrainL = pairwise(Euclidean(),
                                               covariatesValidationL[idx_val_stratum, :],
                                               covariatesTrainL[idx_train_stratum, :],
                                               dims=1);
    sta_distanceXTestU_TrainL = pairwise(Euclidean(),
                                         covariatesTestU[idx_test_stratum, :],
                                         covariatesTrainL[idx_train_stratum, :], dims=1);
    sta_distanceXValidationU_TrainL = pairwise(Euclidean(),
                                               covariatesValidationU[idx_valU_stratum,:],
                                               covariatesTrainL[idx_train_stratum, :],
                                               dims=1);

    # Subsetting the outcome (redshift) according to strata
    sta_zTrainL = zTrainL[idx_train_stratum];
    sta_zValidationL = zValidationL[idx_val_stratum];
    sta_zTestU = zTestU[idx_test_stratum];
    sta_zValidationU = zValidationU[idx_valU_stratum];

    # Final training L and testing sets U
    idx_L_stratum = findall(x -> x in train_strata[stratum], all_data.group[1:nL]);
    idx_U_stratum = findall(x -> x == stratum, all_data.group[(nL + 1):(nL + nU)]);
    sta_distanceXL_L = pairwise(Euclidean(),
                                Matrix(covariatesL[idx_L_stratum, :]),
                                Matrix(covariatesL[idx_L_stratum, :]), dims=1);
    sta_distanceXU_L = pairwise(Euclidean(),
                                Matrix(covariatesU[idx_U_stratum, :]),
                                Matrix(covariatesL[idx_L_stratum, :]), dims=1);

    # Final training L and testing U redshift
    sta_zL = zL[idx_L_stratum];
    sta_zU = zU[idx_U_stratum];
    sta_photo_indices = findall(x -> x == stratum, all_data.group[(nL + 1):(nL + nU)]);
    sta_uniqueID_photo = uniqueID_photo[idx_U_stratum];

    # Keep track of the size of each unlabelled test set used in each strata
    push!(sta_zTestU_size, length(sta_zTestU));
    push!(sta_zU_size, length(sta_zU));

    # StratLearn Stationary Adaptive model fitting and prediction
    # This is the series estimator (Izbicki2017), applied via StratLearn within
    # each stratum
    print("Fit StratLearn Stationary Adaptive (Series cond. density estimator)\n")

    # Initialize variables
    epsGrid = range(0.05, stop=0.4, length=7);
    myerror = fill(NaN, length(epsGrid));

    if hyperparam_selection == "grid_search"
        for (ii, myeps) in enumerate(epsGrid)
            println(ii / length(epsGrid))
            
            # Function calls (assuming these functions are defined in the external file)
            object = condDensityStatio(sta_distanceXTrainL_TrainL,
                                       sta_zTrainL,
                                       nZMax=70,
                                       kernel_function=radialKernelDistance,
                                       extra_kernel= Dict("eps.val" => myeps),
                                       normalization=nothing,
                                       system="Fourier")
            object = estimateErrorEstimatorStatio(object,
                                                  sta_zValidationL,
                                                  sta_distanceXValidationL_TrainL)
            myerror[ii] = object["bestError"]
        end
        myeps = epsGrid[argmin(myerror)]

    elseif hyperparam_selection == "fixed"
        myeps = stored_hyperparams_per_strata["epsilon"][stratum]
    end

    if hyperparam_selection == "grid_search"
        sta_objectStationaryAdaptive_delta = condDensityStatio(
            sta_distanceXTrainL_TrainL,
            sta_zTrainL,
            nZMax=70,
            kernel_function = radialKernelDistance,
            extra_kernel = Dict("eps.val" => myeps),
            normalization = nothing,
            system = "Fourier")
        sta_objectStationaryAdaptive_delta = estimateErrorEstimatorStatio(
            sta_objectStationaryAdaptive_delta,
            sta_zValidationL,
            sta_distanceXValidationL_TrainL)

        sta_bestDeltaStationaryAdaptive = chooseDeltaStatio(
            sta_objectStationaryAdaptive_delta,
            sta_zValidationL,
            sta_distanceXValidationL_TrainL,
            range(0, stop=0.4, step=0.025),
            nr_fzxbins)

        delta_best = sta_bestDeltaStationaryAdaptive["bestDelta"]
        optimal_hyperparams[stratum, "delta"] = delta_best
        optimal_hyperparams[stratum, "epsilon"] = myeps

        nXBest_J = sta_objectStationaryAdaptive_delta["nXBest"]
        nZBest_I = sta_objectStationaryAdaptive_delta["nZBest"]
        optimal_hyperparams[stratum, "nXBest_J"] = nXBest_J
        optimal_hyperparams[stratum, "nZbest_I"] = nZBest_I
        sta_validationL_stratified_predictions_Adaptive =
            estimate_stratifiedpredictions_Statio(sta_objectStationaryAdaptive_delta,
                                                  sta_zValidationL,
                                                  sta_distanceXValidationL_TrainL,
                                                  nr_fzxbins,
                                                  delta=delta_best,
                                                  zMin = 0, zMax = 1)
        push!(sta_validationL_predictedComplete_Adaptive,
              sta_validationL_stratified_predictions_Adaptive["predictedComplete"])
        push!(sta_validationL_ordered_z, sta_zValidationL)

        push!(sta_validationL_finallossAdaptive,
              estimateErrorFinalEstimatorStatio(
                  sta_objectStationaryAdaptive_delta,
                  sta_zValidationL,
                  sta_distanceXValidationL_TrainL,
                  nr_fzxbins,
                  boot=400,
                  delta=delta_best,
                  zMin = 0, zMax = 1,
                  predictedComplete = sta_validationL_stratified_predictions_Adaptive["predictedComplete"],
                  predictedObserved = sta_validationL_stratified_predictions_Adaptive["predictedObserved"]))

        ## labeled validation predictions and loss (on strata)
        sta_validationU_stratified_predictions_Adaptive =
            estimate_stratifiedpredictions_Statio(sta_objectStationaryAdaptive_delta,
                                                  sta_zValidationU,
                                                  sta_distanceXValidationU_TrainL,
                                                  nr_fzxbins,
                                                  delta=delta_best,
                                                  zMin = 0, zMax = 1)
        push!(sta_validationU_predictedComplete_Adaptive,
              sta_validationU_stratified_predictions_Adaptive["predictedComplete"])
        push!(sta_validationU_ordered_z, sta_zValidationU)
        push!(sta_validationU_finallossAdaptive,
              estimateErrorFinalEstimatorStatio(
                  sta_objectStationaryAdaptive_delta,
                  sta_zValidationU,
                  sta_distanceXValidationU_TrainL,
                  nr_fzxbins,
                  boot=400,
                  delta=delta_best,
                  zMin = 0,
                  zMax = 1,
                  predictedComplete = sta_validationU_stratified_predictions_Adaptive["predictedComplete"],
                  predictedObserved = sta_validationU_stratified_predictions_Adaptive["predictedObserved"]))
    elseif hyperparam_selection == "fixed"
        delta_best = stored_hyperparams_per_strata.delta[stratum]
        nXBest_J = stored_hyperparams_per_strata.nXbest_J[stratum] 
        nZBest_I = stored_hyperparams_per_strata.nZbest_I[stratum]
    end
# !!! indentation???
sta_objectStationaryAdaptive = condDensityStatio(sta_distanceXL_L,
                                                 sta_zL,
                                                 nZMax=70,
                                                 kernel_function=radialKernelDistance,
                                                 extra_kernel=Dict("eps.val" => myeps),
                                                 normalization=nothing,
                                                 system="Fourier")
sta_objectStationaryAdaptive["nXBest"] = nXBest_J
sta_objectStationaryAdaptive["nZBest"] = nZBest_I

predict_complete_U = predictDensityStatio(sta_objectStationaryAdaptive,
                                          sta_distanceXU_L,
                                          zTestMin=0,
                                          zTestMax=1,
                                          B=nr_fzxbins,
                                          probabilityInterval=false,
                                          delta=delta_best)

## compute the stratified predictions to merge them together later, to get the combined loss 
sta_stratified_predictions =
    estimate_stratifiedpredictions_Statio(sta_objectStationaryAdaptive,
                                          sta_zU,
                                          sta_distanceXU_L,
                                          nr_fzxbins,
                                          delta=delta_best,
                                          zMin = 0,
                                          zMax = 1,
                                          predictedComplete = predict_complete_U)
# concatenate the predictions for the test sets of each stratum,
# leading to one set of test set preds. in StratLearn order
push!(sta_predictedComplete, predict_complete_U)
push!(sta_predictedObserved, sta_stratified_predictions["predictedObserved"])

# concatenate the true redshift in the order of the StratLearn predictions for the test sets of each strata
sta_ordered_zU = [sta_ordered_zU; sta_zU]
sta_ordered_zTestU = [sta_ordered_zTestU; sta_zTestU]
sta_ordered_uniqueID_photo = [sta_ordered_uniqueID_photo; sta_uniqueID_photo]
sta_ordered_photo_indices = [sta_ordered_photo_indices; sta_photo_indices]

if hyperparam_selection == "grid_search"
    push!(sta_finallossAdaptive,
          estimateErrorFinalEstimatorStatio(
              sta_objectStationaryAdaptive,
              sta_zU,
              sta_distanceXU_L,
              nr_fzxbins,
              boot=400,
              delta=delta_best, zMin = 0, zMax = 1,
              predictedComplete = predict_complete_U,
              predictedObserved = sta_stratified_predictions["predictedObserved"]))
end
end

# Save hyperparameters if optimized via grid search
if hyperparam_selection == "grid_search"
    new_hyperparams_path_name =
        joinpath(result_folder_path,
                 "optimal_hyperparams_$(out_comment)_seed$(selected_seed)_$(time_stamp).rds")
    serialize(new_hyperparams_path_name, optimal_hyperparams)
end

# Calculate combined loss if grid search was performed
if hyperparam_selection == "grid_search"
    sta_finallossAdaptive[nr_groups + 1] =
        estimate_combined_stratified_risk_Statio(
            sta_predictedComplete,
            sta_predictedObserved,
            nr_fzxbins,
            sta_ordered_zU,
            boot=400,
            zMin=0, zMax=1)
end


#-------------------------------- KNN ------------------------------------------
# Initialize variables for storing KNN results
sta_finallossKNN = []
sta_predictedComplete_KNN = []
sta_predictedObserved_KNN = []

sta_validationL_finallossKNN = []
sta_validationL_predictedComplete_KNN = []
sta_validationU_finallossKNN = []
sta_validationU_predictedComplete_KNN = []

# Placeholder for optimal hyperparameters
optimal_hyperparams = DataFrame(epsilon = NaN, delta = NaN,nXbest_J = NaN,
                                nZbest_I = NaN,
                                bandwidth = NaN, nearestNeighbors = NaN, alpha = NaN)

for stratum in first_test_group:nr_groups
    ### Train and validation subsets
    # Distances L
    
    idx_train_stratum = findall(x -> x in train_strata[stratum],
                                all_data.group[idx_TrainL])
    idx_val_stratum = findall(x -> x in val_strata[stratum],
                              all_data.group[idx_ValidationL])
    idx_test_stratum = findall(x -> x == stratum,
                               all_data.group[(nL + 1):(nL + nU)][idx_TestU])
    idx_valU_stratum = findall(x -> x == stratum,
                               all_data.group[(nL + 1):(nL + nU)][idx_ValidationU])


    sta_distanceXTrainL_TrainL = distanceXTrainL_TrainL[idx_train_stratum,
                                                        idx_train_stratum]
    sta_distanceXValidationL_TrainL = pairwise(Euclidean(),
                                               covariatesValidationL[idx_val_stratum, :],
                                               covariatesTrainL[idx_train_stratum, :],
                                               dims=1)
    sta_distanceXTestU_TrainL = pairwise(Euclidean(),
                                         covariatesTestU[idx_test_stratum, :],
                                         covariatesTrainL[idx_train_stratum, :], dims=1)
    sta_distanceXValidationU_TrainL = pairwise(Euclidean(),
                                               covariatesValidationU[idx_valU_stratum,:],
                                               covariatesTrainL[idx_train_stratum, :],
                                               dims=1)

    # Subsetting the outcome (redshift) according to strata
    sta_zTrainL = zTrainL[idx_train_stratum]
    sta_zValidationL = zValidationL[idx_val_stratum]
    sta_zTestU = zTestU[idx_test_stratum]
    sta_zValidationU = zValidationU[idx_valU_stratum]
    
    # Final training L and testing sets U
    idx_L_stratum = findall(x -> x in train_strata[stratum], all_data.group[1:nL])
    idx_U_stratum = findall(x -> x == stratum, all_data.group[(nL + 1):(nL + nU)])
    sta_distanceXL_L = pairwise(Euclidean(),
                                Matrix(covariatesL[idx_L_stratum, :]),
                                Matrix(covariatesL[idx_L_stratum, :]), dims=1)
    sta_distanceXU_L = pairwise(Euclidean(),
                                Matrix(covariatesU[idx_U_stratum, :]),
                                Matrix(covariatesL[idx_L_stratum, :]), dims=1)

    # Final training L and testing U redshift
    sta_zL = zL[idx_L_stratum]
    sta_zU = zU[idx_U_stratum]

    ### KNN hyperparameter optimization
    print("Fit StratLearn KNN (Continuous) (KNN estimator in paper with StratLearn)")

    if hyperparam_selection == "grid_search"
        bandwidthsVec = range(0.0000001, 0.002, length=10)
        nNeighbours = Int.(round.(range(2, 30, length=10)))
        sta_lossKNNBinned = fill(NaN, (length(bandwidthsVec),length(nNeighbours)))

        for ii in 1:length(bandwidthsVec)
            println(ii / length(bandwidthsVec))
            for jj in 1:length(nNeighbours)
                sta_lossKNNBinned[ii,jj] =
                    estimateErrorFinalEstimatorKNNContinuousStatio(
                        nNeighbours[jj],
                        nr_fzxbins,
                        bandwidthsVec[ii],
                        0,
                        1,
                        sta_zTrainL,
                        sta_distanceXValidationL_TrainL,
                        sta_zValidationL, boot=false)["mean"]
            end
        end
        pointMin = findmin(sta_lossKNNBinned)
        sta_bestBandwidthKNN=(bandwidthsVec)[pointMin[2][1]] # 0.0002223111
        sta_bestKNNDensity=(nNeighbours)[pointMin[2][2]] # 10
        
        ## store optimal hyperparameters
        optimal_hyperparams[stratum,"bandwidth"] = sta_bestBandwidthKNN
        optimal_hyperparams[stratum,"nearestNeighbors"] = sta_bestKNNDensity
        
        ### Validation set predictions:
        ### labeled validation predictions and loss (on strata)
        sta_validationL_stratified_predictions_KNN =
            estimate_stratifiedpredictions_Statio_KNN(
                sta_bestKNNDensity,
                nr_fzxbins,
                sta_bestBandwidthKNN,
                0,
                1,
                sta_zTrainL,
                sta_distanceXValidationL_TrainL,
                sta_zValidationL, normalization = true)
        
        push!(sta_validationL_predictedComplete_KNN,
              sta_validationL_stratified_predictions_KNN["predictedComplete"])

        push!(sta_validationL_finallossKNN,
              estimateErrorFinalEstimatorKNNContinuousStatio(
                  sta_bestKNNDensity,
                  nr_fzxbins,
                  sta_bestBandwidthKNN,
                  0,
                  1,
                  sta_zTrainL,
                  sta_distanceXValidationL_TrainL,
                  sta_zValidationL,
                  boot=400,
                  add_pred = false,
                  predictedComplete = sta_validationL_stratified_predictions_KNN["predictedComplete"],
                  predictedObserved = sta_validationL_stratified_predictions_KNN["predictedObserved"]))

        # unlabeled validation predictions and loss (on strata)
        sta_validationU_stratified_predictions_KNN =
            estimate_stratifiedpredictions_Statio_KNN(
                sta_bestKNNDensity,
                nr_fzxbins,
                sta_bestBandwidthKNN,
                0,
                1,
                sta_zTrainL,
                sta_distanceXValidationU_TrainL,
                sta_zValidationU, normalization = true)
        
        push!(sta_validationU_predictedComplete_KNN,
              sta_validationU_stratified_predictions_KNN["predictedComplete"])
        
        push!(sta_validationU_finallossKNN,
              estimateErrorFinalEstimatorKNNContinuousStatio(
                  sta_bestKNNDensity,
                  nr_fzxbins,
                  sta_bestBandwidthKNN,
                  0,
                  1,
                  sta_zTrainL,
                  sta_distanceXValidationU_TrainL,
                  sta_zValidationU,
                  boot=400,
                  add_pred = false,
                  predictedComplete = sta_validationU_stratified_predictions_KNN["predictedComplete"],
                  predictedObserved = sta_validationU_stratified_predictions_KNN["predictedObserved"]))
        GC.gc() 
        
        
    elseif hyperparam_selection == "fixed"
        sta_bestBandwidthKNN = stored_hyperparams_per_strata.bandwidth[stratum]
        sta_bestKNNDensity = stored_hyperparams_per_strata.nearestNeighbors[stratum]
    end

############################################################################
### Predict and evaluate optimized KNN fzx

# Normalized KNN complete (unlabelled test set predictions) test set
predict_complete_KNN_U = predictDensityKNN(sta_distanceXU_L,
                                           sta_zL,
                                           sta_bestKNNDensity,
                                           sta_bestBandwidthKNN,
                                           0,
                                           1,
                                           nr_fzxbins,
                                           normalization = false)

## compute the stratified predictions to merge together later, to get the combined loss 
sta_stratified_predictions_KNN = estimate_stratifiedpredictions_Statio_KNN(
    sta_bestKNNDensity,
    nr_fzxbins,
    sta_bestBandwidthKNN,
    0,
    1,
    sta_zL,
    sta_distanceXU_L,
    sta_zU, 
    predictedComplete = predict_complete_KNN_U)

# concatenate predictions for test sets of each stratum, leading to
# one set of test set preds. in StratLearn order
push!(sta_predictedComplete_KNN, predict_complete_KNN_U)
push!(sta_predictedObserved_KNN, sta_stratified_predictions_KNN["predictedObserved"])

if hyperparam_selection == "grid_search"
    push!(sta_finallossKNN,
          estimateErrorFinalEstimatorKNNContinuousStatio(
              sta_bestKNNDensity,
              nr_fzxbins,
              sta_bestBandwidthKNN,
              0,
              1,
              sta_zL,
              sta_distanceXU_L,
              sta_zU,
              boot=400,
              predictedComplete = predict_complete_KNN_U,
              predictedObserved = sta_stratified_predictions_KNN["predictedObserved"]))
end
end

# Save optimal hyperparameters if optimized via grid search
if hyperparam_selection == "grid_search"
    new_hyperparams_path_name =
        joinpath(result_folder_path,
                 "optimal_hyperparams_$(out_comment)_seed$(selected_seed)_$(time_stamp).rds")
    serialize(new_hyperparams_path_name, optimal_hyperparams)
end


# !!! sono arrivata qua
# Calculate combined loss if grid search was performed
if hyperparam_selection == "grid_search"
    sta_finallossAdaptive[nr_groups + 1] =
        estimate_combined_stratified_risk_Statio_KNN(
            sta_predictedComplete_KNN[stratum],
            sta_predictedObserved_KNN[stratum],
            sta_ordered_zU,
            0,
            1,
            nr_fzxbins,
            boot=400)
end

#--------------------------------------------------------------
# COMB ST
pdf(paste(result_folder_path,"/", "_strata_results_", out_comment, "_scaled_", "seed", as.character(selected_seed),
          "_",  "_" ,time_stamp,"alpha_comb.pdf", sep = "" ),
    width = 13, height = 7) 
par(mfrow=c(2,3))

sta_bestAlpha = DataFrame(fill(NaN, (nr_groups, length(hyperparam_names)),
                               ["epsilon", "delta", "nXBest_J", "nZbest_I",
                                "bandwidth", "nearestNeighbors", "alpha"]))
sta_predictUTestCombined = []
sta_comb_loss =  []
idx_start_tmp = 1

for stratum in first_test_group:nr_groups
    
    if hyperparam_selection == "grid_search"
        sta_alpha = range(0, 1, length=50)
        sta_loss=fill(NaN, length(sta_alpha))

        for ii in 1:length(sta_alpha)
            sta_predictUValidationCombined =
                sta_alpha[ii] * sta_validationU_predictedComplete_KNN[stratum] +
                (1 - sta_alpha[ii])*sta_validationU_predictedComplete_Adaptive[stratum]
            sta_predictLValidationCombined = sta_alpha[ii] *
                sta_validationL_predictedComplete_KNN[stratum] + (1-sta_alpha[ii]) *
                sta_validationL_predictedComplete_Adaptive[stratum]
            
            sta_loss[ii] =
                estimateErrorFinalEstimatorGeneric(
                    sta_predictLValidationCombined,
                    sta_predictUValidationCombined,
                    sta_validationL_ordered_z[stratum],
                    boot=F,
                    zMin = 0,
                    zMax = 1,
                    nBins = nr_fzxbins)["mean"]
        end

        scatter(sta_alpha, sta_loss, shape=:octagon, legend=false,
                title="Stratum $stratum")
        sta_bestAlpha[stratum, "alpha"] = sta_alpha[argmin(sta_loss)]
        sta_bestAlpha[stratum] = sta_alpha[which.min(sta_loss)]

        # store best alpha and save later
        optimal_hyperparams[stratum,"alpha"] = sta_alpha[argmin(sta_loss)]
        
    elseif hyperparam_selection == "fixed"
        sta_bestAlpha[stratum] = stored_hyperparams_per_strata.alpha[stratum]
    end
    
    #######################################################################################
    ### Final combination of predictions:

    println(idx_start_tmp)
    sta_U_idx = idx_start_tmp:(idx_start_tmp + sta_zU_size[stratum - first_test_group + 1] - 1)
    println(sta_U_idx)

    sta_predictUTestCombined[stratum] =
        sta_bestAlpha[stratum] *
        sta_predictedComplete_KNN[sta_U_idx[1]:sta_U_idx[end], :] + 
        (1 - sta_bestAlpha[stratum]) *
        sta_predictedComplete[sta_U_idx[1]:sta_U_idx[end], :]
    
    if hyperparam_selection == "grid_search"
        sta_comb_loss[stratum] =
            comb_test_loss_fct(
                sta_predictUTestCombined[stratum],
                sta_ordered_zU[sta_U_idx[1]:sta_U_idx[2]],
                0,
                1,
                nr_fzxbins,
                boot = 400)
    end
    idx_start_tmp = idx_start_tmp + sta_zU_size[stratum - first_test_group +1] 

end

if hyperparam_selection == "grid_search"
    predictedComplete = vcat(sta_predictUTestCombined...)
    sta_comb_loss[nr_groups + 1] = comb_test_loss_fct(
        predictedComplete,
        sta_ordered_zU,
        0,
        1,
        nr_fzxbins,
        boot = 400)
end

# final StratLearn comb predictions, combining the predictions for the five test strata
# in one matrix (this is ordered according to StratLearn strata), will be reordered later 
sta_ordered_SL_fzx_target = vcat(sta_predictUTestCombined...)

#--------------------------------------------------------------
# Save hyperparameters if optimized via grid search
if hyperparam_selection == "grid_search"
    new_hyperparams_path_name = joinpath(result_folder_path, "optimal_hyperparams_$(out_comment)_seed$(selected_seed)_$(time_stamp).rds")
    # Serialize the `optimal_hyperparams` DataFrame to a file
    serialize(new_hyperparams_path_name, optimal_hyperparams)
end

# Store all StratLearn results
stratlearn_results = (sta_finallossKNN, sta_finallossAdaptive, sta_comb_loss)
serialize(joinpath(result_folder_path, "Stratified_learning_results_$(out_comment)_$(time_stamp).rds"), stratlearn_results)

# Reorder StratLearn predictions to match the initial order of the photo data
# Implement `reorder_array_according_rowindices_vector` or use an equivalent Julia function
uniqueID_photo_reordered = reorder_array_according_rowindices_vector(
    sta_ordered_uniqueID_photo, sta_ordered_photo_indices)

zU_reordered = reorder_array_according_rowindices_vector(
    sta_ordered_zU, sta_ordered_photo_indices)

# Check if reordering is correct
if !all(uniqueID_photo_reordered .== uniqueID_photo)
    error("Reordering of StratLearn predictions (to initial order) not correct!")
end
if !all(zU_reordered .== zU)
    error("Reordering of StratLearn predictions (to initial order) not correct!")
end

# Reorder the StratLearn conditional density predictions
SL_fzx_target_reordered = reorder_array_according_rowindices_vector(
    sta_ordered_SL_fzx_target, sta_ordered_photo_indices)

# Normalize the StratLearn conditional density predictions
SL_fzx_target_normalized = SL_fzx_target_reordered / (rescaled_zGrid[end] - rescaled_zGrid[1])

# Store the predictions and additional results
serialize(
    joinpath(
        result_folder_path,
        "Conditional-Z_fzx-StratLearn_$(out_comment)_seed$(selected_seed)_$(time_stamp)_all_predictions_.rds"),
    Dict("zgrid" => rescaled_zGrid, "fzx" => SL_fzx_target_normalized))

additional_results = Dict("Z_source" => rescaled_zL, "Z_target" => rescaled_zU,
                          "PS_all" => PS, "groups_all" => all_data_group,
                          "ID_spectro" => uniqueID_spectro, "ID_photo" => uniqueID_photo)
serialize(joinpath(
    result_folder_path,
    "Additional_results_and_data_$(out_comment)_seed$(selected_seed)_$(time_stamp).rds"),
          additional_results)

# Exporting results to CSV
CSV.write(joinpath("data", "intermediate", "histograms.csv"),
          DataFrame(fzx = SL_fzx_target_normalized), writeheader = false)
CSV.write(joinpath("data", "intermediate", "grid.csv"),
          DataFrame(zgrid = rescaled_zGrid), writeheader = false)
end

main()
# End of StratLearn code
