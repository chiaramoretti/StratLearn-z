using HypothesisTests
using Statistics

"""
    smd_fct(data, cov_names, split_indicator="source_ind")

Calculates the standardized mean difference (SMD) for a given set of covariates in a dataset. It computes the SMD by taking the absolute difference between the means of two groups (split by a specified indicator variable) and dividing it by the square root of the average of their variances.

# Arguments
- `data`: DataFrame containing the covariates and the split indicator variable.
- `cov_names`: Array of strings representing the names of the covariates.
- `split_indicator` (optional): String representing the name of the indicator variable used to split the data into two groups. Default value is "source_ind".

# Returns
A dictionary where the keys are the covariate names and the values are the corresponding SMD values.

# Example Usage
```julia
data = DataFrame(cov1 = [1, 2, 3, 4, 5], cov2 = [6, 7, 8, 9, 10], source_ind = [1, 1, 0, 0, 1])
cov_names = ["cov1", "cov2"]
smd_vals = smd_fct(data, cov_names, "source_ind")
```
Expected Output:
```
smd_vals = Dict("cov1" => 0.536056, "cov2" => 0.536056)
```
"""
function smd_fct(data, cov_names, split_indicator)
    smd_vals = Dict()

    for cov_name in cov_names
        cov_S = data[!,cov_name][data[!, split_indicator] .== 1]
        cov_T = data[!,cov_name][data[!, split_indicator] .== 0]
        nominator = mean(skipmissing(cov_S)) - mean(skipmissing(cov_T))
        denominator = sqrt((var(skipmissing(cov_S)) +
                            var(skipmissing(cov_T))) / 2)
        
        # Compute SMD value
        smd_vals[cov_name] = abs(nominator / denominator)
    end

    return smd_vals
end


"""
    ks_test_fct(data, cov_names, split_indicator)

The `ks_test_fct` function calculates the p-value of the Kolmogorov-Smirnov test for each covariate in a dataset. It compares the distribution of the covariate values between two groups, split by a specified indicator variable.

# Arguments
- `data`: DataFrame containing the covariates and the split indicator variable.
- `cov_names`: Array of strings representing the names of the covariates.
- `split_indicator`: String representing the name of the indicator variable used to split the data into two groups.

# Returns
A dictionary where the keys are the covariate names and the values are the p-values from the Kolmogorov-Smirnov test.

# Example
```julia
data = DataFrame(cov1 = [1, 2, 3, 4, 5], cov2 = [6, 7, 8, 9, 10], source_ind = [1, 1, 0, 0, 1])
cov_names = ["cov1", "cov2"]
ks_vals = ks_test_fct(data, cov_names, "source_ind")
```
Expected Output:
```
ks_vals = Dict("cov1" => 0.667, "cov2" => 0.667)
```
"""
function ks_test_fct(data, cov_names, split_indicator)
    ks_vals = Dict()

    for cov_name in cov_names
        group1 = data[!, cov_name][data[!, split_indicator] .== 1]
        group2 = data[!, cov_name][data[!, split_indicator] .== 0]

        ks_test = ApproximateTwoSampleKSTest(group1, group2)
        ks_vals[cov_name] = ks_test.δ
    end

    return ks_vals
end


"""
    balance_ingroups_fct(data_set, covariate_names, use_group = [true, true, true, true, true], indicator_var = "source_ind", measure = "smd", strata_var = "group")
Calculate either the standardized mean difference (SMD) or the Kolmogorov-Smirnov (KS) statistics for a given set of covariates, based on different groups defined by a strata variable.

# Arguments
- `data_set`: A DataFrame containing the data.
- `covariate_names`: An array of strings specifying the names of the covariates to be analyzed.
- `use_group`: An array of booleans indicating which groups defined by the strata variable should be included in the analysis. Default is `[true, true, true, true, true]`.
- `indicator_var`: A string specifying the name of the indicator variable used to split the data into groups. Default is `"source_ind"`.
- `measure`: A string specifying the measure to be calculated. It can be either `"smd"` for standardized mean difference or `"ks-statistics"` for Kolmogorov-Smirnov statistics. Default is `"smd"`.
- `strata_var`: A string specifying the name of the strata variable used to define the groups. Default is `"group"`.

# Returns
- If `measure` is `"smd"`, the function returns a tuple containing a dictionary with the SMD values for each covariate, as well as the mean and standard deviation of the SMD values.
- If `measure` is `"ks-statistics"`, the function returns a tuple containing a dictionary with the KS statistics for each covariate, as well as the mean and standard deviation of the KS statistics.

# Example Usage
```julia
data = DataFrame(group = [1, 2, 1, 2, 1, 2],
                 cov1 = [1, 2, 3, 4, 5, 6],
                 cov2 = [7, 8, 9, 10, 11, 12],
                 source_ind = [1, 0, 1, 0, 1, 0])

covariate_names = ["cov1", "cov2"]

# Calculate SMD for covariates within groups 1 and 2
smd_result = balance_ingroups_fct(data, covariate_names, use_group = [true, true], measure = "smd")
# Output: (Dict("cov1" => 0.5, "cov2" => 0.5), mean = 0.5, sd = 0.0)

# Calculate KS statistics for covariates within groups 1 and 2
ks_result = balance_ingroups_fct(data, covariate_names, use_group = [true, true], measure = "ks-statistics")
# Output: (Dict("cov1" => 0.333, "cov2" => 0.333), mean = 0.333, sd = 0.0)
```
"""
function balance_ingroups_fct(data_set, covariate_names;
                              use_group = [true, true, true, true, true],
                              indicator_var = "source_ind", measure = "smd",
                              strata_var = "group")
    if measure == "smd"
        use_rows = [data_set[!, strata_var][i] in findall(use_group)
                    for i in 1:size(data_set, 1)]
        smd = smd_fct(data_set[use_rows, :], covariate_names, indicator_var)
        return (smd, mean = mean(values(smd)), sd = std(values(smd)))
    elseif measure == "ks-statistics"
        use_rows = [data_set[!, strata_var][i] in findall(use_group)
                    for i in 1:size(data_set, 1)]
        ks_all = ks_test_fct(data_set[use_rows, :], covariate_names, indicator_var)
        return (ks_all, mean = mean(values(ks_all)), sd = std(values(ks_all)))
    else
        error("Invalid measure specified")
    end
end


"""
    reorder_array_according_rowindices_vector(array_to_reorder, index_vector)

Reorders the rows of an array according to the index vector.

# Arguments
- `array_to_reorder`: The array that needs to be reordered.
- `index_vector`: The vector that specifies the new order of the rows in the array.

# Returns
- `array_reordered`: The input array with the rows reordered according to the index vector.

# Example
```julia
array_to_reorder = [1 2 3; 4 5 6; 7 8 9]
index_vector = [3, 1, 2]
reordered_array = reorder_array_according_rowindices_vector(array_to_reorder, index_vector)
```
Expected Output:
```
reordered_array = [7 8 9; 1 2 3; 4 5 6]
```
"""
function reorder_array_according_rowindices_vector(array_to_reorder, index_vector)

    array_to_reorder = Matrix(array_to_reorder)
    # Initialize an empty matrix with the same dimensions
    array_reordered = Matrix{eltype(array_to_reorder)}(undef, size(array_to_reorder))
    # Reorder the array according to the index vector
    array_reordered[index_vector, :] = array_to_reorder

    return array_reordered
end
