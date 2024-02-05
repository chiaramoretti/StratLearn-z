using Plots

""" balance_evaluation_plot_fct(balance_list_strata, balance_raw,
balance_measure = "smd", method_name = "", add_method_list = [],
add_method_name = "", result_folder = "")

Create scatter plots comparing the balance measure between different
strata and the raw data. Additional methods can also be plotted. The
function calculates the maximum value of the balance measure, creates
scatter plots for each stratum, adds the raw data and additional
methods if provided, and adds legends to the plots. Finally, it saves
the plots as PNG files.

Parameters:
- `balance_list_strata`: A list of tuples, where each tuple contains a dictionary with the balance measure values for each covariate in a stratum, as well as the mean and standard deviation of the balance measure for that stratum.
- `balance_raw`: A tuple containing a dictionary with the balance measure values for each covariate in the raw data, as well as the mean and standard deviation of the balance measure for the raw data.
- `balance_measure`: A string specifying the balance measure being plotted (default: "smd").
- `method_name`: A string specifying the name of the method used to calculate the balance measure for the strata (default: "").
- `add_method_list`: A list of tuples, where each tuple contains a dictionary with the balance measure values for each covariate calculated using an additional method, as well as the mean and standard deviation of the balance measure for that method (default: []).
- `add_method_name`: A string specifying the name of the additional method (default: "").
- `result_folder`: A string specifying the path to the folder where the plots will be saved (default: "").

Example Usage:
```julia
balance_list_strata = [(Dict("cov1" => 1.0, "cov2" => 1.0), mean = 1.0, sd = 0.0), (Dict("cov1" => 0.8, "cov2" => 0.9), mean = 0.85, sd = 0.05)]
balance_raw = (Dict("cov1" => 1.0, "cov2" => 1.0), mean = 1.0, sd = 0.0)
balance_measure = "smd"
method_name = "Method 1"
add_method_list = [(Dict("cov1" => 0.9, "cov2" => 0.95), mean = 0.925, sd = 0.025)]
add_method_name = "Method 2"
result_folder = "/path/to/results/"

balance_evaluation_plot_fct(balance_list_strata, balance_raw, balance_measure, method_name, add_method_list, add_method_name, result_folder)
```

This function creates scatter plots comparing the balance measure between the raw data and each stratum, as well as any additional methods provided. Legends will be added to the plots, and the plots will be saved as PNG files in the specified result folder.
"""
function plot_balance_evaluation(balance_list_strata, balance_raw;
                                 balance_measure = "smd", method_name = "",
                                 add_method_list = [], add_method_name = "",
                                 result_folder = "")

    max_smd = [maximum([collect(values(balance_raw[1]));
                        collect(values(balance_list_strata[i][1]))])
               for i in 1:length(balance_list_strata)]

    plotlist = []

    for i in 1:length(balance_list_strata)
        push!(plotlist,
              plot([(balance_raw[1][k], balance_list_strata[i][1][k])
                    for k in keys(balance_raw[1])],
                   seriestype = :scatter,
                   markershape = :cross,
                   color = :red,
                   xlim = (0, 1.1*maximum(max_smd)), ylim = (0, 1.1*maximum(max_smd)),
                   title = "$(balance_measure): raw vs. stratum: $i",
                   xlabel = "raw $balance_measure",
                   ylabel = "$balance_measure stratum $i",
                   label=""))

        # Add legend
        legend = method_name * " mean: " *
            string(round(balance_list_strata[i][2], digits=3)) * 
            " sd: " * string(round(balance_list_strata[i][3], digits=3)) *
            "\nraw mean: " * string(round(balance_raw[2], digits=3)) * 
            " sd: " * string(round(balance_raw[3], digits=3))

        if !isempty(add_method_list)
            scatter!(balance_raw[1], add_method_list[i][1], color = :blue)
            push!(legend, add_method_name * " mean: " *
                  string(round(add_method_list[i][2], digits=2)) * 
                  " sd: " * string(round(add_method_list[i][3], digits=2)))
        end

        plot!(x -> x, linestyle = :dot, color = :black, label = "")
        annotate!((0.05,0.95),text(legend, 10, :left, :top))
    end
    plot(plotlist..., layout=5, size=(1400, 800), framestyle=:box)
    savefig(result_folder * "balance_plots_$(balance_measure).pdf")
end
