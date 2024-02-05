using YAML
using CSV

function load_param_file(paramfile)
    
    params=YAML.load_file(paramfile)
    nr_groups = params["nr_groups"]
    first_test_group = params["first_test_group"]
    nr_covariates = params["nr_covariates"]
    selected_seed = params["selected_seed"]
    nr_fzxbins = params["nr_fzxbins"]
    z_scaling_type = params["z_scaling_type"]
    hyperparam_selection = params["hyperparam_selection"]

    if hyperparam_selection == "fixed"
        hyperparam_file_name = params["hyperparam_file_name"]
    else
        hyperparam_file_name = ""
    end
    
    out_folder = params["output_folder"]
    out_comment = params["add_comment"]
    
    return nr_groups, first_test_group, nr_covariates, selected_seed, nr_fzxbins, z_scaling_type, hyperparam_selection, hyperparam_file_name, out_folder, out_comment
end

function load_datasets(paramfile)
    params=YAML.load_file(paramfile)
    train = CSV.read(params["train"], DataFrame)
    test  = CSV.read(params["test"], DataFrame)

    return train, test
end
