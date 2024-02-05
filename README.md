# StratLearn

This repository contains a Julia implementation of the StratLearn method described in
[arXiv:2106.11211](https://ui.adsabs.harvard.edu/abs/2021arXiv210611211A/abstract).

The main script is in `run_stratlearn.jl`, while `src/` holds support functions for the calculations.
The `data\` folder holds a data set already split in training and test samples, originally from 
[here](https://github.com/nataliastylianou/Photo-z/tree/main) and described in this 
[paper](https://ui.adsabs.harvard.edu/abs/2022PASP..134d4501S/abstract)

To run the example, paste in a terminal
```
    julia run_stratlearn.jl
```

The run's specifications are listed in the `params.yaml` file, as well as the paths to the datafiles.
The code will produce and store results in a folder `summary_results\`

Most support functions are documented with docstrings, that can be accessed by typing in the Julia REPL
```
    ?function_name
```
