


#######################################################################################################################################
# general functions to check covariate balance within strata (or raw data) given a data frame with covariate and the vector of strata and source/target indicator

smd_fct = function(data, cov_names, split_indicator = "source_ind"){
  smd_vals = numeric()
  for(i in 1:length(cov_names)){
    cov_S = data[cov_names[i]][data[split_indicator] == 1]
    cov_T = data[cov_names[i]][data[split_indicator] == 0]
    nominator = mean(cov_S, na.rm = T) - mean(cov_T, na.rm = T) 
    denominator =sqrt((sd(cov_S, na.rm = T)^2 +  sd(cov_S, na.rm = T)^2)/2 )
    # compute smd value
    smd_vals[cov_names[i]] = abs(nominator/denominator)
  }
  return(smd_vals)
}


ks_test_fct = function(data, cov_names, split_indicator){
  ks_vals = numeric()
  for(i in 1:length(cov_names)){
    ks_vals[cov_names[i]] = ks.test(x = data[cov_names[i]][data[split_indicator] == 1], 
                                    y = data[cov_names[i]][data[split_indicator] == 0])$statistic
  }
  return(ks_vals)
}  

balance_ingroups_fct = function(data_set, covariate_names, use_group = c(T,T,T,T,T), indicator_var = "source_ind", measure ='smd', strata_var = "group"){
  if (measure == 'smd'){
    use_rows = data_set[,strata_var] %in% (1:5)[use_group]
    #smdtable =  CreateTableOne(vars =covariate_names, strata = indicator_var,  # here (in this fct) strata means source vs target
    #                           data = data_set[use_rows,], test = FALSE)
    #smd = ExtractSmd(smdtable)
    smd = smd_fct(data =  data_set[use_rows,], cov_names = covariate_names, split_indicator = indicator_var)
    return(list(smd, mean = mean(smd), sd = sd(smd)))
  }
  # write same part for ks-statistics
  if (measure == 'ks-statistics'){
    use_rows = data_set[,strata_var] %in% (1:5)[use_group]
    ks_all = ks_test_fct(data = data_set[use_rows,], cov_names = covariate_names, split_indicator = indicator_var)
    return(list(ks_all, mean = mean(ks_all), sd = sd(ks_all)))
  }
}


### plot/illustrate the balance measures  
# function that uses smd of one or (several models) and plots raw vs smd strata (for as many strata as given)
# + add mean smd (of each model in legend or title)
balance_evaluation_plot_fct = function(balance_list_strata, balance_raw, balance_measure = "smd", method_name = "",
                                       add_method_list = list(), add_method_name = "", result_folder = ""){
  
  pdf(paste(result_folder  ,"balance_plots_", balance_measure,"_" , out_comment, "_scaled_", "seed", as.character(selected_seed),
            "_",time_stamp,".pdf", sep = "" ),width = 12, height = 8)
  
  par(mfrow = c(2,3))
  for(i in 1:length(balance_list_strata)){
    max_smd = max(c(balance_raw[[1]], balance_list_strata[[i]][[1]]))
    plot(balance_raw[[1]], balance_list_strata[[i]][[1]], pch = 3, col = "red",
         xlim = c(0, max_smd), ylim = c(0,max_smd),
         main = paste(balance_measure ,": raw vs. stratum:", i, sep = ""), 
         xlab = paste("raw ", balance_measure , sep =""), ylab = paste(balance_measure ," stratum ", i, sep = "" ))
    
    if(length(add_method_list) == 0){
      legend("topleft",legend = c(paste(method_name, " mean: ", round(balance_list_strata[[i]][[2]],2), 
                                        " sd: ", round(balance_list_strata[[i]][[3]],2)  , sep = ""),
                                  paste("raw", " mean: ", round(balance_raw[[2]],2), " sd: ",
                                        round(balance_raw[[3]],2)  , sep = "") ),
             col = c("red", "black"), pch = c(3, NA))
    }else if(length(add_method_list) != 0){
      # add additional methods
      additional_color = c("blue")
      points(balance_raw[[1]], add_method_list[[i]][[1]], pch = 3, col = additional_color)
      legend("topleft",legend = c(paste(method_name, " mean: ", round(balance_list_strata[[i]][[2]],2), 
                                        " sd: ", round(balance_list_strata[[i]][[3]],2)  , sep = ""),
                                  paste(add_method_name, " mean: ", round(add_method_list[[i]][[2]],2), 
                                        " sd: ", round(add_method_list[[i]][[3]],2)  , sep = ""),
                                  paste("raw", " mean: ", round(balance_raw[[2]],2), " sd: ",
                                        round(balance_raw[[3]],2)  , sep = "") ),
             col = c("red", additional_color , "black"), pch = c(3,3 ,NA))
    }
    
    lines(c(0, max_smd), c(0,max_smd), lty = 3)
    
  }
  
  dev.off()
  ## return a matrix with mean and sd for each method and strata
  
  return()
}



reorder_array_according_rowindicees_vector = function(array_to_reorder, index_vector){
  array_to_reorder = as.matrix(array_to_reorder) # make array to cal ncol() and nrow()
  array_reordered = matrix(ncol = ncol(array_to_reorder), nrow = nrow(array_to_reorder))
  
  array_reordered[index_vector,] = array_to_reorder
  return(array_reordered)
}






