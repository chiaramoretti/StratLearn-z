################################################################################################
### Change path to point to your folder

# set working directory
setwd('./')
# delete old results
unlink('summary_results', recursive = TRUE)
# create new results folder
dir.create('summary_results')
# Script parameters

#data: first nr_covariates columns are the covariates (r-band + colors), last columns is spectroscopic redshift
data_spectro_raw = read.csv(file =  "data/train.csv") # CHANGE "" to your filename with the spectroscopic source data
data_photo_raw = read.csv(file = "data/test.csv") #CHANGE "" to your filename with the photometric target data


################################################################################################
### Script parameters

nL = nrow(data_spectro_raw) # nr of source samples used
nU = nrow(data_photo_raw) # nr of target samples used

nr_groups = 5
first_test_group  = 1

selected_seed = 2
add_comment = ""


#data_spectro_raw = data_spectro_raw[ sample(1:nrow(data_spectro_raw) ,size = nL , replace = F),]
#data_photo_raw = data_photo_raw[sample(1:nrow(data_photo_raw) ,size = nU , replace = F),]
### REPLACE covariate_names with the names of your covariates (probably colors and r-band)
## Structure:
# first column is the unique identifier, then nr_covariates as covariates, last column in redshift

nr_covariates = 6
covariate_names = colnames(data_photo_raw)[2:(nr_covariates+1)]

## only keep the covariates and the spectroscopic redshift

uniqueID_spectro = data_spectro_raw[,1]
data_spectro = data_spectro_raw[,  c(covariate_names, "Z")]

uniqueID_photo = data_photo_raw[,1]
data_photo = data_photo_raw[,  c(covariate_names, "Z")]
 
###############################################################################################
############################# START OF THE BLACKBOX CODE ######################################
###############################################################################################

################################
### additional file parameters (moslty default used)

nr_fzxbins = 201 # careful, this is the nr of boundaries including start and end (nr_fzxbins - 1 thus gives the number of bins/grids used). In fcts later use nr_fzxbins when asked for nr_bins.
z_scaling_type = "min_max_spectro_z"    # "min_max_z", "min_max_spectro_z" (scales to min(z) being 0 and max(z) being 1), "fixed" (means scaled via zMin= 0.025, zMax= 1.725)

hyperparam_selection = "grid_search"
time_stamp = format(Sys.time() , "%Y%m%d_%H%M%S_")
set.seed(selected_seed) 
out_comment = paste(time_stamp, add_comment ,sep = "")  # ""

result_folder_path = "summary_results/"

################################################################################################
################################################################################################
################################################################################################
source("additional_functions/Balance_check_fcts.R")

model_path_local = ""
source(paste(model_path_local,  "code_to_fit_models/conditionalDensityCS.r", sep = ""))
source(paste(model_path_local, "code_to_fit_models/conditionalDensity.r", sep = ""))
source(paste(model_path_local, "code_to_fit_models/estimateWeights.r", sep = ""))

################################################################################################
# load beforehand optimized hyperparamters (this is only needed if hyperparam_selection == "fixed")
  # "grid_search" , "fixed" (when fixed, then the values in eps/delta_per_strata_fixed are used as hyperparameters)
hyperparam_file_name_tmp =""
time_stamp_hyperparams_file = ""
hyperparam_file_name = paste(hyperparam_file_name_tmp, sep = "")

eps_per_strata_fixed = c(NA,NA,NA,NA,NA) # these values used if hyperparam_selection = "fixed". Define values e.g. via previous hyperparameter optimization on smaller data
# MA_d: in the normalization step of Series fzx, bumps (separate regions) of fzx are removed when a large delta is applied.
nXBest_J_per_strata_fixed = c(NA,NA,NA,NA,NA)
nZBest_I_per_strata_fixed = c(NA,NA,NA,NA,NA)
delta_per_strata_fixed = c(NA,NA,NA,NA,NA) # these values used if hyperparam_selection = "fixed". Define values e.g. via previous hyperparameter optimization on smaller data
# KNN fzx estimator
bandwidth_per_strata_fixed = c(NA,NA,NA,NA,NA) # these values used if hyperparam_selection = "fixed". Define values e.g. via previous hyperparameter optimization on smaller data
nearestneighbors_per_strata_fixed = c(NA,NA,NA,NA,NA) # these values used hyperparam_selection = "fixed". Define values e.g. via previous hyperparameter optimization on smaller data
# Comb fzx estimator
alpha_per_strata_fixed = c(NA,NA,NA,NA,NA) 
hyperparam_names = c("epsilon", "delta", "nXbest_J", "nZbest_I","bandwidth","nearestNeighbors","alpha")
stored_hyperparams_per_strata = data.frame(eps_per_strata_fixed, nXBest_J_per_strata_fixed, nZBest_I_per_strata_fixed, delta_per_strata_fixed,bandwidth_per_strata_fixed,nearestneighbors_per_strata_fixed, alpha_per_strata_fixed)

################################
if(hyperparam_selection == "fixed"){
  stored_hyperparams_per_strata = readRDS(paste(result_folder_path, time_stamp_hyperparams_file ,"/", "", hyperparam_file_name,".rds", sep = ""))
  stored_hyperparams_per_strata = data.frame(stored_hyperparams_per_strata)
  print(paste("Hyperparameter file: ", hyperparam_file_name, " is loaded!", sep = ""))
}
#stored_hyperparams_per_strata$nXbest_J = c(100,100,100,100,100)
colnames(stored_hyperparams_per_strata) = hyperparam_names
###############################
###############################################################
### Libraries needed:

library("future")
library("furrr")
library("pdist")
library("gplots")
library("coda")
library("truncnorm") # truncated normal, as jumping distribution
library("xtable") #M: for latex tables
library("R.matlab") # to write and load files from matlab (e.g. weights)
library("matlabr") # for confusion matrix
library("cvms")
library("tibble")
library('ggimage')
library('rsvg')
library("ggnewscale")
library("invgamma")
library("finalfit")
library("tableone") # for covariate balance check
library("stats") # for ks test statistics
library("VIM") # knn imputation

###########################################################################################
###########################################################################################
###########################################################################################
###############################################################################################
####################################################################################
### Obtain train and validation subsets (split source set into training and validation, final model is then tained on all)
nTrainL=round(0.5 * nL) 
nValidationL= nL - nTrainL

nValidationU=round(0.5 * nU)
nTestU= nU - nValidationU # round(0.5 * nU)

####################################################################################
# add train indicator
data_spectro['train'] = 1
data_photo['train'] = 0
################################################################################################


data_full = rbind(data_spectro, data_photo)
data_full['Z'] =  data_full$Z

# scale z to be between 0 and 1 here (makes calculations of conditional densities simpler)
zBeforeScale= data_full$Z
if(z_scaling_type == "min_max_z"){
  zMin=min(zBeforeScale)
  zMax=max(zBeforeScale)
}else if(z_scaling_type == "min_max_spectro_z"){
    zMin=min(zBeforeScale[1:nL])
    zMax=max(zBeforeScale[1:nL])
}else if(z_scaling_type == "fixed"){
  zMin= 0.06 #min(zBeforeScale)
  zMax= 1.5  #max(zBeforeScale)
}
# w.l.o.g. scale the redshift z being between 0 and 1 (makes some of the calculations later simpler)
z=(zBeforeScale-zMin)/(zMax-zMin) 
print(zMax)

# grid to evaluate the conditional densities fzx on
zGrid=seq(from=0,to=1,length.out= nr_fzxbins)
rescaled_zGrid =  zGrid * (zMax-zMin) + zMin

##########################################################################################################################
##########################################################################################################################
### covariates used for computation

covariates = data_full[,covariate_names]
covariates = scale(covariates)  # M: from original code; scaled to have mean 0 and sd 1 (needed fro fzx estimation)


covariatesL=covariates[1:nL,]
covariatesU=covariates[-(1:nL),]
zL=z[1:nL]
zU=z[-c(1:nL)]

rescaled_zU = zU  * (zMax-zMin) + zMin 
rescaled_zL = zL * (zMax-zMin) + zMin 
##########################################################################################################################
#Split L Sample
randomPermL=sample(1:nL)
idx_TrainL = randomPermL[1:nTrainL]
idx_ValidationL = randomPermL[(nTrainL+1):(nTrainL+nValidationL)]
covariatesTrainL=covariatesL[idx_TrainL,]
covariatesValidationL=covariatesL[idx_ValidationL,]
zTrainL=zL[idx_TrainL]
zValidationL=zL[idx_ValidationL]

# Split U Sample
randomPermU=sample(1:nU)
idx_ValidationU = randomPermU[1:nValidationU]
idx_TestU = randomPermU[(nValidationU+1):(nValidationU+nTestU)]
covariatesValidationU=covariatesU[idx_ValidationU,]
covariatesTestU=covariatesU[idx_TestU,]
zValidationU=zU[idx_ValidationU]
zTestU=zU[idx_TestU]



# Transform to Matrix
covariatesValidationU=as.matrix(covariatesValidationU)
covariatesTrainL=as.matrix(covariatesTrainL)
covariatesValidationL=as.matrix(covariatesValidationL)
covariatesTestU=as.matrix(covariatesTestU)

#MA_c: this leads too overly large matricees, not fitting into memory, need to change to scale up code!
print("b")
# Distances L
distanceXTrainL_TrainL=as.matrix(pdist(covariatesTrainL,covariatesTrainL))
distanceXValidationL_TrainL=as.matrix(pdist(covariatesValidationL,covariatesTrainL))
distanceXValidationL_TrainL=as.matrix(pdist(covariatesValidationL,covariatesTrainL))
distanceXValidationU_TrainL=as.matrix(pdist(covariatesValidationU,covariatesTrainL))
distanceXValidationL_ValidationL =as.matrix(pdist(covariatesValidationL, covariatesValidationL))
# Distances U to L
distanceXValidationU_TrainL=as.matrix(pdist(covariatesValidationU,covariatesTrainL))
distanceXTestU_TrainL=as.matrix(pdist(covariatesTestU,covariatesTrainL))

####################################################################################################
####################################################################################################
####################################################################################################
### Compute StratLearn

### estimate the propensity scores
source_ind = data_full$train
# MA_c: could remove data_full here (not used afterwards)
all_data = data.frame(data_full["Z"], covariates,  source_ind)

# Logistic regression:
lreg_PS <- glm(source_ind ~ . - Z, family = "binomial", data = all_data) 

summary(lreg_PS)
PS = fitted(lreg_PS)


##############################################################################################################################
### given estimated PS, assign samples to strata
temp = rank(PS) #Rank of the fitted values (i.e. propensity scores)
grp.size <- length(all_data[,1])/nr_groups
all_data$group = rep(NA, length(temp))
for( k in  1:nr_groups){
 all_data$group[temp <= grp.size * (nr_groups - k +1 )  ] =  k
 print(sum(temp <= grp.size * (nr_groups - k +1 ) ))
}

pdf(paste(result_folder_path,"/", "PS_distribution", out_comment, "_scaled_", "seed", as.character(selected_seed),
          "_","_" ,time_stamp,".pdf", sep = "" ),
    width = 7, height = 7) 
par(mfrow=c(1,1))
p1 =  hist(PS[1:nL], breaks = 50, freq = F ) 
p2 =  hist(PS[(nL + 1): (nL +nU)] , breaks = 50, freq = F ) 
plot( p2, col= rgb(1,0,0,1/5), xlim=c(0,1), main = paste("Hist: Estimated propensity scores (source vs. target) resampled data", sep = ""), xlab = "propensity score", freq = F)  # second  # first histogram
plot( p1, col=rgb(0,0,1,1/5), xlim=c(0,1), add=T, freq = F)
legend("topright", legend=c("source propensity score", "target propensity score"),
       fill = c("blue","red"))
dev.off()

####################################################################################################################
#### check proportions in each strata
# table with number of SNIa and non SNIa + proportion for each training and test set
group_proportions = matrix(data = NA, ncol = 2, nrow = (nr_groups+1)*2 )
counter = 1
for( s in 1:nr_groups){
  group_proportions[counter,1] = length(subset(all_data, source_ind == 1 & group == s)$Z)
  group_proportions[counter+1,1] = length(subset(all_data, source_ind == 0 & group == s)$Z)
  group_proportions[counter,2] = round(mean(subset(all_data, source_ind == 1 & group == s)$Z), 2) 
  group_proportions[counter+1,2] =  round(mean(subset(all_data, source_ind == 0 & group == s)$Z), 2) 
  counter = counter+2
}
group_proportions[nr_groups*2 +1 , 1 ] = length(subset(all_data, source_ind == 1)$Z)
group_proportions[nr_groups*2 +1 , 2 ] = round(mean(subset(all_data, source_ind == 1 )$Z), 2) 
group_proportions[nr_groups*2 +2 , 1 ] = length(subset(all_data, source_ind == 0)$Z)
group_proportions[nr_groups*2 +2 , 2 ] = round(mean(subset(all_data, source_ind == 0 )$Z), 2) 

colnames(group_proportions) = c("# samples", "mean redshift")
rownames(group_proportions) =  (rep(c("Source", "Target"), nr_groups + 1))
group_proportions

write.csv(group_proportions,  paste(result_folder_path, "/", "group_proportions_K_", nr_groups ,"_",
                                    out_comment ,time_stamp, ".csv", sep = "" ))
xtable(group_proportions) 

###################################################################################
### compute the covariate balance within strata (only for batch one for computational reasons):

### compute the standardized mean differences for raw data and within all strata  
smd_raw = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                            use_group = c(T,T,T,T,T), indicator_var = "source_ind", measure = "smd")
smd_all = list()
smd_all[[1]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                            use_group = c(T,F,F,F,F), indicator_var = "source_ind", measure = "smd")
smd_all[[2]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                                    use_group = c(F,T,F,F,F), indicator_var = "source_ind", measure = "smd")
smd_all[[3]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                                    use_group = c(F,F,T,F,F), indicator_var = "source_ind", measure = "smd")
smd_all[[4]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                                    use_group = c(F,F,F,T,F), indicator_var = "source_ind", measure = "smd")
smd_all[[5]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                                    use_group = c(F,F,F,F,T), indicator_var = "source_ind", measure = "smd")

### compute the ks-statistics (Kolmogorov-Smirnov test statistics) for raw data and within all strata
### compute the standardized mean differences for raw data and within all strata  
ks_raw = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                               use_group = c(T,T,T,T,T), indicator_var = "source_ind", measure = "ks-statistics")
ks_all = list()
ks_all[[1]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                                    use_group = c(T,F,F,F,F), indicator_var = "source_ind", measure = "ks-statistics")
ks_all[[2]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                                    use_group = c(F,T,F,F,F), indicator_var = "source_ind", measure = "ks-statistics")
ks_all[[3]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                                    use_group = c(F,F,T,F,F), indicator_var = "source_ind", measure = "ks-statistics")
ks_all[[4]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                                    use_group = c(F,F,F,T,F), indicator_var = "source_ind", measure = "ks-statistics")
ks_all[[5]] = balance_ingroups_fct(data_set = all_data, covariate_names = covariate_names, 
                                    use_group = c(F,F,F,F,T), indicator_var = "source_ind", measure = "ks-statistics")

# plot smd
balance_evaluation_plot_fct(balance_list_strata = smd_all, balance_raw = smd_raw, balance_measure = "smd", 
                            result_folder = paste(result_folder_path, "/", sep="") )
# plot ks-statistics
balance_evaluation_plot_fct(balance_list_strata = ks_all, balance_raw = ks_raw, balance_measure = "ks-statistics",
                            result_folder =  paste(result_folder_path,"/", sep=""))

saveRDS(list(smd_all, ks_all)  ,file =  paste(result_folder_path, "/", "balance_results_", out_comment, "seed", as.character(selected_seed),
                                              "_" ,time_stamp,".rds", sep = "" ))
  

### PS estimation done
###################################################################################
###################################################################################
###################################################################################


###################################################################################
###################################################################################
###################################################################################
### Compute conditional densities on each stratum separately 



#### get the distance matricees for each stratum, the lists are only needed in case strata are combined for training, in which case this can be done for each stratum seperately below!
train_strata = list()
train_strata[[1]] = c(1)
train_strata[[2]] = c(2)
train_strata[[3]] = c(3)
train_strata[[4]] = c(4)
train_strata[[5]] = c(5)


# note that for now the validation set is always only from same strata as the test set (not combined)
val_strata = list()
val_strata[[1]] = c(1)
val_strata[[2]] = c(2)
val_strata[[3]] = c(3)
val_strata[[4]] = c(4)
val_strata[[5]] = c(5)

sta_finallossAdaptive = list()
sta_predictedComplete = sta_predictedObserved = numeric()
sta_ordered_zTestU = numeric() # only to track testU strata sizes
sta_ordered_zU = numeric()
sta_ordered_uniqueID_photo = numeric()
sta_ordered_photo_indicees = numeric()

sta_zTestU_size = sta_zU_size = numeric()

#MA_r: are all these lists needed? Can maybe summarize in one list?
sta_validationL_finallossAdaptive  = sta_validationL_predictedComplete_Adaptive = sta_validationL_ordered_z  = list()
sta_validationU_finallossAdaptive  = sta_validationU_predictedComplete_Adaptive  = sta_validationU_ordered_z  = list()

# Series hyperparameters

optimal_hyperparams = matrix(data = NA ,nrow = nr_groups, ncol = length(hyperparam_names))
colnames(optimal_hyperparams) = hyperparam_names

for(stratum in first_test_group:nr_groups){
  
  #### For each stratum we need the distance matrices when computing the density estimators and densities
  ### Train and validation subsets
  # Distances L
  sta_distanceXTrainL_TrainL= distanceXTrainL_TrainL[ all_data$group[idx_TrainL] %in% train_strata[[stratum]], all_data$group[idx_TrainL] %in% train_strata[[stratum]] ]
  sta_distanceXValidationL_TrainL= as.matrix(pdist(covariatesValidationL[ all_data$group[idx_ValidationL] %in% val_strata[[stratum]], ]    ,  covariatesTrainL[all_data$group[idx_TrainL] %in% train_strata[[stratum]], ]   ))
  # Distances U to L
  sta_distanceXTestU_TrainL=as.matrix(pdist(covariatesTestU[all_data$group[(nL +1): (nL + nU)][idx_TestU] == stratum, ]  ,covariatesTrainL[all_data$group[idx_TrainL] %in% train_strata[[stratum]], ]))
  sta_distanceXValidationU_TrainL=as.matrix(pdist(covariatesValidationU[all_data$group[(nL +1): (nL + nU)][idx_ValidationU] == stratum, ]  ,covariatesTrainL[all_data$group[idx_TrainL] %in% train_strata[[stratum]], ]))
  
  # Subsetting the outcome (redshift according to strata)
  sta_zTrainL =  zTrainL[all_data$group[idx_TrainL] %in% train_strata[[stratum]]] 
  sta_zValidationL =  zValidationL[all_data$group[idx_ValidationL] %in% val_strata[[stratum]]]
  sta_zTestU = zTestU[all_data$group[(nL +1): (nL + nU)][idx_TestU] == stratum]
  sta_zValidationU = zValidationU[all_data$group[(nL +1): (nL + nU)][idx_ValidationU] == stratum]
  
  
  ### Final training L and testing sets U:
  sta_distanceXL_L = as.matrix(pdist(covariatesL[all_data$group[1:nL] %in% train_strata[[stratum]],  ] ,  covariatesL[all_data$group[1:nL] %in% train_strata[[stratum]],  ] ))
  sta_distanceXU_L = as.matrix(pdist(covariatesU[all_data$group[(nL +1):(nL + nU)] %in% stratum,  ] ,  covariatesL[all_data$group[1:nL] %in% train_strata[[stratum]],  ] ))
  # Final training L and testing U redshift (testing only for evaluation purposes): 
  sta_zL = zL[all_data$group[1:nL] %in% train_strata[[stratum]]]
  sta_zU = zU[all_data$group[(nL+1):(nL+nU)] %in% stratum]
  sta_photo_indicees = which(all_data$group[(nL+1):(nL+nU)] %in% stratum)
  sta_uniqueID_photo = uniqueID_photo[all_data$group[(nL+1):(nL+nU)] %in% stratum]
  
  # keep track of the size of each unlabelled test set used in each strata
  sta_zTestU_size = c(sta_zTestU_size, length(sta_zTestU))
  sta_zU_size = c(sta_zU_size, length(sta_zU))
  
  
  # StratLearn Stationary Adaptive
  #MA_d: This is the series estimator (Izbicki (2017)) applied via StratLearn within each stratum:
  print("Fit StratLearn Stationary Adaptive (Series cond. density estimator)")
  #MA_c: does epsGrid change with larger sample size? Otherwise, could get value on smaller size and then for larger set used fixed value?
  epsGrid=seq(0.05,0.4,length.out=7)
  error=rep(NA,length(epsGrid))
  if(hyperparam_selection == "grid_search"){
    for(ii in 1:length(epsGrid)){
      print(ii/length(epsGrid))
      eps=epsGrid[ii]
      #MA_d: This is the Series estimator from Izbicki and Lee (2016) (Not adjusted for covariate shift as in Izbicki 2017) only through StratLearn!
      #MA_d: condDensityStatio instantiates the series estimator (computing the basis function and eigenfunctions etc) for trainL set (ther errors computed on ValidationL to get best hyperparameters)
      object=condDensityStatio(sta_distanceXTrainL_TrainL,sta_zTrainL,nZMax=70,kernelFunction=radialKernelDistance,extraKernel=list("eps.val"=eps),normalization=NULL,system="Fourier") #,nXMax=350, ,nXM=1
      object=estimateErrorEstimatorStatio(object,sta_zValidationL,sta_distanceXValidationL_TrainL)
      gc()
      error[ii]=object$bestError
      rm(object)
    }
    eps=epsGrid[which.min(error)]  # 0.1289286
    #pdf(paste(result_folder_path, time_stamp ,"/", "Series_epsilon_delta_opt",   out_comment,"_", "seed", as.character(selected_seed), "_",
    #          time_stamp, "_file_idx_",as.character(file_idx),"_Stratum",stratum ,".pdf", sep = "" ))
    #plot(epsGrid,error,pch=18)
    #points(eps,min(error), pch = 3, cex = 2, col = "blue")

  }else if(hyperparam_selection == "fixed"){
    eps = stored_hyperparams_per_strata$epsilon[stratum]
  }

  gc()
  if(hyperparam_selection == "grid_search"){
    # MA_d: Instantiation of the Series estimator using the best eps value as found in loop above 
    sta_objectStationaryAdaptive_delta=condDensityStatio(sta_distanceXTrainL_TrainL, sta_zTrainL,nZMax=70,kernelFunction=radialKernelDistance,extraKernel=list("eps.val"=eps),normalization=NULL,system="Fourier") #M: ,nXMax=350
    sta_objectStationaryAdaptive_delta=estimateErrorEstimatorStatio(sta_objectStationaryAdaptive_delta, sta_zValidationL,sta_distanceXValidationL_TrainL)
    # 
    gc()
    # MA_d: here nXBest (J in Eq. (17) Izbicki2017) and nZBest (I in Eq. (17) Izbicki2017)  are used (in predictDensityStatio fct), delta is a threshold to reduce bumps, 
    # MA_d: when predicting Series conditional densities fzx (via. predictDensityStatio), a last step is the normalization of the fzx
    # MA_d: in the normalization step, bumps (separate regions) are removed when a large delta is applied.
    # MA_d: If delta is large enough, only one (the largest) connected region will be given as fzx output (removing all bumps, and then being renormalized)

    sta_bestDeltaStationaryAdaptive=chooseDeltaStatio(sta_objectStationaryAdaptive_delta, sta_zValidationL, sta_distanceXValidationL_TrainL, deltaGrid=seq(0,0.4,0.025),
                                                    nBins = nr_fzxbins) 
    delta_best = sta_bestDeltaStationaryAdaptive$bestDelta
    #dev.off()
    # store best epsilon and best delta for each stratum into matrix (save later after all strata)
    optimal_hyperparams[stratum,"delta"] = delta_best
    optimal_hyperparams[stratum,"epsilon"] = eps
    
    #
    nXBest_J = sta_objectStationaryAdaptive_delta$nXBest
    nZBest_I = sta_objectStationaryAdaptive_delta$nZBest
    optimal_hyperparams[stratum,"nXbest_J"] = nXBest_J
    optimal_hyperparams[stratum,"nZbest_I"] = nZBest_I
    
    #####
    ## labeled validation predictions and loss (on strata)
    # For validationn set predictions, the training set (and not the entire labelled set) has to be used, otherwise predictions might be too confident (could lead to overfitting of comb estimator)
    # only when doing parameter optimization
    sta_validationL_stratified_predictions_Adaptive = estimate_stratifiedpredictions_Statio(sta_objectStationaryAdaptive_delta, sta_zValidationL, sta_distanceXValidationL_TrainL, delta=delta_best,
                                                                                            zMin = 0, zMax = 1, nBins = nr_fzxbins) 
    sta_validationL_predictedComplete_Adaptive[[stratum]] =  sta_validationL_stratified_predictions_Adaptive$predictedComplete
    sta_validationL_ordered_z[[stratum]] =  sta_zValidationL
    sta_validationL_finallossAdaptive[[stratum]]=estimateErrorFinalEstimatorStatio(sta_objectStationaryAdaptive_delta, sta_zValidationL, sta_distanceXValidationL_TrainL, boot=400, delta=delta_best,
                                                                                   zMin = 0, zMax = 1, nBins = nr_fzxbins,
                                                                                   predictedComplete = sta_validationL_stratified_predictions_Adaptive$predictedComplete,
                                                                                   predictedObserved = sta_validationL_stratified_predictions_Adaptive$predictedObserved )
    
    ## labeled validation predictions and loss (on strata)
    sta_validationU_stratified_predictions_Adaptive =  estimate_stratifiedpredictions_Statio(sta_objectStationaryAdaptive_delta, sta_zValidationU, sta_distanceXValidationU_TrainL, delta=delta_best,
                                                                                             zMin = 0, zMax = 1, nBins = nr_fzxbins)  # sta_zValidationU only used to estimate risk, but not to obtain the fzx predictions (predictedComplete)
    sta_validationU_predictedComplete_Adaptive[[stratum]] =  sta_validationU_stratified_predictions_Adaptive$predictedComplete
    sta_validationU_ordered_z[[stratum]] =  sta_zValidationU
    sta_validationU_finallossAdaptive[[stratum]]=estimateErrorFinalEstimatorStatio(sta_objectStationaryAdaptive_delta, sta_zValidationU, sta_distanceXValidationU_TrainL, boot=400, delta=delta_best,
                                                                                   zMin = 0, zMax = 1, nBins = nr_fzxbins,
                                                                                   predictedComplete = sta_validationU_stratified_predictions_Adaptive$predictedComplete,
                                                                                   predictedObserved = sta_validationU_stratified_predictions_Adaptive$predictedObserved)
  
  ### end of hyperparameter optimization for Series estimator
  }else if(hyperparam_selection == "fixed"){
    delta_best = stored_hyperparams_per_strata$delta[stratum]
    nXBest_J = stored_hyperparams_per_strata$nXbest_J[stratum] 
    nZBest_I = stored_hyperparams_per_strata$nZbest_I[stratum] 
  } 
  gc()
  
  #####################################################################################################################
  #####################################################################################################################
  ### Final model using the optimized hyperparameter set, and entire labelled set for training, entire unlabelled set for test predictions:
  
  sta_objectStationaryAdaptive=condDensityStatio(sta_distanceXL_L, sta_zL, nZMax=70,kernelFunction=radialKernelDistance,extraKernel=list("eps.val"=eps),normalization=NULL,system="Fourier") #M: ,nXMax=350
  sta_objectStationaryAdaptive$nXBest = nXBest_J
  sta_objectStationaryAdaptive$nZBest = nZBest_I
  #sta_objectStationaryAdaptive=estimateErrorEstimatorStatio(sta_objectStationaryAdaptive, sta_zL,sta_distanceXL_L)
  
  predict_complete_U = predictDensityStatio(sta_objectStationaryAdaptive,zTestMin=0,zTestMax= 1 ,B=nr_fzxbins,sta_distanceXU_L,probabilityInterval=F,delta=delta_best)
  
  ## compute the stratified predictions to merge them together later, to get the combined loss 
  sta_stratified_predictions =estimate_stratifiedpredictions_Statio(sta_objectStationaryAdaptive, sta_zU, sta_distanceXU_L, delta=delta_best,
                                                                    zMin = 0, zMax = 1, nBins = nr_fzxbins,
                                                                    predictedComplete = predict_complete_U) #delta=sta_objectStationaryAdaptive$bestDelta)
  # concatenate the predictions for the test sets of each stratum, leading to one set of test set preds. in StratLearn order 
  sta_predictedComplete = rbind(sta_predictedComplete, predict_complete_U) 
  sta_predictedObserved = c(sta_predictedObserved, sta_stratified_predictions$predictedObserved)
  
  # concatenate the true redshift in the order of the StratLearn predictions for the test sets of each strata
  sta_ordered_zU = c(sta_ordered_zU, sta_zU )
  sta_ordered_zTestU = c(sta_ordered_zTestU, sta_zTestU )
  sta_ordered_uniqueID_photo = c(sta_ordered_uniqueID_photo, sta_uniqueID_photo)
  sta_ordered_photo_indicees = c(sta_ordered_photo_indicees, sta_photo_indicees)
  
  if(hyperparam_selection == "grid_search"){
    sta_finallossAdaptive[[stratum]] =estimateErrorFinalEstimatorStatio(sta_objectStationaryAdaptive, sta_zU, sta_distanceXU_L, boot=400, delta=delta_best,
                                                                      zMin = 0, zMax = 1, nBins = nr_fzxbins,
                                                                      predictedComplete = predict_complete_U,
                                                                      predictedObserved = sta_stratified_predictions$predictedObserved)
  }
  

}
# save hyperparameters if optimized via grid search
if(hyperparam_selection == "grid_search"){
  new_hyperparams_path_name = paste(result_folder_path, "/", "optimal_hyperparams_",   out_comment,"_", "seed", as.character(selected_seed), "_",
                                    time_stamp,".rds", sep = "" )
  saveRDS(optimal_hyperparams, file = new_hyperparams_path_name)
}

if(hyperparam_selection == "grid_search"){
  ### get combined (all unlabelled test sets combined over strata) loss of Series estimator
  sta_finallossAdaptive[[nr_groups + 1]] = estimate_combined_stratified_risk_Statio(sta_predictedComplete, sta_predictedObserved, sta_ordered_zU, boot = 400,
                                                                                  zMin = 0, zMax = 1, nBins = nr_fzxbins)
}


######################################################################################
######################################################################################
######################################################################################
# Fit KNN StratLearn 

sta_finallossKNN = list()
sta_predictedComplete_KNN = sta_predictedObserved_KNN  =  numeric()

sta_validationL_finallossKNN = sta_validationL_predictedComplete_KNN = list()
sta_validationU_finallossKNN = sta_validationU_predictedComplete_KNN = list()

#### get the distance matricees for each stratum
for(stratum in first_test_group:nr_groups){
  
  #### For each stratum we need the distance matricees when computing the density estimators and densities
  
  ### Train and validation subsets
  # Distances L
  sta_distanceXTrainL_TrainL= distanceXTrainL_TrainL[ all_data$group[idx_TrainL] %in% train_strata[[stratum]], all_data$group[idx_TrainL] %in% train_strata[[stratum]] ]
  sta_distanceXValidationL_TrainL= as.matrix(pdist(covariatesValidationL[ all_data$group[idx_ValidationL] %in% val_strata[[stratum]], ]    ,  covariatesTrainL[all_data$group[idx_TrainL] %in% train_strata[[stratum]], ]   ))
  # Distances U to L
  sta_distanceXTestU_TrainL=as.matrix(pdist(covariatesTestU[all_data$group[(nL +1): (nL + nU)][idx_TestU] == stratum, ]  ,covariatesTrainL[all_data$group[idx_TrainL] %in% train_strata[[stratum]], ]))
  sta_distanceXValidationU_TrainL=as.matrix(pdist(covariatesValidationU[all_data$group[(nL +1): (nL + nU)][idx_ValidationU] == stratum, ]  ,covariatesTrainL[all_data$group[idx_TrainL] %in% train_strata[[stratum]], ]))
  # Subsetting the outcome (redshift according to strata)
  sta_zTrainL =  zTrainL[all_data$group[idx_TrainL] %in% train_strata[[stratum]]] 
  sta_zValidationL =  zValidationL[all_data$group[idx_ValidationL] %in% val_strata[[stratum]]]
  sta_zTestU = zTestU[all_data$group[(nL +1): (nL + nU)][idx_TestU] == stratum]
  sta_zValidationU = zValidationU[all_data$group[(nL +1): (nL + nU)][idx_ValidationU] == stratum]
  
  
  
  ### Final training L and testing sets U:
  sta_distanceXL_L = as.matrix(pdist(covariatesL[all_data$group[1:nL] %in% train_strata[[stratum]],  ] ,  covariatesL[all_data$group[1:nL] %in% train_strata[[stratum]],  ] ))
  sta_distanceXU_L = as.matrix(pdist(covariatesU[all_data$group[(nL +1):(nL + nU)] %in% stratum,  ] ,  covariatesL[all_data$group[1:nL] %in% train_strata[[stratum]],  ] ))
  # Final training L and testing U redshift (testing only for evaluation purposes): 
  sta_zL = zL[all_data$group[1:nL] %in% train_strata[[stratum]]]
  sta_zU = zU[all_data$group[(nL+1):(nL+nU)] %in% stratum]
  
  #####################################################################################
  ### KNN hyperparameter optimization
  print("Fit StratLearn KNN (Continuous) -- M: This is the KNN estimator in paper with StratLearn")
  if(hyperparam_selection == "grid_search"){
    bandwidthsVec=seq(0.0000001,0.002,length.out=10)
    nNeighbours=round(seq(2,30,length.out=10))
    sta_lossKNNBinned=array(NA,dim=c(length(bandwidthsVec),length(nNeighbours)))
    #foreach(ii = 1:length(bandwidthsVec)) %dopar% {
    for(ii in 1:length(bandwidthsVec)){
      print(ii/length(bandwidthsVec))
      #foreach(jj = 1:length(nNeighbours)) %dopar% {
      for(jj in 1:length(nNeighbours)){
        cat(".")
        sta_lossKNNBinned[ii,jj]=estimateErrorFinalEstimatorKNNContinuousStatio(nNeighbours[jj],nr_fzxbins,bandwidthsVec[ii],0,1,sta_zTrainL,sta_distanceXValidationL_TrainL,sta_zValidationL,boot=F)$mean
      }
    }
    pointMin=which(sta_lossKNNBinned == min(sta_lossKNNBinned,na.rm=T), arr.ind = TRUE)
    sta_bestBandwidthKNN=(bandwidthsVec)[pointMin[1]] # 0.0002223111
    sta_bestKNNDensity=(nNeighbours)[pointMin[2]] # 10
    
    ## store optimal hyperparameters
    optimal_hyperparams[stratum,"bandwidth"] = sta_bestBandwidthKNN
    optimal_hyperparams[stratum,"nearestNeighbors"] = sta_bestKNNDensity
    
    
    ### Validation set predictions:
    ### labeled validation predictions and loss (on strata)
    sta_validationL_stratified_predictions_KNN = estimate_stratifiedpredictions_Statio_KNN(nNeigh =sta_bestKNNDensity,nBins=nr_fzxbins,bandwidthBinsOpt=sta_bestBandwidthKNN,
                                                                                           zMin=0,zMax=1,zTrain=sta_zTrainL,distanceXTestTrainL=sta_distanceXValidationL_TrainL,
                                                                                           zTest= sta_zValidationL, normalization = T)
    sta_validationL_predictedComplete_KNN[[stratum]] =  sta_validationL_stratified_predictions_KNN$predictedComplete
    sta_validationL_finallossKNN[[stratum]]=estimateErrorFinalEstimatorKNNContinuousStatio(nNeigh=sta_bestKNNDensity,nBins=nr_fzxbins,bandwidthBinsOpt=sta_bestBandwidthKNN,
                                                                                           zMin=0,zMax=1,zTrainL=sta_zTrainL,distanceXTestTrainL=sta_distanceXValidationL_TrainL,
                                                                                           zTestU=sta_zValidationL,boot=400, add_pred = F,
                                                                                           predictedComplete = sta_validationL_stratified_predictions_KNN$predictedComplete,
                                                                                           predictedObserved = sta_validationL_stratified_predictions_KNN$predictedObserved)
    # unlabeled validation predictions and loss (on strata)
    sta_validationU_stratified_predictions_KNN =estimate_stratifiedpredictions_Statio_KNN(nNeigh =sta_bestKNNDensity, nBins=nr_fzxbins,bandwidthBinsOpt=sta_bestBandwidthKNN,
                                                                                          zMin=0,zMax=1,zTrain=sta_zTrainL,distanceXTestTrainL=sta_distanceXValidationU_TrainL,
                                                                                          zTest=sta_zValidationU, normalization = T) # sta_zValidationU only needed to compute loss (not the predictedComplete)!
    sta_validationU_predictedComplete_KNN[[stratum]] =  sta_validationU_stratified_predictions_KNN$predictedComplete
    sta_validationU_finallossKNN[[stratum]]=estimateErrorFinalEstimatorKNNContinuousStatio(nNeigh=sta_bestKNNDensity,nBins=nr_fzxbins,bandwidthBinsOpt=sta_bestBandwidthKNN,
                                                                                           zMin=0,zMax=1,zTrainL=sta_zTrainL,distanceXTestTrainL=sta_distanceXValidationU_TrainL,
                                                                                           zTestU=sta_zValidationU,boot=400, add_pred = F,
                                                                                           predictedComplete = sta_validationU_stratified_predictions_KNN$predictedComplete,
                                                                                           predictedObserved = sta_validationU_stratified_predictions_KNN$predictedObserved)
    gc() 
    
    
  }else if(hyperparam_selection == "fixed"){
    sta_bestBandwidthKNN = stored_hyperparams_per_strata$bandwidth[stratum]
    sta_bestKNNDensity = stored_hyperparams_per_strata$nearestNeighbors[stratum]
  }
  rm(sta_lossKNNBinned)
  gc()
  
  ############################################################################
  ### Predict and evaluate optimized KNN fzx
  
  # Normalized KNN complete (unlabelled test set predictions) test set
  predict_complete_KNN_U = predictDensityKNN(distanceXTestTrain = sta_distanceXU_L, zTrain = sta_zL,
                                      KNNneighbors = sta_bestKNNDensity, KNNbandwidth = sta_bestBandwidthKNN, 
                                      nBins = nr_fzxbins, normalization = T)

  ## compute the stratified predictions to merge them together later, to get the combined loss 
  sta_stratified_predictions_KNN =estimate_stratifiedpredictions_Statio_KNN(nNeigh = sta_bestKNNDensity, nBins=nr_fzxbins, bandwidthBinsOpt=sta_bestBandwidthKNN,
                                                                            zMin=0,zMax=1,zTrain=sta_zL, distanceXTestTrainL=sta_distanceXU_L,
                                                                            zTest = sta_zU, 
                                                                            predictedComplete = predict_complete_KNN_U)
  
  # concatenate the predictions for the test sets of each stratum, leading to one set of test set preds. in StratLearn order 
  sta_predictedComplete_KNN = rbind(sta_predictedComplete_KNN, predict_complete_KNN_U)
  sta_predictedObserved_KNN = c(sta_predictedObserved_KNN, sta_stratified_predictions_KNN$predictedObserved)
  
  if(hyperparam_selection == "grid_search"){
    sta_finallossKNN[[stratum]]=estimateErrorFinalEstimatorKNNContinuousStatio(sta_bestKNNDensity,nr_fzxbins,sta_bestBandwidthKNN,0,1,sta_zL,sta_distanceXU_L,sta_zU,boot=400,
                                                     predictedComplete = predict_complete_KNN_U, predictedObserved = sta_stratified_predictions_KNN$predictedObserved)
  }
  

  
}
# save hyperparameters if optimized via grid search
if(hyperparam_selection == "grid_search"){
  saveRDS(optimal_hyperparams, file = new_hyperparams_path_name)
}

if(hyperparam_selection == "grid_search"){
  ## Compute loss on all test samples, by combining the StratLearn predictions
  sta_finallossKNN[[nr_groups + 1]] = estimate_combined_stratified_risk_Statio_KNN(sta_predictedComplete_KNN,
                                                                                   sta_predictedObserved_KNN,
                                                                                   sta_ordered_zU, 
                                                                                   zMin = 0, zMax = 1, nBins = nr_fzxbins,
                                                                                   boot = 400)
}
###################################################################################################################################
###################################################################################################################################
#### Comb_ST
pdf(paste(result_folder_path,"/", "_strata_results_", out_comment, "_scaled_", "seed", as.character(selected_seed),
          "_",  "_" ,time_stamp,"alpha_comb.pdf", sep = "" ),
    width = 13, height = 7) 
par(mfrow=c(2,3))

sta_bestAlpha = sta_predictUTestCombined = sta_comb_loss =  list()
idx_start_tmp = 1

for(stratum in first_test_group:nr_groups){
  
  if(hyperparam_selection == "grid_search"){
    sta_alpha=seq(0,1,length.out=50)
    sta_loss=rep(NA,length(sta_alpha))
    #foreach(ii = 1:length(sta_alpha)){
    for(ii in 1:length(sta_alpha)){
      sta_predictUValidationCombined=sta_alpha[ii]*sta_validationU_predictedComplete_KNN[[stratum]]+(1-sta_alpha[ii])*sta_validationU_predictedComplete_Adaptive[[stratum]]
      sta_predictLValidationCombined=sta_alpha[ii]*sta_validationL_predictedComplete_KNN[[stratum]]+(1-sta_alpha[ii])*sta_validationL_predictedComplete_Adaptive[[stratum]]
      
      sta_loss[ii]=estimateErrorFinalEstimatorGeneric(sta_predictLValidationCombined, sta_predictUValidationCombined,sta_validationL_ordered_z[[stratum]],boot=F,
                                                      zMin = 0, zMax = 1, nBins = nr_fzxbins)$mean
      # M: estimateErrorFinalEstimatorGeneric() is a fct in the conditionalDensityCS.r file
      # M: function computes the loss according to formula 9 in the Izbicki2017
    }
    plot(sta_alpha,sta_loss,pch=18, main = paste("","Stratum", stratum, sep = "" ))
    sta_bestAlpha[[stratum]]=sta_alpha[which.min(sta_loss)]
    # store best alpha and save later
    optimal_hyperparams[stratum,"alpha"] = sta_alpha[which.min(sta_loss)]
    
  }else if(hyperparam_selection == "fixed"){
    sta_bestAlpha[[stratum]] = stored_hyperparams_per_strata$alpha[stratum]
  }
  
  ############################################################################################
  ### Final combination predictions:
  
  print(idx_start_tmp)
  #sta_testU_idx = c(idx_start_tmp, idx_start_tmp + sta_zTestU_size[stratum - first_test_group +1 ] -1 )
  sta_U_idx = c(idx_start_tmp, idx_start_tmp + sta_zU_size[stratum - first_test_group +1 ] -1 )
  print(sta_U_idx)
  #sta_predictUTestCombined[[stratum]] = sta_bestAlpha[[stratum]]* sta_predictedComplete_KNN[sta_testU_idx[1]:sta_testU_idx[2] , ] + (1-sta_bestAlpha[[stratum]])*  sta_predictedComplete[sta_testU_idx[1]:sta_testU_idx[2],  ]
  sta_predictUTestCombined[[stratum]] = sta_bestAlpha[[stratum]]* sta_predictedComplete_KNN[sta_U_idx[1]:sta_U_idx[2] , ] + (1-sta_bestAlpha[[stratum]])*  sta_predictedComplete[sta_U_idx[1]:sta_U_idx[2],  ]
  
  if(hyperparam_selection == "grid_search"){
    sta_comb_loss[[stratum]] = comb_test_loss_fct(predictedComplete = sta_predictUTestCombined[[stratum]] , zTestU = sta_ordered_zU[sta_U_idx[1]:sta_U_idx[2] ],
                                                zMin = 0, zMax = 1, nBins = nr_fzxbins, boot = 400)
  }
  #idx_start_tmp = idx_start_tmp + sta_zTestU_size[stratum - first_test_group +1 ] 
  idx_start_tmp = idx_start_tmp + sta_zU_size[stratum - first_test_group +1 ] 
  
}
if(hyperparam_selection == "grid_search"){
  sta_comb_loss[[nr_groups +1]] = comb_test_loss_fct(predictedComplete = (do.call(rbind,sta_predictUTestCombined )) , zTestU =  sta_ordered_zU,
                                                   zMin = 0, zMax = 1, nBins = nr_fzxbins, boot = 400)
  sta_comb_loss
}
dev.off()

# final StratLearn comb predictions, combining the predictions for the five test strata in one matrix (this is ordered according to StratLearn strata), will be reordered later 
sta_ordered_SL_fzx_target = (do.call(rbind,sta_predictUTestCombined ))


### save hyperparameter values
# save hyperparameters if optimized via grid search
if(hyperparam_selection == "grid_search"){
  saveRDS(optimal_hyperparams, file = new_hyperparams_path_name)
}

###################################################################################################################################
### store all StratLearn results

saveRDS(list(sta_finallossKNN,sta_finallossAdaptive, sta_comb_loss), file = paste(result_folder_path,"/", "Stratified_learning_results_",
                                                                                  out_comment , "_",time_stamp,".rds", sep = "" ))


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
### store all necessary results from photo_batch (in the order of initially loaded photo batches)
### 
# MA: reorder the StratLearn predictions to have same order as photo data initially (when loaded as .rds files)
# + reorder zU and uniqueID_photo and compare (as a sanity check to make sure ordering is correct)
# check the reordering via saved indicees, reorder vectors to be in previous order as loaded (and saved in photo batches) (for uniqueID, and redshift variable)
uniqueID_photo_tmp = reorder_array_according_rowindicees_vector(array_to_reorder = sta_ordered_uniqueID_photo,
                                                                       index_vector = sta_ordered_photo_indicees)
if(mean(uniqueID_photo_tmp == uniqueID_photo) != 1){
 stop("Reordering of StratLearn predictions (to initial order) not correct!") 
}
zU_tmp = reorder_array_according_rowindicees_vector(array_to_reorder = sta_ordered_zU,
                                                                       index_vector = sta_ordered_photo_indicees)
if(mean(zU_tmp == zU) != 1){
  stop("Reordering of StratLearn predictions (to initial order) not correct!") 
}

# if both checks go through, reorder the StratLearn conditional density predictions (according to sta_ordered_photo_indicees) to have same order as photo data batch initially (when loaded as .rds files)
SL_fzx_target_propto = reorder_array_according_rowindicees_vector(array_to_reorder =sta_ordered_SL_fzx_target,
                                                                       index_vector = sta_ordered_photo_indicees)

SL_fzx_target = SL_fzx_target_propto/(rescaled_zGrid[length(rescaled_zGrid)] - rescaled_zGrid[1])
### Store the predictions:
saveRDS(list("zgrid" = rescaled_zGrid, "fzx"= SL_fzx_target), file = paste(result_folder_path, "/", "Conditional-Z_fzx-StratLearn_", out_comment,"_", "seed", as.character(selected_seed), "_",
                                              time_stamp, "_all_predictions_"  , ".rds", sep = "" ))

### store additional results and data for further analysis:
saveRDS(list("Z_source" = rescaled_zL, "Z_target" = rescaled_zU, "PS_all" = PS , "groups_all" = all_data$group,
             "ID_spectro" = uniqueID_spectro, "ID_photo" = uniqueID_photo),
        file =   paste(result_folder_path, "/", "Additional_results_and_data_", out_comment,"_", "seed", as.character(selected_seed), "_",
                                 time_stamp  , ".rds", sep = "" ))

## comments
# save rescaled_zGrid,
# save redshift (zu rescaled), PS, unique id, groups,  


pdf(paste(result_folder_path,"/", "Sample_plots_of_fzx_" ,time_stamp,".pdf", sep = "" ),
    width = 7, height = 7) 
for(i in 1:20){
  plot(rescaled_zGrid, SL_fzx_target[i,])
  points(rescaled_zU[i],0, cex = 3, col = "green")
}
dev.off()
##################################################################################################
################################ END OF THE BLACKBOX CODE ########################################
##################################################################################################
# Exporting to csv
# results directory
setwd('./')
# saving the results in two files
a = readRDS(list.files('.')[grep(list.files('.'), pattern = 'Conditional')])
write.csv(a$fzx, 'data/intermediate/histograms.csv', row.names = FALSE)
write.csv(a$zgrid, 'data/intermediate/grid.csv', row.names=FALSE)

### End of StratLearn code 
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
