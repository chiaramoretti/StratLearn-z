#M all functions in this file contain "Statio" in function name, this is not the case for the fcts in the estimateWeights.R file and 
# conditionalDensity.r file


condDensityKNNStatio=function(zTrainNearest,nBins,zMin,zMax){
  # calculates estimated density for a single observation
  # estimate density of z given z's of the  nearest neighbour
  binsIntervals=seq(zMin,zMax,length.out=nBins)
  binSize=binsIntervals[2]-binsIntervals[1]
  means=apply(as.matrix(binsIntervals),1,function(xx)
  {
    lower=xx-binSize/2
    upper=xx+binSize/2
    mean(zTrainNearest>=lower & zTrainNearest<upper)/binSize
  })
  output=NULL
  output$means=means
  output$binsIntervals=binsIntervals
  return(output)
}


# MA_d: commented out on March 15th, 2023. Function not applied anywhere, might need to reuse later. (Then change naming of zTestL (and others) not duplicating any vectors/matrices in main.)
# estimateErrorEstimatorStatioNewLoss=function(object,zTestL,distanceXTestL_TrainL,distanceXTestU_TrainL,weightsZTestL,lessInfo=F)
# {
#   # returns a nX by nZ matrix with the errors for the different possibilities
#   # of number of components, plus the best combination
#   if(class(object)!="cDensity") stop("Object should be of class cDensity")
#   
#   kernelNewOldTestL_TrainL=object$kernelFunction(distanceXTestL_TrainL,object$extraKernel)
#   kernelNewOldTestU_TrainL=object$kernelFunction(distanceXTestU_TrainL,object$extraKernel)
#   
#   if(any(is.na(kernelNewOldTestU_TrainL))) stop("Kernel with NA")
#   
#   nX=object$nX
#   nZ=object$nZ
#   
#   mTestL=dim(kernelNewOldTestL_TrainL)[1] # New
#   nTrainL=dim(kernelNewOldTestL_TrainL)[2] # Old
#   mTestU=dim(kernelNewOldTestU_TrainL)[1] # New
#   
#   basisZ=calculateBasis(zTestL,nZ,object$system) # returns matrix length(z)xnZ with the basis for z.
#   
#   eigenVectors=object$eigenX
#   eigenValues=object$eigenValuesX
#   
#   basisXTestL=kernelNewOldTestL_TrainL %*% eigenVectors
#   basisXTestL=1/nTrainL*basisXTestL*matrix(rep(1/eigenValues,mTestL),mTestL,nX,byrow=T)
#   
#   basisXTestU=kernelNewOldTestU_TrainL %*% eigenVectors
#   basisXTestU=1/nTrainL*basisXTestU*matrix(rep(1/eigenValues,mTestU),mTestU,nX,byrow=T)
#   
#   rm(eigenVectors)
#   
#   W=1/mTestU*t(basisXTestU)%*%basisXTestU
#   
#   matrixBeta=matrix(weightsZTestL,mTestL,nX)
#   basisPsiMean=1/mTestL*t(t(basisZ)%*%(basisXTestL*(matrixBeta)))  
#   
#   grid=expand.grid(1:nX,1:nZ) 
# 
#   
#   prodMatrix=lapply(as.matrix(1:nZ),function(xx)
#   {
#     auxMatrix=W*object$coefficients[,xx,drop=F]%*%t(object$coefficients[,xx,drop=F])
#     returnValue=diag(apply(t(apply(auxMatrix, 1, cumsum)), 2, cumsum))
#     return(returnValue[1:nX])
#   })
#   prodMatrix=sapply(prodMatrix,function(xx)xx)
#   D=t(apply(prodMatrix,1,cumsum))
#   
#   rm(prodMatrix)
#   
#   errors=apply(grid,1,function(xx) { sBeta=1/2*D[xx[1],xx[2]]
#                                      sLikeli=sum(object$coefficients[1:xx[1],1:xx[2]]*basisPsiMean[1:xx[1],1:xx[2]]) 
#                                      return(sBeta-sLikeli)} )
#   errors=matrix(errors,nX,nZ)
#   rm(D,W)
#   rm(basisPsiMean)
#   
#   pointMin=which(errors == min(errors,na.rm=T), arr.ind = TRUE)
#   object$nXBest=(1:nX)[pointMin[1]]
#   object$nZBest=(1:nZ)[pointMin[2]]
#   object$errors=errors
#   object$bestError=min(errors)
#   return(object)
# }


chooseDeltaStatio <- function (object, zValidation, distanceXValidationTrain,deltaGrid=seq(0,0.3,0.04), nBins) {
  error=errorSE=rep(NA,length(deltaGrid))
  for(ii in 1:length(deltaGrid))
  {
    print(ii/length(deltaGrid))
    estimateErrors=estimateErrorFinalEstimatorStatio(object,zValidation,distanceXValidationTrain,boot=100,delta=deltaGrid[ii],
                                                     zMin = 0, zMax = 1, nBins = nBins)
    error[ii]=estimateErrors$mean  
    errorSE[ii]=estimateErrors$seBoot  
  }
  plot(deltaGrid,error+errorSE,type="l",lwd=2,ylim=c(min(error),max(error+errorSE)))
  lines(deltaGrid,error,lwd=3)  
  whichMin=(1:length(error))[error==min(error)]
  whichMin=max(whichMin)
  bestDelta=deltaGrid[whichMin]
  bestDeltaoneSD=max(deltaGrid[error<=(error+errorSE)[whichMin]])
  returnValue=NULL
  returnValue$bestDelta=bestDelta
  returnValue$bestDeltaoneSD=bestDeltaoneSD
  return(returnValue)
}


condDensityStatio=function(distancesX,z,nZMax=20,nXMax=NULL,kernelFunction=radialKernelDistance,extraKernel=list("eps.val"=1),normalization=NULL,system="Fourier")
{
  # estimates f(z|X)
  # z is assumed to be between 0 and 1
  # returns all coefficients up to nZMax, the maximum number of components
  # for Z, same for X
  kernelMatrix=kernelFunction(distancesX,extraKernel)
  if(any(is.na(kernelMatrix))) stop("Kernel with NA")
  n = length(distancesX[,1])
  
  normalizationParameters=NULL
  normalizationParameters$kernelMatrixN=kernelMatrix
  kernelMatrixN=kernelMatrix
  if(!is.null(normalization)) # if data (X) needs to be normalized first
  {
    if(normalization=="symmetric")  
    {
      normalizationParameters=symmetricNormalization(kernelMatrix)
      kernelMatrixN=normalizationParameters$KernelMatrixN # normalized matrix
    } else {
      stop("Normalization rule not implemented!")
    }
  }
  print("Computing EigenVectors\n")
  
  #results=eigen(kernelMatrixN,symmetric=T)
  
  
  if(is.null(nXMax)) nXMax=length(z)-10
  p=10
  Omega=matrix(rnorm(n*(nXMax+p),0,1),n,nXMax+p)
  Z=kernelMatrixN%*%Omega
  Y=kernelMatrixN%*%Z
  Q=qr(x=Y)
  Q=qr.Q(Q)
  B=t(Q)%*%Z%*%solve(t(Q)%*%Omega)
  eigenB=eigen(B)
  lambda=eigenB$values
  U=Q%*%eigenB$vectors
  basisX=Re(sqrt(n)*U[,1:nXMax])
  eigenValues=Re((lambda/n)[1:nXMax])
  rm(Omega,Z,Y,Q,B,U,eigenB)
  gc()
  
  nX=nXMax
  
  print("Done")
  
  basisZ=calculateBasis(z,nZMax,system) # returns vector length(z)xnZ with the basis for z.
  
  #coefficients=apply(grid,1,function(xx)mean(basisX[,xx[1]]*basisZ[,xx[2]]))
  
  coefficients=1/n*t(t(basisZ)%*%basisX)
  #coefficients=apply(grid,1,function(xx)sum(basisX[,xx[1]]*basisZ[,xx[2]]))
  
  object=list()
  class(object)="cDensity"
  matrixCoef=matrix(coefficients,nX,nZMax)
  object$coefficients=matrixCoef
  object$system=system
  object$nX=nX
  object$nZ=nZMax
  object$normalizationParameters=normalizationParameters
  if(is.null(normalization)) normalization=0
  object$normalization=normalization
  object$eigenX=basisX
  object$eigenValuesX=eigenValues
  object$kernelFunction=kernelFunction
  object$extraKernel=extraKernel
  return(object)  
}  


estimateErrorFinalEstimatorKNNContinuousStatio=function(nNeigh,nBins,bandwidthBinsOpt,zMin,zMax,zTrainL,distanceXTestTrainL,zTestU,boot=F, add_pred = F,
                                                        predictedComplete = numeric(), predictedObserved = numeric(),
                                                        normalization = F){
  # nNeigh,nBinsOpt,zMin,zMax,zTrainL,zTrainLWeights,c are needed to compute the estimates
  # c is optimal power to use for weights
  zGrid=seq(zMin,zMax,length.out=nBins)
  output=NULL
  output$mean=NULL
  output$seBoot=NULL
  
  # get fzx predictions (complete) if not given as argument
  if(length(predictedComplete) == 0){
    predictedComplete = predictDensityKNN(distanceXTestTrain = distanceXTestTrainL, zTrain = zTrainL,
                                  KNNneighbors = nNeigh, KNNbandwidth = bandwidthBinsOpt, 
                                  nBins = nBins, normalization = normalization)
  }
  
  colmeansComplete=colMeans(predictedComplete^2)
  sSquare=mean(colmeansComplete)
  
  n=length(zTestU)
  if(length(predictedObserved) == 0){
    predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
                                                            return(predictedComplete[xx,index])
    })
  }
  
  likeli=mean(predictedObserved)
  output$mean=1/2*sSquare-likeli
  if(boot==F)
  {
    return(output);
  }

  
  # Bootstrap
  meanBoot=apply(as.matrix(1:boot),1,function(xx){
    sampleBoot=sample(1:n,replace=T)
    
    predictedCompleteBoot=predictedComplete[sampleBoot,]
    zTestBoot=zTestU[sampleBoot]
    
    colmeansComplete=colMeans(predictedCompleteBoot^2)
    sSquare=mean(colmeansComplete)
    
    predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestBoot[xx]-zGrid))
                                                            return(predictedCompleteBoot[xx,index])
    })
    likeli=mean(predictedObserved_boot)
    return(1/2*sSquare-likeli)    
  })
  output$seBoot=sqrt(var(meanBoot))
  
  if(add_pred == TRUE){
    return(list(output = output, predictedComplete = predictedComplete, predictedObserved = predictedObserved))
  }else{
    return(output)
  }
}


###M: Add two functions for KNN 1. to get the predictions seperately and 2. estimate loss by loading the predictedObserved and predictedComplete + the zTest_ordered values


estimate_stratifiedpredictions_Statio_KNN = function(nNeigh,nBins,bandwidthBinsOpt,zMin,zMax,zTrain,distanceXTestTrainL,zTest,
                                                   predictedComplete = numeric(), normalization = F){
  # nNeigh,nBinsOpt,zMin,zMax,zTrainL,zTrainLWeights,c are needed to compute the estimates
  # c is optimal power to use for weights
  # normalization: if normalization = T, and if predictedComplete = numeric(), then the predictions below will be normalized preds
  zGrid=seq(zMin,zMax,length.out=nBins)
  
  if(length(predictedComplete) == 0 ){
    predictedComplete = predictDensityKNN(distanceXTestTrain = distanceXTestTrainL, zTrain = zTrain,
                                          KNNneighbors = nNeigh, KNNbandwidth = bandwidthBinsOpt, 
                                          nBins = nBins, normalization = normalization)
  }
  
  n=length(zTest)
  predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTest[xx]-zGrid))
  return(predictedComplete[xx,index])
  })
  
  return(list(predictedComplete = predictedComplete, predictedObserved = predictedObserved, zTest = zTest))
}
  


estimate_combined_stratified_risk_Statio_KNN=function(predictedComplete, predictedObserved, zTestU_ordered, boot = F, zMin, zMax, nBins)
{
  # nNeigh,nBinsOpt,zMin,zMax,zTrainL,zTrainLWeights,c are needed to compute the estimates
  # c is optimal power to use for weights
  zGrid=seq(zMin,zMax,length.out=nBins)
  n=length(zTestU_ordered)
  
  output=NULL
  output$mean=NULL
  output$seBoot=NULL
  
  ## compute the loss
  colmeansComplete=colMeans(predictedComplete^2)
  sSquare=mean(colmeansComplete)
  
  likeli=mean(predictedObserved)
  output$mean=1/2*sSquare-likeli
  if(boot==F)
  {
    return(output);
  }
  
  
  # Bootstrap
  meanBoot=apply(as.matrix(1:boot),1,function(xx){
    sampleBoot=sample(1:n,replace=T)
    
    predictedCompleteBoot=predictedComplete[sampleBoot,]
    zTestBoot=zTestU_ordered[sampleBoot]
    
    colmeansComplete=colMeans(predictedCompleteBoot^2)
    sSquare=mean(colmeansComplete)
    
    predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestBoot[xx]-zGrid))
    return(predictedCompleteBoot[xx,index])
    })
    likeli=mean(predictedObserved_boot)
    return(1/2*sSquare-likeli)    
  })
  output$seBoot=sqrt(var(meanBoot))
  
  return(output)
}


### KNN (statio) predictions of complete conditional densities
predictDensityKNN = function(distanceXTestTrain, zTrain, KNNneighbors, KNNbandwidth,
                             zMin = 0, zMax = 1, nBins, normalization = F, delta = 0){
  zGrid=seq(from=zMin, to=zMax,length.out=nBins)
  
  estimates=t(apply(distanceXTestTrain,1,function(xx){
    nearest=sort(xx,index.return=T)$ix[1:KNNneighbors]
    densityObject=condDensityKNNContinuousStatio(zTrain[nearest],length(zGrid),KNNbandwidth,0,1)
    return(densityObject$means) 
  }))
  if(normalization == T){
    binSize=(zMax-zMin)/(nBins-1)
    estimates=t(apply(estimates,1,function(xx)normalizeDensity(binSize,xx,delta)))
  }
  return(estimates)
}



########################



condDensityKNNContinuousStatio=function(zTrainNearest,nBins=1000,bandwidth,zMin,zMax)
{
  # calculates estimated density for a single observation
  # weights are previously calculated weights, based on unlabeled data,
  # one weight for each traning sample
  binsMedium=seq(zMin,zMax,length.out=nBins)
  estimates=apply(as.matrix(binsMedium),1,function(xx)
  {
    weightsFinal=exp(-abs(xx-zTrainNearest)^2/(4*bandwidth))/sqrt(pi*4*bandwidth)
    return(sum(weightsFinal)/length(weightsFinal))
  })
  output=NULL
  output$means=estimates
  output$binsIntervals=binsMedium
  return(output)
}


estimateErrorFinalEstimatorKNNStatio=function(nNeigh,nBins,nBinsOpt,zMin,zMax,zTrain,distanceXValidationTrain,zValidation,boot=F, add_pred = F)
{
  zGrid=seq(zMin,zMax,length.out=nBins)
  binsIntervals=seq(zMin,zMax,length.out=nBinsOpt)
  whichClosest=apply(as.matrix(zGrid),1,function(yy)  {
    which.min(abs(binsIntervals-yy))
  })
  
  predictedComplete=t(apply(distanceXValidationTrain,1,function(xx)
  {
    nearest=sort(xx,index.return=T)$ix[1:nNeigh]
    densityObject=condDensityKNNStatio(zTrain[nearest],nBinsOpt,zMin,zMax)
    means=densityObject$means[whichClosest]
    return(means)
  }))  
  
  colmeansComplete=colMeans(predictedComplete^2)
  sSquare=mean(colmeansComplete)
  
  n=length(zValidation)
  predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zValidation[xx]-zGrid))
                                                          return(predictedComplete[xx,index])
  })
  likeli=mean(predictedObserved)
  output=NULL
  output$mean=1/2*sSquare-likeli
  if(boot==F)
  {
    return(output)
  }
  
  # Bootstrap
  meanBoot=apply(as.matrix(1:boot),1,function(xx){
    sampleBoot=sample(1:n,replace=T)
    
    predictedCompleteBoot=predictedComplete[sampleBoot,]
    zValidationBoot=zValidation[sampleBoot]
    
    colmeansComplete=colMeans(predictedCompleteBoot^2)
    sSquare=mean(colmeansComplete)
    
    predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zValidationBoot[xx]-zGrid))
                                                            return(predictedCompleteBoot[xx,index])
    })
    likeli=mean(predictedObserved_boot)
    return(1/2*sSquare-likeli)    
  })
  output$seBoot=sqrt(var(meanBoot))
  
  if(add_pred == TRUE){
    return(list(output = output, predictedComplete = predictedComplete, predictedObserved = predictedObserved))
  }else{
    return(output)
  }
}


#### Add two NN functions, one to gie the predictions (complete and observed) + ordered output, and the other to estimate risk based on the predictions

estimate_stratifiedpredictions_Statio_NN=function(nNeigh,nBins,nBinsOpt,zMin,zMax,zTrain,distanceXValidationTrain,zTestU,boot=F)
{
  zGrid=seq(zMin,zMax,length.out=nBins)
  binsIntervals=seq(zMin,zMax,length.out=nBinsOpt)
  whichClosest=apply(as.matrix(zGrid),1,function(yy)  {
    which.min(abs(binsIntervals-yy))
  })
  
  predictedComplete=t(apply(distanceXValidationTrain,1,function(xx)
  {
    nearest=sort(xx,index.return=T)$ix[1:nNeigh]
    densityObject=condDensityKNNStatio(zTrain[nearest],nBinsOpt,zMin,zMax)
    means=densityObject$means[whichClosest]
    return(means)
  }))  

  
  n=length(zTestU)
  predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
  return(predictedComplete[xx,index])
  })

  return(list(predictedComplete = predictedComplete, predictedObserved = predictedObserved, zTest = zTestU))
}


estimate_combined_stratified_risk_Statio_NN=function(predictedComplete, predictedObserved, zTestU_ordered, boot = F, zMin, zMax, nBins)
{
  zGrid=seq(zMin,zMax,length.out=nBins)
  n=length(zTestU_ordered)  
    
  colmeansComplete=colMeans(predictedComplete^2)
  sSquare=mean(colmeansComplete)
  
  likeli=mean(predictedObserved)
  output=NULL
  output$mean=1/2*sSquare-likeli
  if(boot==F)
  {
    return(output)
  }
  
  # Bootstrap
  meanBoot=apply(as.matrix(1:boot),1,function(xx){
    sampleBoot=sample(1:n,replace=T)
    
    predictedCompleteBoot=predictedComplete[sampleBoot,]
    zTestUBoot=zTestU_ordered[sampleBoot]
    
    colmeansComplete=colMeans(predictedCompleteBoot^2)
    sSquare=mean(colmeansComplete)
    
    predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestUBoot[xx]-zGrid))
    return(predictedCompleteBoot[xx,index])
    })
    likeli=mean(predictedObserved_boot)
    return(1/2*sSquare-likeli)    
  })
  output$seBoot=sqrt(var(meanBoot))
  return(output)
}


################################################################



estimateErrorFinalEstimatorStatio=function(object,zTest,distanceXTestTrain,boot=F,delta=0, add_pred = F,
                                           zMin, zMax, nBins,
                                           predictedComplete = numeric(), predictedObserved = numeric()){
  # estimates the loss of a given estimator, after fitting was done
  # boot==F if no error estimates, otherwise boot is the number of bootstrap samples
  zGrid=seq(zMin,zMax,length.out=nBins)
  
  if(length(predictedComplete) == 0){
    predictedComplete=predictDensityStatio(object,zTestMin=0,zTestMax=1,B=length(zGrid),distanceXTestTrain,probabilityInterval=F,delta=delta)
  }
  
  colmeansComplete=colMeans(predictedComplete^2)
  sSquare=mean(colmeansComplete)
  
  n=length(zTest)
  if(length(predictedObserved) == 0){
    predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTest[xx]-zGrid))
                                                          return(predictedComplete[xx,index]) # 1 indicates that fct is applied over rows
    })
    #M: The function (above) computes the value of the predicted density at the actual (spectroscopically) observed redshift.
  }
  likeli=mean(predictedObserved)
  output=NULL
  output$mean=1/2*sSquare-likeli
  if(boot==F)
  {
    return(output)
  }
  
  # Bootstrap
  meanBoot=apply(as.matrix(1:boot),1,function(xx){
    sampleBoot=sample(1:n,replace=T)
    
    predictedCompleteBoot=predictedComplete[sampleBoot,]
    zTestBoot=zTest[sampleBoot]
    
    colmeansComplete=colMeans(predictedCompleteBoot^2)
    sSquare=mean(colmeansComplete)
    
    predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestBoot[xx]-zGrid))
                                                            return(predictedCompleteBoot[xx,index])
    })
    likeli=mean(predictedObserved_boot)
    return(1/2*sSquare-likeli)    
  })
  output$seBoot=sqrt(var(meanBoot))
  
  if(add_pred == TRUE){
    return(list(output = output, predictedComplete = predictedComplete, predictedObserved = predictedObserved))
  }else{
    return(output)
  }
}



estimate_stratifiedpredictions_Statio=function(object,zTest,distanceXTestTrain,delta=0,
                                               zMin, zMax, nBins,
                                               predictedComplete = numeric()){
  # estimates the loss of a given estimator, after fitting was done
  # boot==F if no error estimates, othewise boot is the number of bootstrap samples
  zGrid=seq(zMin,zMax,length.out=nBins)
  if(length(predictedComplete) == 0){
    # if set of predcited conditional densities is already given as input, then no need to recompute it here!
    predictedComplete=predictDensityStatio(object,zTestMin=0,zTestMax=1,B=length(zGrid),distanceXTestTrain,probabilityInterval=F,delta=delta)
  }
  n=length(zTest)
  predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTest[xx]-zGrid))
  return(predictedComplete[xx,index]) # 1 indicates that fct is applied over rows
  #M: This function (above) computes the "magnitude" of the predicted density at the actual (spectroscopically) observed redshift.
  })
  
  return(list(predictedComplete = predictedComplete, predictedObserved = predictedObserved))
}




estimate_combined_stratified_risk_Statio=function(predictedComplete, predictedObserved, zTest_ordered, boot = F,
                                                  zMin, zMax, nBins){
  # estimates the loss of a given estimator, after fitting was done
  # boot==F if no error estimates, othewise boot is the number of bootstrap samples
  zGrid=seq(zMin,zMax,length.out=nBins)
  
  colmeansComplete=colMeans(predictedComplete^2)
  sSquare=mean(colmeansComplete)
  
  n=length(zTest_ordered)

  likeli=mean(predictedObserved)
  output=NULL
  output$mean=1/2*sSquare-likeli
  if(boot==F)
  {
    return(output)
  }
  
  # Bootstrap
  meanBoot=apply(as.matrix(1:boot),1,function(xx){
    sampleBoot=sample(1:n,replace=T)
    
    predictedCompleteBoot=predictedComplete[sampleBoot,]
    zTestBoot=zTest_ordered[sampleBoot]
    
    colmeansComplete=colMeans(predictedCompleteBoot^2)
    sSquare=mean(colmeansComplete)
    
    predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestBoot[xx]-zGrid))
    return(predictedCompleteBoot[xx,index])
    })
    likeli=mean(predictedObserved_boot)
    return(1/2*sSquare-likeli)    
  })
  output$seBoot=sqrt(var(meanBoot))
  return(output)
}




estimateErrorEstimatorStatio=function(object,zTest,distanceXTestTrain)
{
  # returns a nX by nZ matrix with the errors for the different possibilities
  # of number of components, plus the best combination
  if(class(object)!="cDensity") stop("Object should be of class cDensity")
  
  kernelNewOld=object$kernelFunction(distanceXTestTrain,object$extraKernel)
  
  
  if(object$normalization!=0)
  {
    if(object$normalization=="symmetric")
    {
      sqrtColMeans=object$normalizationParameters$sqrtColMeans
      sqrtRowMeans=sqrt(rowMeans(kernelNewOld))
      kernelNewOld=kernelNewOld/(sqrtRowMeans%*%t(sqrtColMeans))
    }
  }
  
  if(any(is.na(kernelNewOld))) stop("Kernel with NA")
  
  nX=object$nX
  nZ=object$nZ
  
  m=dim(kernelNewOld)[1] # New
  n=dim(kernelNewOld)[2] # Old
  
  #MA_d: basisZ seems to be phi in Izbicki (2017), then nZBest seems to be the parameter I (page 713)
  basisZ=calculateBasis(zTest,nZ,object$system) # returns matrix length(z)xnZ with the basis for z. 
  
  eigenVectors=object$eigenX
  eigenValues=object$eigenValuesX
  
  basisX=kernelNewOld %*% eigenVectors
  basisX=1/n*basisX*matrix(rep(1/eigenValues,m),m,nX,byrow=T)
  rm(eigenVectors)
  
  basisPsiMean=1/m*t(t(basisZ)%*%basisX)  
  W=1/m*t(basisX)%*%basisX
  
  prodMatrix=lapply(as.matrix(1:nZ),function(xx)
  {
    auxMatrix=W*object$coefficients[,xx,drop=F]%*%t(object$coefficients[,xx,drop=F])
    returnValue=diag(apply(t(apply(auxMatrix, 1, cumsum)), 2, cumsum))
    return(returnValue[1:nX])
  })
  prodMatrix=sapply(prodMatrix,function(xx)xx)
  D=t(apply(prodMatrix,1,cumsum))
  rm(W)
  rm(prodMatrix)
  
  grid=expand.grid(1:nX,1:nZ) 
  
  errors=apply(grid,1,function(xx) { sBeta=1/2*D[xx[1],xx[2]]
                                     sLikeli=sum(object$coefficients[1:xx[1],1:xx[2]]*basisPsiMean[1:xx[1],1:xx[2]]) 
                                     return(sBeta-sLikeli)} )
  errors=matrix(errors,nX,nZ)
  rm(D)
  rm(basisPsiMean)
  
  pointMin=which(errors == min(errors,na.rm=T), arr.ind = TRUE)
  nXBest=(1:nX)[pointMin[1]]
  nZBest=(1:nZ)[pointMin[2]]
  object$nXBest=nXBest
  object$nZBest=nZBest
  object$errors=errors
  object$bestError=min(errors)
  return(object)
}


predictDensityStatio=function(object,zTestMin=0,zTestMax=1,B=1000,distanceXTestTrain,probabilityInterval=F,confidence=0.95,delta=0)
{
  # predict density at points zTest=seq(zTestMin,zTestMax,lenght.out=B) and xTest
  # delta is the treshhold to remove bumps
  print("Densities normalized to integrate 1 in the range of z given!")
  
  if(class(object)!="cDensity") stop("Object should be of class cDensity")
  
  zGrid=seq(from=zTestMin,to=zTestMax,length.out=B)
  
  kernelNewOld=object$kernelFunction(distanceXTestTrain,object$extraKernel)
  if(object$normalization!=0)
  {
    if(object$normalization=="symmetric")
    {
      sqrtColMeans=object$normalizationParameters$sqrtColMeans
      sqrtRowMeans=sqrt(rowMeans(kernelNewOld))
      kernelNewOld=kernelNewOld/(sqrtRowMeans%*%t(sqrtColMeans))
    }
  }
  
  nXBest=object$nX
  nZBest=object$nZ
  
  
  if(!is.null(object$nXBest)) # if CV was done
  {
    nXBest=object$nXBest
    nZBest=object$nZBest
  }
  
  
  m=dim(kernelNewOld)[1] # New
  n=dim(kernelNewOld)[2] # Old
  
  basisZ=calculateBasis(zGrid,nZBest,object$system) # returns matrix length(z)xnZ with the basis for z.
  
  
  eigenVectors=object$eigenX[,1:nXBest,drop=F]
  eigenValues=object$eigenValuesX[1:nXBest]
  
  basisX=kernelNewOld %*% eigenVectors
  basisX=1/n*basisX*matrix(rep(1/eigenValues,m),m,nXBest,byrow=T)
  
  nTestZ=length(zGrid) # how many test points for z
  nTextX=dim(kernelNewOld)[1] # how many test points for x
  
  grid=expand.grid(1:nTextX,1:nTestZ)
  
  coefficients=object$coefficients[1:nXBest,1:nZBest,drop=F]
  
  #estimates=apply(grid,1,function(xx)sum(basisX[xx[1],]%*%t(basisZ[xx[2],])*coefficients))
  
  estimates=t(apply(basisX,1,function(yy)
  {
    apply(basisZ,1,function(xx)sum(outer(yy,xx)*coefficients))
  }))
  
  #estimates=matrix(estimates,nTextX,nTestZ)
  
  #estimates[estimates<0]=0
  
  binSize=(zTestMax-zTestMin)/(B-1)
  #estimates=t(apply(estimates,1,function(xx)1/binSize*xx/sum(xx))) # normalize
  
  estimates=t(apply(estimates,1,function(xx)normalizeDensity(binSize,xx,delta)))
  
  if(!probabilityInterval)  return(estimates)
  
  # Gives threshold on density corresponding to probability interval
  thresholds=apply(estimates,1,function(xx)findThreshold(binSize,xx,confidence))
  objectReturn=list()
  objectReturn$estimates=estimates
  objectReturn$thresholdsIntervals=thresholds
  return(objectReturn)
  
}



############################################################################################################
#### M: functions added for loss evaluation


stratified_loss_function = function(all_predictedComplete, all_predictedObserved, all_zTestU, zMin,zMax, nBins, groups, first_group, boot = T){
  zGrid=seq(zMin,zMax,length.out=nBins)
  ### the loaded groups argument already needs to be subset to the actually to be tested set
  all_outputs = list()
  
  if(length(all_predictedObserved) == 0){
    ## compute predictedObserved    
    n=length(all_zTestU)
    all_predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(all_zTestU[xx]-zGrid))
    return(all_predictedComplete[xx,index])
    # 1 indicates that fct is applied over rows
    #M: This function (above) computes the "magnitude" of the predicted density at the actual (spectroscopically) observed redshift.
    })
    print("The vector all_predictedObserved was computed!")
  }
  
  for(i in first_group:max(groups)){
    predictedComplete = all_predictedComplete[groups == i,]
    predictedObserved = all_predictedObserved[groups == i]  
    zTestU = all_zTestU[groups == i]  
    
    output=NULL
    output$mean=NULL
    output$seBoot=NULL
    
    colmeansComplete=colMeans(predictedComplete^2)
    sSquare=mean(colmeansComplete)
    
    n=length(zTestU)
    likeli=mean(predictedObserved)
    
    output$mean=1/2*sSquare-likeli
    if(boot==F)
    {
      return(output)
    }
    
    print("Bootstrap error is computed")
    meanBoot=apply(as.matrix(1:boot),1,function(xx){
      sampleBoot=sample(1:n,replace=T)
      
      predictedCompleteBoot=predictedComplete[sampleBoot,]
      zTestBoot=zTestU[sampleBoot]
      
      colmeansComplete=colMeans(predictedCompleteBoot^2)
      sSquare=mean(colmeansComplete)
      
      predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestBoot[xx]-zGrid))
      return(predictedCompleteBoot[xx,index])
      })
      likeli=mean(predictedObserved_boot)
      return(1/2*sSquare-likeli)    
    })
    output$seBoot=sqrt(var(meanBoot))
    
    all_outputs[[i]] = output
  }
  
  return(all_outputs)
}  




####M: loss computation given the complete predictions (but not observed_pred),
# below comb_test_loss_fct is e.g. used to get loss of combined predictions


comb_test_loss_fct = function(predictedComplete, zTestU, zMin,zMax, nBins, boot = T){
  zGrid=seq(zMin,zMax,length.out=nBins)
  ### the loaded groups argument already needs to be subset to the actually to be tested set

    output=NULL
    output$mean=NULL
    output$seBoot=NULL
    
    colmeansComplete=colMeans(predictedComplete^2)
    sSquare=mean(colmeansComplete)
    
    ## compute predictedObserved    
    n=length(zTestU)
    predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
    return(predictedComplete[xx,index])
    # 1 indicates that fct is applied over rows
    #M: This function (above) computes the "magnitude" of the predicted density at the actual (spectroscopically) observed redshift.
    })
    
    likeli=mean(predictedObserved)
    
    output$mean=1/2*sSquare-likeli
    if(boot==F)
    {
      return(output)
    }
    
    print("Bootstrap error is computed")
    meanBoot=apply(as.matrix(1:boot),1,function(xx){
      sampleBoot=sample(1:n,replace=T)
      
      predictedCompleteBoot=predictedComplete[sampleBoot,]
      zTestBoot=zTestU[sampleBoot]
      
      colmeansComplete=colMeans(predictedCompleteBoot^2)
      sSquare=mean(colmeansComplete)
      
      predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestBoot[xx]-zGrid))
      return(predictedCompleteBoot[xx,index])
      })
      likeli=mean(predictedObserved_boot)
      return(1/2*sSquare-likeli)    
    })
    output$seBoot=sqrt(var(meanBoot))
    
  
  return(output)
}  





