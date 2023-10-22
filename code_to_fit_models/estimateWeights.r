
require(fields)
require(ggplot2)
require(scales)
require(grid)


radialKernel=function(distances,extraKernel=list("eps.val"=1))
{
  # Given the distances and the bandwidth eps.val, computes the matrix of radial kernel
  return(exp(-distances^2/(4*extraKernel$eps.val)))
}


seriesRatio=function(xFTrain=NULL,xGTrain=NULL,xFValidation=NULL,xGValidation=NULL,
                     kernelGTrain_GTrain=NULL,kernelFTrain_GTrain=NULL,
                     kernelGValidation_GTrain=NULL,kernelFValidation_GTrain=NULL,nXMax=100,
                     methodEigen=NULL,kernelFunction=ifelse(is.null(kernelGTrain_GTrain),
                                                            radialKernel,NA),
                     extraKernel=NULL,verbose=TRUE)
{
  # main function
  if(!is.null(kernelGTrain_GTrain)|!is.null(kernelFTrain_GTrain)|!is.null(kernelFValidation_GTrain)|!is.null(kernelGValidation_GTrain))
  {
    if(!is.null(xFTrain)|!is.null(xGTrain)|!is.null(xFValidation)|!is.null(xGValidation))
      stop("Please provide wither xTrain-type of arguments or kernelTrainTrain-type of arguments")
  }
  
  
  if(!is.null(kernelGTrain_GTrain))
  {
    if(is.null(kernelFTrain_GTrain)|is.null(kernelFValidation_GTrain)|is.null(kernelGValidation_GTrain))
      stop("Please provide either all kernelData-type arguments or all XData-type arguments")
  }
  if(!is.null(xFTrain))
  {
    if(is.null(xGTrain)|is.null(xFValidation)|is.null(xGValidation))
      stop("Please provide either all kernelData-type arguments or all XData-type arguments")
  }
  
  if(is.null(xFTrain)&!is.null(xFValidation)) stop("Please provide xTrain if you provide xValidation")
  
  if(!is.null(xFTrain)&is.null(xFValidation)) stop("Please provide xValidation if you provide xTrain")
  
  if(!is.null(xFTrain)&!is.null(kernelGTrain_GTrain)) stop("Please provide either xTrain of kernelTrainTrain")
  
  if(!is.null(xFTrain)&!is.null(kernelFTrain_GTrain)) stop("Please provide either xValidation of kernelValidationTrain")
  
  
  if(is.null(xFTrain)&(is.null(kernelFValidation_GTrain)|is.null(kernelGValidation_GTrain))) stop("You have to provide either kernelTrainTrain and kernelValidationTrain or xTrain and xValidation")
  
  if(!is.null(xGValidation))
  {      
    
    if(!is.matrix(xGValidation))
      stop("xValidation has to be a Matrix")
    
    if(!is.matrix(xGTrain))
      stop("xTrain has to be a Matrix")
    
    if(!is.matrix(xFValidation))
      stop("xValidation has to be a Matrix")
    
    if(!is.matrix(xFTrain))
      stop("xTrain has to be a Matrix")
    
    if(dim(xFTrain)[2]!=dim(xGTrain)[2]) stop("Dimensions of x train don't match!")
    
    if(dim(xFValidation)[2]!=dim(xGValidation)[2]) stop("Dimensions of x validation don't match!")
    
    if(dim(xFTrain)[2]!=dim(xGValidation)[2]) stop("Dimensions of x don't match!")
    
  }
  
  if(is.null(methodEigen))
  {
    if(!is.null(xGValidation))
    {
      methodEigen=ifelse(dim(xGTrain)[1]>500,"SVDRandom","SVD")
    } else {
      if(!is.na(kernelFunction))
        warning("Argument's kernelFunction and extraKernel is being ignored: inputs given by the user are already Gram matrices")
      methodEigen=ifelse(dim(kernelGTrain_GTrain)[1]>500,"SVDRandom","SVD") 
    }
  }
  
  if(verbose) cat("\n Fit model")  
  
  if(!is.null(xGValidation))
  {
    object=.seriesRatioFit(xFTrain=xFTrain,xGTrain=xGTrain,nXMax=nXMax,methodEigen=methodEigen,kernelFunction=kernelFunction,extraKernel=extraKernel,userCall=FALSE,verbose=verbose)
    
    if(verbose) cat("\n Tune I")  
    
    object=.seriesRatioTuning(object=object,xFValidation=xFValidation,xGValidation=xGValidation,userCall=FALSE)
  } else {
    object=.seriesRatioFit(kernelFTrain_GTrain=kernelFTrain_GTrain,kernelGTrain_GTrain=kernelGTrain_GTrain,nXMax=nXMax,methodEigen=methodEigen,kernelFunction=kernelFunction,extraKernel=extraKernel,userCall=FALSE,verbose=verbose)
    
    if(verbose) cat("\n Tune I")  
    
    object=.seriesRatioTuning(object=object,kernelFValidation_GTrain=kernelFValidation_GTrain,kernelGValidation_GTrain=kernelGValidation_GTrain,userCall=FALSE)  
  }
  
  if(verbose) cat("\n Compute Validation set error")
  object$bestError=min(object$errors)
  
  if(abs(object$nXBest-object$nX)<5) warning("nXBest is close to nXMax, you may consider increasing nXMax")
  
  return(object)  
}


.seriesRatioFit=function(xFTrain=NULL,xGTrain=NULL,kernelFTrain_GTrain=NULL,kernelGTrain_GTrain=NULL,nXMax=NULL,methodEigen,kernelFunction=radialKernel,extraKernel=NULL,userCall=TRUE,verbose=TRUE)
{
  if(userCall)
    cat("\n A user will typically use function 'seriesCDE' instead, are you sure about what you are doing?")
  
  object=list()
  class(object)="seriesRatio"
  
  if(is.null(kernelFTrain_GTrain)) # if kernel matrix is not provided
  {
    distancesXG=rdist(xGTrain,xGTrain) # of all training samples G
    
    object$kernelFunction=kernelFunction
    object$xGTrain=xGTrain
    
    if(identical(kernelFunction,radialKernel)&is.null(extraKernel)) # default choice of bandwidth
    {
      extraKernel=list("eps.val"=median(distancesXG^2)/8)
    }
    kernelMatrixG=kernelFunction(distancesXG,extraKernel)    
    object$extraKernel=extraKernel
    rm(distancesXG)
    
    kernelF_G=kernelFunction(rdist(xFTrain,xGTrain),extraKernel)
    rm(xFTrain,xGTrain)
    
  } else{
    kernelMatrixG=kernelGTrain_GTrain
    kernelF_G=kernelFTrain_GTrain
    rm(kernelGTrain_GTrain,kernelFTrain_GTrain)
  }
  
  if(any(is.na(kernelMatrixG))) stop("Kernel with NA")
  
  if(verbose) cat("\n Computing EigenVectors for basis for X")  
  
  nGTrain=dim(kernelMatrixG)[1]
  
  if(methodEigen=="SVD")
  {    
    results=eigen(kernelMatrixG,symmetric=TRUE)
    nX=dim(results$vectors)[2]
    
    if(!is.null(nXMax)) 
    {
      nXMax=min(nX,nXMax)
    } else {
      nXMax=nX
    }
    
    basisXG=sqrt(nGTrain)*results$vectors[,1:nXMax,drop=FALSE]
    eigenValuesG=results$values[1:nXMax]/nGTrain
    
    rm(results)
    
  } else if(methodEigen=="SVDRandom") {
    if(!is.null(nXMax)) 
    {
      nXMax=min(nGTrain-15,nXMax)
    } else {
      nXMax=nGTrain-15
    }
    
    p=10
    
    Omega=matrix(rnorm(nGTrain*(nXMax+p),0,1),nGTrain,nXMax+p)
    Z=kernelMatrixG%*%Omega
    Y=kernelMatrixG%*%Z
    Q=qr(x=Y)
    Q=qr.Q(Q)
    B=t(Q)%*%Z%*%solve(t(Q)%*%Omega)
    eigenB=eigen(B)
    lambda=eigenB$values
    U=Q%*%eigenB$vectors
    
    basisXG=Re(sqrt(nGTrain)*U[,1:nXMax])
    eigenValuesG=Re((lambda/nGTrain)[1:nXMax])
    
    rm(Omega,Z,Y,Q,B,U,eigenB)
    
  } else   {
    stop("Method not implemented")
  }
  
  
  nF=dim(kernelF_G)[1]
  
  basisXF=kernelF_G %*% basisXG
  basisXF=1/nGTrain*basisXF*matrix(rep(1/eigenValuesG,nF),nF,nXMax,byrow=TRUE)
  
  coefficients=colMeans(basisXF)
  
  
  gc(verbose=FALSE)
  object$coefficients=coefficients
  object$methodEigen=methodEigen
  object$nX=nXMax
  object$eigenXG=basisXG
  object$eigenValuesXG=eigenValuesG
  return(object)    
}  



.seriesRatioTuning=function(object,xFValidation=NULL,xGValidation=NULL,kernelFValidation_GTrain=NULL,kernelGValidation_GTrain=NULL,userCall=TRUE)
{
  # returns the best choice of I and J, the number of components to use in each
  # direction
  # zValidation expected to be between 0 and 1
  if(userCall)
    cat("\n A user will typically use function 'seriesCDE' instead, are you sure about what you are doing?")
  
  
  if(class(object)!="seriesRatio") stop("Object should be of class seriesRatio")
  
  if(is.null(kernelFValidation_GTrain))
  {
    kernelFValidation_GTrain=object$kernelFunction(rdist(xFValidation,object$xGTrain),object$extraKernel)
    kernelGValidation_GTrain=object$kernelFunction(rdist(xGValidation,object$xGTrain),object$extraKernel)
  }
  
  if(any(is.na(kernelFValidation_GTrain))) stop("Kernel with NA")
  if(any(is.na(kernelGValidation_GTrain))) stop("Kernel with NA")
  
  nX=object$nX
  
  
  mG=dim(kernelGValidation_GTrain)[1] # G Test
  mF=dim(kernelFValidation_GTrain)[1] # F Test
  nG=dim(kernelFValidation_GTrain)[2] # Labeled Train
  
  
  # basis on G validation
  basisXG=kernelGValidation_GTrain %*% object$eigenX
  basisXG=1/nG*basisXG*matrix(rep(1/object$eigenValuesX,mG),mG,nX,byrow=TRUE)
  
  # basis on F validation
  basisXF=kernelFValidation_GTrain %*% object$eigenX
  basisXF=1/nG*basisXF*matrix(rep(1/object$eigenValuesX,mF),mF,nX,byrow=TRUE)
  
  # Tune
  WG=1/mG*t(basisXG)%*%basisXG
  WF=colMeans(basisXF)
  
  firstTerm=lapply(as.matrix(1:nX),function(xx)
  {
    return(t(object$coefficients[1:xx,drop=FALSE])%*%WG[1:xx,1:xx,drop=FALSE]%*%object$coefficients[1:xx,drop=FALSE])
  })
  firstTerm=sapply(firstTerm,function(xx)as.numeric(xx))
  errors=apply(as.matrix(1:nX),1,function(xx) { sBeta=1/2*firstTerm[xx]
                                                sLikeli=sum(object$coefficients[1:xx,drop=FALSE]*WF[1:xx,drop=FALSE]) 
                                                return(sBeta-sLikeli)} )
  
  # return
  nXBest=(1:nX)[which.min(errors)]
  object$nXBest=nXBest
  object$errors=errors
  object$bestError=min(errors)
  return(object)
}




predictRatio=function(object,xTest=NULL,kernelTest_GTrain=NULL)
{
  # predict density at points zTest=seq(zMin,zMax,lenght.out=B) and xTest
  # delta is the treshhold to remove bumps
  
  if(class(object)!="seriesRatio") stop("Object should be of class seriesRatio")
  
  if(is.null(kernelTest_GTrain)&is.null(object$xGTrain)) 
    stop("It seems you trained the estimator using the
         kernelTrainTrain argument, please use kernelTest_GTrain instead of xTest here")
  
  
  if(is.null(kernelTest_GTrain))
  {
    kernelTest_GTrain=object$kernelFunction(rdist(xTest,object$xGTrain),object$extraKernel)
  }
  
  if(any(is.na(kernelTest_GTrain))) stop("Kernel with NA")
  
  if(is.null(object$nXBest)) 
  {
    nX=object$nX
    warning("Are you sure you want pick the best nX (I)?")
  } else
  {
    nX=object$nXBest
  }
  
  m=dim(kernelTest_GTrain)[1] # Test
  nG=dim(kernelTest_GTrain)[2] # Train  
  
  basisX=kernelTest_GTrain %*% object$eigenX[,1:nX,drop=FALSE]
  basisX=1/nG*basisX*matrix(rep(1/object$eigenValuesX[1:nX,drop=FALSE],m),m,nX,byrow=TRUE)
  
  estimates=basisX%*%object$coefficients[1:nX,drop=FALSE]
  estimates[estimates<0]=0
  return(estimates) 
}


estimateKuLSIF=function(KXTrainG_TrainG,KXTrainG_TrainF,lambda)
{
  # outputs the estimate at the points XTest
  n=dim(KXTrainG_TrainG)[2]
  m=dim(KXTrainG_TrainF)[2]
  object=list()
  object$n=n
  object$m=m
  object$lambda=lambda
  a=1/n*KXTrainG_TrainG+lambda*diag(rep(1,n))
  b=-1/(n*m*lambda)*KXTrainG_TrainF%*%rep(1,m)
  object$alpha=solve(a=a,b=b)
  return(object)
}

predictKuLSIF=function(object,KXTest_TrainG,KXTest_TrainF)
{
  # outputs the estimate at the points XTest
  returnValue=KXTest_TrainG%*%object$alpha+1/(object$m*object$lambda)*rowSums(KXTest_TrainF)
  return(returnValue)
}



radialKernelDistance=function(distances,extraKernel=list("eps.val"=1))
{
  # Given the distances and the bandwidth eps.val, computes the matrix of radial kernel
  return(exp(-distances^2/(4*extraKernel$eps.val)))
}

estimateWeightsCV=function(distancesXLabeled,distancesXUnlabeled_Labeled,nXMax=NULL,kernelFunction=radialKernelDistance,extraKernel=list("eps.val"=1),nFolders=5)
{
  # estimates w(x)=f_tg(x)/f_tr(x)
  # returns all coefficients up to nXMax, the maximum number of components
  kernelMatrix=kernelFunction(distancesXLabeled,extraKernel)
  kernelUnlabeled_Labeled=kernelFunction(distancesXUnlabeled_Labeled,extraKernel)
  if(any(is.na(kernelMatrix))) stop("Kernel with NA")
  
  n=length(distancesXLabeled[,1]) # labeled
  m=dim(distancesXUnlabeled_Labeled)[1]
  
  
  randomPermutationLabeled=sample(1:n)
  randomPermutationUnlabeled=sample(1:m)
  currentIndexesLabeled=1:round(n/nFolders)
  currentIndexesUnlabeled=1:round(m/nFolders)
  estimatedLabeledCV=matrix(NA,n,nXMax)
  estimatedUnlabeledCV=matrix(NA,m,nXMax)
  ff=1
  while(1) # breaks only in the end, when all folders are over
  {
    cat("\nFolder ",ff)
    # Split into folder
    currentIndexesLabeled=currentIndexesLabeled[currentIndexesLabeled<=n]
    currentIndexesUnlabeled=currentIndexesUnlabeled[currentIndexesUnlabeled<=m]
    
    trainingIdsLabeled=randomPermutationLabeled[-currentIndexesLabeled]
    testingIdsLabeled=randomPermutationLabeled[currentIndexesLabeled]
    trainingIdsUnlabeled=randomPermutationUnlabeled[-currentIndexesUnlabeled]
    testingIdsUnlabeled=randomPermutationUnlabeled[currentIndexesUnlabeled]
    
    
    
    # Estimate eigenfunctions
    results=eigen(kernelMatrix[trainingIdsLabeled,trainingIdsLabeled],symmetric=T)
    nX=dim(results$vectors)[2]
    
    if(!is.null(nXMax)) nX=min(nX,nXMax)
    
    nTrainLabeled=length(trainingIdsLabeled)
    basisXLabeled=sqrt(nTrainLabeled)*results$vectors[,1:nX,drop=F]
    eigenValues=results$values[1:nX]/nTrainLabeled
    
    mTrainUnlabeled=length(trainingIdsUnlabeled)
    basisXUnlabeled=kernelUnlabeled_Labeled[trainingIdsUnlabeled,trainingIdsLabeled] %*% basisXLabeled
    basisXUnlabeled=1/nTrainLabeled*basisXUnlabeled*matrix(rep(1/eigenValues,mTrainUnlabeled),mTrainUnlabeled,nX,byrow=T)
    
    # Estimate coefficients
    coefficients=colMeans(basisXUnlabeled)    
    
    # Estimate Weights at new points (hold-out)
    kernelTestLabeled_Train=kernelFunction(distancesXLabeled[testingIdsLabeled,trainingIdsLabeled],extraKernel)
    kernelTestUnlabeled_Train=kernelFunction(distancesXUnlabeled_Labeled[testingIdsUnlabeled,trainingIdsLabeled],extraKernel)
    
    nXVec=1:nX
    for(ii in nXVec)
    {
      nComp=nXVec[ii]
      
      mTestLabeled=dim(kernelTestLabeled_Train)[1] # Test
      nTrainLabeled=dim(kernelTestLabeled_Train)[2] # Train  
      mTestUnlabeled=dim(kernelTestUnlabeled_Train)[1] # Test
      
      # Only first eigenfunctions
      eigenVectorsFirst=basisXLabeled[,1:nComp,drop=F]
      eigenValuesFirst=eigenValues[1:nComp,drop=F]
      
      # Labeled test points
      basisX=kernelTestLabeled_Train %*% eigenVectorsFirst
      basisX=1/nTrainLabeled*basisX*matrix(rep(1/eigenValuesFirst,mTestLabeled),mTestLabeled,nComp,byrow=T)
      estimatesLabeled=basisX%*%coefficients[1:nComp,drop=F]
      estimatesLabeled[estimatesLabeled<0]=0
      estimatedLabeledCV[testingIdsLabeled,ii]=estimatesLabeled
      
      # Unlabeled test points
      basisX=kernelTestUnlabeled_Train %*% eigenVectorsFirst
      basisX=1/nTrainLabeled*basisX*matrix(rep(1/eigenValuesFirst,mTestUnlabeled),mTestUnlabeled,nComp,byrow=T)
      estimatesUnlabeled=basisX%*%coefficients[1:nComp,drop=F]
      estimatesUnlabeled[estimatesUnlabeled<0]=0
      estimatedUnlabeledCV[testingIdsUnlabeled,ii]=estimatesUnlabeled
    }
    
    
    currentIndexesLabeled=currentIndexesLabeled+round(n/nFolders)
    currentIndexesUnlabeled=currentIndexesUnlabeled+round(m/nFolders)
    ff=ff+1
    if(min(currentIndexesLabeled)>n|min(currentIndexesUnlabeled)>m) break;
  }
  
  estimatedLosses=1/2*colMeans(estimatedLabeledCV^2,na.rm=T)-colMeans(estimatedUnlabeledCV,na.rm=T)
  
  varL=apply(estimatedLabeledCV^2,2,function(xx)var(xx,,na.rm=T))
  varU=apply(estimatedUnlabeledCV,2,function(xx)var(xx,,na.rm=T))
  
  estimatedLossesSE=sqrt(varL/(4*n)+varU/(m))
  
  object=list()
  class(object)="adaptiveWeightsCV"
  object$estimatedLosses=estimatedLosses
  object$estimatedLossesSE=estimatedLossesSE
  object$kernelFunction=kernelFunction
  object$extraKernel=extraKernel
  object$nXBest=nXVec[which.min(estimatedLosses)]
  whichMin=which.min(estimatedLosses)
  object$nXBestOnePlus=min(nXVec[estimatedLosses<(estimatedLosses+estimatedLossesSE)[whichMin]])
  object$bestError=estimatedLosses[object$nXBest]
  object$bestErrorOnePlus=estimatedLosses[object$nXBestOnePlus]
  return(object)  
}  

estimateWeightsKNN=function(distanceXTest_TrainLabeled,distanceXTest_TrainUnlabeled,nNeighbours)
{
  # outputs the estimate at the points XTest
  nL=dim(distanceXTest_TrainLabeled)[2]
  nU=dim(distanceXTest_TrainUnlabeled)[2]
  nTest=dim(distanceXTest_TrainUnlabeled)[1] # which nTest and not train here?
  returnValue=apply(as.matrix(1:nTest),1,function(xx)
  {
    distanceMax=sort(distanceXTest_TrainLabeled[xx,])[nNeighbours]
    howManyNeighbours=sum(distanceXTest_TrainUnlabeled[xx,]<=as.numeric(distanceMax))
    return((howManyNeighbours/nNeighbours)*(nL/nU))
  })
  return(returnValue)
}

estimateWeightsKNNAdapt=function(distanceXTest_TrainLabeled,distanceXTest_TrainUnlabeled,nNeighbours)
{
  # outputs the estimate at the points XTest
  nL=dim(distanceXTest_TrainLabeled)[2]
  nU=dim(distanceXTest_TrainUnlabeled)[2]
  nTest=dim(distanceXTest_TrainUnlabeled)[1]
  returnValue=apply(as.matrix(1:nTest),1,function(xx)
  {
    distanceMax=sort(distanceXTest_TrainLabeled[xx,])[nNeighbours[xx]]
    howManyNeighbours=sum(distanceXTest_TrainUnlabeled[xx,]<=distanceMax)
    return((howManyNeighbours/nNeighbours[xx])*(nL/nU))
  })
  return(returnValue)
}


estimateWeightsKNNOne=function(distanceXTest_TrainLabeled,distanceXTest_TrainUnlabeled,nNeighbours)
{
  # outputs the estimate at the point XTest (a single point)
  nL=dim(distanceXTest_TrainLabeled)[2]
  nU=dim(distanceXTest_TrainUnlabeled)[2]
  
  distanceMax=sort(distanceXTest_TrainLabeled)[nNeighbours]
  howManyNeighbours=sum(distanceXTest_TrainUnlabeled<=distanceMax)
  return((howManyNeighbours/nNeighbours)*(nL/nU))
  
}

estimateWeightsKNNPlusOne=function(distanceXTest_TrainLabeled,distanceXTest_TrainUnlabeled,nNeighbours)
{
  # outputs the estimate at the point XTest (a single point)
  nL=dim(distanceXTest_TrainLabeled)[2]
  nU=dim(distanceXTest_TrainUnlabeled)[2]
  
  distanceMax=sort(distanceXTest_TrainLabeled)[nNeighbours]
  howManyNeighbours=sum(distanceXTest_TrainUnlabeled<=distanceMax)
  return(((howManyNeighbours+1)/nNeighbours)*(nL/nU))
  
}

estimateWeights=function(distancesXLabeled,distancesXUnlabeled_Labeled,nXMax=NULL,kernelFunction=radialKernelDistance,extraKernel=list("eps.val"=1))
{
  # estimates w(x)=f_tg(x)/f_tr(x)
  # returns all coefficients up to nXMax, the maximum number of components
  kernelMatrix=kernelFunction(distancesXLabeled,extraKernel)
  if(any(is.na(kernelMatrix))) stop("Kernel with NA")
  
  n=length(distancesXLabeled[,1]) # labeled
  m=dim(distancesXUnlabeled_Labeled)[1]
  
  results=eigen(kernelMatrix,symmetric=T)
  nX=dim(results$vectors)[2]
  
  if(!is.null(nXMax)) nX=min(nX,nXMax)
  
  basisXLabeled=sqrt(n)*results$vectors[,1:nX,drop=F]
  eigenValues=results$values[1:nX]/n
  
  kernelUnlabeled_Labeled=kernelFunction(distancesXUnlabeled_Labeled,extraKernel)
  
  basisXUnlabeled=kernelUnlabeled_Labeled %*% basisXLabeled
  basisXUnlabeled=1/n*basisXUnlabeled*matrix(rep(1/eigenValues,m),m,nX,byrow=T)
  
  coefficients=colMeans(basisXUnlabeled)
  
  object=list()
  class(object)="adaptiveWeights"
  object$coefficients=coefficients
  object$nX=nX
  object$eigenX=basisXLabeled
  object$eigenValuesX=eigenValues
  object$kernelFunction=kernelFunction
  object$extraKernel=extraKernel
  return(object)  
}  

estimateErrorEstimatorWeights=function(object,distanceXValidationLabeled_TrainLabeled,distanceXValidationUnlabeled_TrainLabeled)
{
  # returns a vector with size nX with the errors for the possible nX,
  # plus the best nX
  if(class(object)!="adaptiveWeights") stop("Object should be of class cDensity")
  
  kernelLabeledTest_LabeledTrain=object$kernelFunction(distanceXValidationLabeled_TrainLabeled,object$extraKernel)
  kernelUnlabeledTest_LabeledTrain=object$kernelFunction(distanceXValidationUnlabeled_TrainLabeled,object$extraKernel)
  
  if(any(is.na(kernelLabeledTest_LabeledTrain))|any(is.na(kernelUnlabeledTest_LabeledTrain))) stop("Kernel with NA")
  
  nX=object$nX
  
  
  mL=dim(kernelLabeledTest_LabeledTrain)[1] # Labeled Test
  mU=dim(kernelUnlabeledTest_LabeledTrain)[1] # Unlabed Test
  n=dim(kernelLabeledTest_LabeledTrain)[2] # Labeled Train
  
  
  eigenVectors=object$eigenX
  eigenValues=object$eigenValuesX
  
  basisXLabeled=kernelLabeledTest_LabeledTrain %*% eigenVectors
  basisXLabeled=1/n*basisXLabeled*matrix(rep(1/eigenValues,mL),mL,nX,byrow=T)
  
  basisXUnlabeled=kernelUnlabeledTest_LabeledTrain %*% eigenVectors
  basisXUnlabeled=1/n*basisXUnlabeled*matrix(rep(1/eigenValues,mU),mU,nX,byrow=T)
  
  
  WL=1/mL*t(basisXLabeled)%*%basisXLabeled
  WU=colMeans(basisXUnlabeled)
  
  firstTerm=lapply(as.matrix(1:nX),function(xx)
  {
    return(t(object$coefficients[1:xx,drop=F])%*%WL[1:xx,1:xx,drop=F]%*%object$coefficients[1:xx,drop=F])
  })
  firstTerm=sapply(firstTerm,function(xx)xx)
  errors=apply(as.matrix(1:nX),1,function(xx) { sBeta=1/2*firstTerm[xx]
                                                sLikeli=sum(object$coefficients[1:xx,drop=F]*WU[1:xx,drop=F]) 
                                                return(sBeta-sLikeli)} )
  
  
  nXBest=(1:nX)[which.min(errors)]
  object$nXBest=nXBest
  object$errors=errors
  object$bestError=min(errors)
  return(object)
}

estimateWeightsNewPoints=function(object,distanceTest_Train,onePlus=F)
{
  # onePlus indicates if the best I is defined is nXBestOnePlus or nXBest
  
  if(class(object)!="adaptiveWeights") stop("Object should be of class cDensity")
  
  kernelTest_Train=object$kernelFunction(distanceTest_Train,object$extraKernel)
  
  if(any(is.na(kernelTest_Train))) stop("Kernel with NA")
  
  nX=object$nXBest
  if(onePlus)
  {
    nX=object$nXBestOnePlus
  }
  
  m=dim(kernelTest_Train)[1] # Test
  n=dim(kernelTest_Train)[2] # Train  
  
  eigenVectors=object$eigenX[,1:nX,drop=F]
  eigenValues=object$eigenValuesX[1:nX,drop=F]
  
  basisX=kernelTest_Train %*% eigenVectors
  basisX=1/n*basisX*matrix(rep(1/eigenValues,m),m,nX,byrow=T)
  estimates=basisX%*%object$coefficients[1:nX,drop=F]
  estimates[estimates<0]=0
  return(estimates)
  
}

# M: Function removed as it overwrites a function with the same name to evaluate conditional densities
# 
# estimateErrorFinalEstimator=function(predictedGTest,predictedFTest,se=F)
# {
#   predictedGTest[predictedGTest<=0]=0
#   predictedFTest[predictedFTest<=0]=0
#   
#   output=NULL
#   output$mean=1/2*mean(predictedGTest^2)-mean(predictedFTest)
#   if(se==F)
#   {
#     return(output)
#   }
#   
#   # Standard Error Estimation
#   nL=length(predictedGTest)
#   nU=length(predictedFTest)
#   
#   varL=var(predictedGTest^2)
#   varU=var(predictedFTest)
#   
#   output$se=sqrt(varL/(4*nL)+varU/(nU))
#   
#   return(output)
# }


estimateErrorFinalEstimatorWeights=function(predictedLabeledTest,predictedUnlabeledTest,se=F)
{
  predictedLabeledTest[predictedLabeledTest<=0]=0
  predictedUnlabeledTest[predictedUnlabeledTest<=0]=0
  
  output=NULL
  output$mean=1/2*mean(predictedLabeledTest^2)-mean(predictedUnlabeledTest)
  if(se==F)
  {
    return(output)
  }
  
  # Standard Error Estimation
  nL=length(predictedLabeledTest)
  nU=length(predictedUnlabeledTest)
  
  varL=var(predictedLabeledTest^2)
  varU=var(predictedUnlabeledTest)
  
  output$se=sqrt(varL/(4*nL)+varU/(nU))
  
  return(output)
}

