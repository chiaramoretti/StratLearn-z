radialKernelDistance=function(distances,extraKernel=list("eps.val"=1))
{
  # Given the distances and the bandwidth eps.val, computes the matrix of radial kernel
  return(exp(-distances^2/(4*extraKernel$eps.val)))
}

symmetricNormalization=function(originalMatrix)
{
  # Normalize originalMatrix so that it becomes a Markov Matrix
  # originalMatrix is a kernel matrix, therefore symmetric
  if(any(originalMatrix!=t(originalMatrix))|any(originalMatrix<0) | (length(originalMatrix[1,])!=length(originalMatrix[,1])))
  {
    print("wrong argument, Matrix can't be Normalized")
    return(1);
  }
  sqrtRowMeans=sqrt(rowMeans(originalMatrix))
  sqrtColMeans=sqrt(colMeans(originalMatrix))
  KernelMatrixN=originalMatrix/(sqrtRowMeans%*%t(sqrtColMeans))
  return(list("KernelMatrixN"=KernelMatrixN,"sqrtRowMeans"=sqrtRowMeans,"sqrtColMeans"=sqrtColMeans))
}


calculateBasis=function(z,nZ,system)
{
  if(system=="cosine")
  {
    basisZ=apply(as.matrix(1:(nZ-1)),1,function(xx)sqrt(2)*cos(xx*pi*z))
    basisZ=cbind(rep(1,length(z)),basisZ)
    return(basisZ)
  } 
  if(system=="Fourier")
  {
    sinBasisZ=apply(as.matrix(1:round((nZ)/2)),1,function(xx) sqrt(2)*sin(2*xx*pi*z))
    cosBasisZ=apply(as.matrix(1:round((nZ)/2)),1,function(xx) sqrt(2)*cos(2*xx*pi*z))
    basisZ=matrix(NA,length(z),2*round((nZ)/2))
    basisZ[,seq(from=1,length.out=dim(sinBasisZ)[2],by=2)]=sinBasisZ
    basisZ[,seq(from=2,length.out=dim(cosBasisZ)[2],by=2)]=cosBasisZ
    basisZ=cbind(rep(1,length(z)),basisZ)
    basisZ=basisZ[,1:nZ]
    return(basisZ)
  } else
  {
    stop("System of Basis not known")
  }
  
}


# M: I am almost sure that this "condDensity" function delivers the (in the paper called) Series estimator (here Series_CS). E.g. function predictDensity also belongs to that estimator.
condDensity=function(distanceXTrainU_TrainU,distanceXTrainL_TrainU,weightsZTrainingL,zTrainL,nZMax=20,nXMax=NULL,kernelFunction=radialKernelDistance,extraKernel=list("eps.val"=1),normalization=NULL,system="Fourier",cVec=c(0,1))
{
  # estimates f(z|X)
  # z is assumed to be between 0 and 1
  # returns all coefficients up to nZMax, the maximum number of components
  # for Z, same for X
  # weightsZTrainingL are the importance weights at the training labeled points
  # cVec is a vector with c's to be used when weighting the coefficients
  
  kernelMatrix=kernelFunction(distanceXTrainU_TrainU,extraKernel)
  if(any(is.na(kernelMatrix))) stop("Kernel with NA")
  nU = length(distanceXTrainU_TrainU[,1])
  
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
  #print("Computing EigenVectors\n")
  
  results=eigen(kernelMatrixN,symmetric=T)
  nX=dim(results$vectors)[2]
  
  if(!is.null(nXMax)) nX=min(nX,nXMax)
  
  eigenVectors=sqrt(nU)*results$vectors[,1:nX,drop=F] # at unlabeled points
  eigenValues=results$values[1:nX]/nU
  # Extended basis to labeled (Nystrom)  
  kernelNewOld=kernelFunction(distanceXTrainL_TrainU,extraKernel)
  nL=dim(kernelNewOld)[1] # Labeled points  
  basisX=kernelNewOld %*% eigenVectors
  basisX=1/nU*basisX*matrix(rep(1/eigenValues,nL),nL,nX,byrow=T)
  rm(results)
  
  basisZ=calculateBasis(zTrainL,nZMax,system) # returns vector length(z)xnZ with the basis for z.
  
  #coefficients=apply(grid,1,function(xx)mean(basisX[,xx[1]]*basisZ[,xx[2]]))
  matrixBeta=matrix(weightsZTrainingL,nL,nX)
  
  coefficients=lapply(as.matrix(cVec),function(xx)1/nL*t(t(basisZ)%*%(basisX*(matrixBeta^xx))))
  
  #coefficients=apply(grid,1,function(xx)sum(basisX[,xx[1]]*basisZ[,xx[2]]))
  
  object=list()
  class(object)="cDensity"
  #matrixCoef=matrix(coefficients,nX,nZMax)
  object$coefficients=coefficients
  object$cVec=cVec
  object$system=system
  object$nX=nX
  object$nZ=nZMax
  object$normalizationParameters=normalizationParameters
  if(is.null(normalization)) normalization=0
  object$normalization=normalization
  object$eigenX=eigenVectors
  object$eigenValuesX=eigenValues
  object$kernelFunction=kernelFunction
  object$extraKernel=extraKernel
  return(object)  
}  



#condDensityKNNNoWeights=function(zTrainNearest,nBins,zMin,zMax)
#{
  # calculates estimated density for a single observation
  # estimate density of z given z's of the  nearest neighbour
#  binsIntervals=seq(zMin,zMax,length.out=nBins)
#  binSize=binsIntervals[2]-binsIntervals[1]
#  means=apply(as.matrix(binsIntervals),1,function(xx)
#  {
#    lower=xx-binSize/2
#    upper=xx+binSize/2
#    mean(zTrainNearest>=lower & zTrainNearest<upper)/binSize
#  })
#  output=NULL
#  output$means=means
#  output$binsIntervals=binsIntervals
#  return(output)
#}

condDensityKNNWithWeights=function(zTrainNearest,nBins,zMin,zMax,weightsZTrain,cVec=c(0,1))
{
  # calculates estimated density for a single observation
  # estimate density of z given z's of the  nearest neighbour
  binsIntervals=seq(zMin,zMax,length.out=nBins)
  binSize=binsIntervals[2]-binsIntervals[1]
  output=NULL
  output$means=matrix(NA,length(cVec),length(binsIntervals))
  output$binsIntervals=binsIntervals
  for(ii in 1:length(cVec))
  {
    means=apply(as.matrix(binsIntervals),1,function(xx)
    {
      lower=xx-binSize/2
      upper=xx+binSize/2
      weightsZTrainC=weightsZTrain^cVec[ii]
      if(all(weightsZTrainC==0)) weightsZTrainC=rep(1,length(weightsZTrainC))
      sum(weightsZTrainC[zTrainNearest>=lower & zTrainNearest<upper])/(binSize*sum(weightsZTrainC))
    })
    output$means[ii,]=means
  }
  return(output)
}

condDensityKNNWithWeightsContinuous=function(zTrainNearest,nBins=1000,bandwidth,zMin=0,zMax=1,weightsZTrain,cVec=c(0,1))
{
  # calculates estimated density for a single observation
  # estimate density of z given z's of the  nearest neighbour
  binsIntervals=seq(zMin,zMax,length.out=nBins)
  output=NULL
  output$means=matrix(NA,length(cVec),nBins)
  output$binsIntervals=binsIntervals
  
  for(ii in 1:length(cVec))
  {
    weightsZTrainC=weightsZTrain^cVec[ii]
    if(all(weightsZTrainC==0)) weightsZTrainC=rep(1,length(weightsZTrainC))
    sumWeights=sum(weightsZTrainC)
    
    means=apply(as.matrix(binsIntervals),1,function(xx)
    {
      weightsFinal=weightsZTrainC*exp(-(xx-zTrainNearest)^2/(4*bandwidth))/sqrt(pi*4*bandwidth)/sumWeights
      return(sum(weightsFinal))
    })
    output$means[ii,]=means
  }
  return(output)
}


estimateErrorFinalEstimatorKNNWithWeightsContinuous=function(nNeigh,nBins,bandwidthOpt,zMin,zMax,zTrainL,weightsZTrainL,cVec,distanceXTestUTrainL,distanceXTestLTrainL,zTestL,weightsZTestL,boot=F)
{
  # nNeigh,nBinsOpt,zMin,zMax,zTrainL,zTrainLWeights,c are needed to compute the estimates
  # c is optimal power to use for weights
  zGrid=seq(zMin,zMax,length.out=nBins)
  output=NULL
  output$mean=NULL
  output$seBoot=NULL
  for(ii in 1:length(cVec))
  {
    predictedCompleteU=t(apply(distanceXTestUTrainL,1,function(xx)
    {
      nearest=sort(xx,index.return=T)$ix[1:nNeigh]
      densityObject=condDensityKNNWithWeightsContinuous(zTrainL[nearest],nBins,bandwidthOpt,zMin,zMax,weightsZTrainL[nearest],cVec=cVec[ii])
      return(densityObject$means)
    }))

    
    predictedCompleteL=t(apply(distanceXTestLTrainL,1,function(xx)
    {
      nearest=sort(xx,index.return=T)$ix[1:nNeigh]
      densityObject=condDensityKNNWithWeightsContinuous(zTrainL[nearest],nBins,bandwidthOpt,zMin,zMax,weightsZTrainL[nearest],cVec=cVec[ii])
      return(densityObject$means)
    }))
    
    colmeansComplete=colMeans(predictedCompleteU^2)
    sSquare=mean(colmeansComplete)
    n=length(zTestL)
    predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestL[xx]-zGrid))
                                                            return(predictedCompleteL[xx,index])
    })
    
    likeli=mean(predictedObserved*weightsZTestL)
    output$mean[ii]=1/2*sSquare-likeli
    if(boot==F)
    {
      next;
    }
    
    # Bootstrap
    meanBoot=apply(as.matrix(1:boot),1,function(xx){
      sampleBoot=sample(1:n,replace=T)
      
      predictedCompleteBootL=predictedCompleteL[sampleBoot,]
      predictedCompleteBootU=predictedCompleteU[sampleBoot,]
      zTestBootL=zTestL[sampleBoot]
      weightsZTestBootL=weightsZTestL[sampleBoot]
      colmeansComplete=colMeans(predictedCompleteBootU^2)
      sSquare=mean(colmeansComplete)
      
      predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestBootL[xx]-zGrid))
                                                              return(predictedCompleteBootL[xx,index])
      })
      likeli=mean(predictedObserved_boot*weightsZTestBootL)
      return(1/2*sSquare-likeli)    
    })
    output$seBoot[ii]=sqrt(var(meanBoot))
    
  }
  return(output)
}



estimateErrorFinalEstimatorKNNWithWeightsContinuousWithTesting=function(nNeigh,nBins,bandwidthBinsOpt,zMin,zMax,zTrainL,weightsZTrainL,cVec,distanceXTestUTrainL,zTestU,boot=F, add_pred = F)
{
  # nNeigh,nBinsOpt,zMin,zMax,zTrainL,zTrainLWeights,c are needed to compute the estimates
  # c is optimal power to use for weights
  zGrid=seq(zMin,zMax,length.out=nBins)
  output=NULL
  output$mean=NULL
  output$seBoot=NULL
  for(ii in 1:length(cVec))
  {
    predictedComplete=t(apply(distanceXTestUTrainL,1,function(xx)   {
      nearest=sort(xx,index.return=T)$ix[1:nNeigh]
      densityObject=condDensityKNNWithWeightsContinuous(zTrainL[nearest],nBins,bandwidthBinsOpt,zMin,zMax,weightsZTrainL[nearest],cVec=cVec[ii])
      return(densityObject$means)
    }))
    
    colmeansComplete=colMeans(predictedComplete^2)
    sSquare=mean(colmeansComplete)
    
    n=length(zTestU)
    predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
                                                            return(predictedComplete[xx,index])
    })
    likeli=mean(predictedObserved)
    output$mean[ii]=1/2*sSquare-likeli
    if(boot==F)
    {
      next;
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
    output$seBoot[ii]=sqrt(var(meanBoot))
  }
  if(add_pred == TRUE){
    return = list(output = output, predictedComplete = predictedComplete, predictedObserved = predictedObserved)
  }else{
    return(output)
  }
}


# improved list of objects
.ls.objects <- function (pos = 1, pattern, order.by,
                         decreasing=FALSE, head=FALSE, n=5) {
  napply <- function(names, fn) sapply(names, function(x)
    fn(get(x, pos = pos)))
  names <- ls(pos = pos, pattern = pattern)
  obj.class <- napply(names, function(x) as.character(class(x))[1])
  obj.mode <- napply(names, mode)
  obj.type <- ifelse(is.na(obj.class), obj.mode, obj.class)
  obj.size <- napply(names, object.size)
  obj.dim <- t(napply(names, function(x)
    as.numeric(dim(x))[1:2]))
  vec <- is.na(obj.dim)[, 1] & (obj.type != "function")
  obj.dim[vec, 1] <- napply(names, length)[vec]
  out <- data.frame(obj.type, obj.size, obj.dim)
  names(out) <- c("Type", "Size", "Rows", "Columns")
  if (!missing(order.by))
    out <- out[order(out[[order.by]], decreasing=decreasing), ]
  if (head)
    out <- head(out, n)
  out
}
# shorthand
lsos <- function(..., n=10) {
  .ls.objects(..., order.by="Size", decreasing=TRUE, head=TRUE, n=n)
}

estimateErrorFinalEstimatorKNNWithWeights=function(nNeigh,nBins,nBinsOpt,zMin,zMax,zTrainL,weightsZTrainL,cVec,distanceXTestUTrainL,distanceXTestLTrainL,zTestL,weightsZTestL,boot=F)
{
  # nNeigh,nBinsOpt,zMin,zMax,zTrainL,zTrainLWeights,c are needed to compute the estimates
  # c is optimal power to use for weights
  zGrid=seq(zMin,zMax,length.out=nBins)
  binsIntervals=seq(zMin,zMax,length.out=nBinsOpt)
  whichClosest=apply(as.matrix(zGrid),1,function(yy)
  {
    which.min(abs(binsIntervals-yy))
  })
  output=NULL
  output$mean=NULL
  output$seBoot=NULL
  for(ii in 1:length(cVec))
  {
    predictedCompleteU=t(apply(distanceXTestUTrainL,1,function(xx)
    {
      nearest=sort(xx,index.return=T)$ix[1:nNeigh]
      densityObject=condDensityKNNWithWeights(zTrainL[nearest],nBinsOpt,zMin,zMax,weightsZTrainL[nearest],cVec=cVec[ii])
      means=densityObject$means[whichClosest]
      return(means)
    }))
    
    predictedCompleteL=t(apply(distanceXTestLTrainL,1,function(xx)
    {
      nearest=sort(xx,index.return=T)$ix[1:nNeigh]
      densityObject=condDensityKNNWithWeights(zTrainL[nearest],nBinsOpt,zMin,zMax,weightsZTrainL[nearest],cVec=cVec[ii])
      means=densityObject$means[whichClosest]
      return(means)
    }))
    
    colmeansComplete=colMeans(predictedCompleteU^2)
    sSquare=mean(colmeansComplete)
    n=length(zTestL)
    predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestL[xx]-zGrid))
                                                            return(predictedCompleteL[xx,index])
    })
    
    likeli=mean(predictedObserved*weightsZTestL)
    output$mean[ii]=1/2*sSquare-likeli
    if(boot==F)
    {
      next;
    }
    
    # Bootstrap
    meanBoot=apply(as.matrix(1:boot),1,function(xx){
      sampleBoot=sample(1:n,replace=T)
      
      predictedCompleteBootL=predictedCompleteL[sampleBoot,]
      predictedCompleteBootU=predictedCompleteU[sampleBoot,]
      zTestBootL=zTestL[sampleBoot]
      weightsZTestBootL=weightsZTestL[sampleBoot]
      colmeansComplete=colMeans(predictedCompleteBootU^2)
      sSquare=mean(colmeansComplete)
      
      predictedObserved_boot=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestBootL[xx]-zGrid))
                                                              return(predictedCompleteBootL[xx,index])
      })
      likeli=mean(predictedObserved_boot*weightsZTestBootL)
      return(1/2*sSquare-likeli)    
    })
    output$seBoot[ii]=sqrt(var(meanBoot))
    
  }
  return(output)
}

estimateErrorFinalEstimatorKNNWithWeightsWithTesting=function(nNeigh,nBins,nBinsOpt,zMin,zMax,zTrainL,weightsZTrainL,cVec,distanceXTestUTrainL,zTestU,boot=F, add_pred = F)
{
  # nNeigh,nBinsOpt,zMin,zMax,zTrainL,zTrainLWeights,c are needed to compute the estimates
  # c is optimal power to use for weights
  zGrid=seq(zMin,zMax,length.out=nBins)
  binsIntervals=seq(zMin,zMax,length.out=nBinsOpt)
  whichClosest=apply(as.matrix(zGrid),1,function(yy)  {
    which.min(abs(binsIntervals-yy))
  })
  output=NULL
  output$mean=NULL
  output$seBoot=NULL
  for(ii in 1:length(cVec))
  {
    predictedComplete=t(apply(distanceXTestUTrainL,1,function(xx)   {
      nearest=sort(xx,index.return=T)$ix[1:nNeigh]
      densityObject=condDensityKNNWithWeights(zTrainL[nearest],nBinsOpt,zMin,zMax,weightsZTrainL[nearest],cVec=cVec[ii])
      means=densityObject$means[whichClosest]
      return(means)
    }))
        
    colmeansComplete=colMeans(predictedComplete^2)
    sSquare=mean(colmeansComplete)
    
    n=length(zTestU)
    predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
                                                            return(predictedComplete[xx,index])
    })
    likeli=mean(predictedObserved)
    output$mean[ii]=1/2*sSquare-likeli
    if(boot==F)
    {
      next;
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
    output$seBoot[ii]=sqrt(var(meanBoot))
  }
  if(add_pred == TRUE){
    return = list(output = output, predictedComplete = predictedComplete, predictedObserved = predictedObserved)
  }else{
    return(output)
  }
}

estimateErrorFinalEstimatorMATLAB=function(predictedComplete,zTest,boot=F)
{
  # estimates the loss of a given estimator, after fitting was done
  # boot==F if no error estimates, othewise boot is the number of bootstrap samples
  zGrid=seq(0,1,length.out=dim(predictedComplete)[2])
  
  colmeansComplete=colMeans(predictedComplete^2)
  sSquare=mean(colmeansComplete)
  
  n=length(zTest)
  predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTest[xx]-zGrid))
                                                          return(predictedComplete[xx,index])
  })
  likeli=mean(predictedObserved,na.rm=T)
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
    likeli=mean(predictedObserved_boot,na.rm=T)
    return(1/2*sSquare-likeli)    
  })
  output$seBoot=sqrt(var(meanBoot,na.rm=T))
  return(output)
}

estimateErrorFinalEstimatorGenericWithTesting=function(predictedCompleteUTest,zTestU,boot=F)
{
  # estimates the loss of a given estimator, after fitting was done
  # boot==F if no error estimates, othewise boot is the number of bootstrap samples
  zGrid=seq(0,1,length.out=dim(predictedCompleteUTest)[2])  
  
  colmeansComplete=colMeans(predictedCompleteUTest^2)
  sSquare=mean(colmeansComplete)
  
  n=length(zTestU)
  predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
                                                          return(predictedCompleteUTest[xx,index])
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
    
    predictedCompleteBoot=predictedCompleteUTest[sampleBoot,]
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


estimateErrorFinalEstimatorGeneric=function(predictedCompleteLTest,predictedCompleteUTest,zTestL,weightsZTestL =c(), boot=F,
                                            zMin, zMax, nBins){
#M: function adjusted to get the comb loss without weights. If there is no weightsZTestL specified,
# then Formula 9 is minimized but without beta (weights). Or similar to beta = 1 for every sample.
  
  zGrid=seq(zMin, zMax,length.out=nBins)
  colmeansComplete=colMeans(predictedCompleteUTest^2)
  sSquare=mean(colmeansComplete) #M: sSquare is the first part of formula 9 (integral over unlabeled z)
  
  nU=dim(predictedCompleteUTest)[1]
  nL=length(zTestL)
  predictedObserved=apply(as.matrix(1:nL),1,function(xx) { index=which.min(abs(zTestL[xx]-zGrid))
                                                          return(predictedCompleteLTest[xx,index])
  })
  if(length(weightsZTestL) == 0){
    likeli=mean(predictedObserved)
    print("Formula 9 (loss) WITHOUT beta weights computed")
  }else{
    print("Formula 9 (loss) WITH beta weights computed")
    likeli=mean(predictedObserved*weightsZTestL) 
  }

  output=NULL
  output$mean=1/2*sSquare-likeli
  if(boot==F)
  {
    return(output)
  }
  
  # Bootstrap
  meanBoot=apply(as.matrix(1:boot),1,function(xx){
    sampleBootL=sample(1:nL,replace=T)
    sampleBootU=sample(1:nU,replace=T)
    
    predictedCompleteBootL=predictedCompleteLTest[sampleBootL,]
    predictedCompleteBootU=predictedCompleteUTest[sampleBootU,]
    zTestBootL=zTestL[sampleBootL]
    if(length(weightsZTestL) != 0){
      weightsZTestBootL=weightsZTestL[sampleBootL]
    }
    colmeansComplete=colMeans(predictedCompleteBootU^2)
    sSquare=mean(colmeansComplete)
    
    predictedObserved_boot=apply(as.matrix(1:nL),1,function(xx) { index=which.min(abs(zTestBootL[xx]-zGrid))
                                                            return(predictedCompleteBootL[xx,index])
    })
    if(length(weightsZTestL) == 0){
      likeli=mean(predictedObserved_boot)
    }else{
      likeli=mean(predictedObserved_boot*weightsZTestBootL)
    }
    
    return(1/2*sSquare-likeli)    
  })
  output$seBoot=sqrt(var(meanBoot))
  return(output)
}
 
 
estimateErrorFinalEstimator=function(object,zTestL,distanceXTestL_TrainU,distanceXTestU_TrainU,weightsZTestL,boot=F,delta=0,
                                     zMin, zMax, nBins){
  # estimates the loss of a given estimator, after fitting was done                   #Mon the training set using a weighted loss 
  # boot==F if no error estimates, othewise boot is the number of bootstrap samples
  zGrid=seq(zMin,zMax,length.out=nBins)
  
  predictedCompleteL=predictDensity(object,zTestMin=0,zTestMax=1,B=length(zGrid),distanceXTestL_TrainU,probabilityInterval=F,delta=delta)
  predictedCompleteU=predictDensity(object,zTestMin=0,zTestMax=1,B=length(zGrid),distanceXTestU_TrainU,probabilityInterval=F,delta=delta)
  
  
  colmeansComplete=colMeans(predictedCompleteU^2)
  sSquare=mean(colmeansComplete)
  
  nL=length(zTestL)
  nU=dim(distanceXTestU_TrainU)[1]
  predictedObserved=apply(as.matrix(1:nL),1,function(xx) { index=which.min(abs(zTestL[xx]-zGrid))
                                                          return(predictedCompleteL[xx,index])
  })
  likeli=mean(predictedObserved*weightsZTestL)
  output=NULL
  output$mean=1/2*sSquare-likeli
  if(boot==F)
  {
    return(output)
  }
  
  # Bootstrap
  meanBoot=apply(as.matrix(1:boot),1,function(xx){
    sampleBootL=sample(1:nL,replace=T)
    sampleBootU=sample(1:nU,replace=T)
    
    predictedCompleteBootL=predictedCompleteL[sampleBootL,]
    predictedCompleteBootU=predictedCompleteU[sampleBootU,]
    zTestBootL=zTestL[sampleBootL]
    weightsZTestBootL=weightsZTestL[sampleBootL]
    colmeansComplete=colMeans(predictedCompleteBootU^2)
    sSquare=mean(colmeansComplete)
    
    predictedObserved_boot=apply(as.matrix(1:nL),1,function(xx) { index=which.min(abs(zTestBootL[xx]-zGrid))
                                                            return(predictedCompleteBootL[xx,index])
    })
    likeli=mean(predictedObserved_boot*weightsZTestBootL)
    return(1/2*sSquare-likeli)    
  })
  output$seBoot=sqrt(var(meanBoot))
  return(output)
}


chooseDelta <- function (object, zValidationL, distanceXValidationLTrainU,distanceXValidationUTrainU,weightsZValidationL,deltaGrid=seq(0,0.3,0.04),
                         nBins) {
  # weightsZValidationL correspons to zValidationL
  error=errorSE=rep(NA,length(deltaGrid))
  for(ii in 1:length(deltaGrid))
  {
    print(ii/length(deltaGrid))
    estimateErrors=estimateErrorFinalEstimator(object,zValidationL,distanceXValidationLTrainU,distanceXValidationUTrainU,weightsZValidationL,boot=100,delta=deltaGrid[ii],
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


estimateErrorFinalEstimatorKDE=function(band,covariatesTrain,zTrain,covariatesTest,zTest,boot=F,groups=1,
                                        zMin, zMax, nBins){
  # estimates the loss of a given estimator, after fitting was done
  # boot==F if no error estimates, othewise boot is the number of bootstrap samples
  zGrid=seq(zMin,zMax,length.out=nBins)
  
  nTest=dim(covariatesTest)[1]
  predictedComplete=matrix(NA,nTest,length(zGrid))
  currentIndexes=1:round(nTest/groups)
  ii=1
  while(1) # breaks only in the end, when all folders are over
  {
    print(ii/groups)
    currentIndexes=currentIndexes[currentIndexes<=nTest]
    nInstance=length(currentIndexes)
    covariatesTestInstance=covariatesTest[currentIndexes,,drop=F]
    covariatesTestRep=covariatesTestInstance[sort(rep(1:nInstance,length(zGrid))),]
    zGridExtend=rep(zGrid,nInstance)   
    fit=npcdens(bws=band,txdat=covariatesTrain,tydat=zTrain,exdat=covariatesTestRep,eydat=zGridExtend)
    
    predictedComplete[currentIndexes,]=matrix(fit$condens,nInstance,length(zGrid),byrow=T)
    
    currentIndexes=currentIndexes+round(nTest/groups)
    ii=ii+1
    if(min(currentIndexes)>nTest) break;
  }
  
  colmeansComplete=colMeans(predictedComplete^2)
  sSquare=mean(colmeansComplete)
  
  n=length(zTest)
  predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTest[xx]-zGrid))
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
  return(output)
}


estimateErrorEstimator=function(object,zTestL,distanceXTestL_TrainU,distanceXTestU_TrainU,weightsZTestL,lessInfo=F)
{
  # returns a nX by nZ matrix with the errors for the different possibilities
  # of number of components, plus the best combination
  if(class(object)!="cDensity") stop("Object should be of class cDensity")
  
  kernelNewOldTestL_TrainU=object$kernelFunction(distanceXTestL_TrainU,object$extraKernel)
  kernelNewOldTestU_TrainU=object$kernelFunction(distanceXTestU_TrainU,object$extraKernel)
  
  if(any(is.na(kernelNewOldTestL_TrainU))) stop("Kernel with NA")
  
  nX=object$nX
  nZ=object$nZ
  
  mTestL=dim(kernelNewOldTestL_TrainU)[1] # New
  nTrainU=dim(kernelNewOldTestL_TrainU)[2] # Old
  mTestU=dim(distanceXTestU_TrainU)[1] # New
  
  basisZ=calculateBasis(zTestL,nZ,object$system) # returns matrix length(z)xnZ with the basis for z.
  
  eigenVectors=object$eigenX
  eigenValues=object$eigenValuesX
  
  basisXTestL=kernelNewOldTestL_TrainU %*% eigenVectors
  basisXTestL=1/nTrainU*basisXTestL*matrix(rep(1/eigenValues,mTestL),mTestL,nX,byrow=T)
  
  basisXTestU=kernelNewOldTestU_TrainU %*% eigenVectors
  basisXTestU=1/nTrainU*basisXTestU*matrix(rep(1/eigenValues,mTestU),mTestU,nX,byrow=T)
  
  rm(eigenVectors)
  
  W=1/mTestU*t(basisXTestU)%*%basisXTestU
  
  matrixBeta=matrix(weightsZTestL,mTestL,nX)
  basisPsiMean=1/mTestL*t(t(basisZ)%*%(basisXTestL*(matrixBeta)))  
  

  
  grid=expand.grid(1:nX,1:nZ) 
  cVec=object$cVec
  nXBest=nZBest=errorsByC=errorsByC=NULL
  errorsList=array(NA,dim=c(length(cVec),nX,nZ))
  for(ii in 1:length(cVec))
  {
    coeff=object$coefficients[[ii]]
    prodMatrix=lapply(as.matrix(1:nZ),function(xx)
    {
      auxMatrix=W*coeff[,xx,drop=F]%*%t(coeff[,xx,drop=F])
      returnValue=diag(apply(t(apply(auxMatrix, 1, cumsum)), 2, cumsum))
      return(returnValue[1:nX])
    })
    prodMatrix=sapply(prodMatrix,function(xx)xx)
    D=t(apply(prodMatrix,1,cumsum))
    
    rm(prodMatrix)
    
    errors=apply(grid,1,function(xx) { sBeta=1/2*D[xx[1],xx[2]]
                                       sLikeli=sum(coeff[1:xx[1],1:xx[2]]*basisPsiMean[1:xx[1],1:xx[2]]) 
                                       return(sBeta-sLikeli)} )
    errors=matrix(errors,nX,nZ)
    rm(D)
    
    
    pointMin=which(errors == min(errors,na.rm=T), arr.ind = TRUE)
    nXBest[ii]=(1:nX)[pointMin[1]]
    nZBest[ii]=(1:nZ)[pointMin[2]]
    errorsList[ii,,]=errors
    errorsByC[ii]=min(errors)
  }
  
  
  rm(W)
  rm(basisPsiMean)
  if(!lessInfo)  
  {
    object$errors=errorsList
  }
  object$bestError=min(errorsList)  
  object$nXBest=nXBest
  object$nZBest=nZBest
  object$cBest=cVec[which.min(errorsByC)]
  object$errorsByC=errorsByC
  return(object)
}

estimateErrorEstimatorWithTest=function(object,zTestU,distanceXTestU_TrainU)
{
  # returns a nX by nZ matrix with the errors for the different possibilities
  # of number of components, plus the best combination
  # usually not available in real problems because zTestU is unknown
  if(class(object)!="cDensity") stop("Object should be of class cDensity")
  
  kernelNewOldTestU_TrainU=object$kernelFunction(distanceXTestU_TrainU,object$extraKernel)
  
  if(any(is.na(distanceXTestU_TrainU))) stop("Kernel with NA")
  
  nX=object$nX
  nZ=object$nZ
  
  nTrainU=dim(distanceXTestU_TrainU)[2] # Old
  mTestU=dim(distanceXTestU_TrainU)[1] # New
  
  basisZU=calculateBasis(zTestU,nZ,object$system) # returns matrix length(z)xnZ with the basis for z.
  
  eigenVectors=object$eigenX
  eigenValues=object$eigenValuesX
  
  basisXTestU=kernelNewOldTestU_TrainU %*% eigenVectors
  basisXTestU=1/nTrainU*basisXTestU*matrix(rep(1/eigenValues,mTestU),mTestU,nX,byrow=T)
  
  rm(eigenVectors)
  
  W=1/mTestU*t(basisXTestU)%*%basisXTestU
  
  basisPsiMean=1/mTestU*t(t(basisZU)%*%basisXTestU)  
  
  grid=expand.grid(1:nX,1:nZ) 
  cVec=object$cVec
  nXBest=nZBest=errorsByC=errorsByC=NULL
  errorsList=array(NA,dim=c(length(cVec),nX,nZ))
  for(ii in 1:length(cVec))
  {
    coeff=object$coefficients[[ii]]
    prodMatrix=lapply(as.matrix(1:nZ),function(xx)
    {
      auxMatrix=W*coeff[,xx,drop=F]%*%t(coeff[,xx,drop=F])
      returnValue=diag(apply(t(apply(auxMatrix, 1, cumsum)), 2, cumsum))
      return(returnValue[1:nX])
    })
    prodMatrix=sapply(prodMatrix,function(xx)xx)
    D=t(apply(prodMatrix,1,cumsum))
    
    rm(prodMatrix)
    
    errors=apply(grid,1,function(xx) { sBeta=1/2*D[xx[1],xx[2]]
                                       sLikeli=sum(coeff[1:xx[1],1:xx[2]]*basisPsiMean[1:xx[1],1:xx[2]]) 
                                       return(sBeta-sLikeli)} )
    errors=matrix(errors,nX,nZ)
    rm(D)
    
    
    pointMin=which(errors == min(errors,na.rm=T), arr.ind = TRUE)
    nXBest[ii]=(1:nX)[pointMin[1]]
    nZBest[ii]=(1:nZ)[pointMin[2]]
    errorsList[ii,,]=errors
    errorsByC[ii]=min(errors)
  }
  rm(W)
  rm(basisPsiMean)
  
  object$nXBest=nXBest
  object$nZBest=nZBest
  object$cBest=cVec[which.min(errorsByC)]
  object$errors=errorsList
  object$bestError=min(errorsList)
  object$errorsByC=errorsByC
  return(object)
}


predictDensity=function(object,zTestMin=0,zTestMax=1,B=1000,distanceXTestTrainU,probabilityInterval=F,confidence=0.95,delta=0)
{
  # predict density at points zTest=seq(zTestMin,zTestMax,lenght.out=B) and xTest
  # delta is the treshhold to remove bumps
  print("Densities normalized to integrate 1 in the range of z given!")
  
  if(class(object)!="cDensity") stop("Object should be of class cDensity")
  
  zGrid=seq(from=zTestMin,to=zTestMax,length.out=B)
  
  kernelNewOld=object$kernelFunction(distanceXTestTrainU,object$extraKernel)
  if(object$normalization!=0)
  {
    if(object$normalization=="symmetric")
    {
      sqrtColMeans=object$normalizationParameters$sqrtColMeans
      sqrtRowMeans=sqrt(rowMeans(kernelNewOld))
      kernelNewOld=kernelNewOld/(sqrtRowMeans%*%t(sqrtColMeans))
    }
  }
  whichBestC=which.min(object$errorsByC)
  
  nXBest=object$nX[whichBestC]
  nZBest=object$nZ[whichBestC]
  
  
  if(!is.null(object$nXBest)) # if CV was done
  {
    nXBest=object$nXBest[whichBestC]
    nZBest=object$nZBest[whichBestC]
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
  
  coeff=object$coefficients[[whichBestC]]
  coefficients=coeff[1:nXBest,1:nZBest,drop=F]
  
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


estimateErrorFinalEstimatorWithTest=function(object,zTestU,distanceXTestUTrainU,boot=F,delta=0, add_pred = F,
                                             zMin, zMax, nBins){
  # estimates the loss of a given estimator, after fitting was done           
  #M: Testing is done on unlabelled set, assuming the labels are known only for testing purpose!
  # boot==F if no error estimates, othewise boot is the number of bootstrap samples
  zGrid=seq(zMin,zMax,length.out=nBins)
  
  predictedComplete=predictDensity(object,zTestMin=0,zTestMax=1,B=length(zGrid),distanceXTestUTrainU,probabilityInterval=F,delta=delta)
  
  
  colmeansComplete=colMeans(predictedComplete^2)
  sSquare=mean(colmeansComplete)
  
  n=length(zTestU)
  predictedObserved=apply(as.matrix(1:n),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
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
    return = list(output = output, predictedComplete = predictedComplete, predictedObserved = predictedObserved)
  }else{
    return(output)
  }
}

calculateQuantiles=function(densityValues,probs)
{
  zGrid=seq(0,1,length.out=length(densityValues))
  quantiles=apply(as.matrix(probs),1,function(xx) { cumSum=cumsum(densityValues)/sum(densityValues)
                                                    zGrid[which.max(cumSum>xx)]  })
  return(quantiles)
}



normalizeDensity=function(binSize,estimates,delta=0)
{
  estimates=matrix(estimates,1,length(estimates))
  if(all(estimates<=0)) estimates=matrix(1,1,length(estimates))
  estimatesThresh=estimates
  estimatesThresh[estimatesThresh<0]=0
  whichPositive=estimates>0
  
  if(sum(binSize*estimatesThresh)>1)
  {
    maxDensity=max(estimates)  
    minDensity=0
    newXi=(maxDensity+minDensity)/2
    eps=1
    ii=1
    while(ii<=1000)
    {
      estimatesNew=apply(as.matrix(estimates),2,function(xx)max(0,xx-newXi))
      area=sum(binSize*estimatesNew)
      eps=abs(1-area)
      if(eps<0.0000001) break; # level found
      if(1>area) maxDensity=newXi
      if(1<area) minDensity=newXi
      newXi=(maxDensity+minDensity)/2
      ii=ii+1
    }
    estimatesNew=apply(as.matrix(estimates),2,function(xx)max(0,xx-newXi))
    
    
    runs=rle(estimatesNew>0)
    nRuns=length(runs$values)
    jj=1
    area=lower=upper=NULL
    if(nRuns>2)
    {
      for(ii in 1:nRuns)
      {
        if(runs$values[ii]==FALSE) next;
        whichMin=1
        if(ii>1)
        {
          whichMin=sum(runs$lengths[1:(ii-1)])
        }
        whichMax=whichMin+runs$lengths[ii]
        lower[jj]=whichMin # lower interval of component
        upper[jj]=whichMax # upper interval of component
        area[jj]=sum(binSize*estimatesNew[whichMin:whichMax]) # total area of component
        jj=jj+1
      }
      
      delta=min(delta,max(area))
      for(ii in 1:length(area))
      {
        if(area[ii]<delta)
          estimatesNew[lower[ii]:upper[ii]]=0
      }
      estimatesNew=estimatesNew/(binSize*sum(estimatesNew))
    }
    
    return(estimatesNew)
  }
  
  #c=1-binSize*sum(estimatesThresh)
  #estimatesNew=estimatesThresh+c
  
  ## Previous version:
  ##maxDensity=max(estimates)  
  ##minDensity=0
  ##newXi=(maxDensity+minDensity)/2
  ##eps=1
  ##ii=1
  ##while(ii<=1000)
  ##{
  ##    estimatesNew=apply(as.matrix(estimates),2,function(xx)max(0,xx+newXi))
  ##    area=sum(binSize*estimatesNew)
  ##    eps=abs(1-area)
  ##    if(eps<0.0000001) break; # level found
  ##    if(1>area) minDensity=newXi
  ##    if(1<area) maxDensity=newXi
  ##    newXi=(maxDensity+minDensity)/2
  ##    ii=ii+1
  ##}
  ##estimatesNew=apply(as.matrix(estimates),2,function(xx)max(0,xx+newXi))
  estimatesNew=as.vector(1/binSize*estimatesThresh/sum(estimatesThresh))
  
  runs=rle(estimatesNew>0)
  nRuns=length(runs$values)
  jj=1
  area=lower=upper=NULL
  if(nRuns>2)
  {
    for(ii in 1:nRuns)
    {
      if(runs$values[ii]==FALSE) next;
      whichMin=1
      if(ii>1)
      {
        whichMin=sum(runs$lengths[1:(ii-1)])
      }
      whichMax=whichMin+runs$lengths[ii]
      lower[jj]=whichMin # lower interval of component
      upper[jj]=whichMax # upper interval of component
      area[jj]=sum(binSize*estimatesNew[whichMin:whichMax]) # total area of component
      jj=jj+1
    }
    delta=min(delta,max(area))
    for(ii in 1:length(area))
    {
      if(area[ii]<delta)
        estimatesNew[lower[ii]:upper[ii]]=0
    }
    estimatesNew=estimatesNew/(binSize*sum(estimatesNew))
  }
  
  return(estimatesNew)
  
}
# end normalize density function





findThreshold=function(binSize,estimates,confidence)
{
  estimates=as.vector(estimates)
  maxDensity=max(estimates)  
  minDensity=min(estimates)
  newCut=(maxDensity+minDensity)/2
  eps=1
  ii=1
  while(ii<=1000)
  {
    prob=sum(binSize*estimates*(estimates>newCut))
    eps=abs(confidence-prob)
    if(eps<0.0000001) break; # level found
    if(confidence>prob) maxDensity=newCut
    if(confidence<prob) minDensity=newCut
    newCut=(maxDensity+minDensity)/2
    ii=ii+1
  }
  return(newCut)
}








### #M: below listed are plot functions and density estimation assessmnet, e.g. mode evaluation, or coverage

qqplotConditional=function(object,zTestU,distanceXTestUTrainU,B=1000,delta=0)
{
  # zTest and distanceXTestTrain are drawn from a test set
  if(class(object)!="cDensity") stop("Object should be of class cDensity")
  zTestMax=1
  zTestMin=0
  zGrid=seq(from=zTestMin,to=zTestMax,length.out=B)
  
  estimates=predictDensity(object,zTestMin=0,zTestMax=1,B=length(zGrid),distanceXTestUTrainU,probabilityInterval=F,delta=delta)
  
  estimates=t(apply(estimates,1,function(xx)xx/sum(xx))) # normalize
  estimates=t(apply(estimates,1,cumsum))
  
  percentile=seq(0,1,length.out=B)
  m=dim(estimates)[1]
  grid=expand.grid(1:length(percentile),1:m)
  resultPerc=apply(grid,1,function(xx) { comparison=(estimates[xx[2],]<=percentile[xx[1]])
                                         if(all(comparison==1)) return(length(comparison))
                                         if(all(comparison==0)) return(1)
                                         return(which.min(comparison)) })
  resultPerc=zGrid[resultPerc]  
  resultPerc=matrix(resultPerc,length(percentile),m)
  statistics=apply(resultPerc,1,function(xx)mean(zTestU<=xx))
  par(mgp=c(2.4,1,0))
  plot(percentile,statistics,pch=18,xlab="Expected percentile",ylab="Observed percentile",cex.main=1.7,cex.axis=1.5,cex.lab=1.5,col=4)
  abline(a=0,b=1,lwd=3,col=2)
  resultCumul=apply(as.matrix(1:length(zTestU)),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
                                                                return(estimates[xx,index]) })
  #p=ks.test(resultCumul,runif(length(zTestU),0,1))
  #text(x=0.2,y=0.9,paste("pvalue=",round(p$p.value,3),sep=""),cex=1.5)
  return(resultCumul)
}


qqplotConditionalKNN=function(bestNNDensityContinuous,bestBandwidthContinuous,zTrainL,weightsZTrainL,bestcNNContinuous,distanceXTestU_TrainL,zTestU,B=500)
{
  # zTest and distanceXTestTrain are drawn from a test set
  zTestMax=1
  zTestMin=0
  zGrid=seq(from=zTestMin,to=zTestMax,length.out=B)
  
  
  estimates=t(apply(distanceXTestU_TrainL,1,function(xx)   {
    nearest=sort(xx,index.return=T)$ix[1:bestNNDensityContinuous]
    densityObject=condDensityKNNWithWeightsContinuous(zTrainL[nearest],B,bestBandwidthContinuous,zTestMin,zTestMax,weightsZTrainL[nearest],cVec=bestcNNContinuous)
    return(densityObject$means)
  }))  
    
  m=dim(distanceXTestU_TrainL)[1] # New
  n=dim(distanceXTestU_TrainL)[2] # Old
  
  estimates=t(apply(estimates,1,function(xx)xx/sum(xx))) # normalize
  estimates=t(apply(estimates,1,cumsum))
  
  percentile=seq(0,1,length.out=B)
  grid=expand.grid(1:length(percentile),1:m)
  resultPerc=apply(grid,1,function(xx) { comparison=(estimates[xx[2],]<=percentile[xx[1]])
                                         if(all(comparison==1)) return(length(comparison))
                                         if(all(comparison==0)) return(1)
                                         return(which.min(comparison)) })
  resultPerc=zGrid[resultPerc]  
  resultPerc=matrix(resultPerc,length(percentile),m)
  statistics=apply(resultPerc,1,function(xx)mean(zTestU<=xx))
  par(mgp=c(2.4,1,0))
  plot(percentile,statistics,pch=18,xlab="Expected percentile",ylab="Observed percentile",cex.main=1.7,cex.axis=1.5,cex.lab=1.5,col=4)
  abline(a=0,b=1,lwd=3,col=2)
  resultCumul=apply(as.matrix(1:length(zTestU)),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
                                                                return(estimates[xx,index]) })
  p=ks.test(resultCumul,runif(length(zTestU),0,1))
  text(x=0.2,y=0.9,paste("pvalue=",round(p$p.value,3),sep=""),cex=1.5)
  return(resultCumul)
}

qqplotConditionalGeneric=function(predictedTestLabeled,weightsZTestL,zTestL,B=500)
{
  # zTest and distanceXTestTrain are drawn from a test set
  zTestMax=1
  zTestMin=0
  zGrid=seq(from=zTestMin,to=zTestMax,length.out=dim(predictedTestLabeled)[2])
  
  
  predictedTestLabeled=t(apply(predictedTestLabeled,1,function(xx)xx/sum(xx))) # normalize
  predictedTestLabeled=t(apply(predictedTestLabeled,1,cumsum))
  
  m=dim(predictedTestLabeled)[1]
  
  percentile=seq(0,1,length.out=B)
  grid=expand.grid(1:length(percentile),1:m)
  resultPerc=apply(grid,1,function(xx) { comparison=(predictedTestLabeled[xx[2],]<=percentile[xx[1]])
                                         if(all(comparison==1)) return(length(comparison))
                                         if(all(comparison==0)) return(1)
                                         return(which.min(comparison)) })
  resultPerc=zGrid[resultPerc]  
  resultPerc=matrix(resultPerc,length(percentile),m)
  statistics=apply(resultPerc,1,function(xx)mean(weightsZTestL*(zTestL<=xx)))
  par(mgp=c(6,2.7,0))
  par(mar=c(7.2,9.4,2.1,1.1))
  plot(percentile,statistics,pch=18,xlab="Expected percentile",ylab="Observed percentile",cex.axis=2.5,cex.lab=3,col=4,cex=2.2)
  abline(a=0,b=1,lwd=3,col=2)
  return(statistics)
}

coveragePlotGeneric=function(predictedTestLabeled,weightsZTestL,zTestL,probGrid,type="HPD")
{
  zTestMax=1
  zTestMin=0
  zGrid=seq(from=zTestMin,to=zTestMax,length.out=dim(predictedTestLabeled)[2])  
  
  binSize=(zTestMax-zTestMin)/(dim(predictedTestLabeled)[2]+1)
  n=dim(predictedTestLabeled)[1]
  grid=expand.grid(1:n,1:length(probGrid))
  returnValue=NULL
  if(type=="HPD")
  {
    thresholds=apply(grid,1,function(xx)findThreshold(binSize,predictedTestLabeled[xx[1],],probGrid[xx[2]]))
    
    
    thresholds=matrix(thresholds,n,length(probGrid))
    #upperLenghts=matrix(upperLenghts,n,length(probGrid))
    result=apply(grid,1,function(xx) { index=which.min(abs(zTestL[xx[1]]-zGrid))
                                       thresholds[xx[1],xx[2]]>=predictedTestLabeled[xx[1],index] })
    result=matrix(result,n,length(probGrid))
    empiricalCoverage=1-colMeans(result*weightsZTestL)
    returnValue$thresholds=thresholds # each row is a samples, each column a different probability
  } else {
    stop("Type not implemented")
  }  
  par(mgp=c(6,2.7,0))
  par(mar=c(7.2,9.4,2.1,1.1))
  plot(x=probGrid,y=empiricalCoverage,pch=18,col=4,cex.axis=2.5,cex.lab=3,xlab="Theoretical coverage",ylab="Empirical coverage",xlim=c(0,1),ylim=c(0,1),cex=2.2)
  abline(b=1,a=0,lwd=3,col=2)
  lower=empiricalCoverage-2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
  upper=empiricalCoverage+2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
  apply(as.matrix(1:length(probGrid)),1,function(xx)lines(c(probGrid[xx],probGrid[xx]),c(lower[xx],upper[xx]),col=4,lwd=2))
  return(returnValue)
}



qqplotConditionalKNNBinned=function(bestNNDensity,bestBin,zTrainL,weightsZTrainL,bestcNN,distanceXTestU_TrainL,zTestU,B=500)
{
  # zTest and distanceXTestTrain are drawn from a test set
  zTestMax=1
  zTestMin=0
  zGrid=seq(from=zTestMin,to=zTestMax,length.out=B)
  binsIntervals=seq(zTestMin,zTestMax,length.out=bestBin)
  whichClosest=apply(as.matrix(zGrid),1,function(yy)
  {
    which.min(abs(binsIntervals-yy))
  })
  
  estimates=t(apply(distanceXTestU_TrainL,1,function(xx)   {
    nearest=sort(xx,index.return=T)$ix[1:bestNNDensity]
    densityObject=condDensityKNNWithWeights(zTrainL[nearest],bestBin,zTestMin,zTestMax,weightsZTrainL[nearest],cVec=bestcNNContinuous)
    return(densityObject$means[whichClosest])
  }))  
  
  m=dim(distanceXTestU_TrainL)[1] # New
  n=dim(distanceXTestU_TrainL)[2] # Old
  
  estimates=t(apply(estimates,1,function(xx)xx/sum(xx))) # normalize
  estimates=t(apply(estimates,1,cumsum))
  
  percentile=seq(0,1,length.out=B)
  grid=expand.grid(1:length(percentile),1:m)
  resultPerc=apply(grid,1,function(xx) { comparison=(estimates[xx[2],]<=percentile[xx[1]])
                                         if(all(comparison==1)) return(length(comparison))
                                         if(all(comparison==0)) return(1)
                                         return(which.min(comparison)) })
  resultPerc=zGrid[resultPerc]  
  resultPerc=matrix(resultPerc,length(percentile),m)
  statistics=apply(resultPerc,1,function(xx)mean(zTestU<=xx))
  par(mgp=c(2.4,1,0))
  plot(percentile,statistics,pch=18,xlab="Expected percentile",ylab="Observed percentile",cex.main=1.7,cex.axis=1.5,cex.lab=1.5,col=4)
  abline(a=0,b=1,lwd=3,col=2)
  resultCumul=apply(as.matrix(1:length(zTestU)),1,function(xx) { index=which.min(abs(zTestU[xx]-zGrid))
                                                                 return(estimates[xx,index]) })
  p=ks.test(resultCumul,runif(length(zTestU),0,1))
  text(x=0.2,y=0.9,paste("pvalue=",round(p$p.value,3),sep=""),cex=1.5)
  return(resultCumul)
}


findDistanceCenter=function(binSize,estimates,confidence)
{
  zGrid=seq(0,1,length.out=length(estimates))
  estimates=as.vector(estimates)
  maxDistance=1
  minDistance=0
  newDistance=(maxDistance+minDistance)/2
  center=zGrid[which.max(estimates)]
  eps=1
  ii=1
  while(ii<=1000)
  {
    prob=sum(binSize*estimates[zGrid>(center-newDistance)&zGrid<(center+newDistance)])
    eps=abs(confidence-prob)
    if(eps<0.0000001) break; # level found
    if(confidence>prob) minDistance=newDistance
    if(confidence<prob) maxDistance=newDistance
    newDistance=(maxDistance+minDistance)/2
    ii=ii+1
  }
  return(c(center-newDistance,center+newDistance))
}


coveragePlot=function(object,zGrid,zTestU,distanceXTestUTrainU,probGrid,type="HPD",delta=0)
{
  zTestMin=min(zGrid)
  zTestMax=max(zGrid)
  B=length(zGrid)
  estimates=predictDensity(object,zTestMin=0,zTestMax=1,B=length(zGrid),distanceXTestUTrainU,probabilityInterval=F,delta=delta)
  binSize=(zTestMax-zTestMin)/(B+1)
  n=dim(estimates)[1]
  grid=expand.grid(1:n,1:length(probGrid))
  returnValue=NULL
  if(type=="HPD")
  {
    thresholds=apply(grid,1,function(xx)findThreshold(binSize,estimates[xx[1],],probGrid[xx[2]]))
    
    #upperLenghts=t(apply(as.matrix(1:length(thresholds)),1,function(xx)c(min((1:B)[estimates[grid[xx,1],]>thresholds[xx]]),max(((1:B)[estimates[grid[xx,1],]>thresholds[xx]])))))
    #upperLenghts=t(apply(as.matrix(1:length(thresholds)),1,function(xx)abs(zGrid[upperLenghts[xx,1]]-zGrid[upperLenghts[xx,2]])))
    
    thresholds=matrix(thresholds,n,length(probGrid))
    #upperLenghts=matrix(upperLenghts,n,length(probGrid))
    result=apply(grid,1,function(xx) { index=which.min(abs(zTestU[xx[1]]-zGrid))
                                       thresholds[xx[1],xx[2]]>=estimates[xx[1],index] })
    result=matrix(result,n,length(probGrid))
    empiricalCoverage=1-colMeans(result)
    returnValue$thresholds=thresholds # each row is a samples, each column a different probability
    returnValue$empiricalCoverage=empiricalCoverage
    returnValue$empiricalCoverageInterval=2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
    if(length(probGrid)==1)
    {
      returnValue$lengthInterval=apply(as.matrix(1:n),1,function(xx)mean(estimates[xx,]>thresholds[xx]))
      returnValue$covered=1-result # whether each sample was covered
    }
    
  } else if(type=="Central")
  {
    intervals=t(apply(grid,1,function(xx)findDistanceCenter(binSize,estimates[xx[1],],probGrid[xx[2]])))
    lengths=apply(intervals,1,function(xx)xx[2]-xx[1])
    lengths=matrix(lengths,n,length(probGrid))
    leftInterval=matrix(intervals[,1],n,length(probGrid))
    rightInterval=matrix(intervals[,2],n,length(probGrid))
    
    #result=apply(grid,1,function(xx) (zTest[xx[1]]>intervals[xx[1],1]&zTest[xx[1]]<intervals[xx[1],2]))
    result=apply(grid,1,function(xx) (zTestU[xx[1]]>leftInterval[xx[1],xx[2]]&zTestU[xx[1]]<rightInterval[xx[1],xx[2]]))
    result=matrix(result,n,length(probGrid))
    empiricalCoverage=colMeans(result)
    returnValue$lengths=lengths # each row is a samples, each column a different probability
    returnValue$leftInterval=leftInterval
    returnValue$rightInterval=rightInterval
    returnValue$empiricalCoverage=empiricalCoverage
    returnValue$empiricalCoverageInterval=2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
    
  } else {
    stop("Type not implemented")
  }
  
  
  plot(x=probGrid,y=empiricalCoverage,pch=18,col=4,cex.main=1.7,cex.axis=1.5,cex.lab=1.5,xlab="Theoretical coverage",ylab="Empirical coverage",xlim=c(0,1),ylim=c(0,1),cex=1.4)
  abline(b=1,a=0,lwd=3,col=2)
  lower=empiricalCoverage-2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
  upper=empiricalCoverage+2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
  apply(as.matrix(1:length(probGrid)),1,function(xx)lines(c(probGrid[xx],probGrid[xx]),c(lower[xx],upper[xx]),col=4,lwd=2))
  return(returnValue)
}


coveragePlotKNN=function(bestNNDensityContinuous,bestBandwidthContinuous,zGrid,zTrainL,weightsZTrainL,bestcNNContinuous,distanceXTestU_TrainL,zTestU,probGrid,type="HPD")
{
  zTestMin=min(zGrid)
  zTestMax=max(zGrid)
  B=length(zGrid)
  
  
  estimates=t(apply(distanceXTestU_TrainL,1,function(xx)   {
    nearest=sort(xx,index.return=T)$ix[1:bestNNDensityContinuous]
    densityObject=condDensityKNNWithWeightsContinuous(zTrainL[nearest],B,bestBandwidthContinuous,zTestMin,zTestMax,weightsZTrainL[nearest],cVec=bestcNNContinuous)
    return(densityObject$means)
  }))  
  
  
  
  binSize=(zTestMax-zTestMin)/(B+1)
  n=dim(estimates)[1]
  grid=expand.grid(1:n,1:length(probGrid))
  returnValue=NULL
  if(type=="HPD")
  {
    thresholds=apply(grid,1,function(xx)findThreshold(binSize,estimates[xx[1],],probGrid[xx[2]]))
    
    #upperLenghts=t(apply(as.matrix(1:length(thresholds)),1,function(xx)c(min((1:B)[estimates[grid[xx,1],]>thresholds[xx]]),max(((1:B)[estimates[grid[xx,1],]>thresholds[xx]])))))
    #upperLenghts=t(apply(as.matrix(1:length(thresholds)),1,function(xx)abs(zGrid[upperLenghts[xx,1]]-zGrid[upperLenghts[xx,2]])))
    
    thresholds=matrix(thresholds,n,length(probGrid))
    #upperLenghts=matrix(upperLenghts,n,length(probGrid))
    result=apply(grid,1,function(xx) { index=which.min(abs(zTestU[xx[1]]-zGrid))
                                       thresholds[xx[1],xx[2]]>=estimates[xx[1],index] })
    result=matrix(result,n,length(probGrid))
    empiricalCoverage=1-colMeans(result)
    returnValue$thresholds=thresholds # each row is a samples, each column a different probability
  } else {
    stop("Type not implemented")
  }
  
  
  plot(x=probGrid,y=empiricalCoverage,pch=18,col=4,cex.main=1.7,cex.axis=1.5,cex.lab=1.5,xlab="Theoretical coverage",ylab="Empirical coverage",xlim=c(0,1),ylim=c(0,1),cex=1.4)
  abline(b=1,a=0,lwd=3,col=2)
  lower=empiricalCoverage-2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
  upper=empiricalCoverage+2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
  apply(as.matrix(1:length(probGrid)),1,function(xx)lines(c(probGrid[xx],probGrid[xx]),c(lower[xx],upper[xx]),col=4,lwd=2))
  return(returnValue)
}

coveragePlotKDE=function(band,zGrid,zTrain,covariatesTrain,covariatesTest,zTest,probGrid,type="HPD")
{
  zTestMin=min(zGrid)
  zTestMax=max(zGrid)
  B=length(zGrid)
  
  covariatesTestRep=covariatesTest[sort(rep(1:dim(covariatesTest)[1],length(zGrid))),]
  
  zGridExtend=rep(zGrid,dim(covariatesTest)[1])
  
  
  fit=npcdens(bws=band,txdat=covariatesTrain,tydat=zTrain,exdat=covariatesTestRep,eydat=zGridExtend)
  estimates=matrix(fit$condens,dim(covariatesTest)[1],length(zGrid),byrow=T)
  
  
  binSize=(zTestMax-zTestMin)/(B+1)
  n=dim(estimates)[1]
  grid=expand.grid(1:n,1:length(probGrid))
  returnValue=NULL
  if(type=="HPD")
  {
    thresholds=apply(grid,1,function(xx)findThreshold(binSize,estimates[xx[1],],probGrid[xx[2]]))
    
    #upperLenghts=t(apply(as.matrix(1:length(thresholds)),1,function(xx)c(min((1:B)[estimates[grid[xx,1],]>thresholds[xx]]),max(((1:B)[estimates[grid[xx,1],]>thresholds[xx]])))))
    #upperLenghts=t(apply(as.matrix(1:length(thresholds)),1,function(xx)abs(zGrid[upperLenghts[xx,1]]-zGrid[upperLenghts[xx,2]])))
    
    thresholds=matrix(thresholds,n,length(probGrid))
    #upperLenghts=matrix(upperLenghts,n,length(probGrid))
    result=apply(grid,1,function(xx) { index=which.min(abs(zTest[xx[1]]-zGrid))
                                       thresholds[xx[1],xx[2]]>=estimates[xx[1],index] })
    result=matrix(result,n,length(probGrid))
    empiricalCoverage=1-colMeans(result)
    returnValue$thresholds=thresholds # each row is a samples, each column a different probability
  } else {
    stop("Type not implemented")
  }
  
  
  plot(x=probGrid,y=empiricalCoverage,pch=18,col=4,cex.main=1.7,cex.axis=1.5,cex.lab=1.5,xlab="Theoretical coverage",ylab="Empirical coverage",xlim=c(0,1),ylim=c(0,1),cex=1.4)
  abline(b=1,a=0,lwd=3,col=2)
  lower=empiricalCoverage-2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
  upper=empiricalCoverage+2*sqrt(empiricalCoverage*(1-empiricalCoverage)/n)
  apply(as.matrix(1:length(probGrid)),1,function(xx)lines(c(probGrid[xx],probGrid[xx]),c(lower[xx],upper[xx]),col=4,lwd=2))
  return(returnValue)
}



modesRatio=function(estimates,B=200)
{
  # finds the ratio between the two most important modes
  maxMode=max(estimates)
  
  thresholdVec=seq(0,max(estimates),length.out=B)
  secondMax=rep(NA,B)
  for(kk in 1:B)
  {
    thresholdInstance=thresholdVec[kk]
    runs=rle(estimates>=thresholdInstance)
    positiveRuns=sum(runs$values==1)
    nRuns=length(runs$values)
    currentMax=NULL
    jj=1
    if(positiveRuns>=2)
    {
      for(ii in 1:nRuns)
      {
        if(runs$values[ii]==FALSE) next;
        whichMin=1
        if(ii>1)
        {
          whichMin=sum(runs$lengths[1:(ii-1)])
        }
        whichMax=whichMin+runs$lengths[ii]
        currentMax[jj]=max(estimates[whichMin:whichMax])
        jj=jj+1
      }
      secondMax[kk]=sort(currentMax,decreasing=T)[2]
    }
  }
  if(all(is.na(secondMax))) # one mode only
  {
    return(0)
  }
  return(max(secondMax,na.rm=T)/maxMode)
}

