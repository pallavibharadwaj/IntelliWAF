logs <- read.csv("/home/pallavi/git/course/statistics/final_project/data/log2.csv")
# re-arrange dataframe with Action in 1
data <- logs[, c(5,1,2,3,4,6,7,8,9,10,11,12)]
class(data$Action)

# SMOTE
library(DMwR)
set.seed(8654227, kind="Mersenne-Twister")
smote.1 <- SMOTE(Action~., data=data, perc.over = 200, perc.under=850)
summary(smote.1$Action)

down.sampled <- downSample(x = smote.1[, -1], y = smote.1$Action)
names(down.sampled)[names(down.sampled) == 'Class'] <- 'Action'
data <- down.sampled[, c(12,1,2,3,4,5,6,7,8,9,10,11)]
summary(data)

shuffle = function(X){
  new.order = sample.int(length(X))
  new.X = X[new.order]
  return(new.X)
}

scale.1 <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- mean(x2[,col])
    b <- sd(x2[,col])
    x1[,col] <- (x1[,col]-a)/b
  }
  x1
}

rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

set.seed(8654227, kind="Mersenne-Twister")
K = 10
n = nrow(data)
n.fold = n/K # Approximate number of observations per fold
n.fold = ceiling(n.fold)
ordered.ids = rep(1:10, times = n.fold)
ordered.ids = ordered.ids[1:n]
fold.ids = shuffle(ordered.ids)

### Create a container to store CV MSPEs
### One column per model, and one row per fold
CV.models = c("KNN.1", "KNNmin", "KNNse", "LR", "LR Lasso",
              "NMK", "NBK_pc", "full tree","pruned cvmin", "pruned 1se", "RF",
              "NNet", "SVM")
errs.CV = array(0, dim = c(K,length(CV.models)))
colnames(errs.CV) = CV.models
knn.min.models = list(1:K)
knn.1se.models = list(1:K)
rf.models = list(1:K)
nn.models = list(1:K)
svm.models = list(1:K)

for(fold in 1:K){
  set1 = data[fold.ids != fold,]
  set2 = data[fold.ids == fold,]

  # Creating training and test X matrices, then scaling them.
  x.1.unscaled <- as.matrix(set1[,-1])
  x.1 <- scale.1(x.1.unscaled, x.1.unscaled)
  x.2.unscaled <- as.matrix(set2[,-1])
  x.2 <- scale.1(x.2.unscaled, x.1.unscaled)
  
  # KNN (k=1)
  library(FNN)
  knnfit <- knn(train=x.1, test=x.2, cl=set1[,1], k=1)
  
  (misclass.knn <- 
      mean(ifelse(knnfit == set2[,1], yes=0, no=1)))

  # Tuning KNN
  kmax <- 5
  k <- matrix(c(1:kmax), nrow=kmax)
  runknn <- function(x){
    knncv.fit <- knn.cv(train=x.1, cl=set1[,1], k=x)
    # Fitted values are for deleted data from CV
    mean(ifelse(knncv.fit == set1[,1], yes=0, no=1))
  }
  
  # KNN (kmin)
  (mis <- apply(X=k, MARGIN=1, FUN=runknn))
  (mis.se <- sqrt(mis*(1-mis)/nrow(set2)))
  (mink = which.min(mis))
  knnfitmin <- knn(train=x.1, test=x.2, cl=set1[,1], k=mink)
  (misclass.knnmin <- mean(ifelse(knnfitmin == set2[,1], yes=0, no=1)))
  knn.min.models[[fold]] <- knnfitmin
  
  # KNN (k-1se)
  (serule = max(which(mis<mis[mink]+mis.se[mink])))
  knnfitse <- knn(train=x.1, test=x.2, cl=set1[,1], k=serule)
  (misclass.knnse <- 
      mean(ifelse(knnfitse == set2[,1], yes=0, no=1)))
  knn.1se.models[[fold]] <- knnfitse
  
  # Logistic Regression
  set1.rescale <- data.frame(cbind(rescale(set1[,-1], set1[,-1]), Action=set1$Action))
  set2.rescale <- data.frame(cbind(rescale(set2[,-1], set1[,-1]), Action=set2$Action))
  
  library(nnet)
  mod.fit <- multinom(data=set1.rescale, formula=Action ~ ., 
                      trace=TRUE)
  pred.class.test <- predict(mod.fit, newdata=set2.rescale, 
                             type="class")
  (misclass.lr <- mean(ifelse(pred.class.test == set2$Action, 
                                    yes=0, no=1)))
  
  # LR LASSO
  library(glmnet)
  logit.cv <- cv.glmnet(x=as.matrix(set1.rescale[,1:11]), 
                        y=set1.rescale$Action, family="multinomial")
  lascv.pred.test <- predict(logit.cv, type="class", 
                             s=logit.cv$lambda.min, 
                             newx=as.matrix(set2.rescale[,1:11]))
  (mis.lr.lasso <- 
      mean(ifelse(lascv.pred.test == set2$Action, yes=0, no=1)))
  
  # Naive Bayes
  library(klaR)
  
  # Kernel Density Estimation
  NBk <- NaiveBayes(x=x.1, grouping=set1[,1], usekernel=TRUE)
  
  NBk.pred.train <- predict(NBk, newdata=x.1)
  table(NBk.pred.train$class, set1[,1], dnn=c("Predicted","Observed"))
  warnings()
  
  NBk.pred.test <- predict(NBk, newdata=x.2)
  table(NBk.pred.test$class, set2[,1], dnn=c("Predicted","Observed"))
  warnings()
  head(round(NBk.pred.test$posterior))
  
  (NBkmisclass.train <- mean(ifelse(NBk.pred.train$class == set1$Action, yes=0, no=1)))
  (misclass.nbk <- mean(ifelse(NBk.pred.test$class == set2$Action, yes=0, no=1)))
  
  pc <-  prcomp(x=set1[,-1], scale.=TRUE)
  xi.1 <- data.frame(pc$x, class=as.factor(set1$Action))
  xi.2 <- data.frame(predict(pc, newdata=set2), class=as.factor(set2$Action))

  # Naive Bayes with PC
  # Kernel Density Estimation
  NBk.pc <- NaiveBayes(x=xi.1[,-12], grouping=xi.1[,12], usekernel=TRUE)
  
  NBkpc.pred.train <- predict(NBk.pc, newdata=xi.1[,-12], type="class")
  table(NBkpc.pred.train$class, xi.1[,12], dnn=c("Predicted","Observed"))
  
  NBkpc.pred.test <- predict(NBk.pc, newdata=xi.2[,-12], type="class")
  table(NBkpc.pred.test$class, xi.2[,12], dnn=c("Predicted","Observed"))
  warnings()
  (NBkPCmisclass.train <- mean(ifelse(NBkpc.pred.train$class == xi.1$class, yes=0, no=1)))
  (misclass.nbk.pc <- mean(ifelse(NBkpc.pred.test$class == xi.2$class, yes=0, no=1)))

  # Single Tree
  library(rpart)
  tree <- rpart(data=set1, Action ~ ., method="class", cp=0)

  # Find location of minimum error
  cpt = tree$cptable
  minrow <- which.min(cpt[,4])
  # Take geometric mean of cp values at min error and one step up 
  cplow.min <- cpt[minrow,1]
  cpup.min <- ifelse(minrow==1, yes=1, no=cpt[minrow-1,1])
  (cp.min <- sqrt(cplow.min*cpup.min))
  
  # Find smallest row where error is below +1SE
  se.row <- min(which(cpt[,4] < cpt[minrow,4]+cpt[minrow,5]))
  # Take geometric mean of cp values at min error and one step up 
  cplow.1se <- cpt[se.row,1]
  cpup.1se <- ifelse(se.row==1, yes=1, no=cpt[se.row-1,1])
  (cp.1se <- sqrt(cplow.1se*cpup.1se))
  
  # Creating a pruned tree using a selected value of the CP by CV.
  prune.cv.1se <- prune(tree, cp=cp.1se)
  # Creating a pruned tree using a selected value of the CP by CV.
  prune.cv.min <- prune(tree, cp=cp.min)
  
  prune.cv.1se <- prune(tree, cp=cp.1se)
  prune.cv.min <- prune(tree, cp=cp.min)
  
  # Predict results of classification. "Vector" means store class as a number
  pred.test.cv.1se <- predict(prune.cv.1se, newdata=set2[,-1], type="class")
  pred.test.cv.min <- predict(prune.cv.min, newdata=set2[,-1], type="class")
  pred.test.full <- predict(tree, newdata=set2[,-1], type="class")
  
  (misclass.full <- mean(ifelse(pred.test.full == set2$Action, yes=0, no=1)))
  (misclass.cv.min <- mean(ifelse(pred.test.cv.min == set2$Action, yes=0, no=1)))
  (misclass.cv.1se <- mean(ifelse(pred.test.cv.1se == set2$Action, yes=0, no=1)))
  
  # Random Forest
  library(randomForest)

  reps=5
  all.mtrys = c(2,3,4,5,6)
  all.nodesizes = c(2,4,6,8,10)
  all.pars.rf = expand.grid(mtry = all.mtrys, nodesize = all.nodesizes)
  n.pars = nrow(all.pars.rf)
  M = 5 # Number of times to repeat RF fitting. I.e. Number of OOB errors
  
  ### Container to store OOB errors. This will be easier to read if we name
  ### the columns.
  all.OOB.rf = array(0, dim = c(M, n.pars))
  names.pars = apply(all.pars.rf, 1, paste0, collapse = "-")
  colnames(all.OOB.rf) = names.pars

  for(ii in 1:n.pars){
    print(paste0(ii, " of ", n.pars))
    
    ### Get tuning parameters for this iteration
    this.mtry = all.pars.rf[ii, "mtry"]
    this.nodesize = all.pars.rf[ii, "nodesize"]

    err.rf.final <- 9e99
    for(jj in 1:M){
      ### Fit RF, then get and store OOB errors
      this.fit.rf = randomForest(Action ~ ., data = set1,
                                 mtry = this.mtry, nodesize = this.nodesize)
      
      pred.this.rf = predict(this.fit.rf, set1)
      this.err.rf = mean(set1$Action != pred.this.rf)

      if(this.err.rf < err.rf.final){ 
        err.rf.final <- this.err.rf
        rf.final <- this.fit.rf
      }
      all.OOB.rf[jj, ii] = this.err.rf
    }
  }
  rf.models[[fold]] <- rf.final
  
  (mean.oob.rf <- round(colMeans(all.OOB.rf), 4))
  min(mean.oob.rf)

  pred.rf.test <- predict(rf.final, newdata=set2, type="response")
  (misclass.rf <- mean(ifelse(pred.rf.test == set2$Action, yes=0, no=1)))
  
  # Tuning Neural Net with CV
  library(caret)
  trcon = trainControl(method="repeatedcv", number=10, repeats=2,
                       returnResamp="all")
  parmgrid = expand.grid(size=c(4,5,6,7),decay= c(0,0.001,0.01,0.1))
  
  tuned.nnet <- train(x=x.1, y=set1$Action, method="nnet", preProcess="range", trace=FALSE, 
                      tuneGrid=parmgrid, trControl = trcon)
  # fit the best model
  library(nnet)
  y.1 <- class.ind(set1[,1])
  y.2 <- class.ind(set2[,1])
  
  Mi.final = 1
  
  for(kk in 1:10){
    nn <- nnet(y=y.1, x=x.1, size=tuned.nnet$bestTune$size, decay=tuned.nnet$bestTune$decay,
               maxit=1000, softmax=TRUE, trace=FALSE)
    Pi <- predict(nn, newdata=x.1, type="class")
    Mi <- mean(Pi != as.factor(set1[,1]))
    
    print(Mi)
    
    if(Mi < Mi.final){ 
      Mi.final <- Mi
      nn.final <- nn
    }
  }
  nn.models[[fold]] <- nn.final
  # Train error
  p1.train.nn <-predict(nn.final, newdata=x.1, type="class")
  (misclass.train.nn <- mean(ifelse(p1.train.nn == set1$Action, yes=0, no=1)))
  table(p1.train.nn, as.factor(set1$Action),  dnn=c("Predicted","Observed"))
  
  # Test set error
  p2.test.nn <-predict(nn.final, newdata=x.2, type="class")
  (misclass.nnet <- mean(ifelse(p2.test.nn == set2$Action, yes=0, no=1)))
  table(p2.test.nn, as.factor(set2$Action),  dnn=c("Predicted","Observed"))
  
  # SVM
  library(caret)
  
  # Using 10-fold CV so that training sets are not too small
  #  ( Starting with 200 in training set)
  trcon = trainControl(method="repeatedcv", number=10, repeats=2,
                       returnResamp="all")
  parmgrid = expand.grid(C=10^c(0:5), sigma=10^(-c(5:0)))
  
  tuned.svm <- train(x=set1[,-1], y=set1$Action, method="svmRadial", 
                     preProcess=c("center","scale"), trace=FALSE, 
                     tuneGrid=parmgrid, trControl = trcon)

  library(e1071)
  svm.tun <- svm(data=set1, Action ~ ., kernel="radial", 
                    gamma=tuned.svm$bestTune$sigma, cost=tuned.svm$bestTune$C)
  summary(svm.tun)
  head(svm.tun$decision.values)
  head(svm.tun$fitted)
  
  svm.models[[fold]] <- svm.tun
  
  pred1.tun <- predict(svm.tun, newdata=set1)
  table(pred1.tun, set1$Action,  dnn=c("Predicted","Observed"))
  (misclass1.tun <- mean(ifelse(pred1.tun == set1$Action, yes=0, no=1)))
  
  pred2.tun <- predict(svm.tun, newdata=set2)
  table(pred2.tun, set2$Action,  dnn=c("Predicted","Observed"))
  (misclass.svm <- mean(ifelse(pred2.tun == set2$Action, yes=0, no=1)))

  errs.CV[fold, "KNN.1"] = misclass.knn
  errs.CV[fold, "KNNmin"] = misclass.knnmin
  errs.CV[fold, "KNNse"] = misclass.knnse
  errs.CV[fold, "LR"] = misclass.lr
  errs.CV[fold, "LR Lasso"] = mis.lr.lasso
  errs.CV[fold, "NMK"] = misclass.nbk
  errs.CV[fold, "NBK_pc"] = misclass.nbk.pc
  errs.CV[fold, "full tree"] = misclass.full
  errs.CV[fold, "pruned cvmin"] = misclass.cv.min
  errs.CV[fold, "pruned 1se"] = misclass.cv.1se
  errs.CV[fold, "RF"] = misclass.rf
  errs.CV[fold, "NNet"] = misclass.nnet
  errs.CV[fold, "SVM"] = misclass.svm
}
errs.CV
(mean.errs.CV = apply(errs.CV, 2, mean))

boxplot(errs.CV,
        main = "10-Fold CV Error Estimates")

rel.CV.errs = apply(errs.CV, 1, function(W){
  best = min(W)
  return(W / best)
})
(rel.CV.errs = t(rel.CV.errs))

boxplot(rel.CV.errs,
        main = "10-Fold Relative CV Error Estimates")
