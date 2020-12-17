logs <- read.csv("/home/pallavi/git/course/statistics/final_project/data/log2.csv")
print(head(logs))

# re-arrange dataframe with Action in 1
data <- logs[, c(5,1,2,3,4,6,7,8,9,10,11,12)]
print(summary(data$Action))
class(data$Action)

library(caret)
library(DMwR)

# Class Balancing
set.seed(8654227, kind="Mersenne-Twister")
smote.1 <- SMOTE(Action~., data=data, perc.over = 200, perc.under=850)
summary(smote.1$Action)

down.sampled <- downSample(x = smote.1[, -1], y = smote.1$Action)
names(down.sampled)[names(down.sampled) == 'Class'] <- 'Action'
data <- down.sampled[, c(12,1,2,3,4,5,6,7,8,9,10,11)]
summary(data)
table(data$Class)
summary(data$Action)

set.seed(45836974, kind="Mersenne-Twister")
perm <- sample(x=nrow(data))
set1 <- data[which(perm<=3*nrow(data)/4),]
set2 <- data[which(perm>3*nrow(data)/4),]

# Random Forest
library(randomForest)

reps=5
all.mtrys = c(6,8,9,10,12)
all.nodesizes = c(2,3,4,5,6)
all.pars.rf = expand.grid(mtry = all.mtrys, nodesize = all.nodesizes)
n.pars = nrow(all.pars.rf)
M = 5 # Number of times to repeat RF fitting. I.e. Number of OOB errors

### Container to store OOB errors. This will be easier to read if we name
### the columns.
all.OOB.rf = array(0, dim = c(M, n.pars))
names.pars = apply(all.pars.rf, 1, paste0, collapse = "-")
colnames(all.OOB.rf) = names.pars

for(i in 1:n.pars){
  ### Progress update
  print(paste0(i, " of ", n.pars))
  
  ### Get tuning parameters for this iteration
  this.mtry = all.pars.rf[i, "mtry"]
  this.nodesize = all.pars.rf[i, "nodesize"]
  
  for(j in 1:M){
    ### Fit RF, then get and store OOB errors
    this.fit.rf = randomForest(Action ~ ., data = set1, ntree=500,
                               mtry = this.mtry, nodesize = this.nodesize)
    
    pred.this.rf = predict(this.fit.rf, set1)
    this.err.rf = mean(set1$Action != pred.this.rf)
    
    all.OOB.rf[j, i] = this.err.rf
  }
}

all.OOB.rf
(mean.oob.rf <- round(colMeans(all.OOB.rf), 4))
min(mean.oob.rf)

boxplot(t(apply(all.OOB.rf, 1, function(W) W)), las=2,  # las sets the axis label orientation
        main = "OOB Boxplot")

# applying tuned hyper-parameters on test data
# 8-2 is the chosen combination
set.seed(8654227, kind="Mersenne-Twister")
rf.final = randomForest(Action ~ ., data = set1,
                        mtry = 8, nodesize = 2)
summary(rf.final)
plot(rf.final)

round(importance(rf.final),3) # Print out importance measures
varImpPlot(rf.final)

pred.rf.train <- predict(rf.final, newdata=set1, type="response")
(misclass.train.rf <- mean(ifelse(pred.rf.train == set1$Action, yes=0, no=1)))

pred.rf.test <- predict(rf.final, newdata=set2, type="response")
(misclass.test.rf <- mean(ifelse(pred.rf.test == set2$Action, yes=0, no=1)))
table(pred.rf.test, as.factor(set2$Action),  dnn=c("Predicted","Observed"))