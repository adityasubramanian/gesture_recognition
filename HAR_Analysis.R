# Package and Data load
# install.packages(c('class', 'MASS', 'nnet'))
library(class)
library(MASS)
library(nnet)
X_train <- read.table("~/Downloads/UCI HAR Dataset/train/X_train.txt", quote="\"", comment.char="")
Y_train <- read.table("~/Downloads/UCI HAR Dataset/train/y_train.txt", quote="\"", comment.char="")
X_test <- read.table("~/Downloads/UCI HAR Dataset/test/X_test.txt", quote="\"", comment.char="")
Y_test <- read.table("~/Downloads/UCI HAR Dataset/test/y_test.txt", quote="\"", comment.char="")

# Data Formatting
colnames(Y_train) = c("y")
colnames(Y_test) = c("y")
HAR_train <- data.frame(Y_train, X_train)
HAR_test <- data.frame(Y_test, X_test)


# Data investigation
mean(names(HAR_train)==names(HAR_test)) # Checking for inconsistany bw test/train
range(HAR_train$y) # Response is catagorical on 1-6 (see activity_labels.txt)

# "y" is a catagorical variable that takes values from 1 to 6
# V1 to V561 are numerical variables that represent gyroscope measurements
# There is an additional file "activity_labels.txt" that will associate each V1 to V561 to an activity, however we keep default labels here

# Fitting a simple linear model using all predictors
HAR_LM <- lm(y ~ ., data = HAR_train)
summary(HAR_LM)
TrainPredictions <- predict(HAR_LM, HAR_test)
TrainPredictions <- round(TrainPredictions)
mean(TrainPredictions==HAR_test$y)  # 85% Accuracy

# Using a multinomial regression
HAR_Mul <- multinom(y ~ ., data = HAR_train, MaxNWts=5000)
TestPredict_Mul <- predict(HAR_Mul, HAR_test)
mean(TestPredict_Mul==HAR_test$y)  # 90% Accuracy

# Using KNN
sqrt(dim(HAR_train)[1])
# HAR_KNN <- knn(HAR_train[,-1], HAR_test[,-1], HAR_train$y,k = 85) # Computationally intensive
# mean(HAR_KNN==HAR_test$y)  # 90% Accuracy
