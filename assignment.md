# Human Activity Recognition
========================================================

The goal of this project is to predict the manner in which they did the exercise.

# -
library(caret)
When you look at the basic pml.testing data you can observe a lot of NA value in many variable. There you can handle the out of sample error. The variables are unnessesary so we can use dim reduction and drop out these columns. If we drop out them , drop out needed at the training set too. Whit this step we can also improve our model too.

Load the data and caret for predictions

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.0.3
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.0.3
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.0.3
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
pml.training <- read.csv("data/pml-training.csv")
pml.testing <- read.csv("data/pml-testing.csv")
```


```r
myvars.test <- c("user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window","roll_belt","pitch_belt","yaw_belt","total_accel_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z","accel_belt_x","accel_belt_y","accel_belt_z","magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm","yaw_arm","total_accel_arm","gyros_arm_x","gyros_arm_y","gyros_arm_z","accel_arm_x","accel_arm_y","accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z","roll_dumbbell","pitch_dumbbell","yaw_dumbbell","gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z","roll_forearm","pitch_forearm","yaw_forearm","total_accel_forearm","gyros_forearm_x","gyros_forearm_y","gyros_forearm_z","accel_forearm_x","accel_forearm_y","accel_forearm_z","magnet_forearm_x","magnet_forearm_y","magnet_forearm_z","problem_id")

myvars.train <- c("user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window","roll_belt","pitch_belt","yaw_belt","total_accel_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z","accel_belt_x","accel_belt_y","accel_belt_z","magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm","yaw_arm","total_accel_arm","gyros_arm_x","gyros_arm_y","gyros_arm_z","accel_arm_x","accel_arm_y","accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z","roll_dumbbell","pitch_dumbbell","yaw_dumbbell","gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z","roll_forearm","pitch_forearm","yaw_forearm","total_accel_forearm","gyros_forearm_x","gyros_forearm_y","gyros_forearm_z","accel_forearm_x","accel_forearm_y","accel_forearm_z","magnet_forearm_x","magnet_forearm_y","magnet_forearm_z","classe")
```

Slicing the cols

```r
train <- pml.training[myvars.train]
test <- pml.testing[myvars.test]
```


## Model

There are a lot of variable and observation. So we need to control the randomforest algorithm. 
I used 6 fold cross validated random forest. It's running time was acceptable and I got 1 Accuracy and 0.999 Kappa so the model was really good. After this accuracy no boosting or bagging required.


```r
fit <- train(classe~., method="rf", data=train,  trControl = trainControl(method = "cv", number = 6))
```

```
## Warning: package 'e1071' was built under R version 3.0.3
```


```r
fit
```

```
## Random Forest 
## 
## 19622 samples
##    57 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (6 fold) 
## 
## Summary of sample sizes: 16352, 16352, 16351, 16352, 16351, 16352, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.002        0.002   
##   40    1         1      3e-04        3e-04   
##   80    1         1      4e-04        5e-04   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 40.
```

Predicting and the answers to each problem_id

```r
pred <- predict(fit, test)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.0.3
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
answers = cbind(test$problem_id,as.character(pred))
answers
```

```
##       [,1] [,2]
##  [1,] "1"  "B" 
##  [2,] "2"  "A" 
##  [3,] "3"  "B" 
##  [4,] "4"  "A" 
##  [5,] "5"  "A" 
##  [6,] "6"  "E" 
##  [7,] "7"  "D" 
##  [8,] "8"  "B" 
##  [9,] "9"  "A" 
## [10,] "10" "A" 
## [11,] "11" "B" 
## [12,] "12" "C" 
## [13,] "13" "B" 
## [14,] "14" "A" 
## [15,] "15" "E" 
## [16,] "16" "E" 
## [17,] "17" "A" 
## [18,] "18" "B" 
## [19,] "19" "B" 
## [20,] "20" "B"
```

## Appendix

