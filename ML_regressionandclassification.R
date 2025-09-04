#Setup and data prep----
#Load necessary libraries
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(mgcv)

#Import the dataset
hospital <- read.table("hospital.txt", header = TRUE, sep = " ", stringsAsFactors = TRUE)

#Inspect the dataset
str(hospital)
summary(hospital)

#Convert appropriate columns to factors
hospital <- hospital %>%
  mutate(
    died = as.factor(died),
    gender = as.factor(gender),
    severity = as.factor(severity),
    risk = as.factor(risk),
    avpu = as.factor(avpu)
  )

#Check for missing values
colSums(is.na(hospital))

#Model 1 logit -----
#Logistic Regression Model
logit_model <- glm(died ~ age + gender + bmi + severity + risk + sp02 + sbp + dbp + pulse + respiratory + avpu + temp,
                   data = trainData, family = binomial(link = "logit"))

#Summary of the model
summary(logit_model)

#Evaluate the model on the test set
logit_pred_prob <- predict(logit_model, newdata = testData, type = "response")
logit_pred <- ifelse(logit_pred_prob > 0.5, 1, 0)

#Confusion Matrix
confusionMatrix(as.factor(logit_pred), testData$died)

#Plot ROC Curve
library(pROC)
roc_curve <- roc(testData$died, logit_pred_prob, levels = rev(levels(testData$died)))
plot(roc_curve, main = "ROC Curve for Logistic Regression")
auc(roc_curve)  # Calculate Area Under the Curve
#pretty hgih auc - indicates good model 

#Threshold to maximise Youden index - might not include or only use in appendix if have the time 
optimal_threshold <- coords(roc_curve, "best", ret = "threshold")
logit_pred_adj <- ifelse(logit_pred_prob > optimal_threshold, 1, 0) #Adjust threshold and evaluate

#New confusion matrix
confusionMatrix(as.factor(logit_pred_adj), testData$died)

#Now lets try using step AIC selection for logit for parsimony ----
library(MASS) #ensure we have the right packages 

#Fit a full model with all predictors 
full_model <- glm(died ~ age + gender + bmi + severity + risk + sp02 + sbp + dbp + pulse + respiratory + avpu + temp,
                  data = trainData, family = binomial(link = "logit")) #Full model with all predictors

library(car)
vif(full_model)

#Fit a null model with only the intercept 
null_model <- glm(died ~ 1, data = trainData, family = binomial(link = "logit")) #Null model with intercept only

#Stepwise AIC regression in both directions for parimony 
stepwise_model <- stepAIC(null_model, 
                          scope = list(lower = null_model, upper = full_model),
                          direction = "both", #Stepwise selection in both directions
                          trace = TRUE)  #Set trace = FALSE to suppress output

#Summary of the stepwise model
summary(stepwise_model)

#Now lets evaluate the model 
stepwise_pred_prob <- predict(stepwise_model, newdata = testData, type = "response") #Predict probabilities
stepwise_pred <- ifelse(stepwise_pred_prob > 0.5, 1, 0)#Convert probabilities to binary predictions

#Let us build a Confusion Matrix
confusionMatrix(as.factor(stepwise_pred), testData$died)

#Plot the ROC Curve
roc_curve_stepwise <- roc(testData$died, stepwise_pred_prob, levels = rev(levels(testData$died)))
plot(roc_curve_stepwise, main = "ROC Curve for Stepwise Logistic Regression")
auc(roc_curve_stepwise)
#slightly lower auc but still v high, hence choose this model if want little reg(pars)

#Lasso selection logit for comparison-----
#glmnet for LASSO
library(glmnet)

#Prepare data for glmnet
X <- model.matrix(died ~ age + gender + bmi + severity + risk + sp02 + sbp + dbp + pulse + respiratory + avpu + temp, trainData)[, -1]
y <- trainData$died

#Fit LASSO model with cross validation 
set.seed(123)
lasso_model <- cv.glmnet(X, y, alpha = 1, family = "binomial")

#Plot cross-validated error
plot(lasso_model)

#pick best model - Lambda with minimum error
lambda_min <- lasso_model$lambda.min

# Coefficients of the selected model
lasso_coefficients <- coef(lasso_model, s = lambda_min)
print(lasso_coefficients)

#Coefficients of the best model
#coef(lasso_model, s = "lambda.min") - does same as above 

#Predict probabilities on the test dataset
X_test <- model.matrix(died ~ age + gender + bmi + severity + risk + sp02 + sbp + dbp + pulse + respiratory + avpu + temp, 
                       data = testData)[, -1]
lasso_pred_prob <- predict(lasso_model, newx = X_test, s = lambda_min, type = "response")

#Convert probabilities to binary predictions
lasso_pred <- ifelse(lasso_pred_prob > 0.5, 1, 0)

#Confusion Matrix
confusionMatrix(as.factor(lasso_pred), testData$died)

#ROC Curve and AUC
roc_curve_lasso <- roc(testData$died, lasso_pred_prob, levels = rev(levels(testData$died)))
plot(roc_curve_lasso, main = "ROC Curve for LASSO Logistic Regression")
auc(roc_curve_lasso)

#Extract variables selected by LASSO
selected_variables <- coef(lasso_model, s = "lambda.min")
selected_variables <- selected_variables[selected_variables != 0]  # Retain non-zero coefficients
print(selected_variables)

#Fit logistic regression using selected variables
selected_formula <- as.formula(died ~ (names(selected_variables)[-1], collapse = " + "))  #Exclude intercept
post_lasso_model <- glm(selected_formula, data = trainData, family = "binomial")


#Model comparison for mortality ----

#Function to calculate metrics
evaluate_model <- function(actual, predicted_prob, threshold = 0.5) {
  
#Convert probabilities to binary predictions
predicted <- ifelse(predicted_prob > threshold, 1, 0)
  
#Confusion Matrix
cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
  
#Metrics
accuracy <- cm$overall['Accuracy']
sensitivity <- cm$byClass['Sensitivity']
specificity <- cm$byClass['Specificity']
precision <- cm$byClass['Pos Pred Value']
f1 <- 2 * ((precision * sensitivity) / (precision + sensitivity))
  
#ROC and AUC
roc_curve <- roc(actual, predicted_prob)
auc_value <- auc(roc_curve)
  
#Return results as a list
  return(list(Accuracy = accuracy,
              Sensitivity = sensitivity,
              Specificity = specificity,
              Precision = precision,
              F1 = f1,
              AUC = auc_value))
}

stepwise_metrics <- evaluate_model(testData$died, stepwise_pred_prob)
lasso_metrics <- evaluate_model(testData$died, lasso_pred_prob)

#Now lets compare these metrics 
#Combine metrics into a data frame
model_comparison <- data.frame(
  Model = c("Stepwise Logistic", "LASSO Logistic"),
  Accuracy = c(stepwise_metrics$Accuracy, lasso_metrics$Accuracy),
  Sensitivity = c(stepwise_metrics$Sensitivity, lasso_metrics$Sensitivity),
  Specificity = c(stepwise_metrics$Specificity, lasso_metrics$Specificity),
  Precision = c(stepwise_metrics$Precision, lasso_metrics$Precision),
  F1 = c(stepwise_metrics$F1, lasso_metrics$F1),
  AUC = c(stepwise_metrics$AUC, lasso_metrics$AUC)
)

#Display the comparison table
print(model_comparison)

#Create confusion matrix for two models
model1_predictions <- ifelse(stepwise_pred_prob > 0.5, 1, 0)
model2_predictions <- ifelse(lasso_pred_prob > 0.5, 1, 0)

#Statistical tests
library(mltools)

#Apply McNemar's Test
mcnemar.test(as.factor(model1_predictions), as.factor(model2_predictions))


#Model 2 - GAM for length of stay (los)----
#Load necessary libraries
library(mgcv)
library(ggplot2)
library(caret)

#Split data into training and testing sets - same as model 1 
set.seed(123)
trainIndex <- createDataPartition(hospital$los, p = 0.8, list = FALSE)
trainData <- hospital[trainIndex, ]
testData <- hospital[-trainIndex, ]

#Build the GAM model
gam_model <- gam(los ~ s(age) + s(bmi) + severity + risk + s(sp02) + s(sbp) + s(dbp) +
                   s(pulse) + respiratory + avpu + s(temp), 
                 data = trainData, family = gaussian())

#Summary of the GAM model
summary(gam_model)

#Visualize smooth terms
plot(gam_model, pages = 1)

#Predict and evaluate performance
gam_pred <- predict(gam_model, testData)
cor(gam_pred, testData$los) # Check correlation between predictions and actual values

#Plot for Predicted vs. Actual for Length of Stay
ggplot(data.frame(Predicted = gam_pred, Actual = testData$los), aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(title = "Predicted vs Actual: Length of Stay", x = "Actual LOS", y = "Predicted LOS")


#visualise residuals to evaluate the model 
par(mfrow = c(2, 2))
plot(gam_model, residuals = TRUE, pch = 19, cex = 0.5, main = "Residuals vs Fitted", pages=1)

#Evaluate performance
mse <- mean((gam_pred - testData$los)^2)
rmse <- sqrt(mse)
cat("MSE:", mse, "\nRMSE:", rmse, "\n")

#Visualise predicted vs actual LOS
ggplot(data.frame(Predicted = gam_predictions, Actual = testData$los), aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Predicted vs Actual Length of Stay", x = "Actual LOS", y = "Predicted LOS")

#GAM with interaction terms ----
#Load necessary library
library(mgcv)

#Fit GAM with interaction terms
gam_model_interactions <- gam(
  los ~ s(age) + s(bmi, by = severity) + severity + risk + s(sp02) + s(sbp) + s(dbp) + 
    s(pulse) + respiratory + avpu + s(temp) + severity:risk,
  data = trainData,
  family = gaussian()
)

#Summarise the model
summary(gam_model_interactions)

#Plot smooth terms and interaction effects
par(mfrow = c(2, 2))  # Arrange plots
plot(gam_model_interactions, residuals = TRUE, shade = TRUE, se = TRUE, 
     rug = TRUE, main = "Smooth Terms and Interactions", pages=1)

#model evaluation
#Predict on test data
gam_predictions_interactions <- predict(gam_model_interactions, newdata = testData)

#Evaluate performance metrics
mse <- mean((gam_predictions_interactions - testData$los)^2)
rmse <- sqrt(mse)
cat("MSE with interactions:", mse, "\nRMSE with interactions:", rmse, "\n")

#Visualising the effect of interaction terms ----
#surface plot
#Visualise interaction between two continuous variables (e.g., sp02 and bmi)
vis.gam(
  gam_model_interactions, 
  view = c("bmi", "sp02"),  #Specify variables for the interaction
  plot.type = "persp",      #3D perspective plot
  theta = 30, phi = 20,     #Rotation angles
  main = "Interaction between BMI and SpO2",
  zlab = "LOS"
)


#Contour plot for interaction visualization
vis.gam(
  gam_model_interactions, 
  view = c("bmi", "sp02"),  #Specify variables for the interaction
  plot.type = "contour",    #Contour plot
  color = "terrain",        
  main = "Interaction Contour: BMI vs SpO2"
)


#Prepare data for categorical interactions
predicted <- predict(gam_model_interactions, newdata = testData)
testData$predicted_los <- predicted

#Grouped plot for severity:risk interaction
library(ggplot2)
ggplot(testData, aes(x = risk, y = predicted_los, fill = severity)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Interaction Effect of Severity and Risk on LOS",
       x = "Risk", y = "Predicted LOS") +
  theme_minimal()

#partial dependence 
#Load pdp package
library(pdp)

#Partial dependence plot for an interaction
pdp_severity_risk <- partial(gam_model_interactions, 
                             pred.var = c("severity", "risk"), 
                             grid.resolution = 10)

#Plot PDP
plotPartial(pdp_severity_risk,
            main = "Partial Dependence of Severity and Risk on LOS",
            levelplot = TRUE)  #Contour-style plot

#Residual plot
par(mfrow = c(2, 2))
plot(gam_model_interactions, residuals = TRUE, pch = 19, cex = 0.5, main = "Residuals vs Fitted", pages=1)

#Comparison of GAMs----
#Compare predictions for models with and without interactions
predicted_no_interactions <- predict(gam_model, newdata = testData)
predicted_with_interactions <- predict(gam_model_interactions, newdata = testData)

#Scatter plot comparing predictions
ggplot(data.frame(No_Interaction = predicted_no_interactions, 
                  With_Interaction = predicted_with_interactions), 
       aes(x = No_Interaction, y = With_Interaction)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Comparison of Predictions: Without vs With Interactions",
       x = "Predicted LOS (No Interactions)", 
       y = "Predicted LOS (With Interactions)")

#evaluation (CV, MSE etc.)
#Fit the original GAM
gam_original <- gam(
  los ~ s(age) + s(bmi) + severity + risk + s(sp02) + s(sbp) + s(dbp) + 
    s(pulse) + respiratory + avpu + s(temp),
  data = trainData,
  family = gaussian()
)

#Fit the GAM with interactions
gam_interactions <- gam(
  los ~ s(age) + s(bmi, by = severity) + severity + risk + s(sp02) + 
    s(sbp) + s(dbp) + s(pulse) + respiratory + avpu + s(temp) + severity:risk,
  data = trainData,
  family = gaussian()
)

#Compare models using AIC 
cat("AIC for Original GAM:", AIC(gam_original), "\n")
cat("AIC for GAM with Interactions:", AIC(gam_interactions), "\n")

#Load relevant packages
library(gratia)

#Use built-in diagnostics to validate the models
draw(gam_original)  #Visualise smooth terms
draw(gam_interactions)  #Visualise interaction effects

#Evaluate model performance
appraise(gam_original)
appraise(gam_interactions)

#Load necessary library
library(mgcv)

#Define a custom k-fold cross-validation function
k_fold_cv <- function(model_formula, data, k = 10) {
  #Randomly shuffle the data
  set.seed(123)
  data <- data[sample(1:nrow(data)), ]
  
  #Split data into k folds
  folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)
  
  #Initialise performance metrics
  rmse_values <- c()
  
  #Perform k-fold cross-validation
  for (i in 1:k) {
    # Split into training and testing sets
    test_idx <- which(folds == i, arr.ind = TRUE)
    test_data <- data[test_idx, ]
    train_data <- data[-test_idx, ]
    
    #Fit the GAM model on the training set
    gam_model <- gam(model_formula, data = train_data, family = gaussian())
    
    #Predict on the testing set
    predictions <- predict(gam_model, newdata = test_data)
    
    #Calculate RMSE
    rmse <- sqrt(mean((predictions - test_data$los)^2))
    rmse_values <- c(rmse_values, rmse)
  }
  
  #Return the average RMSE
  return(mean(rmse_values))
}

#Apply k-fold CV to both models
rmse_original <- k_fold_cv(
  los ~ s(age) + s(bmi) + severity + risk + s(sp02) + s(sbp) + s(dbp) + 
    s(pulse) + respiratory + avpu + s(temp),
  data = trainData
)

rmse_interactions <- k_fold_cv(
  los ~ s(age) + s(bmi, by = severity) + severity + risk + s(sp02) + 
    s(sbp) + s(dbp) + s(pulse) + respiratory + avpu + s(temp) + severity:risk,
  data = trainData
)

#Print RMSE results
cat("Average RMSE for Original GAM:", rmse_original, "\n")
cat("Average RMSE for GAM with Interactions:", rmse_interactions, "\n")


