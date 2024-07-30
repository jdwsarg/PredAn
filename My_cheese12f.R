# Load needed libraries
library(readxl)
library(tidyverse)
library(lubridate)
library(xgboost)
library(caret)
library(data.table)

# Set working directory
setwd("~/Cheese")

# Read the data
data <- read_excel("cheese3.xls") # Replace with your actual file path

# Convert DATE column to Date type
data$DATE <- as.Date(data$DATE, format="%Y-%m-%d") # Adjust the format if necessary

# Aggregate to monthly data
data <- as.data.table(data)
data[, Month := floor_date(DATE, "month")]
monthly_data <- data[, .(Monthly_Average = mean(`BLOCK AVERAGE`, na.rm = TRUE)), by = Month]

# Remove any rows with NA or invalid values in Monthly_Average
monthly_data <- monthly_data[!is.na(Monthly_Average) & is.finite(Monthly_Average)]

# Add lag features
monthly_data[, Lag_1 := shift(Monthly_Average, 1, type = "lag")]
monthly_data[, Lag_2 := shift(Monthly_Average, 2, type = "lag")]
monthly_data[, Lag_3 := shift(Monthly_Average, 3, type = "lag")]

# Add time-related features
monthly_data[, Month_Num := month(Month)]
monthly_data[, Year := year(Month)]

# Remove rows with NA values in the new features
monthly_data <- monthly_data[complete.cases(monthly_data)]

# Split the data into training and test sets
train_data <- monthly_data[Month < "2024-01-01"]
test_data <- monthly_data[Month >= "2024-01-01" & Month <= "2024-06-01"]

# Prepare data for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, .(Lag_1, Lag_2, Lag_3, Month_Num, Year)]), label = train_data$Monthly_Average)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, .(Lag_1, Lag_2, Lag_3, Month_Num, Year)]))

# Train the XGBoost model
params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 6)
model <- xgb.train(params, train_matrix, nrounds = 100)

# Make predictions
predicted_train <- predict(model, train_matrix)
predicted_test <- predict(model, test_matrix)

# Combine actual and predicted data
train_data[, Predicted := predicted_train]
test_data[, Predicted := predicted_test]

# Calculate error metrics
mae_train <- mean(abs(train_data$Monthly_Average - train_data$Predicted))
mse_train <- mean((train_data$Monthly_Average - train_data$Predicted)^2)
rmse_train <- sqrt(mse_train)

mae_test <- mean(abs(test_data$Monthly_Average - test_data$Predicted))
mse_test <- mean((test_data$Monthly_Average - test_data$Predicted)^2)
rmse_test <- sqrt(mse_test)

# Print error metrics
cat("Training MAE:", mae_train, "\n")
cat("Training MSE:", mse_train, "\n")
cat("Training RMSE:", rmse_train, "\n")
cat("Test MAE:", mae_test, "\n")
cat("Test MSE:", mse_test, "\n")
cat("Test RMSE:", rmse_test, "\n")

# Forecast future values
future_dates <- seq.Date(from = as.Date("2024-07-01"), by = "month", length.out = 6)
future_data <- data.table(Month = future_dates)

# Initialize lag features with the last known values
last_known <- tail(test_data, 3)
future_data[, Lag_1 := last_known$Monthly_Average[nrow(last_known)]]
future_data[, Lag_2 := last_known$Monthly_Average[nrow(last_known) - 1]]
future_data[, Lag_3 := last_known$Monthly_Average[nrow(last_known) - 2]]
future_data[, Month_Num := month(Month)]
future_data[, Year := year(Month)]

# Forecast future values iteratively
for (i in 1:nrow(future_data)) {
  future_matrix <- xgb.DMatrix(data = as.matrix(future_data[i, .(Lag_1, Lag_2, Lag_3, Month_Num, Year)]))
  future_data[i, Predicted := predict(model, future_matrix)]
  if (i < nrow(future_data)) {
    future_data[i + 1, Lag_1 := future_data[i, Predicted]]
    future_data[i + 1, Lag_2 := future_data[i, Lag_1]]
    future_data[i + 1, Lag_3 := future_data[i, Lag_2]]
  }
}

# Combine all data
all_data <- rbind(train_data, test_data, future_data, fill = TRUE)

# Plot the data for 2024
ggplot(all_data[Month >= "2024-01-01"], aes(x = Month)) +
  geom_line(aes(y = Monthly_Average, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  geom_line(data = future_data, aes(y = Predicted, color = "Future Predicted")) +
  geom_segment(aes(x = as.Date("2024-06-01"), xend = as.Date("2024-07-01"), 
                   y = test_data$Predicted[length(test_data$Predicted)], 
                   yend = future_data$Predicted[1]), 
               color = "black", linetype = "dashed") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red", "Future Predicted" = "green")) +
  labs(title = "Monthly Average Price of Cheese for 2024", x = "Month", y = "Price per Pound", color = "Legend") +
  theme_minimal() +
  annotate("text", x = as.Date("2024-01-15"), y = max(all_data$Monthly_Average, na.rm = TRUE), 
           label = paste("Test RMSE:", round(rmse_test, 4)), color = "red", hjust = 0)

# Save the results to a CSV file
write.csv(all_data, "xgboost_avg_predicted_prices.csv", row.names = FALSE)

