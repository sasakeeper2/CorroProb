import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error

# Load the saved ensemble model
model_path = "ensemble.joblib"
model = load(model_path)
print(f"Loaded model from {model_path}")

# Load test dataset
test_file = "test_data.csv"  # Update with actual test dataset path
df_test = pd.read_csv(test_file)

# Ensure only numeric columns
df_test = df_test.select_dtypes(include=[np.number])

# Drop rows with missing values
df_test = df_test.dropna()

# Define target variable
target_variable = "Price"

# Check if target variable exists
if target_variable not in df_test.columns:
    raise KeyError(f"'{target_variable}' column not found in test dataset.")

# Prepare input features (excluding target variable)
X_test = df_test.drop(columns=[target_variable])
y_test = df_test[target_variable]

# Make predictions
predictions = model.predict(X_test)

# Compute and print evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"MSE: {mse}, RMSE: {rmse}")

# Save predictions
df_test["Predicted_Close"] = predictions
df_test.to_csv("ensemble_predictions.csv", index=False)
print("Predictions saved to ensemble_predictions.csv")
