import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import kagglehub
import os

# Download dataset
dataset_path = kagglehub.dataset_download("khushikyad001/finance-and-economics-dataset-2000-present")

# Find the correct CSV file
csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded dataset.")

file_path = os.path.join(dataset_path, csv_files[0])  # Load the first CSV file

# Load dataset
df = pd.read_csv(file_path)

# Ensure all numeric columns
df = df.select_dtypes(include=[np.number])

# Drop rows with missing values
df = df.dropna()

# Define target variable
target_variable = "Close Price"
if target_variable not in df.columns:
    raise KeyError(f"'{target_variable}' column not found in dataset.")

# Compute correlations
correlations = df.corr()[target_variable].drop(target_variable).abs()
print("Correlations (absolute values) with target:")
print(correlations)

# Create DataFrame for predictions
predictions_df = df.copy()

# Dictionary to store predictions
predictions = {}

# Train models on each feature
for feature in df.columns:
    if feature != target_variable:
        X = df[[feature]]
        y = df[target_variable]

        model = LinearRegression()
        model.fit(X, y)

        # Store predictions
        predictions[feature] = model.predict(X)
        predictions_df[f"pred_{feature}"] = predictions[feature]

# Compute weighted predictions using NumPy
weights = np.array([correlations.get(feature, 0) for feature in predictions])
weighted_sum = np.dot(np.column_stack(list(predictions.values())), weights)
total_weight = np.sum(weights)

# Avoid division by zero
predictions_df['weighted_pred'] = np.where(total_weight != 0, weighted_sum / total_weight, np.nan)

# Display results
print("Weighted prediction (first 5 rows):")
print(predictions_df[['weighted_pred']].head())
print("\nFull predictions DataFrame (first 5 rows):")
print(predictions_df.head())

# Compute MSE for each feature
for feature in predictions:
    mse = mean_squared_error(df[target_variable], predictions[feature])
    print(f"MSE for {feature}: {mse}, sqrt(MSE): {np.sqrt(mse)}")

# Compute weighted MSE (dropping NaNs)
wmse = mean_squared_error(df[target_variable], predictions_df['weighted_pred'].dropna())
print(f"WMSE for {target_variable}: {wmse}, sqrt(WMSE): {np.sqrt(wmse)}")

# Save output
predictions_df.to_csv('predictions_output.csv', index=False)
print('Predictions output saved')
