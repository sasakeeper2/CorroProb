import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('housing_price_dataset.csv/housing_price_dataset.csv')

target_variable = "Price"

# Drop rows with missing values
df = df.dropna()

# Compute correlations between each independent variable and the target
# (This will be used as weights later; here we take the absolute value so that weights are positive)
correlations = df.corr()[target_variable].drop(target_variable).abs()
print("Correlations (absolute values) with target:")
print(correlations)

# Create a copy of the dataset to store predictions
predictions_df = df.copy()

# Dictionary to store predictions for each feature
predictions = {}

# Iterate through each independent variable to train separate models
for feature in df.columns:
    if feature != target_variable:
        X = df[[feature]]  # Make sure X is 2D
        y = df[target_variable]

        # Train linear regression model on the single feature
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions and store them in the dictionary and in the DataFrame
        predictions[feature] = model.predict(X)
        predictions_df[f"pred_{target_variable}"] = predictions[feature]

# Optionally, compute an aggregated prediction as a weighted average of the predictions
# Weight each feature's prediction by its absolute correlation with the target
weighted_predictions = []
for idx in range(len(df)):
    total_weighted_pred = 0
    total_weight = 0
    for feature in predictions:
        weight = correlations.get(feature, 0)
        total_weight += weight
        total_weighted_pred += predictions[feature][idx] * weight
    # Avoid division by zero
    weighted_pred = total_weighted_pred / total_weight if total_weight != 0 else np.nan
    weighted_predictions.append(weighted_pred)

# Store the aggregated prediction in a new column
predictions_df['weighted_pred'] = weighted_predictions

# Display the aggregated prediction and the first few rows of predictions_df
print("Weighted prediction (first 5 rows):")
print(predictions_df[['weighted_pred']].head())
print("\nFull predictions DataFrame (first 5 rows):")
print(predictions_df.head())

for feature in predictions:
    mse = mean_squared_error(df[target_variable], predictions[feature])
    wmse = mean_squared_error(df[target_variable], predictions_df['weighted_pred'])
    print(f"WMSE for {feature}: {wmse}")
    print(f"MSE for {feature}: {mse}")