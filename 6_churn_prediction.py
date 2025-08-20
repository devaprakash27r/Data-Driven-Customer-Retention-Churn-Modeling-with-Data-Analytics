import pandas as pd
import joblib

# Load new customer data
new_data = pd.read_csv('New_Customers.CSV')

# Load trained model
model = joblib.load('Best_Model.pkl')

# Load training data to match feature structure
training_data = pd.read_csv('Model_Buliding_Dataset.CSV')

# Drop target column if present
target_cols = ['churn', 'churn_yes']
for col in target_cols:
    if col in training_data.columns:
        training_data = training_data.drop(col, axis=1)

# Get expected features from training
expected_columns = training_data.columns

# Remove target column(s) from new_data if present
for col in target_cols:
    if col in new_data.columns:
        new_data = new_data.drop(col, axis=1)

# Add missing columns with default 0
for col in expected_columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Ensure order of columns matches training
new_data = new_data[expected_columns]

# Predict
predictions = model.predict(new_data)

# Add predictions
new_data['churn_prediction'] = predictions

# Save results
new_data.to_csv('Churn_Predictions.CSV', index=False)

print("✅ Prediction complete! Output saved to 'churn_predictions.csv'.")