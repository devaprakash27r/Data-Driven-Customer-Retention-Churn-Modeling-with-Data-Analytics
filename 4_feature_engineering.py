# Feature Engineering 

import pandas as pd

dataset = pd.read_csv('Cleaned_Dataset.CSV')  # Getting the data from the cleaned dataset csv file

print(dataset.shape)
dataset.head()


# Checking the null valuses count
dataset.isnull().sum()


# Droping the unwanted columns

dataset.drop(['customer_id', 'customer_name'], axis=1, inplace=True)


# Identify categorical columns
cat_cols = dataset.select_dtypes(include='object').columns

# Fill NaN with 'unknown'
dataset[cat_cols] = dataset[cat_cols].fillna('unknown')

# Check again
dataset.isnull().sum()


# one-hot encoding on your categorical variables for converting to 0 and 1 for Machine Learning modeling
dataset = pd.get_dummies(dataset, drop_first=True) # drop_first=True to remove the churn_no 
print(dataset)


from sklearn.preprocessing import StandardScaler

# Only scale numerical columns for z score
numeric_cols = ['tenure', 'monthly_charges', 'total_charges']
scaler = StandardScaler()   
dataset[numeric_cols] = scaler.fit_transform(dataset[numeric_cols])


print(dataset.info())
print(dataset.head())

# Exporting a CSV file for the cleaned dataset
dataset.to_csv('Model_Buliding_Dataset.CSV', index=False)