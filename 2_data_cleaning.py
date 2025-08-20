# Data cleaning 

import pandas as pd

Raw_Dataset = pd.read_csv('Raw_Dataset.CSV')  # Getting the data from the raw dataset csv file

Raw_Dataset.head() 

Raw_Dataset.info() # To see the non-null values in the columns of the dataset

# To know about the null-values count for each column
print("The total null values are")
print(Raw_Dataset.isnull().sum())

# .describe() will give the descriptive measures of the dataset 
Raw_Dataset.describe(include='all') 

# Using loop to get the first 10 unique values for each column by .unique()[:10]
for col in Raw_Dataset.columns:
    print(f"{col}: {Raw_Dataset[col].unique()[:10]}")

# Creating a list for column names to feed in the function to normalise them 
columns = [
    'gender',
    'partner',
    'dependents',
    'contract',
    'paperless_billing',
    'payment_method',
    'monthly_charges',
    'total_charges',
    'phone_service',
    'multiple_lines',
    'internet_service',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies',
    'churn'
] # columns to change only 
print(columns)

# Creating a function definition for cleaning the column data
def to_normalise_col(raw_data, col_names):
    for col in col_names:
        raw_data[col] = raw_data[col].astype(str).str.strip().str.lower().replace({
        'yes':'yes', 'no':'no', 'none':None, 'nan':None,'':None})
        print(raw_data[col].unique()[:10])
    return raw_data  # You must return df if you're assigning it to Raw_Dataset

Raw_Dataset= to_normalise_col(Raw_Dataset, columns)

# Changing the values 
Raw_Dataset['multiple_lines']=Raw_Dataset['multiple_lines'].replace({'no phone service':'no'})

Raw_Dataset['internet_service']=Raw_Dataset['internet_service'].replace({'no internet':'none'})

# Summary of nulls and uniques
print("Nulls:\n", Raw_Dataset.isnull().sum())

for col in columns:
    print(f"{col}: {Raw_Dataset[col].unique()}")


# Changing the monthly_charges and total_charges Datatypes

print(Raw_Dataset['monthly_charges'].unique()[:10])

print(Raw_Dataset['total_charges'].unique()[:10])


# Creating function definition for cleaning the datatypes of numeric columns

def numeric_col_cleaning(raw_data, col_name):
     # Remove commas, strip spaces, and replace blank strings with NaN
    raw_data[col_name] = (raw_data[col_name]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace({
                '': None,
                'none': None,
                'null': None,
                'nan': None,
                'NaN': None,
                'NA': None
        })                
        )
    # Convert to float with errors='coerce' (invalids become NaN)
    raw_data[col_name] = pd.to_numeric(raw_data[col_name], errors='coerce')
    return raw_data

Raw_Dataset = numeric_col_cleaning(Raw_Dataset, 'monthly_charges') # Function Calling with monthly_charges
Raw_Dataset = numeric_col_cleaning(Raw_Dataset, 'total_charges') # Function Calling with total_charges

# Impute with Median for missing values 
Raw_Dataset['monthly_charges'] = Raw_Dataset['monthly_charges'].fillna(Raw_Dataset['monthly_charges'].median())
Raw_Dataset['total_charges'] = Raw_Dataset['total_charges'].fillna(Raw_Dataset['total_charges'].median())

# Summary and checking 
print(Raw_Dataset[['monthly_charges', 'total_charges']].dtypes)
print(Raw_Dataset[['monthly_charges', 'total_charges']].head(10))
print(Raw_Dataset.isnull().sum()[['monthly_charges', 'total_charges']])
print(Raw_Dataset[['monthly_charges', 'total_charges']].describe())


# Creating Cleaned Dataset
Cleaned_Dataset = Raw_Dataset.copy()

print(Cleaned_Dataset)


# Exporting a CSV file for the cleaned dataset
Cleaned_Dataset.to_csv('Cleaned_Dataset.CSV', index=False)