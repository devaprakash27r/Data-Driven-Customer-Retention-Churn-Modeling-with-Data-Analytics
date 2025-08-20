# Exploratory Data Analysis - EDA

import pandas as pd

dataset = pd.read_csv('Cleaned_Dataset.CSV')  # Getting the data from the cleaned dataset csv file

dataset.head()


dataset.info() # To get the non null values

dataset.isnull().sum() # To get the null value count


# Creating a chart to see the churn distribution
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (6,4)) 
sns.countplot(data=dataset, x='churn')
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()


# Making a list of column names of categorical data

category_cols = [
    'gender',
    'senior_citizen',
    'partner',
    'dependents',
    'contract',
    'paperless_billing',
    'payment_method',
    'phone_service',
    'multiple_lines',
    'internet_service',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies'
]


# Creating a for loop for getting charts for every categorical data column with respect to churn

for col in category_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=dataset, x=col, hue='churn')
    plt.title(f'{col.title()} vs Churn')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Creating a function for getting charts for every numerical data column with respect to churn

def boxplot_numeric(col):
    if col == 'tenure':
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=dataset, x='tenure', hue='churn', fill=True)
        plt.title('KDE Plot of Tenure by Churn')
        plt.show()
    elif col == 'monthly_charges':
        sns.stripplot(data=dataset, x='churn', y='monthly_charges', jitter=True, alpha=0.5)
        plt.title('Monthly Charges Spread by Churn')
        plt.show()
    elif col == 'total_charges':
        sns.histplot(data=dataset, x='total_charges', hue='churn', kde=True, bins=30)
        plt.title('Total Charges Distribution by Churn')
        plt.show()


# Function calling for numerical data charts

boxplot_numeric('tenure')
boxplot_numeric('monthly_charges')
boxplot_numeric('total_charges')


# Copy dataset to avoid modifying the original
correlation_df = dataset.copy()

# Convert churn to numeric: yes = 1, no = 0 for Heatmap 
correlation_df['churn'] = correlation_df['churn'].map({'yes': 1, 'no': 0})


# Correlation Heatmap creation for the dataset of numerical columns

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()