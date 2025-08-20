# Data extraction from database by connection and query 

import pandas as pd # Importing the pandas library 
from sqlalchemy import create_engine # Importing create_engine from sqlalchemy for database connection
from urllib.parse import quote_plus # For password percent-encode

# Database Credentials
# ⚠️ NOTE: Replace with your own database credentials before running
DB_User = "your_username_here"
DB_Password = quote_plus("your_password_here")  # Turns special characters into safe format
DB_Host = 'localhost'
DB_Port = '5432'
DB_Name = 'customers_database'


# Connection string creation for SQLAlchemy engine
Connection_String = f'postgresql+psycopg2://{DB_User}:{DB_Password}@{DB_Host}:{DB_Port}/{DB_Name}'

# Creating SQLAlchemy engine
Engine = create_engine(Connection_String)

# Writing query to get the data from database tables
Query = """ 
SELECT 
    c.customer_id,
    c.customer_name,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,

    a.tenure,
    a.contract,
    a.paperless_billing,
    a.payment_method,
    a.monthly_charges,
    a.total_charges,
    
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,
    
    l.churn
FROM customers_details_table c
JOIN account_info_table a ON c.customer_id = a.customer_id
JOIN service_details_table s ON c.customer_id = s.customer_id
JOIN churn_labels_table l ON c.customer_id = l.customer_id;
"""

Raw_Dataset = pd.read_sql(Query, Engine) # read_sql() to read the data from database and it's tables 
Raw_Dataset.head() # To see the first 5 data records only

# Exporting a CSV file for the raw dataset
Raw_Dataset.to_csv('Raw_Dataset.CSV', index=False)