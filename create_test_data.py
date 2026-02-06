"""
Create complex and unstructured test datasets
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test Case 1: Complex CSV with missing values, mixed types, and outliers
print("Creating complex test datasets...")

# Dataset 1: E-commerce data with issues
np.random.seed(42)
dates = [datetime.now() - timedelta(days=x) for x in range(100)]
categories = ['Electronics', 'Clothing', 'Food', 'Books', None, 'Home & Garden']
regions = ['North', 'South', 'East', 'West', None]

data1 = {
    'Order ID': [f'ORD{i:05d}' for i in range(1, 101)],
    'Date': dates,
    'Product Name': [f'Product {i}' if i % 7 != 0 else None for i in range(100)],
    'Category': np.random.choice(categories, 100),
    'Price': np.random.uniform(10, 1000, 100),
    'Quantity': np.random.randint(1, 20, 100),
    'Revenue': None,  # Will calculate
    'Customer Age': np.random.randint(18, 80, 100),
    'Region': np.random.choice(regions, 100),
    'Rating': np.random.choice([1, 2, 3, 4, 5, None], 100),
    'Is_Premium': np.random.choice(['Yes', 'No', 'Y', 'N', True, False, None], 100),
}

df1 = pd.DataFrame(data1)
# Calculate revenue (with some missing prices)
df1.loc[np.random.choice(df1.index, 15, replace=False), 'Price'] = None
df1['Revenue'] = df1['Price'] * df1['Quantity']

# Add some outliers
df1.loc[np.random.choice(df1.index, 5, replace=False), 'Price'] = np.random.uniform(10000, 50000, 5)

# Add duplicate rows
df1 = pd.concat([df1, df1.iloc[:10]], ignore_index=True)

# Add some weird characters in column names (will be cleaned)
df1_messy = df1.copy()
df1_messy.columns = ['Order  ID!!', 'Date', 'Product-Name', 'Category$', 'Price($)', 
                      'Quantity#', 'Revenue  ', '  Customer_Age', 'Region??', 'Rating***', 'Is Premium?']

# Save to CSV with different encodings and separators
df1_messy.to_csv('test_complex_data.csv', index=False)
print("✓ Created test_complex_data.csv (110 rows with duplicates, missing values, outliers)")

# Dataset 2: Sales data with multiple issues
data2 = {
    'Sales Rep': ['John Doe', 'Jane Smith', 'Bob Johnson', None, 'Alice Brown'] * 20,
    'Sales Amount': [f'${x:,.2f}' for x in np.random.uniform(100, 10000, 100)],  # String with $
    'Commission %': [f'{x}%' for x in np.random.uniform(5, 20, 100)],  # String with %
    'Date Joined': pd.date_range('2020-01-01', periods=100, freq='D'),
    'Active': np.random.choice(['true', 'false', '1', '0', 'Yes', 'No'], 100),
    'Department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR', None], 100),
    'Years Experience': np.random.randint(0, 30, 100).astype(float),
    'Performance Score': np.random.uniform(1.0, 10.0, 100),
}

df2 = pd.DataFrame(data2)
# Add missing values randomly
for col in df2.columns:
    mask = np.random.random(len(df2)) < 0.1
    df2.loc[mask, col] = None

df2.to_csv('test_messy_sales.csv', index=False, sep=';')  # Semicolon separator
print("✓ Created test_messy_sales.csv (semicolon-separated, string numbers, mixed boolean formats)")

# Dataset 3: Excel file with multiple data types
data3 = {
    'Employee ID': range(1, 51),
    'Name': [f'Employee {i}' for i in range(1, 51)],
    'Salary': np.random.uniform(30000, 150000, 50),
    'Hire Date': pd.date_range('2015-01-01', periods=50, freq='ME'),  # ME = Month End
    'Performance Rating': np.random.choice(['Excellent', 'Good', 'Average', 'Poor', None], 50),
    'Bonus': np.random.uniform(0, 20000, 50),
    'Office': np.random.choice(['NY', 'LA', 'Chicago', 'Boston', None], 50),
    'Remote': np.random.choice([True, False], 50),
}

df3 = pd.DataFrame(data3)
# Add duplicates
df3 = pd.concat([df3, df3.iloc[:5]], ignore_index=True)

df3.to_excel('test_employee_data.xlsx', index=False)
print("✓ Created test_employee_data.xlsx (55 rows, XLSX format, dates and booleans)")

print("\n✅ All test datasets created successfully!")
print("\nDatasets created:")
print("1. test_complex_data.csv - E-commerce data with missing values, outliers, duplicates")
print("2. test_messy_sales.csv - Semicolon-separated, string numbers, mixed formats")  
print("3. test_employee_data.xlsx - Excel file with dates, booleans, duplicates")
