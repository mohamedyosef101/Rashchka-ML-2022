import pandas as pd
import os 

# Change directory to the raw data directory
os.chdir('data/raw')

# Define the column names
column_names = [
    'sepal_length', 'sepal_width', 
    'petal_length', 'petal_width', 
    'species'
    ]

# Read the data without headers
iris = pd.read_csv('iris.csv', header=None, names=column_names)

# Print the DataFrame
print(iris)