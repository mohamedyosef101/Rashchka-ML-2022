import pandas as pd
import numpy as np 

def split_iris(class1, class2, features):
    """
    Split the iris dataset and filter it for the selected classes and features.
    Args:
        class1 (str): Name of the first class
        class2 (str): Name of the second class
        features (list of int): Indices of the selected features.
    Returns:
        X (ndarray): Feature matrix
        y (ndarray): Target vector
    Example:
        X, y = split_iris('setosa', 'versicolor', [0, 2])
    """
    # Load the iris dataset
    iris = pd.read_csv('data/iris.csv')
    
    # Filter the dataset for the selected classes
    iris_filtered = iris[iris['species'].isin([class1, class2])]
    
    # Create the target variable
    y = np.where(iris_filtered['species'] == class1, 0, 1)
    
    # Extract the selected features
    X = iris_filtered.iloc[:, features].values
    
    return X, y
