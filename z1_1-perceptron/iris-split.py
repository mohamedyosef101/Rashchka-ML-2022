import pandas as pd
import numpy as np 

# Load the iris dataset
iris = pd.read_csv('data/iris.csv') 

# select setosa and versicolor 
y = iris.iloc[0:100, 4].values 
y = np.where(y == 'setosa', 0, 1)

# extract sepal length and petal length 
X = iris.iloc[0:100, [0, 2]].values 