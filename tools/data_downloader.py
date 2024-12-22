import pandas as pd
import os

def download_data(url, name):

    data = pd.read_csv(url)
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, name)
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")