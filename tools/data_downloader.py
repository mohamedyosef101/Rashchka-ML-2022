import pandas as pd
import os
import requests

def download_data(url, name):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    data = pd.read_csv(pd.compat.StringIO(response.text))
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, name)
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")