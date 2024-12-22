from tools import data_downloader 

url = 'https://archive.ics.uci.edu/ml/' \
'machine-learning-databases/iris/iris.data'

data_downloader.download_data(url, 'iris.csv')