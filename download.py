import pandas as pd
import matplotlib.pyplot as plt
import requests
import os 

url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
r = requests.get(path,allow_redirects=True, stream = True)
path = os.path.join(url)
filename = (url.split('/')[-1])

