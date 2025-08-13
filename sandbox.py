import pandas as pd
import os
import datetime
import json
import math
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pprint
from equipment.intan_rhs import IntanRHS as intan
import time


folder_new = 'C:/Users/3DPrint-Integral/Desktop/temp lifetime/data/LCP Encapsulation Capacitive Ambient'
folder_old = 'C:/Users/3DPrint-Integral/src/Lifetime-Testing/Lifetime-Testing/data/LCP Encapsulation Capacitive Ambient'

files = os.listdir(folder_new)

for file in files:
    if file == 'new' or file == 'raw-data':
        continue
    new = pd.read_csv(f'{folder_new}/{file}')
    old = pd.read_csv(f'{folder_old}/{file}')

    new = new.drop(columns=['Unnamed: 0'])
    old = old.drop(columns=['Unnamed: 0'])

    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates()
    combined = combined.sort_values(by='Real Days').reset_index(drop=True)
    
    # print(new)
    # print(old)
    # print(combined)

    combined.to_csv(f'{folder_new}/new/{file}')