import pandas as pd
import os
from datetime import datetime, timedelta
import json
import math
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pprint
from equipment.intan_rhs import IntanRHS as intan
import time
import re
import csv


folder = f'C:/Users/3DPrint-Integral/src/Lifetime-Testing/Lifetime-Testing/data/LCP Encapsulation Capacitive/raw-data/'

files = os.listdir(folder)

exp_start_date = datetime(year=2024, month=11, day=4, hour=15, minute=25)

ambient = pd.DataFrame(columns=['Measurement Datetime', 'Temperature (C)', 'Real Days', 'Relative Humidity (%)'])
encap_c_100_1 = pd.DataFrame(columns=['Measurement Datetime', 'Temperature (C)', 'Real Days', 'Relative Humidity (%)'])
encap_c_100_2 = pd.DataFrame(columns=['Measurement Datetime', 'Temperature (C)', 'Real Days', 'Relative Humidity (%)'])
encap_c_25_2 = pd.DataFrame(columns=['Measurement Datetime', 'Temperature (C)', 'Real Days', 'Relative Humidity (%)'])

for file in files:
    if file == 'Soak Testing - LCP Encapsulation (Passive).csv':
        counter = 0
        with open (f'{folder}{file}', 'r') as f:
            for line in f:
                if counter >= 4:
                    words = line.strip().split(',')
                    
                    if words[19] == '':
                        continue

                    meas_date = words[0]
                    meas_time = words[1]
                    meas_datetime = datetime.strptime(f'{meas_date} {meas_time}', "%m/%d/%y %I:%M %p")

                    elapsed = meas_datetime - exp_start_date
                    elapsed_days = elapsed.total_seconds() / 60 / 60 / 24

                    ambient_rh = float('nan')
                    ambient_temp = float('nan')
                    encap_c_100_1_rh = float(words[19])
                    encap_c_100_1_temp = float(words[18])
                    encap_c_100_2_rh = float(words[21])
                    encap_c_100_2_temp = float(words[20])
                    encap_c_25_2_rh = float(words[13])
                    encap_c_25_2_temp = float(words[12])

                    ambient.loc[len(ambient)] = {
                        'Measurement Datetime': meas_datetime, 
                        'Temperature (C)': ambient_temp, 
                        'Real Days': elapsed_days, 
                        'Relative Humidity (%)': ambient_rh
                        }
                    encap_c_100_1.loc[len(encap_c_100_1)] = {
                        'Measurement Datetime': meas_datetime, 
                        'Temperature (C)': encap_c_100_1_temp, 
                        'Real Days': elapsed_days, 
                        'Relative Humidity (%)': encap_c_100_1_rh
                        }
                    encap_c_100_2.loc[len(encap_c_100_2)] = {
                        'Measurement Datetime': meas_datetime, 
                        'Temperature (C)': encap_c_100_2_temp, 
                        'Real Days': elapsed_days, 
                        'Relative Humidity (%)': encap_c_100_2_rh
                        }
                    encap_c_25_2.loc[len(encap_c_25_2)] = {
                        'Measurement Datetime': meas_datetime, 
                        'Temperature (C)': encap_c_25_2_temp, 
                        'Real Days': elapsed_days, 
                        'Relative Humidity (%)': encap_c_25_2_rh
                        }

                counter += 1

    elif file[0:7] == 'ENCAP_2':
        counter = 0
        with open (f'{folder}{file}', 'r') as f:
            for line in f:
                # First line has the start date and time
                if counter == 0:
                    words = line.split()

                    start_date = words[3]
                    start_time = words[4]

                    start_datetime = datetime.strptime(f'{start_date} {start_time}', "%Y.%m.%d %H:%M:%S")

                # Remaining lines have the RH and temp data
                else:
                    # Skip any data errors
                    if line[0] != 'm':
                        continue
                    if line[0:2] == 'mm':
                        continue

                    words = re.split(r"[ =]+", line.strip())

                    mins = words[2]
                    meas_datetime = start_datetime + timedelta(minutes=int(mins))

                    elapsed = meas_datetime - exp_start_date
                    elapsed_days = elapsed.total_seconds() / 60 / 60 / 24

                    ambient_rh = float(words[5])
                    ambient_temp = float(words[7])
                    encap_c_100_1_rh = float(words[10])
                    encap_c_100_1_temp = float(words[12])
                    encap_c_100_2_rh = float(words[15])
                    encap_c_100_2_temp = float(words[17])
                    encap_c_25_2_rh = float(words[20])
                    encap_c_25_2_temp = float(words[22])

                    # create new dataframes
                    ambient.loc[len(ambient)] = {
                        'Measurement Datetime': meas_datetime, 
                        'Temperature (C)': ambient_temp, 
                        'Real Days': elapsed_days, 
                        'Relative Humidity (%)': ambient_rh
                        }
                    encap_c_100_1.loc[len(encap_c_100_1)] = {
                        'Measurement Datetime': meas_datetime, 
                        'Temperature (C)': encap_c_100_1_temp, 
                        'Real Days': elapsed_days, 
                        'Relative Humidity (%)': encap_c_100_1_rh
                        }
                    encap_c_100_2.loc[len(encap_c_100_2)] = {
                        'Measurement Datetime': meas_datetime, 
                        'Temperature (C)': encap_c_100_2_temp, 
                        'Real Days': elapsed_days, 
                        'Relative Humidity (%)': encap_c_100_2_rh
                        }
                    encap_c_25_2.loc[len(encap_c_25_2)] = {
                        'Measurement Datetime': meas_datetime, 
                        'Temperature (C)': encap_c_25_2_temp, 
                        'Real Days': elapsed_days, 
                        'Relative Humidity (%)': encap_c_25_2_rh
                        }
                    # new_ambient = pd.DataFrame({
                    #     'Measurement Datetime': [meas_datetime], 
                    #     'Temperature (C)': [ambient_temp], 
                    #     'Real Days': [elapsed_days], 
                    #     'Relative Humidity (%)': [ambient_rh]
                    #     })
                    # new_encap_c_100_1 = pd.DataFrame({
                    #     'Measurement Datetime': [meas_datetime], 
                    #     'Temperature (C)': [encap_c_100_1_temp], 
                    #     'Real Days': [elapsed_days], 
                    #     'Relative Humidity (%)': [encap_c_100_1_rh]
                    #     })
                    # new_encap_c_100_2 = pd.DataFrame({
                    #     'Measurement Datetime': [meas_datetime], 
                    #     'Temperature (C)': [encap_c_100_2_temp], 
                    #     'Real Days': [elapsed_days], 
                    #     'Relative Humidity (%)': [encap_c_100_2_rh]
                    #     })
                    # new_encap_c_25_2 = pd.DataFrame({
                    #     'Measurement Datetime': [meas_datetime], 
                    #     'Temperature (C)': [encap_c_25_2_temp], 
                    #     'Real Days': [elapsed_days], 
                    #     'Relative Humidity (%)': [encap_c_25_2_rh]
                    #     })
                    
                    # # add to old dfs
                    # ambient = pd.concat([ambient, new_ambient], ignore_index=True)
                    # encap_c_100_1 = pd.concat([ambient, new_encap_c_100_1], ignore_index=True)
                    # encap_c_100_2 = pd.concat([ambient, new_encap_c_100_2], ignore_index=True)
                    # encap_c_25_2 = pd.concat([ambient, new_encap_c_25_2], ignore_index=True)

                counter += 1

# sort
ambient = ambient.sort_values('Measurement Datetime').reset_index(drop=True)
encap_c_100_1 = encap_c_100_1.sort_values('Measurement Datetime').reset_index(drop=True)
encap_c_100_2 = encap_c_100_2.sort_values('Measurement Datetime').reset_index(drop=True)
encap_c_25_2 = encap_c_25_2.sort_values('Measurement Datetime').reset_index(drop=True)

ambient.to_csv(f'{folder}ambient.csv')
encap_c_100_1.to_csv(f'{folder}encap_c_100_1.csv')
encap_c_100_2.to_csv(f'{folder}encap_c_100_2.csv')
encap_c_25_2.to_csv(f'{folder}encap_c_25_2.csv')