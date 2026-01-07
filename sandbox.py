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

import subprocess
from datetime import datetime
from pathlib import Path

EQUIPMENT_INFORMATION_PATH = './test_information/equipment.json'
with open(EQUIPMENT_INFORMATION_PATH, 'r') as f:
    EQUIPMENT_INFO = json.load(f)

SAMPLE_INFORMATION_PATH = './test_information/samples'
TEST_INFORMATION_PATH = './test_information/tests.json'
DATA_PATH = './data'
PLOT_PATH = './data/Plots'
IGNORE_PATH = Path('./.gitignore')

def git_commit_and_push(repo_path):
    def run(cmd):
        subprocess.run(cmd, cwd=repo_path, check=True)

    # Stage everything
    run(["git", "add", "."])

    # Commit (will fail if nothing changed)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        run(["git", "commit", "-m", f"Test data update {timestamp}"])
    except subprocess.CalledProcessError:
        # Happens when there is nothing to commit
        return

    # Push
    run(["git", "push"])

def setup_folders_and_gitignore():
    # Check that all data folders exist and add necessary folders to gitignore
    ignore_lines = [
        "# Auto-generated .gitignore",
        "# Do not edit manually\n"
    ]

    # Loop through all groups
    for group in os.listdir(SAMPLE_INFORMATION_PATH):
        # remove ".json" from group name
        group = group[:-5]

        # Open info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Make data folder if it doesn't already exist
        data_folder = Path(f"{DATA_PATH}/{group}")
        raw_data_folder = Path(f"{DATA_PATH}/{group}/raw-data")
        data_folder.mkdir(parents=True, exist_ok=True)
        raw_data_folder.mkdir(parents=True, exist_ok=True)

        # Add to gitignore if specified in group_info
        if not group_info["github_upload"]:
            # Data folder
            ignore_path = f"data/{group}"
            ignore_lines.append(ignore_path)

            # Sample information
            ignore_path = f"test_information/samples/{group}.json"
            ignore_lines.append(ignore_path)

    
    IGNORE_PATH.write_text("\n".join(ignore_lines).rstrip() + "\n")

setup_folders_and_gitignore()

# call whenever you want to sync
git_commit_and_push(EQUIPMENT_INFO["Github"]["path"])


# folder = f'C:/Users/3DPrint-Integral/src/Lifetime-Testing/Lifetime-Testing/data/LCP Encapsulation Capacitive/raw-data/'

# files = os.listdir(folder)

# exp_start_date = datetime(year=2024, month=11, day=4, hour=15, minute=25)

# ambient = pd.DataFrame(columns=['Measurement Datetime', 'Temperature (C)', 'Real Days', 'Relative Humidity (%)'])
# encap_c_100_1 = pd.DataFrame(columns=['Measurement Datetime', 'Temperature (C)', 'Real Days', 'Relative Humidity (%)'])
# encap_c_100_2 = pd.DataFrame(columns=['Measurement Datetime', 'Temperature (C)', 'Real Days', 'Relative Humidity (%)'])
# encap_c_25_2 = pd.DataFrame(columns=['Measurement Datetime', 'Temperature (C)', 'Real Days', 'Relative Humidity (%)'])

# for file in files:
#     if file == 'Soak Testing - LCP Encapsulation (Passive).csv':
#         counter = 0
#         with open (f'{folder}{file}', 'r') as f:
#             for line in f:
#                 if counter >= 4:
#                     words = line.strip().split(',')
                    
#                     if words[19] == '':
#                         continue

#                     meas_date = words[0]
#                     meas_time = words[1]
#                     meas_datetime = datetime.strptime(f'{meas_date} {meas_time}', "%m/%d/%y %I:%M %p")

#                     elapsed = meas_datetime - exp_start_date
#                     elapsed_days = elapsed.total_seconds() / 60 / 60 / 24

#                     ambient_rh = float('nan')
#                     ambient_temp = float('nan')
#                     encap_c_100_1_rh = float(words[19])
#                     encap_c_100_1_temp = float(words[18])
#                     encap_c_100_2_rh = float(words[21])
#                     encap_c_100_2_temp = float(words[20])
#                     encap_c_25_2_rh = float(words[13])
#                     encap_c_25_2_temp = float(words[12])

#                     ambient.loc[len(ambient)] = {
#                         'Measurement Datetime': meas_datetime, 
#                         'Temperature (C)': ambient_temp, 
#                         'Real Days': elapsed_days, 
#                         'Relative Humidity (%)': ambient_rh
#                         }
#                     encap_c_100_1.loc[len(encap_c_100_1)] = {
#                         'Measurement Datetime': meas_datetime, 
#                         'Temperature (C)': encap_c_100_1_temp, 
#                         'Real Days': elapsed_days, 
#                         'Relative Humidity (%)': encap_c_100_1_rh
#                         }
#                     encap_c_100_2.loc[len(encap_c_100_2)] = {
#                         'Measurement Datetime': meas_datetime, 
#                         'Temperature (C)': encap_c_100_2_temp, 
#                         'Real Days': elapsed_days, 
#                         'Relative Humidity (%)': encap_c_100_2_rh
#                         }
#                     encap_c_25_2.loc[len(encap_c_25_2)] = {
#                         'Measurement Datetime': meas_datetime, 
#                         'Temperature (C)': encap_c_25_2_temp, 
#                         'Real Days': elapsed_days, 
#                         'Relative Humidity (%)': encap_c_25_2_rh
#                         }

#                 counter += 1

#     elif file[0:7] == 'ENCAP_2':
#         counter = 0
#         with open (f'{folder}{file}', 'r') as f:
#             for line in f:
#                 # First line has the start date and time
#                 if counter == 0:
#                     words = line.split()

#                     start_date = words[3]
#                     start_time = words[4]

#                     start_datetime = datetime.strptime(f'{start_date} {start_time}', "%Y.%m.%d %H:%M:%S")

#                 # Remaining lines have the RH and temp data
#                 else:
#                     # Skip any data errors
#                     if line[0] != 'm':
#                         continue
#                     if line[0:2] == 'mm':
#                         continue

#                     words = re.split(r"[ =]+", line.strip())

#                     mins = words[2]
#                     meas_datetime = start_datetime + timedelta(minutes=int(mins))

#                     elapsed = meas_datetime - exp_start_date
#                     elapsed_days = elapsed.total_seconds() / 60 / 60 / 24

#                     ambient_rh = float(words[5])
#                     ambient_temp = float(words[7])
#                     encap_c_100_1_rh = float(words[10])
#                     encap_c_100_1_temp = float(words[12])
#                     encap_c_100_2_rh = float(words[15])
#                     encap_c_100_2_temp = float(words[17])
#                     encap_c_25_2_rh = float(words[20])
#                     encap_c_25_2_temp = float(words[22])

#                     # create new dataframes
#                     ambient.loc[len(ambient)] = {
#                         'Measurement Datetime': meas_datetime, 
#                         'Temperature (C)': ambient_temp, 
#                         'Real Days': elapsed_days, 
#                         'Relative Humidity (%)': ambient_rh
#                         }
#                     encap_c_100_1.loc[len(encap_c_100_1)] = {
#                         'Measurement Datetime': meas_datetime, 
#                         'Temperature (C)': encap_c_100_1_temp, 
#                         'Real Days': elapsed_days, 
#                         'Relative Humidity (%)': encap_c_100_1_rh
#                         }
#                     encap_c_100_2.loc[len(encap_c_100_2)] = {
#                         'Measurement Datetime': meas_datetime, 
#                         'Temperature (C)': encap_c_100_2_temp, 
#                         'Real Days': elapsed_days, 
#                         'Relative Humidity (%)': encap_c_100_2_rh
#                         }
#                     encap_c_25_2.loc[len(encap_c_25_2)] = {
#                         'Measurement Datetime': meas_datetime, 
#                         'Temperature (C)': encap_c_25_2_temp, 
#                         'Real Days': elapsed_days, 
#                         'Relative Humidity (%)': encap_c_25_2_rh
#                         }
#                     # new_ambient = pd.DataFrame({
#                     #     'Measurement Datetime': [meas_datetime], 
#                     #     'Temperature (C)': [ambient_temp], 
#                     #     'Real Days': [elapsed_days], 
#                     #     'Relative Humidity (%)': [ambient_rh]
#                     #     })
#                     # new_encap_c_100_1 = pd.DataFrame({
#                     #     'Measurement Datetime': [meas_datetime], 
#                     #     'Temperature (C)': [encap_c_100_1_temp], 
#                     #     'Real Days': [elapsed_days], 
#                     #     'Relative Humidity (%)': [encap_c_100_1_rh]
#                     #     })
#                     # new_encap_c_100_2 = pd.DataFrame({
#                     #     'Measurement Datetime': [meas_datetime], 
#                     #     'Temperature (C)': [encap_c_100_2_temp], 
#                     #     'Real Days': [elapsed_days], 
#                     #     'Relative Humidity (%)': [encap_c_100_2_rh]
#                     #     })
#                     # new_encap_c_25_2 = pd.DataFrame({
#                     #     'Measurement Datetime': [meas_datetime], 
#                     #     'Temperature (C)': [encap_c_25_2_temp], 
#                     #     'Real Days': [elapsed_days], 
#                     #     'Relative Humidity (%)': [encap_c_25_2_rh]
#                     #     })
                    
#                     # # add to old dfs
#                     # ambient = pd.concat([ambient, new_ambient], ignore_index=True)
#                     # encap_c_100_1 = pd.concat([ambient, new_encap_c_100_1], ignore_index=True)
#                     # encap_c_100_2 = pd.concat([ambient, new_encap_c_100_2], ignore_index=True)
#                     # encap_c_25_2 = pd.concat([ambient, new_encap_c_25_2], ignore_index=True)

#                 counter += 1

# # sort
# ambient = ambient.sort_values('Measurement Datetime').reset_index(drop=True)
# encap_c_100_1 = encap_c_100_1.sort_values('Measurement Datetime').reset_index(drop=True)
# encap_c_100_2 = encap_c_100_2.sort_values('Measurement Datetime').reset_index(drop=True)
# encap_c_25_2 = encap_c_25_2.sort_values('Measurement Datetime').reset_index(drop=True)

# ambient.to_csv(f'{folder}ambient.csv')
# encap_c_100_1.to_csv(f'{folder}encap_c_100_1.csv')
# encap_c_100_2.to_csv(f'{folder}encap_c_100_2.csv')
# encap_c_25_2.to_csv(f'{folder}encap_c_25_2.csv')