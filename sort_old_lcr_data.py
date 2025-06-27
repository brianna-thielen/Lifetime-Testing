import pandas as pd
import os
import datetime
import json
import math
import numpy as np

# Update to point to folder with all source data
DATA_PATH = './data/LCP IDEs'

# Update to include all groups included in that source data
GROUPS = ['LCP IDEs 25um', 'LCP IDEs 100um']

# Don't touch these
SAMPLE_INFORMATION_PATH = './test_information/samples'

# Loop through each group
for group in GROUPS:
    print(f"Sorting group: {group}")

    # Load group info
    with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
        group_info = json.load(f)
    
    samples = group_info["samples"]
    start_date = group_info["start_date"]
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M")

    # Create file for each sample
    for sample in samples:
        df = pd.DataFrame({
            "Measurement Datetime": [],
            "Temperature (C)": [],
            "Real Days": [],
            "Accelerated Days": [],
            "Impedance Magnitude at 1000 Hz (ohms)": [],
            "Impedance Phase at 1000 Hz (degrees)": [],
        })

        df.to_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")

    # Open all files
    for file in os.listdir(f"{DATA_PATH}/{group}"):
        if "EIS_" not in file:
            continue

        data = pd.read_csv(f"{DATA_PATH}/{group}/{file}")

        # find sample
        if "IDE-100" in file:
            sample = file[4:13]
        elif "IDE-25" in file:
            sample = file[4:12]

        # Find timestamp
        i = len(file)
        year = int(file[i-18:i-14])
        month = int(file[i-14:i-12])
        day = int(file[i-12:i-10])
        hour = int(file[i-10:i-8])
        minutes = int(file[i-8:i-6])
        seconds = int(file[i-6:i-4])
        
        test_datetime = datetime.datetime(year, month, day, hour, minutes, seconds)

		# Calculate real days from start
        real_days = (test_datetime - start_date).total_seconds() / 60 / 60 / 24

        # Save data to sample
        df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
        df = df.drop(columns=["Unnamed: 0"], axis=1)

        # Find 1k impedance and temperature
        index = (np.abs(data["Frequency"] - 1000)).idxmin()
        z_1k = data["Impedance"][index].item()
        p_1k = data["Phase Angle"][index].item()
        # 3.5 degree offset between measurement and actual temperature
        temp = data["Temperature (Dry Bath)"][index].item() - 3.5

        new_rows = pd.DataFrame({
            "Measurement Datetime": [test_datetime],
            "Temperature (C)": [temp],
            "Real Days": [real_days],
            "Accelerated Days": [float("NaN")],
            "Impedance Magnitude at 1000 Hz (ohms)": [z_1k],
            "Impedance Phase at 1000 Hz (degrees)": [p_1k],
        })

        # drop any columns that are all NA
        new_rows = new_rows.dropna(axis=1, how="all")

        if df.empty:
            new_df = new_rows.copy()
        elif new_rows.empty:
            new_df = df.copy()
        else:
            new_df = pd.concat([df, new_rows], ignore_index=True)

        new_df.to_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")