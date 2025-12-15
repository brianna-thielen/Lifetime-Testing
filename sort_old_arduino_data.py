import pandas as pd
import os
import datetime
import json
import math
import re

# Use this script to sort old data into new format

# Update to point to folder with all source data
DATA_PATH = './data/LCP Encapsulation Capacitive'

# Update to include all groups included in that source data
GROUPS = ['LCP Encapsulation Capacitive', 'LCP Encapsulation Capacitive Ambient']

# Don't touch these
SAMPLE_INFORMATION_PATH = './test_information/samples'

# Loop through each group
for group in GROUPS:
    print(f"Prepping group: {group}")

    # Skip Plots and temp folders
    if "Plots" in group or "temp" in group:
        continue

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
            "Relative Humidity (%)": [],
        })

        df.to_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")

# Start with spreadsheet
data = pd.read_csv(f"{DATA_PATH}/LCP Encapsulation Capacitive/Soak Testing - LCP Encapsulation (Passive).csv")

# Delete irrelevant columns
data.drop(columns=data.columns[[2,6,7,8,9,10,11,12,14,16,18,20]], inplace=True)

# Set row 1 to header titles
data.columns = data.iloc[1]

# Rename columns with lost info
data.columns.values[5] = "RH: 25 um LCP (C2) (%)"
data.columns.values[6] = "RH: 100 um LCP (R1) (%)"
data.columns.values[7] = "RH: 100 um LCP (R2) (%)"
data.columns.values[8] = "RH: 100 um LCP (C1) (%)"
data.columns.values[9] = "RH: 100 um LCP (C2) (%)"

# Data starts at row 9
data = data[3:]

# Sort through and convert to datetime
data["Datetime"] = data["Date"] + " " + data["Time"]
data["Datetime"] = pd.to_datetime(data["Datetime"], format="%m/%d/%y %I:%M %p")

# Delete date and time columns
data.drop(columns=["Date", "Time", "Real Days (Current Step)", "Accel. Days (Total)"], inplace=True)

# Add real days column
data["Real Days"] = None

# Go row by row
for idx, row in data.iterrows():
    current_time = row["Datetime"]
    real_days = (current_time - start_date).total_seconds() / 60 / 60 / 24
    data.loc[idx, "Real Days"] = real_days
    
    for group in GROUPS:
        # Load group info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)
        
        samples = group_info["samples"]

        # Load into file for each sample
        for sample in samples:

            # Save data to sample
            df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            if "C-25-2" in sample:
                rh_sample = row["RH: 25 um LCP (C2) (%)"]
            elif "C-100-1" in sample:
                rh_sample = row["RH: 100 um LCP (C1) (%)"]
            elif "C-100-2" in sample:
                rh_sample = row["RH: 100 um LCP (C2) (%)"]

            rh_sample = float(rh_sample)

            if math.isnan(rh_sample):
                continue
            
            new_rows = pd.DataFrame({
                "Measurement Datetime": [current_time],
                "Temperature (C)": [row["Temperature (C)"]],
                "Real Days": [real_days],
                "Accelerated Days": [float("NaN")],
                "Relative Humidity (%)": [rh_sample],
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

# Open all automated files
for file in os.listdir(f"{DATA_PATH}/LCP Encapsulation Capacitive"):
    print(f"processing {file}")
    if "ENCAP_" not in file:
        continue

    data = pd.read_csv(f"{DATA_PATH}/{group}/{file}")

    header = str(data.iloc[0])

    date_pattern = r'\d{4}[/.]\d{2}[/.]\d{2}[/ ]\d{2}[/:]\d{2}[/:]\d{2}'
    match = re.search(date_pattern, header)
    match = match.group()

    start_time = datetime.datetime.strptime(match, "%Y.%m.%d %H:%M:%S")

    # Loop through each time point and save data (if any)
    for idx, row in data.iterrows():
        pd.set_option('display.max_colwidth', None)
        row = data.iloc[idx,0]
        
        # Find minutes
        min_index = row.index("mins elapsed=") + 13
        mins = row[min_index:]
        min_index = mins.index(" ")
        mins = float(mins[:min_index])

        # Separate into strings for each device 
        ambient_index = row.index("ambient") + 9
        encap_c_100_1_index = row.index("ENCAP-C-100-1") + 14
        encap_c_100_2_index = row.index("ENCAP-C-100-2") + 14
        encap_c_25_2_index = row.index("ENCAP-C-25-2") + 13

        # Get info for ambient
        ambient = row[ambient_index:(encap_c_100_1_index-15)]
        rh_index = ambient.index("RH=") + 3
        temp_index = ambient.index("T=") + 2

        rh_ambient = float(ambient[rh_index:(temp_index-3)])
        temp_ambient = float(ambient[temp_index:])

        # Get info for ENCAP-C-100-1
        encap_c_100_1 = row[encap_c_100_1_index:(encap_c_100_2_index-15)]
        rh_index = encap_c_100_1.index("RH=") + 3
        temp_index = encap_c_100_1.index("T=") + 2

        rh_c_100_1 = float(encap_c_100_1[rh_index:(temp_index-3)])
        temp_c_100_1 = float(encap_c_100_1[temp_index:])

        # Get info for ENCAP-C-100-2
        encap_c_100_2 = row[encap_c_100_2_index:(encap_c_25_2_index-14)]
        rh_index = encap_c_100_2.index("RH=") + 3
        temp_index = encap_c_100_2.index("T=") + 2

        rh_c_100_2 = float(encap_c_100_2[rh_index:(temp_index-3)])
        temp_c_100_2 = float(encap_c_100_2[temp_index:])

        # Get info for ENCAP-C-25-2
        encap_c_25_2 = row[encap_c_25_2_index:]
        rh_index = encap_c_25_2.index("RH=") + 3
        temp_index = encap_c_25_2.index("T=") + 2

        rh_c_25_2 = float(encap_c_25_2[rh_index:(temp_index-3)])
        temp_c_25_2 = float(encap_c_25_2[temp_index:])

        # Find average temperature of soaked samples
        avg_temp = (temp_c_100_1 + temp_c_100_2 + temp_c_25_2) / 3

        # Convert minutes to datetime and calculate aging
        current_time = start_time + datetime.timedelta(minutes=mins)
        real_days = (current_time - start_date).total_seconds() / 60 / 60 / 24
        
        for group in GROUPS:
            # Load group info
            with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
                group_info = json.load(f)
            
            samples = group_info["samples"]

            # Load into file for each sample
            for sample in samples:

                # reopen df
                df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
                df = df.drop(columns=["Unnamed: 0"], axis=1)

                if "C-25-2" in sample:
                    rh_sample = rh_c_25_2
                elif "C-100-1" in sample:
                    rh_sample = rh_c_100_1
                elif "C-100-2" in sample:
                    rh_sample = rh_c_100_2

                new_rows = pd.DataFrame({
                    "Measurement Datetime": [current_time],
                    "Temperature (C)": [avg_temp],
                    "Real Days": [real_days],
                    "Accelerated Days": [float("NaN")],
                    "Relative Humidity (%)": [rh_sample],
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