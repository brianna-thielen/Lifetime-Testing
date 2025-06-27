import pandas as pd
import os
import datetime
import json
import math

# Update to point to folder with all source data
DATA_PATH = './data/SIROF vs Pt'

# Update to include all groups included in that source data
GROUPS = ['LCP Pt Grids', 'Pt Foil', 'SIROF on Pt Foil']

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
            "Pulsing On": [],
            "Temperature (C)": [],
            "Real Days": [],
            "Impedance Magnitude at 1000 Hz (ohms)": [],
            "Impedance Phase at 1000 Hz (degrees)": [],
            "Charge Injection Capacity @ 1000 us (uC/cm^2)": []
        })

        df.to_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")

    # Open all files
    for file in os.listdir(f"{DATA_PATH}/{group}"):
        if "intanimpedance" not in file:
            continue

        data = pd.read_csv(f"{DATA_PATH}/{group}/{file}")

        # Find timestamp
        year = int(file[18:22])
        month = int(file[22:24])
        day = int(file[24:26])
        hour = int(file[26:28])
        minutes = int(file[28:30])
        seconds = int(file[30:32])
        
        test_end_datetime = datetime.datetime(year, month, day, hour, minutes, seconds)

        # Calculate real days from start date
        real_days_end = test_end_datetime - start_date
        real_days_end = real_days_end.total_seconds() / 60 / 60 / 24

        # If there was a CIC measurement, the test start is ~8 mins before save time, and we need to save to df_cic_intan
        if not math.isnan(data.loc[0, "Charge Injection Capacity @ 1000 us (uC/cm^2)"]):
            test_start_datetime = datetime.datetime(year, month, day, hour, minutes, seconds) - datetime.timedelta(minutes=8)

            # Calculate real days from start date
            real_days_start = test_start_datetime - start_date
            real_days_start = real_days_start.total_seconds() / 60 / 60 / 24

        # If there was not a CIC measurement, the test start is ~10 seconds before save time, and we don't need to save df_cic_intan
        else:
            test_start_datetime = datetime.datetime(year, month, day, hour, minutes, seconds) - datetime.timedelta(seconds=10)

            # Calculate real days from start date
            real_days_start = test_start_datetime - start_date
            real_days_start = real_days_start.total_seconds() / 60 / 60 / 24

        # Loop through and save data to each sample
        for idx in range(len(data)):
            sample = data.iloc[idx]["Channel Name"]
            df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            new_rows = pd.DataFrame({
                "Measurement Datetime": [test_start_datetime, test_end_datetime],
                "Pulsing On": [False, True],
                "Temperature (C)": [data.iloc[idx]["Temperature (C)"], data.iloc[idx]["Temperature (C)"]],
                "Real Days": [real_days_start, real_days_end],
                "Impedance Magnitude at 1000 Hz (ohms)": [data.iloc[idx]["Impedance Magnitude at 1000 Hz (ohms)"], float("NaN")],
                "Impedance Phase at 1000 Hz (degrees)": [data.iloc[idx]["Impedance Phase at 1000 Hz (degrees)"], float("NaN")],
                "Charge Injection Capacity @ 1000 us (uC/cm^2)": [data.iloc[idx]["Charge Injection Capacity @ 1000 us (uC/cm^2)"], float("NaN")],
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