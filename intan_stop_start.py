# Adds timestamps to acceleration summaries if intan is stopped or started manually
import os
import json
import pandas as pd
import datetime
from equipment.phidget_4input_temperature import Phidget22TemperatureSensor as phidget
import time

# Edit these constants to import new date
INTAN_TIMESTAMP = datetime.datetime(2025, 6, 18, 15, 45)
STIM_ON = False # False if stim was turned off at that time, True if stim was turned on

# These constants do not need editing
SAMPLE_INFORMATION_PATH = './test_information/samples'
DATA_STORAGE_PATH = './data'
EQUIPMENT_INFORMATION_PATH = './test_information/equipment.json'

def main():
    # Loop through each group to see if it's connected to the intan
    for group in os.listdir(DATA_STORAGE_PATH):
        # Skip plots and temp folders
        if "Plots" in group or "temp" in group:
            continue
        
        # Open group info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # If none of the tests use the Intan, skip
        if not any("Intan" in test for test in group_info["test_info"]["tests"]):
            continue

        # If it is an intan group, add the start and/or stop timestamps to the summary
        accel_summary = pd.read_csv(f'{DATA_STORAGE_PATH}/{group}/acceleration_summary.csv')

        # Find number of days since start
        first_day = accel_summary["Measurement Datetime"].iloc[0]
        first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M:%S")

        real_days = INTAN_TIMESTAMP - first_day
        real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days

        # Either select current temperature (if datetime is recent) or closest temperature in the data
        time_to_now = abs(datetime.datetime.now() - INTAN_TIMESTAMP)
        time_threshold = datetime.timedelta(hours=6)

        if time_to_now <= time_threshold:
            with open(EQUIPMENT_INFORMATION_PATH, 'r') as f:
                equipment_info = json.load(f)

            temperature = measure_temperature(group_info["test_info"]["thermocouple"], group_info["test_info"]["temp_offset"], equipment_info["thermocouple"]["delay"])
        else:
            time_diff = abs(accel_summary["Real Days"] - real_days)
            ind_closest = time_diff.idxmin()
            temperature = accel_summary.loc[ind_closest, "Temperature (C)"]

        new_row = pd.DataFrame({
            "Measurement Datetime": INTAN_TIMESTAMP,
            "Pulsing On": STIM_ON,
            "Temperature (C)": temperature,
            "Real Days": real_days,
        }, index=[0])
        print(new_row)

        accel_summary = pd.concat([accel_summary, new_row], ignore_index=True)
        accel_summary.drop('Unnamed: 0', axis=1, inplace=True)
        accel_summary.to_csv(f'{DATA_STORAGE_PATH}/{group}/acceleration_summary.csv')


def measure_temperature(phidget_channel, offset, delay):
    temp_sensor = phidget(phidget_channel)
    temp_sensor.open_connection()
    temp_sensor.set_thermocouple_type(1)
    time.sleep(delay)
    temperature = temp_sensor.get_temperature() - offset
    time.sleep(delay)
    temp_sensor.close()
    time.sleep(delay)

    return temperature

if __name__ == '__main__':
    main()