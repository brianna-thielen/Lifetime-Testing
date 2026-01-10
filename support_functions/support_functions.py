import time
import datetime
import pandas as pd
import json
import serial
import numpy as np
import math
from equipment.phidget_4input_temperature import Phidget22TemperatureSensor as phidget

def measure_temperature(sample, group_info, equipment_info):
    # Get sensor and equipment information
    sensor_type = group_info["test_info"]["temp_sensor_type"]
    sensor_channel = group_info["samples"][sample.upper()]["temp_sensor_id"]
    delay = equipment_info[sensor_type]["delay"]
    offset = group_info["test_info"]["temp_offset"]

    # Measure temperature
    if sensor_type == "phidget":
        thermocouple_type = group_info["test_info"]["thermocouple_type"]

        temp_sensor = phidget(sensor_channel)
        temp_sensor.open_connection()
        temp_sensor.set_thermocouple_type(thermocouple_type)
        time.sleep(delay)
        temperature = temp_sensor.get_temperature() - offset
        time.sleep(delay)
        temp_sensor.close()
        time.sleep(delay)
    elif sensor_type == "rh-temp":
        # Setup serial connection
        port = group_info["test_info"]["arduino_port"]
        baudrate = group_info["test_info"]["arduino_baudrate"]
        timeout_sec = 120 # Arduino prints output every minute, allow 2 minutes to check for data
        
        ser = serial.Serial(port, baudrate, timeout=timeout_sec)

        # Wait for data
        try:
            # print("Waiting for serial data...")
            line = ser.readline().decode('utf-8').strip()
            # if line:
            #     print("Latest output:", line)
            # else:
            #     print("No data received within timeout period.")
        finally:
            ser.close()

        # Find temperature
        # To use the measurement from the cap sensor in the same vial, replace the "R" in sample with "C"
        sample_i = sample.replace('-R-', '-C-')
        index_i = line.index(sample_i) + len(sample_i) + 2

        data = line[index_i:(index_i+15)]
        temp_index = data.index("T=") + 2

        temperature = float(data[temp_index:])
    else:
        temperature = float("NaN")

    return temperature

def record_timestamp(stim_on, groups, group_info_path, equipment_info, data_path):
    timestamp = datetime.datetime.now()

    # Loop through each group
    for group in groups:
        with open(f"{group_info_path}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Find number of days since start
        first_day = group_info["start_date"]
        first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

        real_days = timestamp - first_day
        real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days

        # Loop through each sample
        for sample in group_info["samples"]:
            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Measure temperature for the given sample
            temperature_i = measure_temperature(sample, group_info, equipment_info)
            
            # Check if it's a stim device, and create new row accordingly
            if "Pulsing On" in df.columns:
                new_row = pd.DataFrame({
                    "Measurement Datetime": [timestamp],
                    "Pulsing On": [stim_on],
                    "Temperature (C)": [temperature_i],
                    "Real Days": [real_days],
                    # "Accelerated Days": [accel_days],
                    "Impedance Magnitude at 1000 Hz (ohms)": [float("NaN")],
                    "Impedance Phase at 1000 Hz (degrees)": [float("NaN")],
                    "Charge Injection Capacity @ 1000 us (uC/cm^2)": [float("NaN")]
                })
            else:
                new_row = pd.DataFrame({
                    "Measurement Datetime": [timestamp],
                    "Temperature (C)": [temperature_i],
                    "Real Days": [real_days],
                    # "Accelerated Days": [accel_days],
                    "Impedance Magnitude at 1000 Hz (ohms)": [float("NaN")],
                    "Impedance Phase at 1000 Hz (degrees)": [float("NaN")]
                })

            # drop any columns that are all NA
            new_row = new_row.dropna(axis=1, how="all")

            if df.empty:
                new_df = new_row.copy()
            elif new_row.empty:
                new_df = df.copy()
            else:
                new_df = pd.concat([df, new_row], ignore_index=True)

            new_df.to_csv(f"{data_path}/{group}/{sample}_data_summary.csv")

def record_impedance_data_to_summary(group, sample, measurement_time, impedance, phase, temperature, data_path, group_info):
    # Convert all values to float (some come in scientific notation)
    impedance = float(impedance)
    phase = float(phase)
    temperature = float(temperature)

    # Open data summary
    df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
    df = df.drop(columns=["Unnamed: 0"], axis=1)

    # Find number of days since start
    first_day = group_info["start_date"]
    first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

    real_days = measurement_time - first_day
    real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days

    # Check if it's a stim device, and create new row accordingly
    if "Pulsing On" in df.columns:
        new_row = pd.DataFrame({
            "Measurement Datetime": [measurement_time],
            "Pulsing On": [False],
            "Temperature (C)": [temperature],
            "Real Days": [real_days],
            # "Accelerated Days": [accel_days],
            "Impedance Magnitude at 1000 Hz (ohms)": [impedance],
            "Impedance Phase at 1000 Hz (degrees)": [phase],
            "Charge Injection Capacity @ 1000 us (uC/cm^2)": [float("NaN")]
        })
    else:
        new_row = pd.DataFrame({
            "Measurement Datetime": [measurement_time],
            "Temperature (C)": [temperature],
            "Real Days": [real_days],
            # "Accelerated Days": [accel_days],
            "Impedance Magnitude at 1000 Hz (ohms)": [impedance],
            "Impedance Phase at 1000 Hz (degrees)": [phase],
        })

    # drop any columns that are all NA
    new_row = new_row.dropna(axis=1, how="all")

    if df.empty:
        new_df = new_row.copy()
    elif new_row.empty:
        new_df = df.copy()
    else:
        new_df = pd.concat([df, new_row], ignore_index=True)
    # print(f"saving: {data_path}/{group}/{sample}_data_summary.csv")

    new_df.to_csv(f"{data_path}/{group}/{sample}_data_summary.csv")

def record_rh_data_to_summary(group, sample, measurement_time, rh, temperature, data_path, group_info):
    # Open data summary
    df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
    df = df.drop(columns=["Unnamed: 0"], axis=1)

    # Find number of days since start
    first_day = group_info["start_date"]
    first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

    real_days = measurement_time - first_day
    real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days

    # Create a new row
    new_row = pd.DataFrame({
        "Measurement Datetime": [measurement_time],
        "Temperature (C)": [temperature],
        "Real Days": [real_days],
        # "Accelerated Days": [accel_days],
        "Relative Humidity (%)": [rh]
    })

    # drop any columns that are all NA
    new_row = new_row.dropna(axis=1, how="all")

    if df.empty:
        new_df = new_row.copy()
    elif new_row.empty:
        new_df = df.copy()
    else:
        new_df = pd.concat([df, new_row], ignore_index=True)

    new_df.to_csv(f"{data_path}/{group}/{sample}_data_summary.csv")

def calculate_accel_days_single(days, real_days_list, accel_days_list):
    # Find two values closest to days in real_days_list
    # Convert to numpy array

    real_days_list = np.array(real_days_list['Real Days'])
    accel_days_list = np.array(accel_days_list)

    # Find where "days" would be inserted
    idx = np.searchsorted(real_days_list, days)

    # If there's an exact match, return accel_days
    if idx < len(real_days_list) and real_days_list[idx] == days:
        accel_days = accel_days_list[idx]

        return accel_days
    
    # Clamp indexes at the ends
    if idx == 0:
        idx = 1
    elif idx == len(real_days_list):
        idx = len(real_days_list) - 1

    # Extract surrounding real_days, corresponding accel_days
    real0, real1 = real_days_list[idx - 1], real_days_list[idx]
    accel0, accel1 = accel_days_list[idx - 1], accel_days_list[idx]

    # Interpolate
    accel_days = accel0 + ((days - real0) / (real1 - real0)) * (accel1 - accel0)

    return accel_days

def calculate_accel_days(real_days, temperature):
    accel_days = [0]

    # First entry should be zero
    if real_days['Real Days'].loc[0] > 0.05:
        day_zero = pd.DataFrame({'Real Days': [0]})
        temp_zero = pd.DataFrame({'Temperature (C)': [temperature['Temperature (C)'].loc[0]]})
        real_days = pd.concat([day_zero, real_days], ignore_index=True)
        temperature = pd.concat([temp_zero, temperature], ignore_index=True)

    # Loop through each day and calculate
    for d in range(1, len(real_days)):
        elapsed_days = real_days['Real Days'].loc[d] - real_days['Real Days'].loc[d-1]

        # If temperature is not nan, save it
        if not math.isnan(temperature['Temperature (C)'].loc[d]):
            temp = temperature['Temperature (C)'].loc[d]

        # If there is a previous saved value in locals, use it (ie do nothing)

        # Calculate accel_days
        accel_days_d = accel_days[d-1] + elapsed_days * 2 ** ((temp - 37) / 10)
        accel_days.append(accel_days_d)

    return accel_days