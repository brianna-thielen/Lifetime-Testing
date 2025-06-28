import time
import datetime
import pandas as pd
from equipment.phidget_4input_temperature import Phidget22TemperatureSensor as phidget

def measure_temperature(sample, group_info, equipment_info):
    # Get sensor and equipment information
    sensor_type = group_info["test_info"]["temp_sensor_type"]
    sensor_channel = group_info["samples"][sample]["temp_sensor_id"]
    delay = equipment_info[sensor_type]["delay"]
    offset = group_info["test_info"]["temp_offset"]
    thermocouple_type = group_info["test_info"]["thermocouple_type"]

    # Measure temperature
    if sensor_type == "phidget":
        temp_sensor = phidget(sensor_channel)
        temp_sensor.open_connection()
        temp_sensor.set_thermocouple_type(thermocouple_type)
        time.sleep(delay)
        temperature = temp_sensor.get_temperature() - offset
        time.sleep(delay)
        temp_sensor.close()
        time.sleep(delay)
    elif sensor_type == "rh":
        # This is captured in run_lifetime_testing.py under perform_arduino_measurements
        # If these sensors are used later for other groups, you can update this section based on that function
        temperature = float("NaN")
    else:
        temperature = float("NaN")

    return temperature

def record_timestamp(stim_on, groups, group_info, equipment_info, data_path):
    timestamp = datetime.datetime.now()

    # Loop through each group
    for group in groups:
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

            # Calculate accelerated days
            accel_days = calculate_accel_days(real_days, temperature_i, df)
            
            # Check if it's a stim device, and create new row accordingly
            if "Pulsing On" in df.columns:
                new_row = pd.DataFrame({
                    "Measurement Datetime": [timestamp],
                    "Pulsing On": [stim_on],
                    "Temperature (C)": [temperature_i],
                    "Real Days": [real_days],
                    "Accelerated Days": [accel_days],
                    "Impedance Magnitude at 1000 Hz (ohms)": [float("NaN")],
                    "Impedance Phase at 1000 Hz (degrees)": [float("NaN")],
                    "Charge Injection Capacity @ 1000 us (uC/cm^2)": [float("NaN")]
                })
            else:
                new_row = pd.DataFrame({
                    "Measurement Datetime": [timestamp],
                    "Temperature (C)": [temperature_i],
                    "Real Days": [real_days],
                    "Accelerated Days": [accel_days],
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
    # Open data summary
    df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
    df = df.drop(columns=["Unnamed: 0"], axis=1)

    # Find number of days since start
    first_day = group_info["start_date"]
    first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

    real_days = measurement_time - first_day
    real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days

    # Calculate accelerated days
    accel_days = calculate_accel_days(real_days, temperature, df)

    # Check if it's a stim device, and create new row accordingly
    if "Pulsing On" in df.columns:
        new_row = pd.DataFrame({
            "Measurement Datetime": [measurement_time],
            "Pulsing On": [False],
            "Temperature (C)": [temperature],
            "Real Days": [real_days],
            "Accelerated Days": [accel_days],
            "Impedance Magnitude at 1000 Hz (ohms)": [impedance],
            "Impedance Phase at 1000 Hz (degrees)": [phase],
            "Charge Injection Capacity @ 1000 us (uC/cm^2)": [float("NaN")]
        })
    else:
        new_row = pd.DataFrame({
            "Measurement Datetime": [measurement_time],
            "Temperature (C)": [temperature],
            "Real Days": [real_days],
            "Accelerated Days": [accel_days],
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
    print(f"saving: {data_path}/{group}/{sample}_data_summary.csv")

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

    # Calculate accelerated days
    accel_days = calculate_accel_days(real_days, temperature, df)

    # Create a new row
    new_row = pd.DataFrame({
        "Measurement Datetime": [measurement_time],
        "Temperature (C)": [temperature],
        "Real Days": [real_days],
        "Accelerated Days": [accel_days],
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

def calculate_accel_days(real_days, temperature, data_summary):
    # Pull last value of real and accelerated days from data_summary
    last_accel_days = data_summary["Accelerated Days"].iloc[-1]
    last_real_days = data_summary["Real Days"].iloc[-1]

    # Calculate new accelerated days
    real_days_elapsed = real_days - last_real_days
    accel_days_elapsed = real_days_elapsed * 2 ** ((temperature - 37) / 10)

    accel_days = last_accel_days + accel_days_elapsed

    return accel_days