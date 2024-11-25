import argparse
import datetime
import time
import os

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scipy.integrate as integrate
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from keithley_sourcemeter import KeithleySourceMeter as ksourcemeter

SOURCEMETER_ADDRESS = "GPIB0::16::INSTR"
START_VOLTAGE = 0.0  # V
PEAK_VOLTAGE = 0.8    # V
VALLEY_VOLTAGE = -0.6    # V
END_VOLTAGE = 0.0    # V
SWEEP_RATE = 0.25      # V/s
STEP_SIZE = 0.01      # V
CURRENT_THRESHOLD = 10e-3  # Current threshold in amperes (e.g., 1 mA)
TIME_OFFSET = 0.03  # s
TRANSIENT_TEST_DURATION = 5  # Duration for voltage transient testing in seconds


def main():
    args = parse_args()
    perform_cyclic_voltammetry(args.days, args.cycles, args.electrode, args.testtype)


def parse_args():
    parser = argparse.ArgumentParser(description="Perform cyclic voltammetry testing.")
    parser.add_argument("-d", "--days", type=int, default=0, help="Number of days in soak (accelerated).")
    parser.add_argument("-n", "--cycles", type=int, default=1, help="Number of cycles for cyclic voltammetry testing.")
    parser.add_argument("-e", "--electrode", type=str, default=f"unknown_electrode", help="Electrode name.")
    parser.add_argument("-t", "--testtype", type=str, default=f"coatings", help="Test type (coatings or IDE).")
    return parser.parse_args()


def perform_cyclic_voltammetry(days: int, cycles: int, electrode: str, testtype: str):
    """
    Perform cyclic voltammetry testing.

    Parameters:
    - days (int): The number of days in soak (accelerated time).
    - cycles (int): The number of cycles for cyclic voltammetry testing.
    - electrode (str): The electrode name.

    Returns:
    None
    """
    # Initialize the sourcemeter
    sm = ksourcemeter(SOURCEMETER_ADDRESS)
    sm.reset_and_initialize()
    time.sleep(1)
    sm.set_source("VOLT")
    time.sleep(1)
    sm.set_sense("'CURR'")
    time.sleep(1)
    sm.set_current_compliance(CURRENT_THRESHOLD)
    time.sleep(1)
    sm.sourcemeter.write('SENS:VOLT:NPLC 0.01')

    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=(
            "CV",
        ),
    )

    for i in range(cycles):
        print(f"Cycle {i + 1}/{cycles}")
        # Perform cyclic voltammetry testing
        log_time, voltages, currents = cyclic_voltammetry(sm)

        # Save the data to a CSV file
        data = {
            "Time (s)": log_time,
            "Voltage (V)": voltages,
            "Current (A)": currents
        }


        df = pd.DataFrame(data)

        if days < 10:
            daysstr = "000" + str(days)
        elif days < 100:
            daysstr = "00" + str(days)
        elif days < 1000:
            daysstr = "0" + str(days)
        else:
            daysstr = str(days)

        # Create directory if it doesn't exist
        if os.path.exists(f"./data/{testtype}/{daysstr}") == False:
            os.makedirs(f"./data/{testtype}/{daysstr}")
            
        filename = f"CV_{days}_{electrode}_cycle{i+1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        file_path = f"./data/{testtype}/{daysstr}/{filename}.csv"

        # Create directory if it doesn't exist
        if os.path.exists(f"./data/{testtype}/{daysstr}") == False:
            os.makedirs(f"./data/{testtype}/{daysstr}")

        df.to_csv(file_path, index=False)
        
        print(f"Dataframe saved as CSV file: {file_path}")

        fig.add_trace(
            go.Scatter(
                x=df["Voltage (V)"],
                y=df["Current (A)"],
                mode="lines",
            ),
            row=1,
            col=1,
        )

    # Plot the cyclic voltammetry curve
    # plot_cyclic_voltammetry(df)
    fig.show()

    # Close the sourcemeter connection
    sm.close()


def cyclic_voltammetry(sm: ksourcemeter):
    """
    Perform cyclic voltammetry testing.

    Parameters:
    - sm (sourcemeter): The sourcemeter object.

    Returns:
    - log_time (list): The list of time points.
    - voltages (list): The list of voltages.
    - currents (list): The list of currents.
    """
    log_time = []
    voltages = []
    currents = []
    start_time = time.time()
    # Perform the forward sweep
    log_time, voltages, currents = forward_sweep(sm, voltages, currents, log_time, start_time)

    # Perform the reverse sweep
    log_time, voltages, currents = reverse_sweep(sm, voltages, currents, log_time, start_time)

    # Perform the settle sweep to 0.0 V
    log_time, voltages, currents = settle_sweep(sm, voltages, currents, log_time, start_time)

    return log_time, voltages, currents


def forward_sweep(sm: ksourcemeter, voltages: list, currents: list, log_time: list, start_time: float):
    """
    Perform the forward sweep of cyclic voltammetry testing.

    Parameters:
    - sm (sourcemeter): The sourcemeter object.
    - voltages (list): The list of voltages.
    - currents (list): The list of currents.
    - log_time (list): The list of time points.

    Returns:
    - log_time (list): The updated list of time points.
    - voltages (list): The updated list of voltages.
    - currents (list): The updated list of currents.
    """
    # print("Performing forward sweep...")
    for voltage in np.arange(START_VOLTAGE, PEAK_VOLTAGE + STEP_SIZE, STEP_SIZE):
        sm.set_voltage(voltage)
        time.sleep(STEP_SIZE / SWEEP_RATE)
        current = float(sm.measure_current())
        # current = 1
        log_time.append(time.time() - start_time - (TIME_OFFSET*len(log_time)))
        voltages.append(voltage)
        currents.append(current)

    return log_time, voltages, currents


def reverse_sweep(sm: ksourcemeter, voltages: list, currents: list, log_time: list, start_time: float):
    """
    Perform the reverse sweep of cyclic voltammetry testing.

    Parameters:
    - sm (sourcemeter): The sourcemeter object.
    - voltages (list): The list of voltages.
    - currents (list): The list of currents.
    - log_time (list): The list of time points.
    - start_time (float): The start time of the cyclic voltammetry testing.

    Returns:
    - log_time (list): The updated list of time points.
    - voltages (list): The updated list of voltages.
    - currents (list): The updated list of currents.
    """
    # print("Performing reverse sweep...")
    for voltage in np.arange(PEAK_VOLTAGE, VALLEY_VOLTAGE - STEP_SIZE, -STEP_SIZE):
        sm.set_voltage(voltage)
        time.sleep(STEP_SIZE / SWEEP_RATE)
        current = float(sm.measure_current())
        # current = 1
        log_time.append(time.time() - start_time - (TIME_OFFSET*len(log_time)))
        voltages.append(voltage)
        currents.append(current)

    return log_time, voltages, currents


def settle_sweep(sm: ksourcemeter, voltages: list, currents: list, log_time: list, start_time: float):
    """
    Perform the settle sweep of cyclic voltammetry testing.

    Parameters:
    - sm (sourcemeter): The sourcemeter object.
    - voltages (list): The list of voltages.
    - currents (list): The list of currents.
    - log_time (list): The list of time points.
    - start_time (float): The start time of the cyclic voltammetry testing.

    Returns:
    - log_time (list): The updated list of time points.
    - voltages (list): The updated list of voltages.
    - currents (list): The updated list of currents.
    """
    # print("Performing settle sweep...")
    for voltage in np.arange(VALLEY_VOLTAGE, END_VOLTAGE + STEP_SIZE, STEP_SIZE):
        sm.set_voltage(voltage)
        time.sleep(STEP_SIZE / SWEEP_RATE)
        current = float(sm.measure_current())
        # current = 1
        log_time.append(time.time() - start_time - (TIME_OFFSET*len(log_time)))
        voltages.append(voltage)
        currents.append(current)

    return log_time, voltages, currents


def plot_cyclic_voltammetry(data: pd.DataFrame):
    """
    Plot the cyclic voltammetry curve.

    Parameters:
    - data (pd.DataFrame): The cyclic voltammetry data.

    Returns:
    None
    """
    plt.plot(data["Voltage (V)"], data["Current (A)"])
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("Cyclic Voltammetry Curve")
    plt.show()
    

if __name__ == "__main__":
    main()