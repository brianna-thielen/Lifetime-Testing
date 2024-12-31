import argparse
import datetime
import time
import os

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from phidget_4input_temperature import Phidget22TemperatureSensor as phidget
from rs_lcx100 import LCX100 as lcx

# _START_FREQ_HZ = 5
# _END_FREQ_HZ = 50000
# _NUM_FREQ_POINTS = 25
_START_FREQ_HZ = 10
_END_FREQ_HZ = 10000
_NUM_FREQ_POINTS = 10
_MEASURE_INTERVAL = "SHOR"
_MEASURE_RANGE = 1e2
_INPUT_VOLTAGE = 0.025
_DATA_COLLECTION_DELAY_FIRST_CONTACT_FREQ_SEC = 2.0
_DATA_COLLECTION_DELAY_FREQ_SEC = 0.0
_DATA_COLLECTION_DELAY_TEMP_SEC = 0.5
_LCR_RESOURCE_ADDRESS = "USB0::0x0AAD::0x0197::3629.8856k02-102189::INSTR"
_MEASURE_TYPE = "R"
_VISA_TIMEOUT_MS = 3000

_MEASURE_TEMP = True
_TEMP_SENSOR_DRY_BATH_CHANNEL = 2
_THEMOCOUPLE_TYPE_J = 1

def main():
    """
    This is the main function that performs impedance testing for a range of frequencies.
    It generates frequencies to test and collects impedance data for each frequency.
    """
    args = parse_args()
    frequencies = generate_frequencies_to_test()

    frequency_data = (
        collect_impedance_vs_frequency(frequencies)
    )
    save_dataframe_as_csv(
        frequency_data,
        f"EIS_{args.days}_{args.electrode}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        args.days,
        args.testtype
    )
    plot_impedance_vs_frequency(
        freq_df=frequency_data,
    )


def parse_args():
    """
    Parses the command line arguments.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Impedance Testing")
    parser.add_argument(
        "-e", 
        "--electrode",
        type=str, 
        default=f"unknown_electrode", 
        help="Electrode name.",
    )
    parser.add_argument(
        "-d", 
        "--days",
        type=int, 
        default=f"unknown_time", 
        help="Number of days in soak (accelerated).",
    )
    parser.add_argument(
        "-t",
        "--testtype",
        type=str,
        help="Test type (coatings or IDE).",
    )
    return parser.parse_args()


def generate_frequencies_to_test():
    """
    Generates a list of frequencies to be tested against.

    Returns:
        frequencies (list): A list of frequencies to be tested.
    """
    frequencies = np.logspace(
        np.log10(_START_FREQ_HZ), np.log10(_END_FREQ_HZ), _NUM_FREQ_POINTS
    )
    print("Frequencies to be tested against: ", frequencies)
    return frequencies

def collect_impedance_vs_frequency(frequencies):
    """
    Collects impedance and phase angle data for different frequencies and contact areas.
    Args:
        frequencies (list): List of frequencies at which impedance is measured.
    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - freq_data_df: DataFrame containing impedance and phase angle data for each contact and frequency.
    """

    if _MEASURE_TEMP:
        temperature_sensor_dry_bath = phidget(_TEMP_SENSOR_DRY_BATH_CHANNEL)
        temperature_sensor_dry_bath.open_connection()
        temperature_sensor_dry_bath.set_thermocouple_type(_THEMOCOUPLE_TYPE_J)
        temperature_sensor_dry_bath.get_thermocouple_type()

    lcx100 = lcx(_LCR_RESOURCE_ADDRESS)
    lcx100.set_aperture(_MEASURE_INTERVAL)
    lcx100.set_visa_timeout(_VISA_TIMEOUT_MS)
    lcx100.reset_and_initialize()
    lcx100.set_measurement_type(_MEASURE_TYPE)
    lcx100.set_measurement_range(_MEASURE_RANGE)
    lcx100.set_voltage(_INPUT_VOLTAGE)
    freq_data = []
    counter = 0

    print("Initializing Impedance vs Frequency Sweep...")
    for freq in frequencies:
        lcx100.set_frequency(freq)
        if counter == 0:
            time.sleep(_DATA_COLLECTION_DELAY_FIRST_CONTACT_FREQ_SEC)
            counter = counter + 1
        else:
            time.sleep(_DATA_COLLECTION_DELAY_FREQ_SEC)
        impedance, phase_angle = lcx100.get_impedance()

        if _MEASURE_TEMP:
            temp_dry_bath = temperature_sensor_dry_bath.get_temperature()
        else:
            temp_dry_bath = np.nan

        time.sleep(_DATA_COLLECTION_DELAY_TEMP_SEC)
        freq_data.append(
            {
                "Frequency": freq,
                "Impedance": float(impedance),
                "Phase Angle": float(phase_angle),
                "Temperature (Dry Bath)": float(temp_dry_bath),
            }
        )
        counter = 0
        
    if _MEASURE_TEMP:
        temperature_sensor_dry_bath.close()
    lcx100.close()
    freq_data_df = pd.DataFrame(freq_data)
    return freq_data_df


def save_dataframe_as_csv(df, filename, days, testtype):
    """
    Saves the generated dataframe as a CSV file in the same directory.

    Args:
        df (pandas.DataFrame): The dataframe to be saved.

    Returns:
        None
    """
    if days < 10:
        daysstr = "000" + str(days)
    elif days < 100:
        daysstr = "00" + str(days)
    elif days < 1000:
        daysstr = "0" + str(days)
    else:
        daysstr = str(days)

    file_path = f"./data/{testtype}/{daysstr}/{filename}.csv"
    # Create directory if it doesn't exist
    if os.path.exists(f"./data/{testtype}/{daysstr}") == False:
        os.makedirs(f"./data/{testtype}/{daysstr}")

    df.to_csv(file_path, index=False)
    print(f"Dataframe saved as CSV file: {file_path}")


def save_plotly_figure_as_image_and_html(fig, filename):
    """
    Saves the generated Plotly figure as an image in the same directory.

    Args:
        fig (plotly.graph_objects.Figure): The Plotly figure to be saved.
        filename (str): The name of the image file.

    Returns:
        None
    """
    file_path = f"./images/{filename}.png"
    fig.write_image(file_path)
    file_path_html = f"./images/{filename}.html"
    fig.write_html(file_path_html)
    print(f"Plotly figure saved as HTML: {file_path_html}")
    print(f"Plotly figure saved as image: {file_path}")


def plot_impedance_vs_frequency(
    freq_df,
):

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Impedance vs Frequency",
            "Phase Angle vs Frequency",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=freq_df["Frequency"],
            y=freq_df["Impedance"],
            mode="markers",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=freq_df["Frequency"],
            y=freq_df["Phase Angle"],
            mode="markers",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Impedance (Ohms)", type="log", row=1, col=1)

    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=2, col=1)
    fig.update_yaxes(
        title_text="Phase Angle (degrees)",
        range=[-90, 0],
        tickmode="linear",
        tick0=0,
        dtick=10,
        row=2,
        col=1,
    )
    
    # save_plotly_figure_as_image_and_html(
    #     fig,
    #     f"impedance_vs_frequency_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
    # )
    fig.show()


if __name__ == "__main__":
    main()
