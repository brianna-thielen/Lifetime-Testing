import os
import json
from pathlib import Path

from support_functions.plotting_functions import plot_cic, plot_z, plot_rh

SAMPLE_INFORMATION_PATH = './test_information/samples'
EQUIPMENT_INFORMATION_PATH = './test_information/equipment.json'
TEST_INFORMATION_PATH = './test_information/tests.json'
DATA_PATH = './data'
PLOT_PATH = './data/Plots'
IGNORE_PATH = Path('./.gitignore')


# Loop through each group
for group in os.listdir(DATA_PATH):
    flagged_samples_group = ""

    # Skip Plots and temp folders
    if "Plots" in group or "temp" in group or "Archive" in group:
        continue

    # Load group info
    with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
        group_info = json.load(f)

    # Loop through each test - those will define the plots that are generated
    for test in group_info["test_info"]["tests"]:
        # For VT, plot CIC vs time
        if "VT" in test:
            title = f"{group}: Charge Injection Capacity vs Time (1000 us pulse)"
            cic_last, cic_norm_last, accel_days = plot_cic(group, DATA_PATH, SAMPLE_INFORMATION_PATH, PLOT_PATH, title, False)

        # For EIS, plot Z vs time
        elif "EIS" in test:
            title = f"{group}: Impedance Magnitude vs Time"
            z_last, z_norm_last, accel_days = plot_z(group, DATA_PATH, SAMPLE_INFORMATION_PATH, PLOT_PATH, title, False)

        # For RH, plot RH vs time and Temp vs time:
        elif "RH" in test:
            title = f"{group}: Relative Humidity vs Time"
            rh_last, accel_days = plot_rh(group, DATA_PATH, SAMPLE_INFORMATION_PATH, PLOT_PATH, title, False)


