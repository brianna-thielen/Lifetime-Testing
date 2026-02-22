# To run lifetime testing: python run_lifetime_testing.py
# To change interpreter: CTRL+SHIFT+P, Python: Select Interpreter, Python 3.12.4 ('base')

# To pause lifetime testing, click ctrl-C (this will pause testing and stop stimulation)
# OR, change the setting CONTINUOUS_TESTING below to allow stim to continue after ctrl-c, then use pause_lifetime_testing.py to stop stim and record timestamps

# Various json files control all testing parameters and sample information:

# Each group of samples should have an associated json file under test_information/samples
#   - start_date: the day and time that testing began (format "YYYY-MM-DD HH:MM" in 24 hour format)
#   - end_date: the day and time that testing ended (format "YYYY-MM-DD HH:MM" in 24 hour format)
#       note: this is only required if you want to keep the data in the directory. Alternately, you
#           can remove the sample group json under test_information/samples and the data folder under
#           data to stop testing
#   - flagged_dates: dictionary, initially empty, then add any dates (same format) with any important timestamps (e.g. "power loss", or "saline replaced")
#       note, if there are multiple dates for one flag (e.g. saline was replaced twice), enter those as a list under a single key (e.g. "replaced PBS": ["2025-1-1 8:00", "2025-2-1 16:35"])
#   - samples: dictionary of sample names, each containing: (note: a "sample" is a single electrode, IDE, or humidity sensor)
#       For parts connected to the Intan
#       - intan_channel (format "a-001")
#       - geom_surf_area: geometric surface area in mm^2
#       - pulse_amplitude: continuous stim amplitude in uA
#       - pulse_width: continuous stim pulse width in us
#       - pulse_interphase: continuous stim interphase delay in us
#       - pulse_frequency: continuous stim frequency in Hz
#       - initial_i_max: starting point for CIC measurement (should not exceed stim amplitude - this will update with each test iteration)
#       - temp_sensor_id: ID for which temperature sensor to use (which phidget port it's plugged into)
#       For parts connected to the LCR meter
#       - mux_channels: list of mux bus and channels to measure between (format "[bus, channel]", e.g. [[1, 10], [1, 30]])
#           for IDEs: [1, 10] and [1, 30] would be the two contacts of a single IDE
#           for EIS of electrodes: [1, 10] would be the working electrode and [1, 30] would be the counter electrode
#       - temp_sensor_id: ID for which temperature sensor to use (which phidget port it's plugged into)
#       For parts connected to the Arduino
#       - temp_sensor_id: -1 (tells the code to use the humidity sensor's internal temp sensor)
#   - broken_devices: list of broken sample names, testing will be skipped on these samples
#   - flags: dictionary of pass/fail criteria for measurements ("Z" (ohms), "Z change" (decimal), "CIC" (uC/cm^2), "CIC change" (decimal), "RH" (%) accepted)
#   - test_info: dictionary containing:
#       - tests: list of tests to be run (must be listed in tests.json, details below)
#       - temp_sensor_type: type of temperature sensor (must be listed in equipment.json, details below)
#       - thermocouple_type: type of thermocouple used (J, K, etc)
#       - temp_sensor_offset: offset from temperature reading (e.g. if the thermocouple reads 65 when the saline temperature is 60, enter 5)
#       - cadence_hrs: how many hours to wait between tests
#       - last_test: date and time of last test (format "MM/DD/YYYY HH:MM" in 24 hour format), start value anytime prior to start_date (this will update automatically)
#       For parts connected via the arduino:
#       - arduino_port: usb port the arduino is connected via (e.g. COM4)
#       - arduino_baudrate: baudrate set in the arduino code (9600 default)
#   - slack_updates: dictionary containing:
#       - cadence_months: how many accelerated months to wait between slack updates (recommend starting at low value (~0.5-2) then increasing to 12 once stable)
#           a value of -1 means no testing updates will be sent (crash notifications will still be sent)
#       - last_update_months: start at 0 (will update automatically after sending updates)
#   - github_upload: true if you want automatic data upload to the public github, false if you do not

# tests.json contains all testing information
#   this should not be edited unless a new type of test is added
#   note: tests with Intan take a long time (lots of delays are added because the equipment is slow to respond and buffers need to be cleared often)
#   - EIS-LCR-3: EIS measured with the LCR meter from 10 to 10k Hz at 25 mV with 3 points per decade
#   - EIS-LCR-5: EIS measured with the LCR meter from 10 to 10k Hz at 25 mV with 5 points per decade
#   - EIS-LCR-10: EIS measured with the LCR meter from 10 to 10k Hz at 25 mV with 10 points per decade
#   - EIS-Intan-1: EIS measured with the Intan at 1k Hz with 30, 3, 0.3 nA currents (they measure all 3, use an algorithm to select best result)
#   - EIS-Intan-3: EIS measured with the Intan from 30 to 5060 Hz with 30, 3, 0.3 nA currents with 3 points per decade (Intan will not measure below 30 or above 5060 Hz)
#   - EIS-Intan-5: EIS measured with the Intan from 30 to 5060 Hz with 30, 3, 0.3 nA currents with 5 points per decade
#   - EIS-Intan-10: EIS measured with the Intan from 30 to 5060 Hz with 30, 3, 0.3 nA currents with 10 points per decade
#   - VT-Intan: VT measured with the intan with 1000 us pulse width, 500 us interphase delay, and 10 pulses in 1 second (quicker measurement not possible with Intan's recording resolution)
#   - RH-LCR: RH via analog humidity sensor measured with the LCR meter at 1k Hz and 1 V
#   - RH-Arduino: RH via I2C sensor measured with the Arduino

# equipment.json contains all equipment information
#   this should not be edited unless new equipment is added or existing equipment is reconfigured
#   - LCR: Rohde and Schwarz LCX100 LCR meter
#   - MUX: Keithley 7002 Switch System
#   - Intan: Intan RHS 128 channel Stim/Record System
#   - phidget: Phidget 4-input temperature sensor: https://www.phidgets.com/?prodid=1222&pcid=87
#   - rh-temp: Amphenol Advanced Sensors humidity/temperature sensor: https://www.digikey.com/en/products/detail/amphenol-advanced-sensors-telaire-/CC2D25-SIP/4732676
#       note: it's not listed here as it doesn't have a software connection, but here's the link for the analog rh sensor: https://www.digikey.com/en/products/detail/amphenol-advanced-sensors-telaire/hs30p/4780893
#   - Slack: instructions for replacing the webhook are in the README
#   - Github: please don't change this path - this is referenced in a published paper
#       note: if you don't want data pushed to github, you can override publishing in the sample information json


import time
import datetime
import os
import pandas as pd
import statistics
import numpy as np
import requests
import traceback
import json
import serial
import subprocess
from pathlib import Path

from equipment.keithley_mux import KeithleyMUX as kmux
from equipment.rs_lcx100 import LCX100 as lcx
from equipment.intan_rhs import IntanRHS as intan

from support_functions.support_functions import measure_temperature, record_timestamp, record_impedance_data_to_summary, record_rh_data_to_summary
from support_functions.plotting_functions import plot_cic, plot_z, plot_rh
from pause_lifetime_testing import pause_testing

SAMPLE_INFORMATION_PATH = './test_information/samples'
EQUIPMENT_INFORMATION_PATH = './test_information/equipment.json'
TEST_INFORMATION_PATH = './test_information/tests.json'
DATA_PATH = './data'
PLOT_PATH = './data/Plots'
IGNORE_PATH = Path('./.gitignore')

# Set CONTINUOUS_TESTING to true to continue stim after stopping code (with ctrl-C, this only stops testing, but continues stim)
# If you do this, you need to use pause_lifetime_testing.py to stop testing (this logs timestamps and stops stim)
CONTINUOUS_TESTING = False

# Import test, equipment, and plot information
with open(TEST_INFORMATION_PATH, 'r') as f:
    TEST_INFO = json.load(f)

with open(EQUIPMENT_INFORMATION_PATH, 'r') as f:
    EQUIPMENT_INFO = json.load(f)

# Import slack webhook
# This is saved in a file called "slack.json" in the "test_information" folder, under the key "webhook"
# This is included in the gitignore so that slack doesn't invalidate the webhook
with open('./test_information/slack.json') as f:
    SLACK_WEBHOOK = json.load(f)
    SLACK_WEBHOOK = SLACK_WEBHOOK["webhook"]

# Import github folder
# This is saved in a file called "github.json" in the "test_information" folder, under the key "path"
# This is included in the gitignore for privacy
with open('./test_information/github.json') as f:
    GITHUB_FOLDER = json.load(f)
    GITHUB_FOLDER = GITHUB_FOLDER["path"]

def main():
    # Starts automated lifetime testing

    # Initialize the Intan, setup stim, and start
    rhx, sample_frequency = initialize_intan()
    rhx.set_display()
    setup_all_stim_intan(rhx, True) #True triggers setting stim from json values (False disables stim)
    rhx.start_board()
    print('Starting stim.')

    # Setup data folders and gitignore
    setup_folders_and_gitignore()

    # Setup counter for github failures
    consecutive_failures = 0

    # Start testing loop
    run_test = True
    try:
        while run_test:
            current_datetime = datetime.datetime.now()

            # Check if it's time to run tests for any group, and save intan frequencies for those groups
            intan_groups, lcr_groups, arduino_groups, intan_eis_frequencies = check_for_due_tests(current_datetime)

            # If tests are due, print timestamp
            if len(intan_groups + lcr_groups + arduino_groups) > 0:
                now = datetime.datetime.now()
                now = now.strftime("%m/%d %H:%M:%S")
                print(f'Starting testing at {now}')

                # Also update folders and gitignore
                setup_folders_and_gitignore()

            # Run Intan tests
            if len(intan_groups) > 0:
                # Stop the intan
                rhx.stop_board()
                print('Stopping stim for Intan measurements.')

                # Perform measurements and save data
                perform_intan_measurements(rhx, intan_eis_frequencies, intan_groups, sample_frequency)

                # Restart Intan
                setup_all_stim_intan(rhx, True) #True triggers setting stim from json values (False disables stim)
                rhx.start_board()
                print('Resuming stim.')
                time.sleep(0.12)

                # Record restart time in data summaries
                record_timestamp(True, intan_groups, SAMPLE_INFORMATION_PATH, EQUIPMENT_INFO, DATA_PATH)

            # Then, run LCR tests
            if len(lcr_groups) > 0:
                now = datetime.datetime.now()
                now = now.strftime("%m/%d %H:%M:%S")
                print(f'Starting LCR measurements at {now}.')

                # Perform measurements
                perform_lcr_measurements(lcr_groups)

            # Then, arduino measurements
            if len(arduino_groups) > 0:
                now = datetime.datetime.now()
                now = now.strftime("%m/%d %H:%M:%S")
                print(f'Starting Arduino measurements at {now}.')

                # Read Arduino
                perform_arduino_measurements(arduino_groups)

            # Update last measurement in tested groups
            for group in intan_groups + lcr_groups + arduino_groups:
                with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
                    group_info = json.load(f)

                group_info["test_info"]["last_test"] = f"{current_datetime.year}-{current_datetime.month}-{current_datetime.day} {current_datetime.hour:02}:{current_datetime.minute:02}"

                with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'w') as f:
                    json.dump(group_info, f, indent=4)

            # Print that testing is done
            # If any tests were done, print timestamp
            if len(intan_groups + lcr_groups + arduino_groups) > 0:
                now = datetime.datetime.now()
                now = now.strftime("%m/%d %H:%M:%S")
                print(f'Finished testing at {now}')

            # Process data to generate plots and flag any issues
            if len(intan_groups + lcr_groups + arduino_groups) > 0:
                process_all_data()
                
                # Push data to github
                consecutive_failures = git_commit_and_push(GITHUB_FOLDER, consecutive_failures)

            # Write a heartbeat
            write_heartbeat()

            # Wait a minute before the next loop iteration
            time.sleep(60)

    except KeyboardInterrupt:
        # If CONTINUOUS_TESTING is true, continue stimulation but stop code
        if CONTINUOUS_TESTING:
            print("Automated test stopped, stimulation remains on.")
            notify_slack(SLACK_WEBHOOK, f"Automated test stopped at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. Stimulation remains on.")

        # If CONTINUOUS_TESTING is false, pause stim and record timestamps
        else:
            now = datetime.datetime.now()
            now = now.strftime("%m/%d %H:%M:%S")

            print(f"Automated test stopped at {now}, stopping stimulation.")
            pause_testing(rhx)
            notify_slack(SLACK_WEBHOOK, f"Automated test stopped at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. Stimulation turned off.")

def initialize_intan():
    print('Connecting to Intan.')
    # Connect to RHX software via TCP
    rhx = intan()

    # Query sample rate from RHX software.
    sample_frequency = rhx.find_sample_frequency()

    # Clear data output and disable all TCP channels
    rhx.reset()

    # Connect to RHX software via TCP
    rhx.connect_to_waveform_server()

    return rhx, sample_frequency

def initialize_lcr_mux():
    print('Connecting to LCR and mux.')
    # Connect equipment
    lcx100 = lcx(EQUIPMENT_INFO["LCR"]["resource_address"])
    mux = kmux(EQUIPMENT_INFO["MUX"]["resource_address"])

    # Set measurement settings and initialize LCR
    lcx100.set_aperture(EQUIPMENT_INFO["LCR"]["measure_interval"])
    lcx100.set_visa_timeout(EQUIPMENT_INFO["LCR"]["visa_timeout"])
    lcx100.reset_and_initialize()
    lcx100.set_measurement_type(EQUIPMENT_INFO["LCR"]["measure_type"])
    lcx100.set_measurement_range(EQUIPMENT_INFO["LCR"]["measure_range"])

    # Set low voltage and 1k frequency to start
    lcx100.set_voltage(0.025)
    lcx100.set_frequency(1000)

    # Initialize mux
    mux.reset_and_initialize()

    # Open all channels
    for bus in range(1, 9):
        mux.open_channels(bus, range(1, 41))

    return lcx100, mux

def setup_all_stim_intan(rhx, stim_on):
    # When stim_on is True, all channels are set to values under test_information/samples
    # When stim_on is False, all channels are set to zero and disabled
    now = datetime.datetime.now()
    now = now.strftime("%m/%d %H:%M:%S")
    if stim_on:
        print(f"Setting up stim at {now} (this takes ~5-10 seconds per channel).")
    else:
        print(f"Disabling stim at {now} (this takes ~5-10 seconds per channel).")


    for group in os.listdir(DATA_PATH):
        if "Plots" in group or "temp" in group or "Archive" in group:
            continue

        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Ignore any groups where testing has concluded
        if group_info["end_date"] is not None:
            continue

        # Check if the group uses intan
        first_sample, first_params = next(iter(group_info["samples"].items()))
        if "pulse_amplitude" not in first_params:
            continue

        # Check for any broken samples
        broken_samples = group_info["broken_devices"]

        # Loop through each sample in the group
        for sample in group_info["samples"]:
            sample_info = group_info["samples"][sample]

            # If stim_on is False, set zero stim
            if not stim_on:
                rhx.set_stim_parameters(sample_info["intan_channel"], 0, 0, 0, 0, sample)
            # If it's broken, set zero stim
            elif sample in broken_samples:
                rhx.set_stim_parameters(sample_info["intan_channel"], 0, 0, 0, 0, sample)

            # Otherwise, set the defined stim
            else:
                rhx.set_stim_parameters(
                    sample_info["intan_channel"],
                    sample_info["pulse_amplitude"],
                    sample_info["pulse_width"], 
                    sample_info["pulse_interphase"], 
                    sample_info["pulse_frequency"], 
                    sample
                )

def check_for_due_tests(current_datetime):
    intan_groups = []
    lcr_groups = []
    arduino_groups = []
    intan_eis_frequencies = set()
    # Intan tests impedance for all electrodes at the same time, so this information gets 
    # collected here rather than repeating EIS multiple times when iterating through each group

    # Loop through each group to see if testing is due
    for group in os.listdir(DATA_PATH):
        # Ignore temp and plot folders
        if "Plots" in group or "temp" in group or "Archive" in group:
            continue

        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Ignore any groups where testing has concluded
        if group_info["end_date"] is not None:
            continue

        # Pull out the datetime of the last test and calculate the next test
        last_test = group_info["test_info"]["last_test"]
        last_test = datetime.datetime.strptime(last_test, "%Y-%m-%d %H:%M")

        next_test = last_test + datetime.timedelta(hours=group_info["test_info"]["cadence_hrs"])

        # If current_datetime is after next_test, add group to the relevant testing list
        if current_datetime > next_test:
            # Loop through each test to find the relevant testing group
            for test in group_info["test_info"]["tests"]:
                if "Intan" in test:
                    intan_groups.append(group)
                elif "LCR" in test:
                    lcr_groups.append(group)
                elif "Arduino" in test:
                    arduino_groups.append(group)

                # If the group includes an Intan EIS test, add frequencies to the list
                if "EIS-Intan" in test:
                    intan_eis_frequencies.update(TEST_INFO[test]["eis_frequencies"])

    # Convert intan_eis_frequencies from set to list
    intan_eis_frequencies = list(intan_eis_frequencies)
    intan_eis_frequencies.sort()

    # Remove duplicates from group lists
    intan_groups = list(set(intan_groups))
    lcr_groups = list(set(lcr_groups))
    arduino_groups = list(set(arduino_groups))

    return intan_groups, lcr_groups, arduino_groups, intan_eis_frequencies

def perform_intan_measurements(rhx, intan_eis_frequencies, intan_groups, sample_frequency):
    # Measure impedance (and temperature)
    measure_intan_impedance(rhx, intan_groups, intan_eis_frequencies)

    # Measure CIC
    measure_intan_vt(rhx, intan_groups, sample_frequency)

def measure_intan_impedance(rhx, groups, frequencies):
    # Take timestamp for measurement
    measurement_time = datetime.datetime.now()

    # Loop through each frequency and measure
    now = datetime.datetime.now()
    now = now.strftime("%m/%d %H:%M:%S")
    print(f'Measuring impedances at {now} (this takes ~2-3 minutes).')
    freq_actual = []
    filenames = []
    for freq in frequencies:
        if freq < 30 or freq > 5060: # Intan won't test outside this range
            continue

        # Measure impedance and store to temp folder
        directory = os.getcwd()
        file, freq_a = rhx.measure_impedance(f"{directory}/data/temp/", freq)

        # Save filename and actual frequency
        freq_actual.append(freq_a)
        filenames.append(file)

    # Loop through each group to organize data
    for group in groups:
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # If EIS testing is not included, skip the group
        if not any("EIS-Intan" in test for test in group_info["test_info"]["tests"]):
            continue

        # Measure temperature, looping through each sensor for the given group
        sample_ids = list(group_info["samples"].keys())
        temp_sensor_ids = group_info["temp_sensors"].keys()
        temperature_i = []
        for sensor_id in temp_sensor_ids:
            temperature_i.append(measure_temperature(sample_ids, sensor_id, group_info, EQUIPMENT_INFO))
        temperature_i = statistics.mean(temperature_i)

        # Pull group sample and channel info
        channels_g = [sample["intan_channel"] for sample in group_info["samples"].values()]

        # Initiate impedance and phase lists
        impedance_g = []
        phase_g = []

        # Loop through each sample
        for channel_i in channels_g:
            channel_i_cap = channel_i.upper()

            # Loop through each file
            for f, file in enumerate(filenames):
                # Find frequency
                freq_a = freq_actual[f]

                # Import saved impedance data
                saved_impedances = pd.read_csv(f"{directory}/data/temp/{file}.csv")

                # Pull only data corresponding to current group from the file
                sample_i = saved_impedances.loc[saved_impedances['Channel Number'] == channel_i_cap].iloc[0, 1]
                sample_i_cap = sample_i.upper()
                
                impedance_i = saved_impedances.loc[saved_impedances['Channel Number'] == channel_i_cap].iloc[0, 4]
                phase_i = saved_impedances.loc[saved_impedances['Channel Number'] == channel_i_cap].iloc[0, 5]

                # Update groups
                impedance_g.append(impedance_i)
                phase_g.append(phase_i)

                if round(freq_a) == 1000:
                    impedance_i_1k = impedance_i
                    phase_i_1k = phase_i

                    # If the sample is broken, change values to NaN
                    if sample_i in group_info["broken_devices"]:
                        impedance_i_1k = float("NaN")
                        phase_i_1k = float("NaN")

                    # Save 1k impedance magnitude and phase to sample_data_summary
                    record_impedance_data_to_summary(group, sample_i_cap, measurement_time, impedance_i_1k, phase_i_1k, temperature_i, DATA_PATH, group_info)

            # If sample is not broken, save EIS data separately by channel
            if sample_i_cap not in group_info["broken_devices"]:
                freq_data = {
                    "Frequency": freq_actual,
                    "Impedance": impedance_i,
                    "Phase Angle": phase_i,
                    "Temperature (Dry Bath)": [temperature_i] * len(freq_actual),
                }
                freq_data_df = pd.DataFrame(freq_data)

                file_path = f"{DATA_PATH}/{group}/raw-data/intan-eis/EIS_{sample_i}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                
                freq_data_df.to_csv(file_path, index=False)

    # When done, delete temp files
    for file in filenames:
        os.remove(f"{directory}/data/temp/{file}.csv")

def measure_intan_vt(rhx, groups, sample_frequency):
    # Counter to pause and reset every 5 channels - otherwise intan gets overwhelmed
    counter = 0

    now = datetime.datetime.now()
    now = now.strftime("%m/%d %H:%M:%S")
    print(f'Starting VT tests at {now} (this takes ~2 minutes per sample).')

    # Loop through each group for testing
    for group in groups:
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # If VT testing is not included, skip the group
        if "VT-Intan" not in group_info["test_info"]["tests"]:
            continue

        # Get relevant information from group_info
        sample_list = list(group_info["samples"].keys())
        channel_list = [sample["intan_channel"] for sample in group_info["samples"].values()]
        gsas = [sample["geom_surf_area"] for sample in group_info["samples"].values()]
        vt_start = [sample["initial_i_max"] for sample in group_info["samples"].values()]
        amplitudes = [sample["pulse_amplitude"] for sample in group_info["samples"].values()]

        # Get relevant information from test_info
        vt_pulse_width = TEST_INFO["VT-Intan"]["vt_pulse_width"]
        vt_interphase = TEST_INFO["VT-Intan"]["vt_interphase"]
        vt_frequency = TEST_INFO["VT-Intan"]["vt_frequency"]

        rhx.reset()

        # Disable all channels
        setup_all_stim_intan(rhx, False)

        # Loop through each channel and run VT test
        for i in range(len(channel_list)):
            channel_i = channel_list[i]
            sample_i = sample_list[i]
            gsa_i = gsas[i]

            # If the sample is broken, skip it
            if sample_i in group_info["broken_devices"]:
                continue

            # If vt_start is zero, the channel is broken or not listed for testing
            if vt_start[i] == 0:
                continue
            # If vt_start is small, reset it to 100 uA
            elif vt_start[i] < 100:
                vt_start[i] = 100
            # Intan cannot exceed 2500 uA - scale down if needed
            elif vt_start[i] > 2500:
                vt_start[i] = 2500
            else:
                vt_start[i] = int(vt_start[i])

            currents_tested = []
            eps_calculated = []

            # Open data summary
            df = pd.read_csv(f"{DATA_PATH}/{group}/{sample_i}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Start testing
            # Measure at 4 points, starting at vt_start[i] and scaling down
            for current_i in [vt_start[i], int(0.9*vt_start[i]), int(0.8*vt_start[i]), int(0.7*vt_start[i])]:
                # Enable current channel and set current
                rhx.set_stim_parameters(channel_i, current_i, vt_pulse_width, vt_interphase, vt_frequency, sample_i)
                rhx.enable_data_output(channel_i)

                # Stim for 1 second
                rhx.start_board()
                time.sleep(1)
                rhx.stop_board()
                time.sleep(0.5)
                
                # Read data from board
                buffer_size = EQUIPMENT_INFO["Intan"]["waveform_buffer_per_second_per_channel"]
                time_seconds, voltage_microvolts, repeat_test, failure = rhx.read_data(buffer_size, sample_frequency, False)

                # If the magic number is incorrect, repeat stim and try again
                if repeat_test:
                    # Wait 30 seconds to reset
                    time.sleep(30)

                    # Stim for 1 second
                    rhx.start_board()
                    time.sleep(1)
                    rhx.stop_board()
                    time.sleep(0.5)
                    
                    # Read data from board
                    buffer_size = EQUIPMENT_INFO["Intan"]["waveform_buffer_per_second_per_channel"]
                    time_seconds, voltage_microvolts, repeat_test, failure = rhx.read_data(buffer_size, sample_frequency, True)

                    if failure:
                        print('Data read failed, skipping data point - no action needed.')
            
                # If the test was successful, create dataframe to calculate ep
                if not failure:
                    vt_data = {
                        'Time (s)': time_seconds,
                        'Voltage (uV)': voltage_microvolts
                    }
                    vt_data = pd.DataFrame(vt_data)

                    # Calculate and store ep
                    ep = calculate_ep(vt_data, vt_pulse_width, vt_interphase)
                    currents_tested.append(current_i)
                    eps_calculated.append(ep)

            # Disable current channel
            rhx.set_stim_parameters(channel_i, 0, 0, 0, 0, sample_i)
            rhx.disable_stim(channel_i)
            rhx.disable_data_output(channel_i)

            # If there's enough data, calculate max current and CIC
            if len(eps_calculated) > 2:
                coeff = np.polyfit(eps_calculated, currents_tested, 1)
                bestfit = np.poly1d(coeff)
                max_current = bestfit(0.6)

                cic = max_current * (vt_pulse_width / 1000000) / (gsa_i / 100) # uA * s / cm^2

                # If CIC is unreasonably high, set to NaN
                if cic > 50000:
                    max_current = 1
                    cic = float('NaN')
            
            else:
                max_current = 1
                cic = float('NaN')

            # Force poorly performing devices to 1 so they continue being tested
            if max_current <= 0:
                max_current = 1
                cic = 0
            
            # Save cic to sample_data_summary
            df.loc[len(df)-1, 'Charge Injection Capacity @ 1000 us (uC/cm^2)'] = cic
            df.to_csv(f"{DATA_PATH}/{group}/{sample_i}_data_summary.csv")

            # Save max current back to json
            # Scale back max current if needed - we don't want to do VT test above stim level (high stim can accelerate aging)
            stim_amplitude = amplitudes[i]
            if stim_amplitude == 0: # If it's a no stim sample, we still want to measure CIC, but below average stim value for group
                stim_amplitude = sum(amplitudes) / len(amplitudes)

            if max_current * 0.9 > stim_amplitude:
                max_current = stim_amplitude / 0.9

            group_info["samples"][sample_i]["initial_i_max"] = max_current*0.9

            # Save sample information and reopen new file
            with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'w') as f:
                json.dump(group_info, f, indent=4)

            with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
                group_info = json.load(f)

            # Iterate counter and flush buffer every 5 electrodes
            counter += 1
            if counter >= 5:
                counter = 0
                # Wait 30 seconds to reset
                print("Pausing to prevent data loss.")
                time.sleep(30)
                print('Resuming VT test.')

def calculate_ep(vt_data, pulse_width, interphase):
    vt_adjusted = vt_data

    # Find and subtract baseline
    baseline = statistics.mode(vt_adjusted["Voltage (uV)"])
    vt_adjusted["Voltage (uV)"] = vt_adjusted["Voltage (uV)"] - baseline

    # Find voltage peak
    voltage_at_max = max(vt_adjusted["Voltage (uV)"])
    time_at_max = vt_adjusted.loc[vt_adjusted["Voltage (uV)"] == voltage_at_max, "Time (s)"].iloc[0]

    # If peak is too close to the beginning or end, find next highest peak
    if time_at_max < 0.1 or time_at_max > 1.9:
        vt_adjusted = vt_adjusted[vt_adjusted["Time (s)"] < 1.89]
        vt_adjusted = vt_adjusted[vt_adjusted["Time (s)"] > 0.11]

        voltage_at_max = max(vt_adjusted["Voltage (uV)"])
        time_at_max = vt_adjusted.loc[vt_adjusted["Voltage (uV)"] == voltage_at_max, "Time (s)"].iloc[0]

    # Move backward and forward to find beginning of pulse and end of interphase
    time_list = vt_adjusted["Time (s)"].tolist()
    time_at_baseline = time_at_max - (pulse_width / 1000000)
    time_at_interphase = time_at_max + (interphase / 1000000)

    ind_at_baseline = len([t for t in time_list if t < time_at_baseline])
    ind_at_interphase = len([t for t in time_list if t < time_at_interphase])

    voltage_at_baseline = vt_adjusted["Voltage (uV)"].iloc[ind_at_baseline]
    voltage_at_interphase = vt_adjusted["Voltage (uV)"].iloc[ind_at_interphase]

    # Calculate Ep
    ep = float(voltage_at_interphase - voltage_at_baseline) / 10.0 # manual adjustment to match o-scope

    return ep

def perform_lcr_measurements(lcr_groups):
    # Initialize the LCR and mux
    lcx100, mux = initialize_lcr_mux()

    # Loop through each group
    for group in lcr_groups:
        # Pull sample information
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Check for any broken samples
        broken_samples = group_info["broken_devices"]

        # Load test info
        tests = group_info["test_info"]["tests"]

        # Measure temperature, looping through each sensor for the given group
        sample_ids = list(group_info["samples"].keys())
        temp_sensor_ids = group_info["temp_sensors"].keys()
        temperature = []
        for sensor_id in temp_sensor_ids:
            temperature.append(measure_temperature(sample_ids, sensor_id, group_info, EQUIPMENT_INFO))
        temperature = statistics.mean(temperature)

        # Loop through each LCR test
        for test in tests:
            # Get test information
            test_frequencies = TEST_INFO[test]["eis_frequencies"]
            test_voltage = TEST_INFO[test]["eis_amplitude"]

            # Loop through each sample in the group
            for sample in sample_ids:
                # If the sample is broken, skip it
                if sample in broken_samples:
                    continue

                # Pull relevant information from group info
                sample_info = group_info["samples"][sample]
                mux_channel_1 = sample_info["mux_channels"][0]
                mux_channel_2 = sample_info["mux_channels"][1]

                # Close mux channels for the current sample
                mux.close_channels(mux_channel_1[0], [mux_channel_1[1]])
                mux.close_channels(mux_channel_2[0], [mux_channel_2[1]])

                # Run LCR EIS and save data
                measure_lcr_impedance(sample, group, lcx100, test_frequencies, test_voltage, group_info, temperature)

                # Open mux channels
                mux.open_channels(mux_channel_1[0], [mux_channel_1[1]])
                mux.open_channels(mux_channel_2[0], [mux_channel_2[1]])

    # When measurements are done, set LCR back to 25 mV for safety, close connection
    lcx100.set_voltage(25.0 / 1000) # Set voltage back to 25 mV for safety
    lcx100.close()

    # For some reason bus 1/channel 1 wants to stay closed, so open it manually, then close the mux connection
    mux.open_channels(1, [1])
    mux.close()

def measure_lcr_impedance(sample, group, lcx100, test_frequencies, test_voltage, group_info, temperature):
    counter = 0
    impedance_temperature = []

    # Save timestamp
    measurement_time = datetime.datetime.now()

    # Set LCR voltage
    lcx100.set_voltage(test_voltage)

    # Loop through each test frequency
    for freq in test_frequencies:
        # Set frequency
        lcx100.set_frequency(freq)
        if counter == 0:
            time.sleep(EQUIPMENT_INFO["LCR"]["delay_first"])
            counter = counter + 1
        else:
            time.sleep(EQUIPMENT_INFO["LCR"]["delay"])

        # Measure impedance and temperature
        impedance, phase = lcx100.get_impedance()

        # Save data to array
        impedance_temperature.append(
            {
                "Frequency": freq,
                "Impedance": float(impedance),
                "Phase Angle": float(phase),
                "Temperature (Dry Bath)": float(temperature),
            }
        )

        # If freq = 1000, save data to sample_data_summary
        if freq == 1000:
            for test in group_info["test_info"]["tests"]:
                # If it's an RH test, convert to humidity first
                if "RH-LCR" in test:
                    # Impedance -> RH conversion is from the datasheet, corrected for temperature
                    impedance_kohms = float(impedance) / 1000
                    rh = (impedance_kohms / (1.92e22 * temperature ** -7.57)) ** (1 / (-6.99 + 0.052 * temperature - 0.000225 * temperature ** 2))

                    record_rh_data_to_summary(group, sample, measurement_time, rh, temperature, DATA_PATH, group_info)

                # Otherwise, save raw Z data
                else:
                    record_impedance_data_to_summary(group, sample, measurement_time, impedance, phase, temperature, DATA_PATH, group_info)

    # Save EIS data
    impedance_temperature = pd.DataFrame(impedance_temperature)

    filename = f"EIS_{sample}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    file_path = f"./data/{group}/raw-data/{filename}.csv"
    impedance_temperature.to_csv(file_path, index=False)

def perform_arduino_measurements(arduino_groups):
    # Take timestamp for measurement
    measurement_time = datetime.datetime.now()

    # Loop through each group
    for group in arduino_groups:
        # Load group info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # If RH testing is not included, skip the group
        if "RH-Arduino" not in group_info["test_info"]["tests"]:
            continue

        # Measure temperature, looping through each sensor for the given group
        sample_ids = list(group_info["samples"].keys())
        temp_sensor_ids = group_info["temp_sensors"].keys()
        temperature_i = []
        for sensor_id in temp_sensor_ids:
            temperature_i.append(measure_temperature(sample_ids, sensor_id, group_info, EQUIPMENT_INFO))
        temperature_i = statistics.mean(temperature_i)

        # Setup serial connection
        port = group_info["test_info"]["arduino_port"]
        baudrate = group_info["test_info"]["arduino_baudrate"]
        timeout_sec = 120 # Arduino prints output every minute, allow 2 minutes to check for data
        
        ser = serial.Serial(port, baudrate, timeout=timeout_sec)

        # Wait for data
        try:
            line = read_valid_line(ser)
        finally:
            ser.close()

        # Sort data
        for sample_i in group_info["samples"]:
            index_i = line.index(sample_i) + len(sample_i) + 2

            data = line[index_i:(index_i+15)]
            rh_index = data.index("RH=") + 3
            temp_index = data.index("T=") + 2

            rh = float(data[rh_index:(temp_index-3)])
            # temperature = float(data[temp_index:])

            # Save data to summary
            record_rh_data_to_summary(group, sample_i, measurement_time, rh, temperature_i, DATA_PATH, group_info)

def read_valid_line(ser):
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line.startswith("mins"):
            return line
        # otherwise keep looping

def process_all_data():
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
        
        # Look at all flags for the current group
        flags = group_info["flags"]
        for flag, value in flags.items():
            if flag == "CIC":
                flagged_samples_flag = [sample for s, sample in enumerate(group_info["samples"]) if cic_last[s] < value]
            elif flag == "CIC change":
                flagged_samples_flag = [sample for s, sample in enumerate(group_info["samples"]) if cic_norm_last[s] < value]
            elif flag == "Z":
                flagged_samples_flag = [sample for s, sample in enumerate(group_info["samples"]) if z_last[s] < value]
            elif flag == "Z change":
                flagged_samples_flag = [sample for s, sample in enumerate(group_info["samples"]) if z_norm_last[s] < value]
            elif flag == "RH":
                flagged_samples_flag = [sample for s, sample in enumerate(group_info["samples"]) if rh_last[s] < value]
            
            # Add to flagged_samples
            if flagged_samples_group == "":
                flagged_samples_group = f"{flagged_samples_flag} outside of {flag} range"
            else:
                flagged_samples_group = f"{flagged_samples_group}, {flagged_samples_flag} outside of {flag} range"

        # Create group summary
        if flagged_samples_group == "":
            summary = f"{group} testing at {round(accel_days/365.25, 1)} accelerated years, all parts within expected range."
        else:
            summary = f"{group} testing at {round(accel_days/365.25, 1)} accelerated years, {flagged_samples_group}."

        # Save summary, if it's time for an update
        last_update = group_info["slack_updates"]["last_update_accel_months"]
        update_cadence = group_info["slack_updates"]["cadence_accel_months"]

        start_date = group_info["start_date"]
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M")

        current_test = round(accel_days/365.25*12, 1)

        time_elapsed = current_test - float(last_update)

        if time_elapsed > update_cadence and update_cadence > 0:
            notify_slack(SLACK_WEBHOOK, summary)
            print(summary)

            # Update group info
            group_info["slack_updates"]["last_update_accel_months"] = current_test

            with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'w') as f:
                json.dump(group_info, f, indent=4)

def notify_slack(webhook, message):
    payload = {'text': message}
    requests.post(webhook, json=payload)

def setup_folders_and_gitignore():
    # Check that all data folders exist and add necessary folders to gitignore
    ignore_lines = [
        "# Auto-generated .gitignore",
        "# Do not edit manually\n"
    ]

    # Add github information
    ignore_lines.append("test_information/github.json")

    # Add slack information
    ignore_lines.append("test_information/slack.json")

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
            print(ignore_lines)
    
    IGNORE_PATH.write_text("\n".join(ignore_lines).rstrip() + "\n")

def git_commit_and_push(repo_path, consecutive_failures):
    print(f"Starting Github commit and push")
    def run(cmd):
        subprocess.run(cmd, cwd=repo_path, check=True, stdout=subprocess.DEVNULL)

    # add a try statement so code continues running in case of git failures
    try:
        # Stage everything
        run(["git", "add", "."])

        # Commit (will fail if nothing changed)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            run(["git", "commit", "-m", f"Auto data update {timestamp}"])
            print(f"Committing data update to github as: Auto data update {timestamp}")
        except subprocess.CalledProcessError:
            # Happens when there is nothing to commit
            return

        # Push
        run(["git", "push"])
        print(f"Github push complete")

    except subprocess.CalledProcessError as e:
        print("Git operation failed: ", e)
        consecutive_failures += 1

        if consecutive_failures < 3:
            print("Likely network error, no action needed")
        else:
            print(f"{consecutive_failures} consecutive failures, try restarting system")
            notify_slack(SLACK_WEBHOOK, f"Git operation failed: {e}")
            notify_slack(SLACK_WEBHOOK, f"{consecutive_failures} consecutive failures, try restarting system")

    return consecutive_failures

def write_heartbeat():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("heartbeat.txt", "w") as f:
        f.write(f"Last heartbeat: {now}\n")

if __name__ == '__main__':
    # Declare buffer size for reading from TCP command socket
    # This is the maximum number of bytes expected for 1 read. 1024 is plenty for a single text command.
    # Increase if many return commands are expected.
    COMMAND_BUFFER_SIZE = 1024
    try:
        main()
    except Exception as e:
        error_msg = f"Lifetime testing script crashed at {datetime.datetime.now()}:\n{traceback.format_exc()}"

        if "KeyboardInterrupt" not in error_msg:
            notify_slack(SLACK_WEBHOOK, error_msg)

        print(error_msg)        