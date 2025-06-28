# To run lifetime testing: python run_lifetime_testing.py
# To change interpreter: CTRL+SHIFT+P, Python: Select Interpreter, Python 3.12.4 ('base')

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

from equipment.keithley_mux import KeithleyMUX as kmux
from equipment.rs_lcx100 import LCX100 as lcx
from equipment.intan_rhs import IntanRHS as intan
from equipment.phidget_4input_temperature import Phidget22TemperatureSensor as phidget

from support_functions.support_functions import measure_temperature, record_timestamp, record_impedance_data_to_summary, record_rh_data_to_summary
from support_functions.plotting_functions import plot_cic, plot_z, plot_rh

from data_processing.lcp_encapsulation_data_processing import process_encapsulation_soak_data
from data_processing.lcp_ide_data_processing import process_ide_soak_data
from data_processing.sirof_vs_pt_data_processing import process_coating_soak_data
from data_processing.lcp_pt_grids_data_processing import process_lcp_pt_grids_soak_data

SAMPLE_INFORMATION_PATH = './test_information/samples'
EQUIPMENT_INFORMATION_PATH = './test_information/equipment.json'
TEST_INFORMATION_PATH = './test_information/tests.json'
PLOT_INFORMATION_PATH = './test_information/special_plots.json'
DATA_PATH = './data'
PLOT_PATH = './data/Plots'

# Import test, equipment, and plot information
with open(TEST_INFORMATION_PATH, 'r') as f:
    TEST_INFO = json.load(f)

with open(EQUIPMENT_INFORMATION_PATH, 'r') as f:
    EQUIPMENT_INFO = json.load(f)

with open(PLOT_INFORMATION_PATH, 'r') as f:
    PLOT_INFO = json.load(f)

def main():
    """
    Starts automated lifetime testing

    Various json files control all testing parameters and sample information:

    - Each group of samples should have an associated json file under test_information/samples, containing:
        - start_date: the day and time that testing began (format "YYYY/MM/DD HH:MM" in 24 hour format)
        - flagged_dates: dictionary, initially empty, then add any dates (same format) with any important timestamps (e.g. "power loss", or "saline replaced")
            Note, if there are multiple dates for one flag (e.g. saline was replaced twice), enter those as a list under a single key (e.g. "saline replaced": ["2025-1-1 8:00", "2025-2-1 16:35"])
        - samples: dictionary of sample names, each containing:
            For parts connected to the Intan
            - intan_channel (format "a-001")
            - geom_surf_area: geometric surface area in mm^2
            - pulse_amplitude: continuous stim amplitude in uA
            - pulse_width: continuous stim pulse width in us
            - pulse_interphase: continuous stim interphase delay in us
            - pulse_frequency: continuous stim frequency in Hz
            - initial_i_max: starting point for CIC measurement (should not exceed stim amplitude)
            - temp_sensor_id: ID for which temperature sensor to use (which port it's plugged into)
            - temp_sensor_serial: serial number for phidget device (thermocouples only)
            For parts connected to the LCR meter
            - mux_channels: list of mux bus and channels to measure (format "[bus, channel]")
                For EIS tests, should contain two lists of channels to measure between (e.g. [[1, 10], [1, 30]])
                For crosstalk tests, should contain a single channel to measure from to all other channels (e.g. [1, 10]), and the counter electrode in solution must be entered as its own sample
            - temp_sensor_id: ID for which temperature sensor to use (which port it's plugged into)
            - temp_sensor_serial: serial number for phidget device (thermocouples only)
        - broken_devices: list of broken sample names, testing will be skipped on these samples
        - flags: dictionary of pass/fail criteria for measurements ("Z" (ohms), "Z change" (decimal), "CIC" (uC/cm^2), "CIC change" (decimal) "RH" (%) accepted)
        - test_info: dictionary containing:
            - tests: list of tests to be run (must be listed in tests.json)
            - temp_sensor_type: type of temperature sensor (must be listed in equipment.json)
            - thermocouple_type: type of thermocouple used (J, K, etc)
            - temp_sensor_offset: offset from temperature reading (e.g. if the thermocouple reads 65 when the saline temperature is 60, enter 5)
            - cadence_hrs: how many hours to wait between tests
            - last_test: date and time of last test (format "MM/DD/YYYY HH:MM" in 24 hour format), start value anytime prior to start_date
            For parts connected via the arduino:
            - arduino_port: usb port the arduino is connected via (e.g. COM4)
            - arduino_baudrate: baudrate set in the arduino code
        - slack_updates: dictionary containing:
            - cadence_months: how many accelerated months to wait between slack updates (recommend starting at low value (~0.5-2) then increasing to 12 once stable)
            - last_update_months: start at 0, will update automatically after sending updates
    
    - special_plots.json contains a list of extra plots
        by default, plots are generated within each group reflecting tests performed
        any groups which should be compared to each other should be included here
        this should be in the format of "Plot Title": ["group 1", "group 2", ...] (accepts up to 5 groups)
    
    - tests.json contains all testing information
        this should not be edited unless a new type of test is added

    - equipment.json contains all equipment information
        this should not be edited unless new equipment is added or existing equipment is reconfigured
    """

    # Initialize the Intan, setup stim, and start
    rhx, sample_frequency = initialize_intan()
    # setup_all_stim_intan(rhx, True) #True triggers setting stim from json values (False disables stim)
    rhx.start_board()
    print('Starting stim.')

    # Initialize the LCR and mux
    lcx100, mux = initialize_lcr_mux()

    # Start testing loop
    run_test = True
    try:
        while run_test:
            current_datetime = datetime.datetime.now()

            # Check if it's time to run tests for any group, and save intan frequencies for those groups
            intan_groups, lcr_groups, arduino_groups, intan_eis_frequencies = check_for_due_tests(current_datetime)

            # Run Intan tests
            if len(intan_groups) > 0:
                # Stop the intan
                rhx.stop_board()
                print('Stopping stim for Intan measurements.')

                # # Record stop time in data summary
                # record_timestamp(intan_stop, False, intan_groups)

                # Perform measurements and save data
                perform_intan_measurements(rhx, intan_eis_frequencies, intan_groups, sample_frequency)

                # Restart Intan
                setup_all_stim_intan(rhx, True) #True triggers setting stim from json values (False disables stim)
                rhx.start_board()
                print('Resuming stim.')
                time.sleep(0.12)

                # Record restart time in data summaries
                record_timestamp(True, intan_groups, group_info, EQUIPMENT_INFO, DATA_PATH)

                # # Sort data and save
                # sort_intan_data(impedance_temperature_cic, intan_groups, filename)

            # Then, run LCR tests
            if len(lcr_groups) > 0:
                print('Staring LCR measurements.')

                # Perform measurements
                perform_lcr_measurements(lcx100, mux, lcr_groups)

                # # Record timestamp
                # record_timestamp(measurement_time, False, lcr_groups)

            # Then, arduino measurements
            if len(arduino_groups) > 0:
                print('Starting Arduino measurements.')
                measurement_time = datetime.datetime.now()

                # Read Arduino
                perform_arduino_measurements()

            # Update last measurement in tested groups
            for group in intan_groups + lcr_groups + arduino_groups:
                with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
                    group_info = json.load(f)

                group_info["test_info"]["last_test"] = f"{current_datetime.year}-{current_datetime.month}-{current_datetime.day} {current_datetime.hour:02}:{current_datetime.minute:02}"

                with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'w') as f:
                    json.dump(group_info, f, indent=4)

            # Process data to generate plots and flag any issues
            if len(intan_groups + lcr_groups + arduino_groups) > 0:
                process_all_data()

            # Write a heartbeat
            write_heartbeat()

            # Wait a minute before the next loop iteration
            time.sleep(60)

    except KeyboardInterrupt:
        print("Automated test stopped, stimulation remains on.")
        notify_slack(EQUIPMENT_INFO["Slack"]["webhook"], f"Automated test stopped at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. Stimulation remains on.")

def initialize_intan():
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
    if stim_on:
        print("Setting up stim (this takes a few minutes).")
    else:
        print("Disabling stim (this takes a few minutes).")


    for group in os.listdir(DATA_PATH):
        if "Plots" in group or "temp" in group or "Archive" in group:
            continue

        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

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
    # Setup dataframe to save intan test data
    ztc_dict = {
        'Channel Number': [None], 
        'Channel Name': [None], 
        'Impedance Magnitude at 1000 Hz (ohms)': [None],
        'Impedance Phase at 1000 Hz (degrees)': [None],
        'Temperature (C)': [None],
        'Charge Injection Capacity @ 1000 us (uC/cm^2)': [None],
        'Geometric Surface Area (mm^2)': [None]
    }
    impedance_temperature_cic = pd.DataFrame(ztc_dict)

    # Measure impedance (and temperature)
    measure_intan_impedance(rhx, intan_groups, intan_eis_frequencies, impedance_temperature_cic)

    # Measure CIC
    measure_intan_vt(rhx, intan_groups, sample_frequency, impedance_temperature_cic)

def measure_intan_impedance(rhx, groups, frequencies, impedance_temperature_cic):
    # Take timestamp for measurement
    measurement_time = datetime.datetime.now()

    # Create a list of intan channels, samples of interest from groups
    channels = []
    samples = []
    # temperatures = [] # Will have equal length to groups, not channels/samples
    for group in groups:
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # If EIS testing is not included, skip the group
        if not any("EIS-Intan" in test for test in group_info["test_info"]["tests"]):
            continue

        channels_g = [sample["intan_channel"] for sample in group_info["samples"].values()]
        channels = channels + channels_g

        samples_g = list(group_info["samples"].keys())
        samples = samples + samples_g

        # # also measure temperature for each group and save it
        # temperature = measure_temperature(sample, group_info, EQUIPMENT_INFO)
        # temperatures.append(temperature)

    # Empty lists will be added to as tests are run
    impedances = []
    phases = []
    frequencies_updated = []

    # Loop through each frequency
    print('Measuring impedances...')
    for freq in frequencies:
        if freq < 30 or freq > 5060: # Intan won't test outside this range
            continue

        # Measure impedance and store to temp folder
        directory = os.getcwd()
        filename, freq = rhx.measure_impedance(f"{directory}/data/temp/", freq)

        # Import saved impedance data
        saved_impedances = pd.read_csv(f"{directory}/data/temp/{filename}.csv")

        # Delete the temp file
        os.remove(f"{directory}/data/temp/{filename}.csv")

        # Pull data from only the current channels under test
        for channel_i in channels:
            channel_i = channel_i.capitalize()
            impedance_i = saved_impedances.loc[saved_impedances['Channel Number'] == channel_i].iloc[0, 4]
            phase_i = saved_impedances.loc[saved_impedances['Channel Number'] == channel_i].iloc[0, 5]
        
            impedances.append(impedance_i)
            phases.append(phase_i)
            channels.append(channel_i)
            frequencies_updated.append(freq)

    # Once through all frequencies, separate EIS data by group and channel
    print(channels)
    for g, group in enumerate(groups):
        print(group)
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # # Find number of days since start
        # first_day = group_info["start_date"]
        # first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

        # real_days = measurement_time - first_day
        # real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days
        
        for i, channel_i in enumerate(channels):
            print(channel_i, group_info["samples"])
            # If the channel is not in the group, skip it
            if channel_i not in [sample["intan_channel"] for sample in group_info["samples"].values()]:
                print('continuing')
                continue

            channel_i_cap = channel_i.capitalize()
            sample_i = samples[i]

            # List frequencies and impedances corresponding to the current channel
            frequencies_i = [f for ch, f in zip(channels, frequencies_updated) if ch == channel_i_cap]
            impedance_i = [imp for ch, imp in zip(channels, impedances) if ch == channel_i_cap]
            impedance_i_1k = impedance_i[frequencies_i.index(1000)]

            phase_i = [ph for ch, ph in zip(channels, phases) if ch == channel_i_cap]
            phase_i_1k = phase_i[frequencies_i.index(1000)]

            # temperature_i = temperatures[g]

            # # Open data summary
            # df = pd.read_csv(f"{DATA_PATH}/{group}/{sample_i}_data_summary.csv")
            # df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Measure temperature for the given sample
            temperature_i = measure_temperature(sample_i, group_info, EQUIPMENT_INFO)

            # If the sample is broken, change values to NaN
            if sample_i in group_info["broken_devices"]:
                impedance_i_1k = float("NaN")
                phase_i_1k = float("NaN")
            
            # Save 1k impedance magnitude and phase to sample_data_summary
            record_impedance_data_to_summary(group, sample_i, measurement_time, impedance_i_1k, phase_i_1k, temperature_i, DATA_PATH, group_info)
            # new_row = pd.DataFrame({
            #     "Measurement Datetime": [measurement_time],
            #     "Pulsing On": [False],
            #     "Temperature (C)": [temperature_i],
            #     "Real Days": [real_days],
            #     "Impedance Magnitude at 1000 Hz (ohms)": [impedance_i_1k],
            #     "Impedance Phase at 1000 Hz (degrees)": [phase_i_1k],
            #     "Charge Injection Capacity @ 1000 us (uC/cm^2)": [float("NaN")]
            # })

            # # drop any columns that are all NA
            # new_row = new_row.dropna(axis=1, how="all")

            # if df.empty:
            #     new_df = new_row.copy()
            # elif new_row.empty:
            #     new_df = df.copy()
            # else:
            #     new_df = pd.concat([df, new_row], ignore_index=True)

            # new_df.to_csv(f"{DATA_PATH}/{group}/{sample_i}_data_summary.csv")

            # impedance_temperature_cic.loc[impedance_temperature_cic['Channel Number'] == channel_i, f'Impedance Magnitude at 1000 Hz (ohms)'] = impedance_i_1k
            # impedance_temperature_cic.loc[impedance_temperature_cic['Channel Number'] == channel_i, f'Impedance Phase at 1000 Hz (degrees)'] = phase_i_1k
            # impedance_temperature_cic.loc[impedance_temperature_cic['Channel Number'] == channel_i, 'Temperature (C)'] = temperature_i

            # If sample is not broken, save EIS data separately by channel
            if sample_i not in group_info["broken_devices"]:
                freq_data = {
                    "Frequency": frequencies_i,
                    "Impedance": impedance_i,
                    "Phase Angle": phase_i,
                    "Temperature (Dry Bath)": [temperature_i] * len(frequencies_i),
                }
                freq_data_df = pd.DataFrame(freq_data)

                for group, devices in groups.items():
                    if sample_i in devices:
                        file_path = f"{DATA_PATH}/{group}/intan-eis/EIS_{sample_i}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                
                freq_data_df.to_csv(file_path, index=False)

def measure_intan_vt(rhx, groups, sample_frequency, impedance_temperature_cic):
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

        print(f"Running VT test for group: {group}.")
        rhx.reset()

        # max_current_list = []
        # cic_list = []

        # Disable all channels
        print("Disabling all currents for test (this takes a few seconds)...")
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
            print(f"Testing sample {sample_i} (channel {channel_i})")   

            # Measure at 4 points, starting at vt_start[i] and scaling down
            for current_i in [vt_start[i], int(0.9*vt_start[i]), int(0.8*vt_start[i]), int(0.7*vt_start[i])]:
                # Enable current channel and set current
                rhx.set_stim_parameters(channel_i, current_i, vt_pulse_width, vt_interphase, vt_frequency, sample_i)
                rhx.enable_data_output(channel_i)

                # Stim for 1 second
                rhx.start_board()
                time.sleep(1)
                rhx.stop_board()
                
                # Read data from board
                buffer_size = EQUIPMENT_INFO["Intan"]["waveform_buffer_per_second_per_channel"]
                time_seconds, voltage_microvolts = rhx.read_data(buffer_size, sample_frequency)
            
                # Create dataframe
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

            # Calculate max current and CIC
            coeff = np.polyfit(eps_calculated, currents_tested, 1)
            bestfit = np.poly1d(coeff)
            max_current = bestfit(0.6)

            cic = max_current * (vt_pulse_width / 1000000) / (gsa_i / 100) # uA * s / cm^2

            # Force poorly performing devices to 1 so they continue being tested
            if max_current <= 0:
                max_current = 1
                cic = 0
            
            # Save cic to sample_data_summary
            df.loc[len(df)-1, 'Charge Injection Capacity @ 1000 us (uC/cm^2)'] = cic
            df.to_csv(f"{DATA_PATH}/{group}/{sample_i}_data_summary.csv")
            print(f"saved vt to: {DATA_PATH}/{group}/{sample_i}_data_summary.csv")

            # # Save CIC to impedance_temperature_cic
            # impedance_temperature_cic.loc[impedance_temperature_cic['Channel Number'] == channel_i, 'Charge Injection Capacity @ 1000 us (uC/cm^2)'] = cic

            print(f"CIC at 1000 us pulse width: {cic} uC/cm^2")

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

def perform_lcr_measurements(lcx100, mux, lcr_groups):
    # Loop through each group
    for group in lcr_groups:
        # Pull sample information
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

            # Check for any broken samples
            broken_samples = group_info["broken_devices"]

            # Load test info
            tests = group_info["test_info"]["tests"]

            # Loop through each LCR test
            for test in tests:
                # Get test information
                test_frequencies = TEST_INFO[test]["eis_frequencies"]
                test_voltage = TEST_INFO[test]["eis_amplitude"]

                # Loop through each sample in the group
                for sample in group_info["samples"]:

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
                    measure_lcr_impedance(sample, group, lcx100, test_frequencies, test_voltage)

                    # Open mux channels
                    mux.open_channels(mux_channel_1[0], [mux_channel_1[1]])
                    mux.open_channels(mux_channel_2[0], [mux_channel_2[1]])

    # When measurements are done, set LCR back to 25 mV for safety, close connection
    lcx100.set_voltage(25.0 / 1000) # Set voltage back to 25 mV for safety
    lcx100.close()

    # For some reason bus 1/channel 1 wants to stay closed, so open it manually, then close the mux connection
    mux.open_channels(1, [1])
    mux.close()

def measure_lcr_impedance(sample, group, lcx100, test_frequencies, test_voltage, group_info):
    counter = 0
    freq_data = []

    # Save timestamp
    measurement_time = datetime.datetime.now()
    
    # # Find number of days since start
    # first_day = group_info["start_date"]
    # first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

    # real_days = measurement_time - first_day
    # real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days

    # # Open data summary
    # df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
    # df = df.drop(columns=["Unnamed: 0"], axis=1)

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
        
        temperature = measure_temperature(sample, group_info, EQUIPMENT_INFO)

        # Save data to array
        freq_data.append(
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
                if "LCR-RH" in test:
                    # Impedance -> RH conversion is from the datasheet, corrected for temperature
                    impedance_kohms = impedance / 1000
                    rh = (impedance_kohms / (1.92e22 * temperature ** -7.57)) ** (1 / (-6.99 + 0.052 * temperature - 0.000225 * temperature ** 2))

                    record_rh_data_to_summary(group, sample, measurement_time, rh, temperature, DATA_PATH, group_info)

                # Otherwise, save raw Z data
                else:
                    record_impedance_data_to_summary(group, sample, measurement_time, impedance, phase, temperature, DATA_PATH, group_info)
        # new_row = pd.DataFrame({
        #     "Measurement Datetime": [measurement_time],
        #     "Temperature (C)": [temperature],
        #     "Real Days": [real_days],
        #     "Impedance Magnitude at 1000 Hz (ohms)": [float("NaN")],
        #     "Impedance Phase at 1000 Hz (degrees)": [float("NaN")]
        # })

        # # drop any columns that are all NA
        # new_row = new_row.dropna(axis=1, how="all")

        # if df.empty:
        #     new_df = new_row.copy()
        # elif new_row.empty:
        #     new_df = df.copy()
        # else:
        #     new_df = pd.concat([df, new_row], ignore_index=True)

        # new_df.to_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")

    # Save EIS data
    impedance_temperature = pd.DataFrame(impedance_temperature)

    filename = f"EIS_{sample}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    file_path = f"./data/{group}/{filename}.csv"
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
        if "Arduino-RH" not in group_info["test_info"]["tests"]:
            continue

        # Setup serial connection
        port = group_info["test_info"]["arduino_port"]
        baudrate = group_info["test_info"]["arduino_baudrate"]
        timeout_sec = 120 # Arduino prints output every minute, allow 2 minutes to check for data
        
        ser = serial.Serial(port, baudrate, timeout=timeout_sec)

        # Wait for data
        try:
            print("Waiting for serial data...")
            line = ser.readline().decode('utf-8').strip()
            # if line:
            #     print("Latest output:", line)
            # else:
            #     print("No data received within timeout period.")
        finally:
            ser.close()

        # Sort data
        for sample_i in group_info["samples"]:
            index_i = line.index(sample_i) + len(sample_i) + 2

            data = line[index_i:(index_i+18)]
            rh_index = data.index("RH=") + 3
            temp_index = data.index("T=") + 2

            rh = float(data[rh_index:(temp_index-3)])
            temperature = float(data[temp_index:])

            # Save data to summary
            record_rh_data_to_summary(group, sample_i, measurement_time, rh, temperature, DATA_PATH, group_info)

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

        # # Check if there are any special plots
        # special_plots = []
        # for plot_title, plot_groups in PLOT_INFO.items():
        #     if group in plot_groups:
        #         special_plots.append(plot_title)

        # Loop through each test - those will define the plots that are generated
        for test in group_info["test_info"]["tests"]:
            # For VT, plot CIC vs time
            if "VT" in test:
                title = f"{group}: Charge Injection Capacity vs Time (1000 us pulse)"
                cic_last, cic_norm_last, accel_days = plot_cic([group], DATA_PATH, SAMPLE_INFORMATION_PATH, PLOT_PATH, title, True)

            # For EIS, plot Z vs time
            elif "EIS" in test:
                title = f"{group}: Impedance Magnitude vs Time"
                z_last, z_norm_last, accel_days = plot_z([group], DATA_PATH, SAMPLE_INFORMATION_PATH, PLOT_PATH, title, True)

            # For RH, plot RH vs time and Temp vs time:
            elif "RH" in test:
                title = f"{group}: Relative Humidity vs Time"
                rh_last, accel_days = plot_rh([group], DATA_PATH, SAMPLE_INFORMATION_PATH, PLOT_PATH, title)
        
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
        last_update = group_info["slack_updates"]["last_update_months"]
        update_cadence = group_info["slack_updates"]["cadence_months"]

        last_test = group_info["test_info"]["last_test"]
        last_test = datetime.datetime.strptime(last_test, "%Y-%m-%d %H:%M")
        last_test = last_test.total_seconds() / 24 / 60 / 60 / 365.25 * 12 # convert to months

        if last_test - update_cadence > last_update:
            notify_slack(EQUIPMENT_INFO["Slack"]["webhook"], summary)
            print(summary)

def notify_slack(webhook, message):
    payload = {'text': message}
    requests.post(webhook, json=payload)

def write_heartbeat():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("heartbeat.txt", "w") as f:
        f.write(f"Last heartbeat: {now}\n")

if __name__ == '__main__':
    # Declare buffer size for reading from TCP command socket
    # This is the maximum number of bytes expected for 1 read. 1024 is plenty
    # for a single text command.
    # Increase if many return commands are expected.
    COMMAND_BUFFER_SIZE = 1024
    try:
        main()
    except Exception as e:
        error_msg = f"Lifetime testing script crashed at {datetime.datetime.now()}:\n{traceback.format_exc()}"

        if "KeyboardInterrupt" not in error_msg:
            notify_slack(EQUIPMENT_INFO["Slack"]["webhook"], error_msg)
        print(error_msg)
        