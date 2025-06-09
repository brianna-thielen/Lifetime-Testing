# Notes for changing interpreter
# CTRL+SHIFT+P, Python: Select Interpreter, Python 3.12.4 ('base')
import time
import datetime
import os
import pandas as pd
import statistics
import numpy as np
import requests
import traceback

from equipment.keithley_mux import KeithleyMUX as kmux
from equipment.rs_lcx100 import LCX100 as lcx
from equipment.intan_rhs import IntanRHS as intan
from equipment.phidget_4input_temperature import Phidget22TemperatureSensor as phidget

from data_processing.lcp_encapsulation_data_processing import process_encapsulation_soak_data
from data_processing.lcp_ide_data_processing import process_ide_soak_data
from data_processing.sirof_vs_pt_data_processing import process_coating_soak_data
from data_processing.lcp_pt_grids_data_processing import process_lcp_pt_grids_soak_data

SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'

IMPEDANCE_TEST_TIME = (8, 20) #tests at 8am and 8pm every day

GROUPS = {
    "SIROF vs Pt": ["IR01", "IR02", "IR03", "IR04", "IR05", "IR06", "IR07", "IR08", "IR09", "IR10", "PT01", "PT02", "PT03", "PT04", "PT05", "PT06", "PT07", "PT08", "PT09", "PT10"],
    "LCP Pt Grids": ["G1X1-1", "G1X1-2", "G3X3S-1", "G3X3S-2", "G2X2S-1", "G2X2S-2", "G2X2L-1", "G2X2L-2"],
    "LCP IDEs": ["IDE-25-1", "IDE-25-2", "IDE-25-3", "IDE-25-4", "IDE-25-5", "IDE-25-6", "IDE-25-7", "IDE-25-8", "IDE-100-1", "IDE-100-2", "IDE-100-3", "IDE-100-4", "IDE-100-5", "IDE-100-6", "IDE-100-7", "IDE-100-8"],
    "LCP Encapsulation": ["ENCAP-R-100-1", "ENCAP-R-100-2", "ENCAP-C-25-2", "ENCAP-C-100-1", "ENCAP-C-100-2"],
}

FLAGS = {
    "SIROF (vs Pt) - Z": 10000, # ohms
    "SIROF (vs Pt) - CIC": 30, # uC/cm2
    "Pt (vs SIROF) - Z": 10000, # ohms
    "Pt (vs SIROF) - CIC": 30, # uC/cm2
    "LCP Pt Grids - Z": 10000, # ohms
    "LCP Pt Grids - CIC": 10, # uC/cm2
    "LCP IDEs - value": 10000, # ohms
    "LCP IDEs - change": 0.8, # difference from start
    "LCP Encapsulation - Cap": 20, # %RH
    "LCP Encapsulation - Res": 40, # %RH
}

# SIROF SAMPLE CONSTANTS
BROKEN_ELECTRODES = ["IR07", "PT01"]
# SAMPLES = [
#     "IR01", "IR02", "IR03", "IR04", "IR05", "IR06", "IR07", "IR08", "IR09", "IR10", 
#     "PT01", "PT02", "PT03", "PT04", "PT05", "PT06", "PT07", "PT08", "PT09", "PT10"
# ]
SAMPLES = [
    "IR01", "IR02", "IR03", "IR04", "IR05", "IR06", "IR07", "IR08", "IR09", "IR10", 
    "PT01", "PT02", "PT03", "PT04", "PT05", "PT06", "PT07", "PT08", "PT09", "PT10",
    "G1X1-1", "G1X1-2", "G3X3S-1", "G3X3S-2", "G2X2S-1", "G2X2S-2", "G2X2L-1", "G2X2L-2",
]
# CHANNELS = [
#     "d-024", "d-025", "d-026", "d-027", "d-011", "d-023", "d-022", "d-021", "d-020", "d-004",
#     "d-028", "d-029", "d-030", "d-031", "d-015", "d-019", "d-018", "d-017", "d-016", "d-000"
# ]
CHANNELS = [
    "d-024", "d-025", "d-026", "d-027", "d-007", "d-028", "d-029", "d-030", "d-031", "d-006",
    "d-008", "d-009", "d-010", "d-011", "d-005", "d-012", "d-013", "d-014", "d-015", "d-004",
    "d-023", "d-022", "d-021", "d-020", "d-019", "d-018", "d-017", "d-016", 
]
# PULSE_AMPLITUDES = [
#     600, 600, 600, 600, 0, 400, 400, 400, 400, 0,
#     400, 400, 400, 400, 0, 200, 200, 200, 200, 0
# ] # uA
# PULSE_AMPLITUDES = [
#     600, 600, 1800, 1800, 0, 400, 400, 1200, 1200, 0,
#     400, 400, 1200, 1200, 0, 200, 200, 600, 600, 0
# ] # uA - updated 1/10/25 11:16am
PULSE_AMPLITUDES = [
    600, 600, 1800, 1800, 0, 400, 400, 1200, 1200, 0,
    400, 400, 1200, 1200, 0, 200, 200, 600, 600, 0,
    400, 400, 400, 400, 400, 400, 400, 400, 
] # uA - updated 4/25/25 5:06pm to include LCP test samples
GEOM_SURF_AREAS = [
    1.7272, 0.8103, 1.1716, 1.2028, 1.4268, 1.377, 0.594, 0.8375, 1.008, 1.2387,
    3.2472, 2.1784, 2.0868, 2.3458, 2.0832, 2.6302, 2.1168, 1.7199, 1.607, 2.8304,
    4, 4, 3.9204, 3.9204, 3.9601, 3.9601, 4, 4
] # mm^2
SIROF_IMPEDANCE_THRESHOLD = 100000 # ohms

# CONTINUOUS STIM CONSTANTS
PULSE_WIDTH = 200 # us
INTERPHASE_DELAY = 50 # us
PULSE_FREQUENCY = 50 # Hz

# INTAN CONSTANTS
WAVEFORM_BUFFER_PER_SECOND_PER_CHANNEL = 200000  # buffer size for reading from TCP waveform socket (maximum expected for 1 second)
# There is TCP lag in starting/stopping acquisition; exact number of data blocks may vary slightly
# From intan: 30 kHz, 1 channel, 1 second of wideband waveform data is 181,420 byte

# VT CONSTANTS
VT_INITIAL_I_FILE = 'vt_starting_currents.csv'
INITIAL_I_MAX = [
    1000, 1000, 1000, 1000, 500, 1000, 100, 1000, 1000, 500, 
    500, 500, 500, 500, 300, 500, 500, 100, 500, 300
] # uA
INITIAL_I_MAX = [
    527, 296, 2022, 1707, 1308, 243, 0, 392, 297, 180, 
    0, 708, 114, 563, 621, 1062, 843, 1000, 1372, 1155, 
    400, 400, 400, 400, 400, 400, 400, 400
]
VT_PULSE_FREQUENCY = 10 # Hz
VT_PULSE_WIDTH = 1000 # us
VT_INTERPHASE_DELAY = 500 # us

# LCR CONSTANTS
LCR_MEASURE_RANGE = 1e2
LCR_DATA_COLLECTION_DELAY_FIRST_CONTACT_SEC = 5.0
LCR_DATA_COLLECTION_DELAY_SEC = 1.0
LCR_MEASURE_INTERVAL = "SHOR"
LCR_RESOURCE_ADDRESS = "USB0::0x0AAD::0x0197::3629.8856k02-102189::INSTR"
LCR_VISA_TIMEOUT_MS = 3000
LCR_MEASURE_TYPE = "R"

# MUX CONSTANTS
MUX_RESOURCE_ADDRESS = "GPIB0::15::INSTR"
MUX_DEVICE_TO_CHANNEL_MAP = {
    "IDE-25-1": [10, 30],
    "IDE-25-2": [20, 40],
    "IDE-25-3": [9, 29],
    "IDE-25-4": [19, 39],
    "IDE-25-5": [8, 28],
    "IDE-25-6": [18, 38],
    "IDE-25-7": [7, 27],
    "IDE-25-8": [17, 37],
    "IDE-100-1": [6, 26],
    "IDE-100-2": [16, 36],
    "IDE-100-3": [5, 25],
    "IDE-100-4": [15, 35],
    "IDE-100-5": [4, 24],
    "IDE-100-6": [14, 34],
    "IDE-100-7": [3, 23],
    "IDE-100-8": [13, 33],
    "ENCAP-R-100-1": [2, 22],
    "ENCAP-R-100-2": [12, 32],
}

# IDE CONSTANTS
IDE_START_FREQ_HZ = 10
IDE_END_FREQ_HZ = 10000
IDE_NUM_FREQ_POINTS = 10
IDE_IMPEDANCE_THRESHOLD = 10000 # ohms

# THERMOCOUPLE CONSTANTS
THERMOCOUPLE_DELAY_TEMP_SEC = 0.5
TEMP_SENSOR_DRY_BATH_CHANNEL = 2
THERMOCOUPLE_TYPE_J = 1

def main():
    """
    Connects via TCP to RHX software
    Sets up stimulation parameters on all channels listed above
    Measures impedance every interval defined by IMPEDANCE_TEST_INTERVAL
    Stimulates all other times

    Intterupts with keypress "q"
    """

    # Generate EIS frequencies
    frequencies = generate_frequencies_to_test()

    # Connect to RHX software via TCP
    rhx = intan()

    # Query sample rate from RHX software.
    sample_frequency = rhx.find_sample_frequency()

    # Clear data output and disable all TCP channels
    rhx.reset()

    # Connect to RHX software via TCP
    rhx.connect_to_waveform_server()

    # Set up stimulation parameters for all channels
    setup_stim_channels(rhx, CHANNELS, SAMPLES, PULSE_AMPLITUDES, PULSE_WIDTH, INTERPHASE_DELAY, PULSE_FREQUENCY)

    # Start board running
    rhx.start_board()
    print('Starting stim.')

    run_test = True

    # Set the last check to the hour before the first test to trigger a test immediately if we're at test time
    last_check = min(IMPEDANCE_TEST_TIME) - 1

    # Set the last update to year 0, will trigger slack update every year thereafter
    last_update_encap = 0
    last_update_ide = 0
    last_update_sirof = 0
    last_update_grids = 0

    try:
        while run_test:
            current_hour = datetime.datetime.now().hour

            if current_hour in IMPEDANCE_TEST_TIME and current_hour != last_check:
                rhx.stop_board()

                # Create a dataframe to store impedance, temperature, and CIC data
                ztc_dict = {
                    'Channel Number': CHANNELS, 
                    'Channel Name': SAMPLES, 
                    'Impedance Magnitude at 1000 Hz (ohms)': None,
                    'Impedance Phase at 1000 Hz (degrees)': None,
                    'Temperature (C)': None,
                    'Charge Injection Capacity @ 1000 us (uC/cm^2)': None,
                    'Geometric Surface Area (mm^2)': GEOM_SURF_AREAS
                }
                impedance_temperature_cic = pd.DataFrame(ztc_dict)

                # Measure impedance and temperature
                print("Stimulation off. Running impedance check for SIROF parts.")
                impedance_temperature_cic, filename = measure_intan_impedance(rhx, frequencies, impedance_temperature_cic)

                # Measure CIC and max currents
                impedance_temperature_cic = measure_vt(rhx, CHANNELS, SAMPLES, sample_frequency, GEOM_SURF_AREAS, impedance_temperature_cic)

                # Save data, sorted to group
                for group, devices in GROUPS.items():
                    impedance_temperature_cic_group = impedance_temperature_cic[impedance_temperature_cic["Channel Name"].isin(devices)]
                    if not impedance_temperature_cic_group.empty:
                        impedance_temperature_cic_group.to_csv(f"./data/{group}/{filename}.csv")

                # Set up stimulation parameters for all channels
                print("Setting up stimulation parameters:")
                setup_stim_channels(rhx, CHANNELS, SAMPLES, PULSE_AMPLITUDES, PULSE_WIDTH, INTERPHASE_DELAY, PULSE_FREQUENCY)

                # Start board running
                rhx.start_board()
                time.sleep(0.12)
                
                # Wait until next impedance check
                print("Stimulation on; starting IDE EIS tests.")

                collect_impedance_vs_frequency(frequencies)

                # save hour of last impedance/VT check
                last_check = current_hour

                # Process data to flag any issues
                summary_encap, summary_ide, summary_sirof, summary_grids, last_update_encap, last_update_ide, last_update_sirof, last_update_grids = process_all_data(last_update_encap, last_update_ide, last_update_sirof, last_update_grids)

                # Print summary and notify slack (if applicable)
                print(f"Testing complete at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}.")
                print(summary_encap)
                print(summary_ide)
                print(summary_sirof)
                print("Press ctrl-c to stop stimulation and testing loop.")

                # Write a heartbeat with current time to save progress in case of crash
                write_heartbeat()

            else:
                # Save the heartbeat, wait a minute, and check again
                write_heartbeat()
                time.sleep(60)

    except KeyboardInterrupt:
        # rhx.stop_board()
        print("Automated test stopped, stimulation remains on.")
        notify_slack(f"Automated test stopped at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. Stimulation remains on.")

    # Close TCP socket
    rhx.close_tcp()

def notify_slack(message):
    payload = {'text': message}
    requests.post(SLACK_WEBHOOK_URL, json=payload)

def write_heartbeat():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("heartbeat.txt", "w") as f:
        f.write(f"Last heartbeat: {now}\n")

def setup_stim_channels(rhx, channel_list, sample_list, amplitude_list, pulse_width, interphase_delay, frequency):
    """
    Set up stimulation parameters for all channels
    """

    # If disabling all channels, don't print values
    if sum(amplitude_list) == 0:
        print_updates = False
    else:
        print_updates = True

    # Set up stimulation parameters for each channel
    for i in range(len(channel_list)):
        channel = channel_list[i]
        pulse_amplitude = amplitude_list[i]
        sample = sample_list[i]

        if sample not in BROKEN_ELECTRODES:
            if print_updates:
                print(f"{sample}: {pulse_amplitude} uA on channel {channel}")

            rhx.set_stim_parameters(channel, pulse_amplitude, pulse_width, interphase_delay, frequency, sample)
            
        else:
            if print_updates:
                print(f"{sample} listed as broken, no stim programmed to channel {channel}")

            rhx.set_stim_parameters(channel, 0, pulse_width, interphase_delay, frequency, sample)
            rhx.disable_stim(channel)

        if pulse_amplitude == 0:
            rhx.disable_stim(channel)

def measure_temperature():
    temp_sensor = phidget(TEMP_SENSOR_DRY_BATH_CHANNEL)
    temp_sensor.open_connection()
    temp_sensor.set_thermocouple_type(THERMOCOUPLE_TYPE_J)
    time.sleep(0.5)
    temperature = temp_sensor.get_temperature() - 5 # offset between dry bath and saline
    time.sleep(0.5)
    temp_sensor.close()

    return temperature

def measure_intan_impedance(rhx, frequencies, impedance_temperature_cic):
    impedances = []
    phases = []
    temperatures = []
    channels = []
    frequencies_updated = []

    # Measure temperature
    temperature = measure_temperature()

    # Loop through each frequency
    print('Measuring impedances...')
    for freq in frequencies:
        if freq > 30 and freq < 5060: # Intan won't test outside this range
            directory = os.getcwd()
            filename = rhx.measure_impedance(f"{directory}/data/temp/", freq)
            # Import saved impedance data
            saved_impedances = pd.read_csv(f"{directory}/data/temp/{filename}.csv")

            # Delete the file
            os.remove(f"{directory}/data/temp/{filename}.csv")

            # Add tested channels to list
            for channel_i in CHANNELS:
                channel_i = channel_i.capitalize()
                impedance_i = saved_impedances.loc[saved_impedances['Channel Number'] == channel_i].iloc[0, 4]
                phase_i = saved_impedances.loc[saved_impedances['Channel Number'] == channel_i].iloc[0, 5]
            
                impedances.append(impedance_i)
                phases.append(phase_i)
                temperatures.append(temperature)
                channels.append(channel_i)
                frequencies_updated.append(freq)

    # Once through all frequencies, separate EIS data by channel
    for i, channel_i in enumerate(CHANNELS):
        channel_i_cap = channel_i.capitalize()
        frequencies_i = [f for ch, f in zip(channels, frequencies_updated) if ch == channel_i_cap]

        sample_i = SAMPLES[i]
        
        impedance_i = [imp for ch, imp in zip(channels, impedances) if ch == channel_i_cap]
        impedance_i_1k = impedance_i[frequencies_i.index(1000)]

        phase_i = [ph for ch, ph in zip(channels, phases) if ch == channel_i_cap]
        phase_i_1k = phase_i[frequencies_i.index(1000)]

        temperature_i = [temp for ch, temp in zip(channels, temperatures) if ch == channel_i_cap]
        temperature_i = sum(temperature_i)/len(temperature_i)  # Average temperature for the channel

        # Add summary data to dataframe
        impedance_temperature_cic.loc[impedance_temperature_cic['Channel Number'] == channel_i, f'Impedance Magnitude at 1000 Hz (ohms)'] = impedance_i_1k
        impedance_temperature_cic.loc[impedance_temperature_cic['Channel Number'] == channel_i, f'Impedance Phase at 1000 Hz (degrees)'] = phase_i_1k
        impedance_temperature_cic.loc[impedance_temperature_cic['Channel Number'] == channel_i, 'Temperature (C)'] = temperature_i

        # Save EIS data separately by channel
        freq_data = {
            "Frequency": frequencies_i,
            "Impedance": impedance_i,
            "Phase Angle": phase_i,
            "Temperature (Dry Bath)": [temperature_i] * len(frequencies_i),
        }
        freq_data_df = pd.DataFrame(freq_data)

        for group, devices in GROUPS.items():
            if sample_i in devices:
                file_path = f"./data/{group}/intan-eis/EIS_{sample_i}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        
        freq_data_df.to_csv(file_path, index=False)
    
    return impedance_temperature_cic, filename

def measure_vt(rhx, channel_list, sample_list, sample_frequency, gsas, impedance_temperature_cic):
    """"
    Runs VT test
    """

    max_current_list = []
    cic_list = []

    print("Running VT test for SIROF parts:")
    rhx.reset()

    # Check for saved starting currents
    if os.path.exists(VT_INITIAL_I_FILE):
        vt_start = pd.read_csv(VT_INITIAL_I_FILE)
        vt_start = vt_start['Starting Current for VT (uA)'].tolist()
    else:
        vt_start = INITIAL_I_MAX

    # Disable all channels
    # Disabling stim doesn't seem to work, so set all currents to zero
    print("Disabling all currents for test (this takes a few seconds)...")
    setup_stim_channels(rhx, CHANNELS, SAMPLES, len(PULSE_AMPLITUDES) * [0], 0, 0, PULSE_FREQUENCY)

    # Loop through each channel and run VT test
    for i in range(len(channel_list)):
        channel = channel_list[i]
        sample = sample_list[i]
        gsa = gsas[i]

        # Only measure parts which are still functional
        if sample not in BROKEN_ELECTRODES:
            # if vt_start is small, reset it to 100 uA
            if vt_start[i] < 100:
                vt_start[i] = 100

            currents_tested = []
            eps_calculated = []

            # Intan cannot exceed 2500 uA - scale down if needed
            if vt_start[i] > 2500:
                start_current = 2500
            else:
                start_current = int(vt_start[i])

            print(f"Testing sample {sample} (channel {channel}) with currents: {start_current}, {int(0.9*start_current)}, {int(0.8*start_current)}, {int(0.7*start_current)} uA")   

            # Measure at 4 points, starting at vt_start[i] and scaling down
            for current_i in [start_current, int(0.9*start_current), int(0.8*start_current), int(0.7*start_current)]:
                # Enable current channel and set current
                rhx.set_stim_parameters(channel, current_i, VT_PULSE_WIDTH, VT_INTERPHASE_DELAY, VT_PULSE_FREQUENCY, sample)
                rhx.enable_data_output(channel)

                # Stim for 1 second
                rhx.start_board()
                time.sleep(1)
                rhx.stop_board()

                # Read data from board
                buffer_size = WAVEFORM_BUFFER_PER_SECOND_PER_CHANNEL
                time_seconds, voltage_microvolts = rhx.read_data(buffer_size, sample_frequency)
            
                # Create dataframe
                vt_data = {
                    'Time (s)': time_seconds,
                    'Voltage (uV)': voltage_microvolts
                }
                vt_data = pd.DataFrame(vt_data)

                # Calculate and store ep
                ep = calcluate_ep(vt_data)
                currents_tested.append(current_i)
                eps_calculated.append(ep)

            # Disable current channel
            rhx.set_stim_parameters(channel, 0, VT_PULSE_WIDTH, VT_INTERPHASE_DELAY, VT_PULSE_FREQUENCY, sample)
            rhx.disable_stim(channel)
            rhx.disable_data_output(channel)

            # Calculate max current and CIC
            coeff = np.polyfit(eps_calculated, currents_tested, 1)
            bestfit = np.poly1d(coeff)
            max_current = bestfit(0.6)

            cic = max_current * (VT_PULSE_WIDTH / 1000000) / (gsa / 100) # uA * s / cm^2

            print(f"CIC at 1000 us pulse width: {cic} uC/cm^2")

        else:
            print(f"Sample {sample} listed as broken. No test performed.")
            max_current = 0
            cic = 0

        max_current_list.append(max_current*0.9)
        cic_list.append(cic)

    # Save CICs to dataframe
    impedance_temperature_cic['Charge Injection Capacity @ 1000 us (uC/cm^2)'] = cic_list

    # Save new max current
    vt_start = max_current_list

    i_dict = {
        'Channel Number': CHANNELS, 
        'Channel Name': SAMPLES, 
        'Starting Current for VT (uA)': vt_start
    }
    vt_start = pd.DataFrame(i_dict)
    vt_start.to_csv(VT_INITIAL_I_FILE, index=False)

    return impedance_temperature_cic

def calcluate_ep(vt_data):
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
    time_at_baseline = time_at_max - (VT_PULSE_WIDTH / 1000000)
    time_at_interphase = time_at_max + (VT_INTERPHASE_DELAY / 1000000)

    ind_at_baseline = len([t for t in time_list if t < time_at_baseline])
    ind_at_interphase = len([t for t in time_list if t < time_at_interphase])

    voltage_at_baseline = vt_adjusted["Voltage (uV)"].iloc[ind_at_baseline]
    voltage_at_interphase = vt_adjusted["Voltage (uV)"].iloc[ind_at_interphase]

    # Calculate Ep
    ep = float(voltage_at_interphase - voltage_at_baseline) / 10.0 # manual adjustment to match o-scope

    return ep

def find_closest_timestamp(time, time_list):
    temp_time = time_list < time

def generate_frequencies_to_test():
    """
    Generates a list of frequencies to be tested against.

    Returns:
        frequencies (list): A list of frequencies to be tested.
    """
    frequencies = np.logspace(
        np.log10(IDE_START_FREQ_HZ), np.log10(IDE_END_FREQ_HZ), IDE_NUM_FREQ_POINTS
    )
    
    if 1000 not in frequencies:
        frequencies = np.append(frequencies, 1000)
        
    if 10000 not in frequencies:
        frequencies = np.append(frequencies, 10000)

    frequencies = np.sort(frequencies).tolist()

    return frequencies

def collect_impedance_vs_frequency(frequencies):
    """
    Collects impedance and phase angle data for different frequencies and contact areas.
    Args:
        frequencies (list): List of frequencies at which impedance is measured.
        area_to_contact_mapping (dict): Mapping of contact areas to contact numbers.
    Returns:
        freq_data_df (pandas.DataFrame): Dataframe containing impedance and phase angle data.
    """

    lcx100 = lcx(LCR_RESOURCE_ADDRESS)
    mux = kmux(MUX_RESOURCE_ADDRESS)
    lcx100.set_aperture(LCR_MEASURE_INTERVAL)
    lcx100.set_visa_timeout(LCR_VISA_TIMEOUT_MS)
    lcx100.reset_and_initialize()
    lcx100.set_measurement_type(LCR_MEASURE_TYPE)
    lcx100.set_measurement_range(LCR_MEASURE_RANGE)
    lcx100.set_voltage(0.5)
    mux.reset_and_initialize()
    mux.open_channels(1, range(1, 41))
    counter = 0

    print("Initializing Impedance vs Frequency Sweep...")
    for sample in MUX_DEVICE_TO_CHANNEL_MAP:
        freq_data = []

        channels = MUX_DEVICE_TO_CHANNEL_MAP[sample]

        mux.close_channels(
            1, channels,
        )
        print(f"Testing sample {sample}")

        # For IDEs, test full frequency spectrum at 25 mV and point to IDE data folder
        if "IDE" in sample:
            test_frequencies = frequencies
            folder = "./data/LCP IDEs"
            voltage = 25.0 / 1000
        # For Encapsulated samples, test only 1 kHz at .5 V and point to encapsulation data folder
        elif "ENCAP" in sample:
            test_frequencies = [1000]
            folder = "./data/LCP Encapsulation"
            voltage = 0.5

        for freq in test_frequencies:
            lcx100.set_voltage(voltage)
            lcx100.set_frequency(freq)
            if counter == 0:
                time.sleep(LCR_DATA_COLLECTION_DELAY_FIRST_CONTACT_SEC)
                counter = counter + 1
            else:
                time.sleep(LCR_DATA_COLLECTION_DELAY_SEC)
            impedance, phase_angle = lcx100.get_impedance()
            temp_dry_bath = measure_temperature()
            time.sleep(THERMOCOUPLE_DELAY_TEMP_SEC)
            freq_data.append(
                {
                    "Frequency": freq,
                    "Impedance": float(impedance),
                    "Phase Angle": float(phase_angle),
                    "Temperature (Dry Bath)": float(temp_dry_bath),
                }
            )

            if float(impedance) < IDE_IMPEDANCE_THRESHOLD and freq == 1000 and "IDE" in sample:
                print(f"Low impedance detected in sample {sample}: {impedance} ohms")
            elif freq == 1000:
                print(f"Impedance at 1 kHz: {impedance} ohms")
        mux.open_channels(
            1, channels,
        )
        counter = 0

        freq_data_df = pd.DataFrame(freq_data)
        
        filename = f"EIS_{sample}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        file_path = f"{folder}/{filename}.csv"
        freq_data_df.to_csv(file_path, index=False)

    lcx100.set_voltage(25.0 / 1000) # Set voltage back to 25 mV for safety
    lcx100.close()
    mux.close()

def process_all_data(last_update_encap, last_update_ide, last_update_sirof, last_update_grids):
    # Processes all soak data and flags any issues.
    # Encapsulation data
    days_encap, rh_cap_sensors, rh_res_sensors = process_encapsulation_soak_data()
    flagged_indices_r = [i for i, rh in enumerate(rh_res_sensors) if rh > FLAGS["LCP Encapsulation - Res"]]
    flagged_indices_c = [i for i, rh in enumerate(rh_cap_sensors) if rh > FLAGS["LCP Encapsulation - Cap"]]

    if len(flagged_indices_r) + len(flagged_indices_c) == 0:
        summary_encap = f"Encapsulation test at {round(days_encap/365.25, 1)} accelerated years, all parts within expected range."
    else:
        rh_sensors_r = GROUPS["LCP Encapsulation"][0:2]
        rh_sensors_c = GROUPS["LCP Encapsulation"][2:5]
        failing_devices = [rh_sensors_r[i] for i in flagged_indices_r] + [rh_sensors_c[i] for i in flagged_indices_c]
        summary_encap = f"LCP encapsulation test at {round(days_encap/365.25, 1)} accelerated years, {', '.join(failing_devices)} above expected range."

    # IDE data
    days_ide, z_IDE_25, z_IDE_100, norm_IDE_25, norm_IDE_100 = process_ide_soak_data()

    flagged_indices_z25 = [i for i, z in enumerate(z_IDE_25) if z < FLAGS["LCP IDEs - value"]]
    flagged_indices_z100 = [i for i, z in enumerate(z_IDE_100) if z < FLAGS["LCP IDEs - value"]]
    flagged_indices_norm25 = [i for i, n in enumerate(norm_IDE_25) if n < FLAGS["LCP IDEs - change"]]
    flagged_indices_norm100 = [i for i, n in enumerate(norm_IDE_100) if n < FLAGS["LCP IDEs - change"]]

    flagged_indices_25 = [x for x in flagged_indices_z25 if x in flagged_indices_norm25]
    flagged_indices_100 = [x for x in flagged_indices_z100 if x in flagged_indices_norm100]

    if len(flagged_indices_25) + len(flagged_indices_100) == 0:
        summary_ide = f"LCP IDE test at {round(days_ide/365.25, 1)} accelerated years, all parts within expected range."
    else:
        ides_25 = GROUPS["LCP IDEs"][0:8]
        ides_100 = GROUPS["LCP IDEs"][8:16]

        failing_devices = [ides_25[i] for i in flagged_indices_25] + [ides_100[i] for i in flagged_indices_100]
        summary_ide = f"LCP IDE test at {round(days_ide/365.25, 1)} accelerated years, {', '.join(failing_devices)} below expected range."

    # SIROF vs Pt data
    days_sirof, cic_pt, cic_ir, z_pt, z_ir = process_coating_soak_data()

    flagged_indices_cicpt = [i for i, c in enumerate(cic_pt) if c < FLAGS["Pt (vs SIROF) - CIC"]]
    flagged_indices_cicir = [i for i, c in enumerate(cic_ir) if c < FLAGS["SIROF (vs Pt) - CIC"]]
    flagged_indices_zpt = [i for i, z in enumerate(z_pt) if z > FLAGS["Pt (vs SIROF) - Z"]]
    flagged_indices_zir = [i for i, z in enumerate(z_ir) if z > FLAGS["SIROF (vs Pt) - Z"]]

    if len(flagged_indices_cicpt) + len(flagged_indices_cicir) + len(flagged_indices_zpt) + len(flagged_indices_zir) == 0:
        summary_sirof = f"SIROF vs Pt test at {round(days_sirof/365.25, 1)} accelerated years, all parts within expected range."
    else:
        sirof = GROUPS["SIROF vs Pt"][0:10]
        pt = GROUPS["SIROF vs Pt"][10:20]

        failing_devices_cic = [pt[i] for i in flagged_indices_cicpt] + [sirof[i] for i in flagged_indices_cicir]
        failing_devices_z = [pt[i] for i in flagged_indices_zpt] + [sirof[i] for i in flagged_indices_zir]

        if len(failing_devices_cic) == 0:
            summary_sirof = f"SIROF vs Pt test at {round(days_sirof/365.25, 1)} accelerated years, {', '.join(failing_devices_z)} above expected Z range."
        elif len(failing_devices_z) == 0:
            summary_sirof = f"SIROF vs Pt test at {round(days_sirof/365.25, 1)} accelerated years, {', '.join(failing_devices_cic)} below expected CIC range."
        else:
            summary_sirof = f"SIROF vs Pt test at {round(days_sirof/365.25, 1)} accelerated years, {', '.join(failing_devices_cic)} below expected CIC range, {', '.join(failing_devices_z)} above expected Z range."

    # LCP Grids
    days_grids, cic_grids, z_grids = process_lcp_pt_grids_soak_data()

    flagged_indices_cic = [i for i, c in enumerate(cic_grids) if c < FLAGS["LCP Pt Grids - CIC"]]
    flagged_indices_z = [i for i, z in enumerate(z_grids) if z > FLAGS["LCP Pt Grids - Z"]]
    
    if len(flagged_indices_cic) + len(flagged_indices_z) == 0:
        summary_grids = f"LCP Pt Grids test at {round(days_grids/365.25, 1)} accelerated years, all parts within expected range."
    else:
        grids = GROUPS["LCP Pt Grids"]

        failing_devices_cic = [grids[i] for i in flagged_indices_cic]
        failing_devices_z = [grids[i] for i in flagged_indices_z]

        if len(failing_devices_cic) == 0:
            summary_grids = f"LCP Pt Grids test at {round(days_grids/365.25, 1)} accelerated years, {', '.join(failing_devices_z)} above expected Z range."
        elif len(failing_devices_z) == 0:
            summary_grids = f"LCP Pt Grids test at {round(days_grids/365.25, 1)} accelerated years, {', '.join(failing_devices_cic)} below expected CIC range."
        else:
            summary_grids = f"LCP Pt Grids test at {round(days_grids/365.25, 1)} accelerated years, {', '.join(failing_devices_cic)} below expected CIC range, {', '.join(failing_devices_z)} above expected Z range."

    # If we reach a next year, notify slack
    if round(days_encap/365.25) > last_update_encap:
        last_update_encap = round(days_encap/365.25)
        notify_slack(summary_encap)

    if round(days_ide/365.25) > last_update_ide:
        last_update_ide = round(days_ide/365.25)
        notify_slack(summary_ide)

    if round(days_sirof/365.25) > last_update_sirof:
        last_update_sirof = round(days_sirof/365.25)
        notify_slack(summary_sirof)
    
    if round(days_grids/365.25) > last_update_grids:
        last_update_grids = round(days_grids/365.25)
        notify_slack(summary_grids)

    return summary_encap, summary_ide, summary_sirof, summary_grids, last_update_encap, last_update_ide, last_update_sirof, last_update_grids
    
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
            notify_slack(error_msg)
        print(error_msg)
        