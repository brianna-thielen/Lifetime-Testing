# To pause lifetime testing: python pause_lifetime_testing.py
# Use this script if you need to pause stim - stopping run_lifetime_testing.py will not stop stim, 
# and manually stopping stim will not be logged, resulting in inaccurate data capture

# If you need to move the setup or change the temperature, make the change while running 
# run_lifetime_testing.py, or run this script immediately before and after the temperature change
# starts and finishes to log accurate acceleration

import os
import json
import datetime

from equipment.intan_rhs import IntanRHS as intan

from support_functions.support_functions import record_timestamp

SAMPLE_INFORMATION_PATH = './test_information/samples'
EQUIPMENT_INFORMATION_PATH = './test_information/equipment.json'
TEST_INFORMATION_PATH = './test_information/tests.json'
DATA_PATH = './data'

# Import test, equipment, and plot information
with open(TEST_INFORMATION_PATH, 'r') as f:
    TEST_INFO = json.load(f)

with open(EQUIPMENT_INFORMATION_PATH, 'r') as f:
    EQUIPMENT_INFO = json.load(f)

def main():
    """
    Pauses intan stimulation during lifetime testing
    """

    # Initialize the Intan
    rhx, sample_frequency = initialize_intan()

    # Turn off stim
    rhx.stop_board()
    print('Stopping stim.')

    # Find list of intan groups
    intan_groups = []

    for group in os.listdir(DATA_PATH):
        # Ignore temp and plot folders
        if "Plots" in group or "temp" in group or "Archive" in group:
            continue

        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Loop through each test to find if the group is part of intan testing
        for test in group_info["test_info"]["tests"]:
            if "Intan" in test:
                intan_groups.append(group)

    # Remove duplicates
    intan_groups = list(set(intan_groups))

    # Record timestamp for all intan groups
    print("Recording pause timestamps (this takes a few seconds)")
    record_timestamp(False, intan_groups, SAMPLE_INFORMATION_PATH, EQUIPMENT_INFO, DATA_PATH)

    # Reset stim
    reset_stim_intan(rhx)

    print("Done")

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

def reset_stim_intan(rhx):
    # When stim_on is True, all channels are set to values under test_information/samples
    # When stim_on is False, all channels are set to zero and disabled
    now = datetime.datetime.now()
    now = now.strftime("%m/%d %H:%M:%S")
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

        # Loop through each sample in the group
        for sample in group_info["samples"]:
            sample_info = group_info["samples"][sample]

            # Turn off stim
            rhx.set_stim_parameters(sample_info["intan_channel"], 0, 0, 0, 0, sample)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Lifetime pause failed')
        