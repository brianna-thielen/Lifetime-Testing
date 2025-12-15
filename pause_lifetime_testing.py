# To pause lifetime testing: python pause_lifetime_testing.py

import os
import json

from equipment.intan_rhs import IntanRHS as intan

from support_functions.support_functions import record_timestamp

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
    record_timestamp(False, intan_groups, SAMPLE_INFORMATION_PATH, EQUIPMENT_INFO, DATA_PATH)


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


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Lifetime pause failed')
        