import pandas as pd
import os
import datetime
import json
import math
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pprint
from equipment.intan_rhs import IntanRHS as intan
import time

SAMPLE_INFORMATION_PATH = './test_information/samples'
DATA_PATH = './data'
EQUIPMENT_INFORMATION_PATH = './test_information/equipment.json'
PLOT_INFORMATION_PATH = './test_information/special_plots.json'
TEST_INFORMATION_PATH = './test_information/tests.json'

# # Connect to RHX software via TCP
# rhx = intan()

# # Query sample rate from RHX software.
# sample_frequency = rhx.find_sample_frequency()

# # Clear data output and disable all TCP channels
# rhx.reset()

# # Connect to RHX software via TCP
# rhx.connect_to_waveform_server()

# directory = os.getcwd()
# filename, freq = rhx.measure_impedance(f"{directory}/data/temp/", 46.4)

g = ['EIS-Intan-3', 'VT-Intan', 'VT-Intan']
g = list(set(g))
print(g)