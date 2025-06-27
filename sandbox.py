import pandas as pd
import os
import datetime
import json
import math
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from support_functions import measure_temperature

SAMPLE_INFORMATION_PATH = './test_information/samples'
DATA_PATH = './data'
EQUIPMENT_INFORMATION_PATH = './test_information/equipment.json'
PLOT_INFORMATION_PATH = './test_information/special_plots.json'

TEMP_GROUP = "Pt Foil"
TEMP_FILES = ["intanimpedance_1k_20250425151936.csv", "intanimpedance_1k_20250426033747.csv"]
flagged_dates_all = {'replaced PBS': '3-28-25 17:50'}

flagged_dates_group = {
    "replaced PBS": ["3-27-25 17:50", '12'],
    "test": "4-24-34 17:50"
}

flagged_samples = {}
flagged_samples_f = {}

flag = "a"
flagged_samples_g = ['n', 'm']
group = 'b'
flagged_samples_f[flag] = flagged_samples_g
flagged_samples[group] = flagged_samples_f

payload = {'text': 'test'}
requests.post("https://hooks.slack.com/services/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", json=payload)