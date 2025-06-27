from data_processing.lcp_encapsulation_data_processing import process_encapsulation_soak_data
from data_processing.lcp_ide_data_processing import process_ide_soak_data
from data_processing.sirof_vs_pt_data_processing import process_coating_soak_data
from data_processing.lcp_pt_grids_data_processing import process_lcp_pt_grids_soak_data

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

DATA_PATH = './data'
GROUPS = ['LCP IDEs 25um']
SAMPLE_INFORMATION_PATH = './test_information/samples'

fig = make_subplots(
    rows=1,
    cols=1,
)

for group in GROUPS:
    # Load group info
    with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
        group_info = json.load(f)
    
    samples = group_info["samples"]

    for sample in samples:

        df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
        df = df.drop(columns=["Unnamed: 0"], axis=1)

        df_sorted = df.sort_values(by='Real Days')

        df_clean = df_sorted.dropna(subset=['Impedance Magnitude at 1000 Hz (ohms)'])
        x = df_clean["Real Days"]
        y = df_clean["Impedance Magnitude at 1000 Hz (ohms)"]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
            ),
            row=1,
            col=1,
        )

fig.update_yaxes(type="log", tick0=100, dtick=1, range=[1, 6])
fig.show()
