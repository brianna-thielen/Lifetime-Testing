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

GROUPS = {
    "SIROF vs Pt": ["IR01", "IR02", "IR03", "IR04", "IR05", "IR06", "IR07", "IR08", "IR09", "IR10", "PT01", "PT02", "PT03", "PT04", "PT05", "PT06", "PT07", "PT08", "PT09", "PT10"],
    "LCP Pt Grids": ["G1X1-1", "G1X1-2", "G3X3S-1", "G3X3S-2", "G2X2S-1", "G2X2S-2", "G2X2L-1", "G2X2L-2"],
    "LCP IDEs": ["IDE-25-1", "IDE-25-2", "IDE-25-3", "IDE-25-4", "IDE-25-5", "IDE-25-6", "IDE-25-7", "IDE-25-8", "IDE-100-1", "IDE-100-2", "IDE-100-3", "IDE-100-4", "IDE-100-5", "IDE-100-6", "IDE-100-7", "IDE-100-8"],
    "LCP Encapsulation": ["ENCAP-R-100-1", "ENCAP-R-100-2"]
}

# SIROF SAMPLE CONSTANTS
BROKEN_ELECTRODES = ["IR07", "PT01"]

SAMPLES = [
    "IR01", "IR02", "IR03", "IR04", "IR05", "IR06", "IR07", "IR08", "IR09", "IR10", 
    "PT01", "PT02", "PT03", "PT04", "PT05", "PT06", "PT07", "PT08", "PT09", "PT10",
    "G1X1-1", "G1X1-2", "G3X3S-1", "G3X3S-2", "G2X2S-1", "G2X2S-2", "G2X2L-1", "G2X2L-2",
]

CHANNELS = [
    "d-024", "d-025", "d-026", "d-027", "d-007", "d-028", "d-029", "d-030", "d-031", "d-006",
    "d-008", "d-009", "d-010", "d-011", "d-005", "d-012", "d-013", "d-014", "d-015", "d-004",
    "d-023", "d-022", "d-021", "d-020", "d-019", "d-018", "d-017", "d-016", 
]

folder = f'C:/Users/brian/Documents/GitHub/Lifetime-Testing/data/temp/'

list_of_files = os.listdir(folder)

for file in list_of_files:
    if file.endswith('.csv'):
        df = pd.read_csv(folder + file)

        for group, devices in GROUPS.items():
            df_new = df[df["Channel Name"].isin(devices)]
            if len(df_new) > 0:
                df_new.to_csv(f"{folder}{group}/{file}", index=False)