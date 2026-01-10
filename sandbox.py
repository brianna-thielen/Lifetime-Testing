import pandas as pd
import os
from datetime import datetime, timedelta
import json
import math
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pprint
from equipment.intan_rhs import IntanRHS as intan
import time
import re
import csv

import subprocess
from datetime import datetime
from pathlib import Path

df = pd.read_csv(f"data/LCP Encapsulation Capacitive Ambient/Ambient_data_summary.csv")
df = df.drop(columns=["Unnamed: 0"], axis=1)

df = df.sort_values("Real Days").reset_index(drop=True)

threshold = 2 / (24 * 60)  # 2 minutes in days

dt = df["Real Days"].diff()

group_id = (dt > threshold).cumsum()

df["group_id"] = group_id

idx = (
    df.dropna(subset=["Relative Humidity (%)"])
      .groupby("group_id")["Relative Humidity (%)"]
      .idxmin()
)

df_clean = (
    df.loc[idx]
      .sort_values("Real Days")
      .reset_index(drop=True)
)

df_clean = df_clean.drop(columns=["group_id"])

fig = make_subplots(rows=1, cols=1)

fig.add_trace(
    go.Scatter(
        x=df['Real Days'],
        y=df['Relative Humidity (%)'],
        mode="lines",
        line=dict(width=1, color='black'),
        connectgaps=True
    )
)

fig.add_trace(
    go.Scatter(
        x=df_clean['Real Days'],
        y=df_clean['Relative Humidity (%)'],
        mode="lines",
        line=dict(width=1, color='blue'),
        connectgaps=True
    )
)

fig.show()
df = pd.read_csv(f"data/LCP Encapsulation Capacitive Ambient/Ambient_data_summary.csv")
df_clean.to_csv(f"data/LCP Encapsulation Capacitive Ambient/Ambient_clean_data_summary.csv")