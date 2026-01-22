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
import numpy as np

import subprocess
from datetime import datetime
from pathlib import Path


dfc1001 = pd.read_csv(f"publication_plots/data/LCP Encapsulation Capacitive/ENCAP-C-100-1_data_summary.csv")
dfc1002 = pd.read_csv(f"publication_plots/data/LCP Encapsulation Capacitive/ENCAP-C-100-2_data_summary.csv")
dfc252 = pd.read_csv(f"publication_plots/data/LCP Encapsulation Capacitive/ENCAP-C-25-2_data_summary.csv")
dfr1001 = pd.read_csv(f"publication_plots/data/LCP Encapsulation Resistive/ENCAP-R-100-1_data_summary.csv")
dfr1002 = pd.read_csv(f"publication_plots/data/LCP Encapsulation Resistive/ENCAP-R-100-2_data_summary.csv")

dfc1001 = dfc1001.drop(columns=["Unnamed: 0"], axis=1)
dfc1002 = dfc1002.drop(columns=["Unnamed: 0"], axis=1)
dfc252 = dfc252.drop(columns=["Unnamed: 0"], axis=1)
dfr1001 = dfr1001.drop(columns=["Unnamed: 0"], axis=1)
dfr1002 = dfr1002.drop(columns=["Unnamed: 0"], axis=1)

dfc1001 = dfc1001.sort_values("Real Days").reset_index(drop=True)
dfc1002 = dfc1002.sort_values("Real Days").reset_index(drop=True)
dfc252 = dfc252.sort_values("Real Days").reset_index(drop=True)
dfr1001 = dfr1001.sort_values("Real Days").reset_index(drop=True)
dfr1002 = dfr1002.sort_values("Real Days").reset_index(drop=True)

out = [df.copy() for df in [dfc1001, dfc1002, dfc252, dfr1001, dfr1002]]
anchor_df = dfc1001.copy()
anchor_tbl = anchor_df[["Real Days"]].rename(columns={"Real Days": "anchor_day"}).reset_index(drop=True)
anchor_tbl["anchor_id"] = np.arange(len(anchor_tbl), dtype=int)

match_tables = []
for i, df in enumerate(out):
    left = df[["Real Days", "Temperature (C)"]].copy().reset_index().rename(columns={"index": "row_idx"})
    right = anchor_tbl.sort_values("anchor_day")

    m = pd.merge_asof(
        left.sort_values("Real Days"),
        right,
        left_on="Real Days",
        right_on="anchor_day",
        direction="nearest",
        tolerance=None
    )

    m["df_i"] = i
    match_tables.append(m)

matches = pd.concat(match_tables, ignore_index=True)

matches = matches.dropna(subset=["anchor_id"]).copy()
matches["anchor_id"] = matches["anchor_id"].astype(int)

avg_by_anchor = (
    matches.groupby("anchor_id")["Temperature (C)"]
    .mean()
    .rename("avg_temp")
    .reset_index()
)

matches = matches.merge(avg_by_anchor, on="anchor_id", how="left")

for i in range(len(out)):
    sub = matches[matches["df_i"] == i]
    # Update only the matched rows
    out[i].loc[sub["row_idx"].to_numpy(), "Temperature (C)"] = sub["avg_temp"].to_numpy()

fig = make_subplots(
    rows=2,
    cols=3
)

fig.add_trace(
    go.Scatter(
        x=dfc1001["Real Days"],
        y=dfc1001["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='black', dash='solid'),
        connectgaps=True
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=out[0]["Real Days"],
        y=out[0]["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='blue', dash='solid'),
        connectgaps=True
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=dfc1002["Real Days"],
        y=dfc1002["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='black', dash='solid'),
        connectgaps=True
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=out[1]["Real Days"],
        y=out[1]["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='blue', dash='solid'),
        connectgaps=True
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=dfc252["Real Days"],
        y=dfc252["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='black', dash='solid'),
        connectgaps=True
    ),
    row=1,
    col=3,
)

fig.add_trace(
    go.Scatter(
        x=out[2]["Real Days"],
        y=out[2]["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='blue', dash='solid'),
        connectgaps=True
    ),
    row=1,
    col=3,
)

fig.add_trace(
    go.Scatter(
        x=dfr1001["Real Days"],
        y=dfr1001["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='black', dash='solid'),
        connectgaps=True
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=out[3]["Real Days"],
        y=out[3]["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='blue', dash='solid'),
        connectgaps=True
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=dfr1002["Real Days"],
        y=dfr1002["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='black', dash='solid'),
        connectgaps=True
    ),
    row=2,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=out[4]["Real Days"],
        y=out[4]["Temperature (C)"],
        mode="lines",
        line=dict(width=1, color='blue', dash='solid'),
        connectgaps=True
    ),
    row=2,
    col=2,
)

fig.show()

out[0].to_csv(f"publication_plots/data/LCP Encapsulation Capacitive/ENCAP-C-100-1_data_summary.csv")
out[1].to_csv(f"publication_plots/data/LCP Encapsulation Capacitive/ENCAP-C-100-2_data_summary.csv")
out[2].to_csv(f"publication_plots/data/LCP Encapsulation Capacitive/ENCAP-C-25-2_data_summary.csv")
out[3].to_csv(f"publication_plots/data/LCP Encapsulation Resistive/ENCAP-R-100-1_data_summary.csv")
out[4].to_csv(f"publication_plots/data/LCP Encapsulation Resistive/ENCAP-R-100-2_data_summary.csv")