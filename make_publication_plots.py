import datetime
import pandas as pd
import numpy as np
import json
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import numpy as np
import math
import datetime

from support_functions.support_functions import calculate_accel_days, calculate_accel_days_single

PLOT_PATH = './publication_plots'
DATA_PATH = './publication_plots/data'
SAMPLE_INFORMATION_PATH = './test_information/samples'

BASE_COLORS_RH = [
    pc.sequential.Greys,
    pc.sequential.Greys,
    pc.sequential.Purples,
    pc.sequential.Greens
]

BASE_COLORS_IDE = [
    pc.sequential.Blues,
    pc.sequential.Reds
]

LINE_WIDTH = 1.5

HUMIDITY_GROUPS = ["LCP Encapsulation Capacitive Ambient", "LCP Encapsulation Capacitive", "LCP Encapsulation Resistive"]
IDE_GROUPS = ["LCP IDEs 25um", "LCP IDEs 100um"]

HUMIDITY_AMBIENT_COLOR = 'rgb(0,100,0)'
HUMIDITY_AMBIENT_SOLID = 'solid'
# HUMIDITY_R_COLOR = 'rgb(100,40,170)'
# HUMIDITY_C_COLOR = 'rgb(100,40,170)'
HUMIDITY_R_COLOR = 'rgb(110,40,185)'
HUMIDITY_C_COLOR = 'rgb(70,30,130)'
HUMIDITY_C_SOLID = 'solid'
HUMIDITY_R_SOLID = 'solid'
HUMIDITY_LINE_WIDTH = 2

TEMPERATURE_COLOR = 'rgb(100,100,100)'
TEMPERATURE_LINE_SOLID = 'dot'
TEMPERATURE_LINE_WIDTH = 1

IDE_ENG_COLOR = 'rgb(150,150,150)'
IDE_IDE_COLOR = 'rgb(200,120,40)'
IDE_BOTH_COLOR = 'rgb(150,10,10)'
IDE_GOOD_COLOR = 'rgb(65,135,165)'
IDE_ENG_SOLID = 'solid'
IDE_IDE_SOLID = 'solid'
IDE_BOTH_SOLID = 'solid'
IDE_GOOD_SOLID = 'solid'
IDE_SAMPLE_LINE_WIDTH = 1

IDE_AVG_COLOR = 'rgb(8,50,110)'
IDE_AVG_SOLID = 'solid'
IDE_AVG_LINE_WIDTH = 2

IDE_AXIS_COLOR = 'rgb(0,0,0)'

SAMPLE_PLOT_HEIGHT = 0.35
SUPPLEMENT_PLOT_HEIGHT = 0.15
GRID_COLOR = 'rgb(230,230,230)'


def plot_humidity_temp_real(groups, rh_range_25, rh_range_100, rh_range_amb, temp_range_ambient, temp_range_sample, t_range):
    # Setup plot
    fig_rh = make_subplots(
        rows=3,
        cols=1,
        row_heights=[SAMPLE_PLOT_HEIGHT, SAMPLE_PLOT_HEIGHT, SUPPLEMENT_PLOT_HEIGHT],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
        x_title="Time Elapsed (weeks)",
        y_title="Relative Humidity (%)",
    )

    fig_rh.add_annotation(
        text="Temperature (°C)",
        xref="paper",
        yref="paper",
        x=0.975,
        y=0.5,
        showarrow=False,
        textangle=90,
    )

    fig_rh.update_annotations(font=dict(size=22, family="Arial", color="black"))

    fig_rh.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=18, family="Arial"),
        yaxis=dict(side='right'), yaxis2=dict(side='left'), yaxis3=dict(side='right'), yaxis4=dict(side='left'), yaxis5=dict(side='right'), yaxis6=dict(side='left')
    )
    
    fig_rh.update_xaxes(
        range=t_range, tick0=0, dtick=4,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR
    )
    
    fig_rh.update_yaxes(
        # title_text="Temp. (C)", 
        range=temp_range_sample, row=1,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR, tickfont=dict(color=TEMPERATURE_COLOR)
    )
    
    fig_rh.update_yaxes(
        # title_text="RH (%)", 
        range=rh_range_25, row=1, secondary_y=True,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR, tickfont=dict(color=HUMIDITY_C_COLOR)
    )

    fig_rh.update_yaxes(
        # title_text="Temp. (C)", 
        range=temp_range_sample, row=2, 
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR, tickfont=dict(color=TEMPERATURE_COLOR)
    )

    fig_rh.update_yaxes(
        # title_text="RH (%)", 
        range=rh_range_100, row=2, secondary_y=True,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR, tickfont=dict(color=HUMIDITY_C_COLOR)
    )

    fig_rh.update_yaxes(
        # title_text="Temp. (C)", 
        range=temp_range_ambient, row=3,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR, tickfont=dict(color=TEMPERATURE_COLOR)
    )

    fig_rh.update_yaxes(
        # title_text="RH (%)", 
        range=rh_range_amb, row=3, secondary_y=True, 
        tick0=0, dtick=25,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR, tickfont=dict(color=HUMIDITY_AMBIENT_COLOR)
    )

    fig_rh.update_layout(showlegend=False)

    row1temp = 0
    row2temp = 0
    row3temp = 0

    for g, group in enumerate(groups):
        # Load group info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            if "C-25" in sample:
                row = 1
                rowtemp = row1temp
                row1temp += 1
            elif "C-100" in sample:
                row = 2
                rowtemp = row2temp
                row2temp += 1
            elif "Ambient" in group:
                row = 3
                rowtemp = row3temp
                row3temp += 1

            if "-C-" in sample:
                shade = HUMIDITY_C_COLOR
                solid = HUMIDITY_C_SOLID
            elif "-R-" in sample:
                shade = HUMIDITY_R_COLOR
                solid = HUMIDITY_R_SOLID
            else:
                shade = HUMIDITY_AMBIENT_COLOR
                solid = HUMIDITY_AMBIENT_SOLID
            
            # Open data summary
            df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Pull data
            rh = df["Relative Humidity (%)"]
            temperature = df["Temperature (C)"]
            real_days = df["Real Days"]
            real_days = [a / 7 for a in real_days]  # convert to weeks

            # Plot
            if rowtemp == 0:
                fig_rh.add_trace(
                    go.Scatter(
                        x=real_days,
                        y=temperature,
                        mode="lines",
                        line=dict(width=TEMPERATURE_LINE_WIDTH, color=TEMPERATURE_COLOR, dash=TEMPERATURE_LINE_SOLID),
                        name="Sample Temperature",
                        connectgaps=True
                    ),
                    # secondary_y=True,
                    row=row,
                    col=1,
                )

                r_d = np.asarray(real_days)
                tm = np.asarray(temperature)
                mask = np.isfinite(r_d) & np.isfinite(tm)
                avg_temp = (
                    np.trapezoid(tm[mask], x=r_d[mask])
                    / (r_d[mask].max() - r_d[mask].min())
                )
                
                print(f'average temp rh row {row}: {avg_temp}')
            
            fig_rh.add_trace(
                go.Scatter(
                    x=real_days,
                    y=rh,
                    mode="lines",
                    line=dict(width=HUMIDITY_LINE_WIDTH, color=shade, dash=solid),
                    name=sample,
                    connectgaps=True
                ),
                secondary_y=True,
                row=row,
                col=1,
            )

    # # remove colons and parentheses from title to produce filename
    # file_title = title.replace(":", "")
    # file_title = title.replace("(", "")
    # file_title = title.replace(")", "")
    # fig_rh.write_html(f"{PLOT_PATH}/{file_title}.html")

    # Show plot
    fig_rh.show()

    return fig_rh

def plot_ide_good_notemp(groups, deltaz_range, t_range, eng_failures, ide_failures, both_failures):
    # Setup plot
    fig_z = make_subplots(
        rows=2,
        cols=1,
        row_heights=[SAMPLE_PLOT_HEIGHT, SAMPLE_PLOT_HEIGHT],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        x_title="Time Elapsed (weeks)",
        y_title="Impedance Magnitude Change (|Z| / |Z|<sub>0</sub>)",
    )

    fig_z.update_annotations(font=dict(size=22, family="Arial", color="black"))

    fig_z.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=18, family="Arial"),
        yaxis=dict(tickmode='array', tickvals=[0.001, 0.01, 0.1, 1, 10], ticktext=['10<sup>-3</sup>', '10<sup>-2</sup>', '10<sup>-1</sup>', '1', '10']),
        yaxis2=dict(tickmode='array', tickvals=[0.001, 0.01, 0.1, 1, 10], ticktext=['10<sup>-3</sup>', '10<sup>-2</sup>', '10<sup>-1</sup>', '1', '10'])
    )

    fig_z.update_xaxes(
        range=t_range, tick0=0, dtick=4,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR
    )

    fig_z.update_yaxes(
        # title_text="Impedance Magnitude Change (|Z| / |Z|<sub>0</sub>)",
        range=deltaz_range, row=1,
        type="log", tick0=0, dtick=1,  
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR, tickfont=dict(color=IDE_AXIS_COLOR)
    )

    fig_z.update_yaxes(
        # title_text="Impedance Magnitude Change (|Z| / |Z|<sub>0</sub>)",
        range=deltaz_range, row=2,
        type="log", tick0=0, dtick=1,  
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor=GRID_COLOR, tickfont=dict(color=IDE_AXIS_COLOR)
    )

    # fig_z.update_layout(showlegend=False)

    for g, group in enumerate(groups):
        if "25um" in group:
            row = 1
        elif "100um" in group:
            row = 2

        # Load group info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Sample evenly spaced color shades from base color
        # shades = pc.sample_colorscale(BASE_COLORS_IDE[g], list(np.linspace(0.3, 1, 6)))

        # Create array to save averaged data
        all_z = []
        t_ref = None

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            # Select color based on good/bad
            if sample in eng_failures:
                line_color = IDE_ENG_COLOR
                line_solid = IDE_ENG_SOLID
            elif sample in ide_failures:
                line_color = IDE_IDE_COLOR
                line_solid = IDE_IDE_SOLID
            elif sample in both_failures:
                line_color = IDE_BOTH_COLOR
                line_solid = IDE_BOTH_SOLID
            else:
                line_color = IDE_GOOD_COLOR
                line_solid = IDE_GOOD_SOLID

            # Open data summary
            df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Pull data
            z = df["Impedance Magnitude at 1000 Hz (ohms)"]
            temperature = df["Temperature (C)"]
            real_days = df["Real Days"]
            # real_days_df = pd.DataFrame(real_days)
            real_days = [a / 7 for a in real_days]  # convert to weeks
            # temperature = pd.DataFrame(temperature)
            # accel_days = calculate_accel_days(real_days_df, temperature)
            # accel_days = [a / 365.25 for a in accel_days]  # convert to years

            if sample == 'IDE-100-1':
                st = 101
                nd = 141
                z[st:nd] = [float('nan')]*(nd-st)

            # Save data to all_z for average and stddev calculation later
            if sample not in [eng_failures+ide_failures]:
                z_temp = [r / z[0] for r in z]

                # if sample == 'IDE-100-1':
                #     st = 101
                    # nd = 141
                    # z[st:nd] = [float('nan')]*(nd-st)

                if t_ref is None:
                    t_ref = np.asarray(real_days, dtype=float)
                    n_ref = len(t_ref)
                    all_z.append(z_temp)
                else:
                    # nearest indeces in time for each accel_days point
                    idx = np.array([np.argmin(np.abs(t_ref - d)) for d in real_days])
                    idx = np.clip(idx, 0, n_ref - 1)

                    z_aligned = np.full(n_ref, np.nan)

                    for i_idx, val in zip(idx, z_temp):
                        if np.isnan(z_aligned[i_idx]):
                            z_aligned[i_idx] = val
                        else:
                            pass

                    all_z.append(z_aligned)

            # Remove nans for plotting
            non_nans = [i for i, x in enumerate(z.tolist()) if not math.isnan(x)]
            # accel_days = [accel_days[i] for i in non_nans]
            real_days = [real_days[i] for i in non_nans]
            z = [z[i] for i in non_nans]

            # Normalize impedance to t=0
            z_norm = [r / z[0] for r in z]

            # Plot
            fig_z.add_trace(
                go.Scatter(
                    x=real_days,
                    y=z_norm,
                    mode="lines",
                    line=dict(width=IDE_SAMPLE_LINE_WIDTH, color=line_color, dash=line_solid),
                    name=sample,
                ),
                row=row,
                col=1,
            )

        # Once we're through all samples, calculate average
        Z = np.vstack(all_z)

        mean_z = np.nanmean(Z, axis=0)
        std_z = np.nanstd(Z, axis=0)

        mask = ~np.isnan(mean_z)

        t = t_ref[mask]
        mean_z = mean_z[mask]
        std_z = std_z[mask]

        # print(t)
        fig_z.add_trace(
            go.Scatter(
                x=t,
                y=mean_z,
                mode="lines",
                line=dict(width=IDE_AVG_LINE_WIDTH, color=IDE_AVG_COLOR, dash=IDE_AVG_SOLID),
                name="Mean"
            ),
            row=row,
            col=1,
        )

    real_days = np.asarray(real_days)
    temperature = np.asarray(temperature)
    mask = np.isfinite(real_days) & np.isfinite(temperature)
    avg_temp = (
        np.trapezoid(temperature[mask], x=real_days[mask])
        / (real_days[mask].max() - real_days[mask].min())
    )

    print(f'average temp ide: {avg_temp}')

    # # remove colons and parentheses from title to produce filename
    # file_title = title.replace(":", "")
    # file_title = title.replace("(", "")
    # file_title = title.replace(")", "")
    # fig_z.write_html(f"{PLOT_PATH}/{file_title}.html")

    # Show plot
    fig_z.show()

    return fig_z

# Plot Humidity Samples
fig_rh = plot_humidity_temp_real(
    HUMIDITY_GROUPS, 
    [0, 60], 
    [0, 60],
    [0, 75],
    [15, 30],
    [56, 68],
    [0, 61]
)

# Plot IDEs
fig_ide = plot_ide_good_notemp(
    IDE_GROUPS,
    [-3.8, 0.8],
    [0, 59],
    ['IDE-100-3', 'IDE-100-4'], #eng failures
    ['IDE-25-5', 'IDE-25-7', 'IDE-100-6'], #ide failures
    ['IDE-25-2', 'IDE-25-6', 'IDE-25-8', 'IDE-100-7', 'IDE-100-8'] #both
)