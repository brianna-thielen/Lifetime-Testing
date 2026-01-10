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

HUMIDITY_GROUPS = ["LCP Encapsulation Capacitive Ambient", "LCP Encapsulation Capacitive Ambient", "LCP Encapsulation Capacitive", "LCP Encapsulation Resistive"]
HUMIDITY_AMBIENT_COLOR = 'rgb(180,180,180)'
HUMIDITY_AMBIENT_SOLID = 'dot'
HUMIDITY_R_COLOR = 'rgb(0,140,40)'
HUMIDITY_C_COLOR = 'rgb(100,40,170)'
HUMIDITY_SAMPLE_SOLID = 'solid'

IDE_GROUPS = ["LCP IDEs 25um", "LCP IDEs 100um"]
IDE_BAD_COLOR = 'rgba(200,200,200,0.5)'
IDE_GOOD_COLOR = 'rgb(80,160,200)'
IDE_AVG_COLOR = 'rgb(8,50,110)'

ELECTRODE_GROUPS= ["LCP Pt Grids"]

def add_vert_line(fig, rows, cols, xline, label, color, show_text):
    for r in range(1, rows+1):
        for c in range(1, cols+1):
            # draw the vertical line in subplot (r,c):
            fig.add_vline(
                x=xline,
                line_width=1,
                line_dash="dash",
                line_color=color,
                row=r,                 # target row
                col=c,                 # target column
            )

            # draw the vertical text next to it, rotated 90°
            if show_text:
                fig.add_annotation(
                    x=xline,
                    y=1,                       # top of that subplot
                    xref="x",                  # use that subplot’s x-axis
                    yref="y domain",              # [0…1] relative height of the subplot
                    text=f"<i>{label}</i>",
                    showarrow=False,
                    textangle=270,             # 90 + 180 = 270 (flipped)
                    xanchor="right",           # place text to the left of x=982.98
                    yanchor="top",             # align the top of the text at y=1
                    font=dict(size=10, color="black"),
                    row=r,
                    col=c,
                )

def plot_humidity_temp(groups, title, rh_range, max_temp, t_range, vert_lines=False):
    ambient_num = 1
    # Setup plot
    fig_rh = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            "IDEs built from 25 um thick LCP",
            "IDEs built from 100 um thick LCP",
        ),
    )

    fig_rh.update_layout(
        title_text=title,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    fig_rh.update_xaxes(
        title_text="Accelerated Time (years)", range=t_range,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )
    fig_rh.update_yaxes(
        title_text="Relative Humidity (%)", range=rh_range, row=1,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )
    fig_rh.update_yaxes(
        title_text="Temperature (C)", range=[0, max_temp], row=2,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )

    # Save all temperatures for average temp later
    all_temperatures = []

    for g, group in enumerate(groups):
        # Load group info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # # Sample evenly spaced color shades from base color
        # shades = pc.sample_colorscale(BASE_COLORS_RH[g], list(np.linspace(0.6, 0.9, len(group_info["samples"]))))

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            if "C-25" in sample:
                column = 1
            elif "C-100" in sample:
                column = 2

            if "-C-" in sample:
                shade = HUMIDITY_C_COLOR
                solid = HUMIDITY_SAMPLE_SOLID
            elif "-R-" in sample:
                shade = HUMIDITY_R_COLOR
                solid = HUMIDITY_SAMPLE_SOLID
            else:
                shade = HUMIDITY_AMBIENT_COLOR
                solid = HUMIDITY_AMBIENT_SOLID
            
            # Open data summary
            df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Pull data
            rh = df["Relative Humidity (%)"]
            temperature = df["Temperature (C)"]
            # Save temp for averaging if it's not the ambient group
            if "Ambient" not in group:
                all_temperatures.extend(temperature)
            temperature = pd.DataFrame(temperature)
            real_days = df["Real Days"]
            real_days = pd.DataFrame(real_days)
            accel_days = calculate_accel_days(real_days, temperature)
            accel_days = [a / 365.25 for a in accel_days]  # convert to years

            # If ambient, take moving average, and use average temp to scale real_days to artificial accel_days
            if "Ambient" in group:
                column = ambient_num
                ambient_num += 1
                rh = pd.Series(rh)
                rh = rh.rolling(window=4).mean()

                # Calculate average temperature and acceleration
                # all_temperatures = [t for t in all_temperatures if not math.isnan(t)]
                # avg_temp = sum(all_temperatures) / len(all_temperatures)
                avg_temp = 64.97070218185101  # Pre-calculated average so we can plot ambient first in the background
                temp_accel = 2 ** ((avg_temp - 37) / 10)
                real_days = real_days["Real Days"].tolist()
                accel_days = [d * temp_accel for d in real_days]
                accel_days = [a / 365.25 for a in accel_days]  # convert to years

                # # Also use lighter gray color
                # shades = pc.sample_colorscale(BASE_COLORS_RH[g], list(np.linspace(0.3, 0.9, len(group_info["samples"]))))


            # Plot
            fig_rh.add_trace(
                go.Scatter(
                    x=accel_days,
                    y=rh,
                    mode="lines",
                    line=dict(width=LINE_WIDTH, color=shade, dash=solid),
                    name=sample,
                    connectgaps=True
                ),
                row=1,
                col=column,
            )

            fig_rh.add_trace(
                go.Scatter(
                    x=accel_days,
                    y=temperature["Temperature (C)"],
                    mode="lines",
                    line=dict(width=LINE_WIDTH, color=shade, dash=solid),
                    name=sample,
                    connectgaps=True,
                    showlegend=False
                ),
                row=2,
                col=column,
            )

            # Plot flagged dates
            if vert_lines:
                flagged_dates_group = group_info["flagged_dates"]
                for flag_g, dates_g in flagged_dates_group.items():
                    # If there's only one date, put it in a list
                    if isinstance(dates_g, str):
                        dates_g = [dates_g]

                    # Loop through each date in the flag
                    for date_g in dates_g:
                        # Convert to datetime
                        date_g = datetime.datetime.strptime(date_g, "%Y-%m-%d %H:%M")

                        # Calculate real days
                        first_day = group_info["start_date"]
                        first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

                        days = date_g - first_day
                        days = days.total_seconds() / 24 / 60 / 60 # convert to days

                        # Calculate accelerated days
                        accel_days_flag = calculate_accel_days_single(days, real_days, accel_days)

                        add_vert_line(fig_rh, 2, 1, accel_days_flag, f"{flag_g} ({sample} @ {round(accel_days_flag)} days)", shade, True)

    # remove colons and parentheses from title to produce filename
    file_title = title.replace(":", "")
    file_title = title.replace("(", "")
    file_title = title.replace(")", "")
    fig_rh.write_html(f"{PLOT_PATH}/{file_title}.html")

    # Show plot
    fig_rh.show()

    return fig_rh

def plot_humidity(groups, title, rh_range, t_range, vert_lines=False):
    ambient_num = 1
    # Setup plot
    fig_rh = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Sensors Encapsulated in 25 um thick LCP",
            "Sensors Encapsulated in 100 um thick LCP",
        ),
    )

    fig_rh.update_layout(
        title_text=title,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    fig_rh.update_xaxes(
        title_text="Accelerated Time (years)", range=t_range,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )
    fig_rh.update_yaxes(
        title_text="Relative Humidity (%)", range=rh_range,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )

    # Save all temperatures for average temp later
    all_temperatures = []

    for g, group in enumerate(groups):
        # Load group info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # # Sample evenly spaced color shades from base color
        # shades = pc.sample_colorscale(BASE_COLORS_RH[g], list(np.linspace(0.6, 0.9, len(group_info["samples"]))))

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            if "C-25" in sample:
                column = 1
            elif "C-100" in sample:
                column = 2

            if "-C-" in sample:
                shade = HUMIDITY_C_COLOR
                solid = HUMIDITY_SAMPLE_SOLID
            elif "-R-" in sample:
                shade = HUMIDITY_R_COLOR
                solid = HUMIDITY_SAMPLE_SOLID
            else:
                shade = HUMIDITY_AMBIENT_COLOR
                solid = HUMIDITY_AMBIENT_SOLID
            
            # Open data summary
            df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Pull data
            rh = df["Relative Humidity (%)"]
            temperature = df["Temperature (C)"]
            # Save temp for averaging if it's not the ambient group
            if "Ambient" not in group:
                all_temperatures.extend(temperature)
            temperature = pd.DataFrame(temperature)
            real_days = df["Real Days"]
            real_days = pd.DataFrame(real_days)
            accel_days = calculate_accel_days(real_days, temperature)
            accel_days = [a / 365.25 for a in accel_days]  # convert to years

            # If ambient, take moving average, and use average temp to scale real_days to artificial accel_days
            if "Ambient" in group:
                column = ambient_num
                ambient_num += 1
                rh = pd.Series(rh)
                rh = rh.rolling(window=4).mean()

                # Calculate average temperature and acceleration
                # all_temperatures = [t for t in all_temperatures if not math.isnan(t)]
                # avg_temp = sum(all_temperatures) / len(all_temperatures)
                avg_temp = 64.97070218185101  # Pre-calculated average so we can plot ambient first in the background
                temp_accel = 2 ** ((avg_temp - 37) / 10)
                real_days_temp = real_days["Real Days"].tolist()
                accel_days = [d * temp_accel for d in real_days_temp]
                accel_days = [a / 365.25 for a in accel_days]  # convert to years

                # # Also use lighter gray color
                # shades = pc.sample_colorscale(BASE_COLORS_RH[g], list(np.linspace(0.3, 0.9, len(group_info["samples"]))))


            # Plot
            fig_rh.add_trace(
                go.Scatter(
                    x=accel_days,
                    y=rh,
                    mode="lines",
                    line=dict(width=LINE_WIDTH, color=shade, dash=solid),
                    name=sample,
                    connectgaps=True
                ),
                row=1,
                col=column,
            )

            # Plot flagged dates
            if vert_lines:
                flagged_dates_group = group_info["flagged_dates"]
                for flag_g, dates_g in flagged_dates_group.items():
                    # If there's only one date, put it in a list
                    if isinstance(dates_g, str):
                        dates_g = [dates_g]

                    # Loop through each date in the flag
                    for date_g in dates_g:
                        # Convert to datetime
                        date_g = datetime.datetime.strptime(date_g, "%Y-%m-%d %H:%M")

                        # Calculate real days
                        first_day = group_info["start_date"]
                        first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

                        days = date_g - first_day
                        days = days.total_seconds() / 24 / 60 / 60 # convert to days

                        # Calculate accelerated days
                        accel_days_flag = calculate_accel_days_single(days, real_days, accel_days)

                        add_vert_line(fig_rh, 1, 2, accel_days_flag, f"{flag_g} ({sample} @ {round(accel_days_flag)} days)", shade, True)

    # remove colons and parentheses from title to produce filename
    file_title = title.replace(":", "")
    file_title = title.replace("(", "")
    file_title = title.replace(")", "")
    fig_rh.write_html(f"{PLOT_PATH}/{file_title}.html")

    # Show plot
    fig_rh.show()

    return fig_rh

def plot_ide_group(groups, title, deltaz_range, ignore_samples, vert_lines=False):
    # Setup plot
    fig_z = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Individual IDEs (25 um thick LCP)",
            "Individual IDEs (100 um thick LCP)",
            "Average Values (25 um thick LCP)",
            "Average Values (100 um thick LCP)"
        ),
        x_title="Accelerated Time (years)",
        y_title="Impedance Magnitude Change (|Z|/|Z|_0)"
    )

    fig_z.update_layout(
        title_text=title,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig_z.update_yaxes(
        type="log", tick0=0, dtick=1, range=deltaz_range, 
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )
    fig_z.update_xaxes(
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )

    for g, group in enumerate(groups):
        if "25um" in group:
            column = 1
        elif "100um" in group:
            column = 2

        # Load group info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Sample evenly spaced color shades from base color
        shades = pc.sample_colorscale(BASE_COLORS_IDE[g], list(np.linspace(0.4, 0.9, len(group_info["samples"]))))

        # Create array to save averaged data
        z_norm_average = []
        all_z = []
        t_ref = None

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            # Skip ignored samples
            if sample in ignore_samples:
                continue

            # Open data summary
            df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Pull data
            z = df["Impedance Magnitude at 1000 Hz (ohms)"]
            temperature = df["Temperature (C)"]
            temperature = pd.DataFrame(temperature)
            real_days = df["Real Days"]
            real_days = pd.DataFrame(real_days)
            accel_days = calculate_accel_days(real_days, temperature)
            accel_days = [a / 365.25 for a in accel_days]  # convert to years

            # Save data to all_z for average and stddev calculation later
            z_temp = [r / z[0] for r in z]
            z_temp = np.asarray(z_temp, dtype=float)
            if t_ref is None:
                t_ref = np.asarray(accel_days, dtype=float)
                n_ref = len(t_ref)
                all_z.append(z_temp)
            else:
                # nearest indeces in time for each accel_days point
                idx = np.array([np.argmin(np.abs(t_ref - d)) for d in accel_days])
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
            accel_days = [accel_days[i] for i in non_nans]
            real_days = real_days.loc[non_nans]
            z = [z[i] for i in non_nans]

            # Normalize impedance to t=0
            z_norm = [r / z[0] for r in z]

            # Plot
            fig_z.add_trace(
                go.Scatter(
                    x=accel_days,
                    y=z_norm,
                    mode="lines",
                    line=dict(width=LINE_WIDTH, color=shades[s]),
                    name=sample,
                    showlegend=True
                ),
                row=1,
                col=column,
            )

            # Save z norm data for averaging
            if len(z_norm_average) == 0:
                z_norm_average = z_norm
            else:
                z_norm_average = [a + b for a, b in zip(z_norm_average, z_norm)]

            # Plot flagged dates
            if vert_lines:
                flagged_dates_group = group_info["flagged_dates"]
                for flag_g, dates_g in flagged_dates_group.items():
                    # If there's only one date, put it in a list
                    if isinstance(dates_g, str):
                        dates_g = [dates_g]

                    # Loop through each date in the flag
                    for date_g in dates_g:
                        # Convert to datetime
                        date_g = datetime.datetime.strptime(date_g, "%Y-%m-%d %H:%M")

                        # Calculate real days
                        first_day = group_info["start_date"]
                        first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

                        days = date_g - first_day
                        days = days.total_seconds() / 24 / 60 / 60 # convert to days

                        # Calculate accelerated days
                        accel_days_flag = calculate_accel_days_single(days, real_days, accel_days)

                        add_vert_line(fig_z, 2, 2, accel_days_flag, f"{flag_g} ({sample} @ {round(accel_days_flag)} days)", shades[s], True)

        # Once we're through all samples, calculate average
        z_norm_average = [x / len(group_info["samples"]) for x in z_norm_average]

        Z = np.vstack(all_z)

        mean_z = np.nanmean(Z, axis=0)
        std_z = np.nanstd(Z, axis=0)

        mask = ~np.isnan(mean_z)

        t = t_ref[mask]
        mean_z = mean_z[mask]
        std_z = std_z[mask]

        # Plot in the darkest shade
        shade = pc.sample_colorscale(BASE_COLORS_IDE[g], [1.0])[0]
        # fig_z.add_trace(
        #     go.Scatter(
        #         x=accel_days, #This will take accelerated days from the last sample - need to update this later to account for different accel days between samples within a group
        #         y=z_norm_average,
        #         mode="lines",
        #         line=dict(width=LINE_WIDTH, color=shade),
        #         name=f"{group} Average",
        #         showlegend=True
        #     ),
        #     row=2,
        #     col=column,
        # )
        fig_z.add_trace(
            go.Scatter(
                x=t,
                y=mean_z+std_z,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            row=2,
            col=column,
        )
        fig_z.add_trace(
            go.Scatter(
                x=t,
                y=mean_z-std_z,
                mode="lines",
                fill="tonexty",
                fillcolor="lightgrey",
                line=dict(width=0),
                name="+/- 1 SD",
            ),
            row=2,
            col=column,
        )
        fig_z.add_trace(
            go.Scatter(
                x=t,
                y=mean_z,
                mode="lines",
                line=dict(width=1, color=shade),
                name="Mean"
            ),
            row=2,
            col=column,
        )

    # remove colons and parentheses from title to produce filename
    file_title = title.replace(":", "")
    file_title = title.replace("(", "")
    file_title = title.replace(")", "")
    fig_z.write_html(f"{PLOT_PATH}/{file_title}.html")

    # Show plot
    fig_z.show()

    return fig_z

def plot_ide_good(groups, title, deltaz_range, t_range, bad_samples, vert_lines=False):
    # Setup plot
    fig_z = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "IDEs Laminated with 25 um thick LCP",
            "IDEs Laminated with 100 um thick LCP",
        )
    )

    fig_z.update_layout(
        title_text=title,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig_z.update_yaxes(
        title_text="Impedance Magnitude Change (|Z| / |Z|<sub>0</sub>)",
        type="log", tick0=0, dtick=1, range=deltaz_range, 
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )
    fig_z.update_xaxes(
        title_text="Accelerated Time (years)", range=t_range,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )

    for g, group in enumerate(groups):
        if "25um" in group:
            column = 1
        elif "100um" in group:
            column = 2

        # Load group info
        with open(f"{SAMPLE_INFORMATION_PATH}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Sample evenly spaced color shades from base color
        shades = pc.sample_colorscale(BASE_COLORS_IDE[g], list(np.linspace(0.3, 1, 6)))

        # Create array to save averaged data
        all_z = []
        t_ref = None

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            # Select color based on good/bad
            if sample in bad_samples:
                line_color = IDE_BAD_COLOR
            else:
                line_color = IDE_GOOD_COLOR

            # Open data summary
            df = pd.read_csv(f"{DATA_PATH}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Pull data
            z = df["Impedance Magnitude at 1000 Hz (ohms)"]
            temperature = df["Temperature (C)"]
            temperature = pd.DataFrame(temperature)
            real_days = df["Real Days"]
            real_days = pd.DataFrame(real_days)
            accel_days = calculate_accel_days(real_days, temperature)
            accel_days = [a / 365.25 for a in accel_days]  # convert to years

            # Save data to all_z for average and stddev calculation later
            if sample not in bad_samples:
                z_temp = [r / z[0] for r in z]
                z_temp = np.asarray(z_temp, dtype=float)
                if t_ref is None:
                    t_ref = np.asarray(accel_days, dtype=float)
                    n_ref = len(t_ref)
                    all_z.append(z_temp)
                else:
                    # nearest indeces in time for each accel_days point
                    idx = np.array([np.argmin(np.abs(t_ref - d)) for d in accel_days])
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
            accel_days = [accel_days[i] for i in non_nans]
            real_days = real_days.loc[non_nans]
            z = [z[i] for i in non_nans]

            # Normalize impedance to t=0
            z_norm = [r / z[0] for r in z]

            # Plot
            fig_z.add_trace(
                go.Scatter(
                    x=accel_days,
                    y=z_norm,
                    mode="lines",
                    line=dict(width=1, color=line_color),
                    name=sample,
                    showlegend=True
                ),
                row=1,
                col=column,
            )

            # Plot flagged dates
            if vert_lines:
                flagged_dates_group = group_info["flagged_dates"]
                for flag_g, dates_g in flagged_dates_group.items():
                    # If there's only one date, put it in a list
                    if isinstance(dates_g, str):
                        dates_g = [dates_g]

                    # Loop through each date in the flag
                    for date_g in dates_g:
                        # Convert to datetime
                        date_g = datetime.datetime.strptime(date_g, "%Y-%m-%d %H:%M")

                        # Calculate real days
                        first_day = group_info["start_date"]
                        first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

                        days = date_g - first_day
                        days = days.total_seconds() / 24 / 60 / 60 # convert to days

                        # Calculate accelerated days
                        accel_days_flag = calculate_accel_days_single(days, real_days, accel_days)

                        add_vert_line(fig_z, 2, 2, accel_days_flag, f"{flag_g} ({sample} @ {round(accel_days_flag)} days)", shades[s], True)

        # Once we're through all samples, calculate average
        Z = np.vstack(all_z)

        mean_z = np.nanmean(Z, axis=0)
        std_z = np.nanstd(Z, axis=0)

        mask = ~np.isnan(mean_z)

        t = t_ref[mask]
        mean_z = mean_z[mask]
        std_z = std_z[mask]

        # Plot in the darkest shade
        # fig_z.add_trace(
        #     go.Scatter(
        #         x=t,
        #         y=mean_z+std_z,
        #         mode="lines",
        #         line=dict(width=0),
        #         showlegend=False,
        #     ),
        #     row=1,
        #     col=column,
        # )
        # fig_z.add_trace(
        #     go.Scatter(
        #         x=t,
        #         y=mean_z-std_z,
        #         mode="lines",
        #         fill="tonexty",
        #         fillcolor='rgba(100,100,100,0.5)',
        #         line=dict(width=0),
        #         name="+/- 1 SD",
        #     ),
        #     row=1,
        #     col=column,
        # )
        fig_z.add_trace(
            go.Scatter(
                x=t,
                y=mean_z,
                mode="lines",
                line=dict(width=2, color=IDE_AVG_COLOR),
                name="Mean"
            ),
            row=1,
            col=column,
        )

    # remove colons and parentheses from title to produce filename
    file_title = title.replace(":", "")
    file_title = title.replace("(", "")
    file_title = title.replace(")", "")
    fig_z.write_html(f"{PLOT_PATH}/{file_title}.html")

    # Show plot
    fig_z.show()

    return fig_z

# Plot Humidity Samples
# fig_rh = plot_humidity_temp(
#     HUMIDITY_GROUPS, 
#     "Relative Humidity and Temperature vs Accelerated Time for Encapsulated RH Sensors", 
#     [0, 75], 
#     75,
#     [0, 8.1],
#     False
# )

# Plot Humidity Samples
fig_rh = plot_humidity(
    HUMIDITY_GROUPS, 
    "Relative Humidity vs Accelerated Time for Encapsulated RH Sensors", 
    [0, 75], 
    [0, 8.1],
    False
)

# Plot IDEs
# fig_ide = plot_ide_group(
#     IDE_GROUPS,
#     "Impedance Maginitude Change vs Accelerated Time for LCP IDEs",
#     [-3, 3],
#     [],
#     False
# )

# Plot IDEs (good data)
fig_ide = plot_ide_good(
    IDE_GROUPS,
    "Impedance Magnitude Change vs Accelerated Time for Laminated Interdigitated Electrodes",
    [-3, 1.5],
    [0, 9],
    ['IDE-25-2', 'IDE-25-5', 'IDE-25-6', 'IDE-25-7','IDE-25-8', 'IDE-100-3', 'IDE-100-4', 'IDE-100-6', 'IDE-100-7', 'IDE-100-8'],
    False
)



