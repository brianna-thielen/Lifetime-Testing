import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import numpy as np
import datetime
import math
import statistics

from support_functions.support_functions import calculate_accel_days, calculate_accel_days_single

_Z_MEAN_UPPER_SCALAR = 80
_Z_MEAN_LOWER_SCALAR = 100
_Z_NORM_MEAN_UPPER_SCALAR = 10
_Z_NORM_MEAN_LOWER_SCALAR = 10
_Z_PLOT_SCALAR = 1.6

_CIC_MEAN_SCALAR = 50
_CIC_NORM_MEAN_SCALAR = 1.5
_CIC_PLOT_SCALAR = 5
_CIC_UPPER_LIMIT = 1e6
_CIC_NORM_UPPER_LIMIT = 1e6

# Base color options - add more if you ever want to plot more than 5 groups in one plot
BASE_COLORS = [
    pc.sequential.Blues,
    pc.sequential.Reds,
    pc.sequential.Greens,
    pc.sequential.Oranges,
    pc.sequential.Purples,
]

# Plotting is off by default. Turn on for debugging.
plot_on = False

def plot_z(groups, data_path, sample_info_path, plot_path, title, plot_norm=False, plot_flags=False):
    # Save all Z data to later find plot limits
    all_z = np.array([])
    all_z_norm = np.array([])

    # If you're plotting normalized data, include second column
    if plot_norm:
        fig_z = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Individual Electrodes",
                "Individual Electrodes (normalized to t=0)",
                "Average",
                "Average (normalized to t=0)"
            )
        )

        fig_z.update_yaxes(
            title_text="Impedance Magnitude Change (|Z| / |Z|<sub>0</sub>)",
            type="log", tick0=0, dtick=1, col=2,
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )
    else:
         fig_z = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Individual Electrodes"
                "Average"
            )
        )
         
    fig_z.update_layout(
            title_text=title,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    fig_z.update_xaxes(
            title_text="Accelerated Time (days)",
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )
    fig_z.update_yaxes(
            title_text="Impedance Magnitude (ohms)",
            type="log", tick0=0, dtick=1, col=1,
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )

    # Loop through groups
    for g, group in enumerate(groups):
        # Open group info
        with open(f"{sample_info_path}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Sample evenly spaced color shades from base color
        shades = pc.sample_colorscale(BASE_COLORS[g], list(np.linspace(0.2, 0.8, len(group_info["samples"]))))

        # Create arrays to save averaged data for the group
        z_average = []
        z_norm_average = []

        # Create array to save the last value for each sample and smallest last accel days (only updates if processing one group)
        z_last = []
        z_norm_last = []
        min_accel_days = [float('inf')] * len(groups)

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Pull z and save to z_average, z_last
            z = df["Impedance Magnitude at 1000 Hz (ohms)"]
            z_last_numpy = z.iloc[-1]
            z_last.append(float(z_last_numpy))
            temperature = df["Temperature (C)"]
            temperature = pd.DataFrame(temperature)
            real_days = df["Real Days"]
            real_days = pd.DataFrame(real_days)
            accel_days = calculate_accel_days(real_days, temperature)

            if min_accel_days[g] > max(accel_days): # want to save the smallest of last accel days so we don't oversell progress
                min_accel_days[g] = max(accel_days)

            # Remove nans for plotting
            non_nans = [i for i, x in enumerate(z.tolist()) if not math.isnan(x)]
            accel_days = [accel_days[i] for i in non_nans]
            real_days = real_days.loc[non_nans]
            z = [z[i] for i in non_nans]

            fig_z.add_trace(
                go.Scatter(
                    x=accel_days,
                    y=z,
                    mode="lines",
                    line=dict(width=1, color=shades[s]),
                    name=sample
                ),
                row=1,
                col=1,
            )

            # Save Z data
            all_z = np.concatenate([all_z, z])

            if len(z_average) == 0:
                z_average = z
            else:
                z_average = [a + b for a, b in zip(z_average, z)]

            # Plot z_norm and save to z_norm_average, z_norm_last
            if plot_norm:
                # Normalize impedance to t=0
                z_norm = [r / z[0] for r in z]
                z_norm_last.append(z_norm[-1])

                fig_z.add_trace(
                    go.Scatter(
                        x=accel_days,
                        y=z_norm,
                        mode="lines",
                        line=dict(width=1, color=shades[s]),
                        name=sample,
                        showlegend=False
                    ),
                    row=1,
                    col=2,
                )

                # Save Z norm data
                all_z_norm = np.concatenate([all_z_norm, z_norm])

                if len(z_norm_average) == 0:
                    z_norm_average = z_norm
                else:
                    z_norm_average = [a + b for a, b in zip(z_norm_average, z_norm)]

        # Once we're through all samples, calculate the average(s)
        z_average = [x / len(group_info["samples"]) for x in z_average]
        if plot_norm:
            z_norm_average = [x / len(group_info["samples"]) for x in z_norm_average]

        # Then plot them in the darkest color of the same color scale
        shade = pc.sample_colorscale(BASE_COLORS[g], [1.0])[0]

        fig_z.add_trace(
            go.Scatter(
                x=accel_days, #This will take accelerated days from the last sample - need to update this later to account for different accel days between samples within a group
                y=z_average,
                mode="lines",
                line=dict(width=1, color=shade),
                name=f"{group} Average"
            ),
            row=2,
            col=1,
        )

        if plot_norm:
            fig_z.add_trace(
                go.Scatter(
                    x=accel_days, #This will take accelerated days from the last sample - need to update this later to account for different accel days between samples within a group
                    y=z_norm_average,
                    mode="lines",
                    line=dict(width=1, color=shade),
                    name=f"{group} Average",
                    showlegend=False
                ),
                row=2,
                col=2,
            )
        
        # Plot flagged dates
        if plot_flags:
            flagged_dates_group = group_info["flagged_dates"]
            # for flag_g, date_g in flagged_dates_group.items():
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

                    add_vert_line(fig_z, 2, 2, accel_days_flag, f"{flag_g} ({sample} @ {round(accel_days_flag)} days)", shade, True)

    # Exclude Z outliers and find axis limits (extraneous outliers appeared when setup was moved)
    # first remove any instances of inf
    all_z = all_z[np.isfinite(all_z)]
    all_z_norm = all_z_norm[np.isfinite(all_z_norm)]

    mean_z = statistics.median(all_z)
    z_upper = mean_z * _Z_MEAN_UPPER_SCALAR
    z_lower = mean_z / _Z_MEAN_LOWER_SCALAR

    all_z_filtered = all_z[(all_z >= z_lower) & (all_z <= z_upper)]

    # for normalized, look at upper and lower limits separately
    vals_over_one = all_z_norm[all_z_norm > 1]
    z_norm_upper = vals_over_one.mean()*_Z_MEAN_UPPER_SCALAR if vals_over_one.size > 0 else 1.2

    vals_under_one = all_z_norm[all_z_norm < 1]
    z_norm_lower = vals_under_one.mean()/_Z_MEAN_LOWER_SCALAR if vals_under_one.size > 0 else 0.8

    all_z_norm_filtered = all_z_norm[(all_z_norm >= z_norm_lower) & (all_z_norm <= z_norm_upper)]

    # If everything gets filtered out, all points are near the mean
    if all_z_filtered.size == 0:
        all_z_filtered = all_z
    if all_z_norm_filtered.size == 0:
        all_z_norm_filtered = all_z_norm

    # Calculate ranges, checking for values <=0
    if min(all_z_filtered) <= 0:
        z_plot_range = [
            0, 
            math.log10(max(all_z_filtered))*_Z_PLOT_SCALAR
            ]
    else:
        z_plot_range = [
            math.log10(min(all_z_filtered))/_Z_PLOT_SCALAR, 
            math.log10(max(all_z_filtered))*_Z_PLOT_SCALAR
            ]
    
    if min(all_z_norm_filtered) <= 0:
        z_norm_plot_range = [
            0, 
            math.log10(max(all_z_norm_filtered))*_Z_PLOT_SCALAR
            ]
    else:
        z_norm_plot_range = [
            math.log10(min(all_z_norm_filtered))/_Z_PLOT_SCALAR, 
            math.log10(max(all_z_norm_filtered))*_Z_PLOT_SCALAR
            ]

    # Update y axes
    fig_z.update_yaxes(range=z_plot_range, row=1, col=1)
    fig_z.update_yaxes(range=z_plot_range, row=2, col=1)
    fig_z.update_yaxes(range=z_norm_plot_range, row=1, col=2)
    fig_z.update_yaxes(range=z_norm_plot_range, row=2, col=2)

    # remove colon from title to produce filename
    file_title = title.replace(":", "")
    fig_z.write_html(f"{plot_path}/{file_title}.html")

    if plot_on:
        fig_z.show()

    # min_accel_days is sorted by group - take the smallest value to not oversell progress
    min_accel_days = min(min_accel_days)

    return z_last, z_norm_last, min_accel_days

def plot_cic(groups, data_path, sample_info_path, plot_path, title, plot_norm=False, plot_flags=False):
    # Save all CIC data to later find plot limits
    all_cic = np.array([])
    all_cic_norm = np.array([])

    # If you're plotting normalized data, include second column
    if plot_norm:
        fig_cic = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Individual Electrodes",
                "Individual Electrodes (normalized to t=0)",
                "Average",
                "Average (normalized to t=0)"
            )
        )

        fig_cic.update_yaxes(
            title_text="Charge Injection Capacity Change (|CIC| / |CIC|<sub>0</sub>)",
            col=2,
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )

    else:
         fig_cic = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Individual Electrodes"
                "Average"
            )
        )

    fig_cic.update_layout(
            title_text=title,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    fig_cic.update_xaxes(
            title_text="Accelerated Time (days)",
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )
    fig_cic.update_yaxes(
            title_text="Impedance Magnitude (ohms)", 
            col=1,
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )

    # Loop through groups
    for g, group in enumerate(groups):
        # Open group info
        with open(f"{sample_info_path}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Sample evenly spaced color shades from base color
        shades = pc.sample_colorscale(BASE_COLORS[g], list(np.linspace(0.2, 0.9, len(group_info["samples"]))))

        # Create arrays to save averaged data for the group
        cic_average = []
        cic_norm_average = []

        # Create array to save the last value for each sample and smallest last accel days (only updates if processing one group)
        cic_last = []
        cic_norm_last = []
        min_accel_days = [float('inf')] * len(groups)

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Plot cic and save to cic_average, cic_last
            cic = df["Charge Injection Capacity @ 1000 us (uC/cm^2)"]
            idx = cic.last_valid_index()
            cic_last.append(cic[idx])
            temperature = df["Temperature (C)"]
            temperature = pd.DataFrame(temperature)
            real_days = df["Real Days"]
            real_days = pd.DataFrame(real_days)
            accel_days = calculate_accel_days(real_days, temperature)

            if min_accel_days[g] > max(accel_days): # want to save the smallest of last accel days so we don't oversell progress
                min_accel_days[g] = max(accel_days)

            # Remove nans for plotting
            non_nans = [i for i, x in enumerate(cic.tolist()) if not math.isnan(x)]
            accel_days = [accel_days[i] for i in non_nans]
            real_days = real_days.loc[non_nans]
            cic = [cic[i] for i in non_nans]

            fig_cic.add_trace(
                go.Scatter(
                    x=accel_days,
                    y=cic,
                    mode="lines",
                    line=dict(width=1, color=shades[s]),
                    name=sample,
                ),
                row=1,
                col=1,
            )

            # Save CIC data
            all_cic = np.concatenate([all_cic, cic])

            if len(cic_average) == 0:
                cic_average = cic
            else:
                cic_average = [a + b for a, b in zip(cic_average, cic)]

            # Plot cic_norm and save to cic_norm_average, cic_norm_last
            if plot_norm:
                # check if cic[0] is good
                if cic[0] == 0:
                    continue

                # Normalize cic to t=0
                cic_norm = [r / cic[0] for r in cic]
                cic_norm_last.append(cic_norm[-1])

                fig_cic.add_trace(
                    go.Scatter(
                        x=accel_days,
                        y=cic_norm,
                        mode="lines",
                        line=dict(width=1, color=shades[s]),
                        name=sample,
                        showlegend=False
                    ),
                    row=1,
                    col=2,
                )

                # Save CIC norm data
                all_cic_norm = np.concatenate([all_cic_norm, cic_norm])

                if len(cic_norm_average) == 0:
                    cic_norm_average = cic_norm
                else:
                    cic_norm_average = [a + b for a, b in zip(cic_norm_average, cic_norm)]
        
        # Once we're through all samples, calculate the average(s)
        cic_average = [x / len(group_info["samples"]) for x in cic_average]
        if plot_norm:
            cic_norm_average = [x / len(group_info["samples"]) for x in cic_norm_average]

        # Then plot them in the darkest color of the same color scale
        shade = pc.sample_colorscale(BASE_COLORS[g], [1.0])[0]

        fig_cic.add_trace(
            go.Scatter(
                x=accel_days, #This will take accelerated days from the last sample - need to update this later to account for different accel days between samples within a group
                y=cic_average,
                mode="lines",
                line=dict(width=1, color=shade),
                name=f"{group} Average"
            ),
            row=2,
            col=1,
        )

        if plot_norm:
            fig_cic.add_trace(
            go.Scatter(
                x=accel_days, #This will take accelerated days from the last sample - need to update this later to account for different accel days between samples within a group
                y=cic_norm_average,
                mode="lines",
                line=dict(width=1, color=shade),
                name=f"{group} Average",
                showlegend=False
            ),
            row=2,
            col=2,
        )
            
        # Plot flagged dates
        if plot_flags:
            flagged_dates_group = group_info["flagged_dates"]
            # for flag_g, date_g in flagged_dates_group.items():
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

                    add_vert_line(fig_cic, 2, 2, accel_days_flag, f"{flag_g} ({sample} @ {round(accel_days_flag)} days)", shade, True)

    # Exclude CIC outliers and find axis limits (extraneous outliers appeared when setup was moved)
    # first remove any instances of inf and any values <0
    all_cic = all_cic[np.isfinite(all_cic)]
    all_cic = all_cic[all_cic < _CIC_UPPER_LIMIT]
    all_cic = all_cic[all_cic >= 0]
    all_cic_norm = all_cic_norm[np.isfinite(all_cic_norm)]
    all_cic_norm = all_cic_norm[all_cic_norm < _CIC_NORM_UPPER_LIMIT]
    all_cic_norm = all_cic_norm[all_cic_norm >= 0]

    mean_cic = statistics.median(all_cic)
    cic_upper = mean_cic * _CIC_MEAN_SCALAR
    cic_lower = mean_cic / _CIC_MEAN_SCALAR

    all_cic_filtered = all_cic[(all_cic >= cic_lower) & (all_cic <= cic_upper)]

    # for normalized, look at upper and lower limits separately
    vals_over_one = all_cic_norm[all_cic_norm > 1]
    cic_norm_upper = vals_over_one.mean()*_CIC_NORM_MEAN_SCALAR if vals_over_one.size > 0 else 1.2

    vals_under_one = all_cic_norm[all_cic_norm < 1]
    cic_norm_lower = vals_under_one.mean()/_CIC_NORM_MEAN_SCALAR if vals_under_one.size > 0 else 0.8

    all_cic_norm_filtered = all_cic_norm[(all_cic_norm >= cic_norm_lower) & (all_cic_norm <= cic_norm_upper)]

    # If everything gets filtered out, all points are near the mean
    if all_cic_filtered.size == 0:
        all_cic_filtered = all_cic
    if all_cic_norm_filtered.size == 0:
        all_cic_norm_filtered = all_cic_norm

    # Calculate ranges, checking for values <=0
    if min(all_cic_filtered) <= 0:
        cic_plot_range = [
            0, 
            max(all_cic_filtered)*_CIC_PLOT_SCALAR
            ]
    else:
        cic_plot_range = [
            min(all_cic_filtered)/_CIC_PLOT_SCALAR, 
            max(all_cic_filtered)*_CIC_PLOT_SCALAR
            ]
    
    if min(all_cic_norm_filtered) <= 0:
        cic_norm_plot_range = [
            0, 
            max(all_cic_norm_filtered)*_CIC_PLOT_SCALAR
            ]
    else:
        cic_norm_plot_range = [
            min(all_cic_norm_filtered)/_CIC_PLOT_SCALAR, 
            max(all_cic_norm_filtered)*_CIC_PLOT_SCALAR
            ]

    # Update y axes
    fig_cic.update_yaxes(range=cic_plot_range, row=1, col=1)
    fig_cic.update_yaxes(range=cic_plot_range, row=2, col=1)
    fig_cic.update_yaxes(range=cic_norm_plot_range, row=1, col=2)
    fig_cic.update_yaxes(range=cic_norm_plot_range, row=2, col=2)

    # remove colon from title to produce filename
    file_title = title.replace(":", "")
    fig_cic.write_html(f"{plot_path}/{file_title}.html")

    if plot_on:
        fig_cic.show()

    # min_accel_days is sorted by group - take the smallest value to not oversell progress
    min_accel_days = min(min_accel_days)

    return cic_last, cic_norm_last, min_accel_days

def plot_rh(groups, data_path, sample_info_path, plot_path, title, plot_flags=False):
    fig_rh = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3]
    )

    fig_rh.update_layout(
            title_text=title,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    fig_rh.update_xaxes(
            title_text="Accelerated Time (days)",
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )
    fig_rh.update_yaxes(
            title_text="Relative Humidity (%)",
            row=1,
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )
    fig_rh.update_yaxes(
            title_text="Temperature (C)",
            row=2,
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )

    # Save all RH data and max temp to later find plot limits
    all_rh = np.array([])
    max_temp = 0

    # Loop through groups
    first_group = True
    for g, group in enumerate(groups):
        # Open group info
        with open(f"{sample_info_path}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Sample evenly spaced color shades from base color
        shades = pc.sample_colorscale(BASE_COLORS[g], list(np.linspace(0.4, 0.9, len(group_info["samples"]))))

        # Create array to save the last value for each sample and smallest last accel days (only updates if processing one group)
        rh_last = []
        min_accel_days = [float('inf')] * len(groups)

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Pull data
            rh = df["Relative Humidity (%)"]
            temperature = df["Temperature (C)"]
            temperature = pd.DataFrame(temperature)
            real_days = df["Real Days"]
            real_days = pd.DataFrame(real_days)
            accel_days = calculate_accel_days(real_days, temperature)

            # Save the smallest of the last accel days so we don't oversell progress
            if min_accel_days[g] > max(accel_days): # want to save the smallest of last accel days so we don't oversell progress
                min_accel_days[g] = max(accel_days)

            # Save the last rh value
            rh_last.append(list(rh)[-1])

            # Plot
            fig_rh.add_trace(
                go.Scatter(
                    x=accel_days,
                    y=rh,
                    mode="lines",
                    line=dict(width=1, color=shades[s]),
                    name=sample,
                    connectgaps=True
                ),
                row=1,
                col=1,
            )

            fig_rh.add_trace(
                go.Scatter(
                    x=accel_days,
                    y=temperature["Temperature (C)"],
                    mode="lines",
                    line=dict(width=1, color=shades[s]),
                    name=sample,
                    connectgaps=True,
                    showlegend=False
                ),
                row=2,
                col=1,
            )

            # Save rh data and max temp
            all_rh = np.concatenate([all_rh, rh])
            if max(temperature["Temperature (C)"]) > max_temp:
                max_temp = max(temperature["Temperature (C)"])
        
            # Plot flagged dates
            if plot_flags:
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

                        add_vert_line(fig_rh, 2, 1, accel_days_flag, f"{flag_g} ({sample} @ {round(accel_days_flag)} days)", shades[s], first_group)

            # After first group, we don't want to plot labels on vertical lines anymore
            first_group = False

    # Exclude rh outliers and find axis limits (extraneous outliers appeared when setup was moved)
    mean_rh = statistics.mean(all_rh)
    rh_upper = mean_rh * 1.5
    rh_lower = 0

    all_rh_filtered = all_rh[(all_rh >= rh_lower) & (all_rh <= rh_upper)]

    # If everything gets filtered out, all points are near the mean
    if all_rh_filtered.size == 0:
        all_rh_filtered = all_rh

    rh_plot_range = [0, max(all_rh_filtered)*1.3]

    # Update y axes
    fig_rh.update_yaxes(range=rh_plot_range, row=1, col=1)
    fig_rh.update_yaxes(range=[0, max_temp*1.3], row=2, col=1)

    # remove colon from title to produce filename
    file_title = title.replace(":", "")
    fig_rh.write_html(f"{plot_path}/{file_title}.html")

    if plot_on:
        fig_rh.show()

    # min_accel_days is sorted by group - take the smallest value to not oversell progress
    min_accel_days = min(min_accel_days)

    return rh_last, min_accel_days
    
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