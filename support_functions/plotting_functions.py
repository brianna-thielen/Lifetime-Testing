import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import numpy as np
import datetime
import math
import statistics
import warnings

from support_functions.support_functions import calculate_accel_days, calculate_accel_days_single

PERCENTAGE_FOR_SCALING_ELECTRODE_DATA = 0.5 # percentage of data to exclude on either end when calculating plot limits (e.g. 0.5 = 0.5th and 99.5th percentiles)
PERCENTAGE_FOR_SCALING_RH_DATA = 0 # percentage of data to exclude on either end when calculating plot limits (e.g. 0.5 = 0.5th and 99.5th percentiles)

# Base color options - add more if you ever want to plot more than 5 groups in one plot
# 03 Feb 2026 - Multiple group plotting has been removed for now to simplify the analysis code
BASE_COLORS = [
    pc.sequential.Blues,
    pc.sequential.Reds,
    pc.sequential.Greens,
    pc.sequential.Oranges,
    pc.sequential.Purples,
]

# Plotting is off by default. Turn on for debugging.
plot_on = False

def plot_z(group, data_path, sample_info_path, plot_path, title, plot_flags=False):
    y_axis_label = "Impedance Magnitude (ohms)"
    y_norm_axis_label = "Impedance Magnitude Change (|Z| / |Z|<sub>0</sub>)"
    y_axis_type = "log"
    data_column = "Impedance Magnitude at 1000 Hz (ohms)"
    
    fig = setup_electrode_plot(y_axis_label, y_axis_type, y_norm_axis_label)

    z_last, z_norm_last, accel_days_last = plot_electrode_data(fig, group, data_path, sample_info_path, plot_path, title, y_axis_type, data_column, plot_flags)

    return z_last, z_norm_last, accel_days_last

def plot_cic(group, data_path, sample_info_path, plot_path, title, plot_flags=False):
    y_axis_label = "Charge Injection Capacity (uC/cm<sup>2</sup>)"
    y_norm_axis_label = "Charge Injection Capacity Change (|CIC| / |CIC|<sub>0</sub>)"
    y_axis_type = "linear"
    data_column = "Charge Injection Capacity @ 1000 us (uC/cm^2)"

    fig = setup_electrode_plot(y_axis_label, y_axis_type, y_norm_axis_label)
    
    cic_last, cic_norm_last, accel_days_last = plot_electrode_data(fig, group, data_path, sample_info_path, plot_path, title, y_axis_type, data_column, plot_flags)

    return cic_last, cic_norm_last, accel_days_last

def setup_electrode_plot(y_axis_label, y_axis_type, y_norm_axis_label):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Individual Electrodes",
            "Individual Electrodes (normalized to t=0)",
            "Average",
            "Average (normalized to t=0)"
        )
    )

    fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    fig.update_xaxes(
            title_text="Time (weeks)",
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )
    fig.update_yaxes(
            title_text=y_axis_label,
            type=y_axis_type,
            col=1,
            showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
        )
    fig.update_yaxes(
        title_text=y_norm_axis_label,
        type="log",
        col=2,
        showline=True, linewidth=2, linecolor='black', ticks='outside', tickcolor='black',tickwidth=2, gridcolor='lightgrey'
    )

    return fig

def plot_electrode_data(fig, group, data_path, sample_info_path, plot_path, title, y_axis_type, data_column, plot_flags=False):
    # Open group info
    with open(f"{sample_info_path}/{group}.json", 'r') as f:
        group_info = json.load(f)

    # Sample evenly spaced color shades from base color
    shades = pc.sample_colorscale(BASE_COLORS[0], list(np.linspace(0.2, 0.9, len(group_info["samples"]))))

    # Create arrays to save averaged data for the group
    all_data = []
    all_data_norm = []
    t_ref = None

    # Create array to save the last value for each sample and smallest last accel days (only updates if processing one group)
    data_last = [float('nan')] * len(group_info["samples"])
    data_norm_last = [float('nan')] * len(group_info["samples"])

    # Loop through each sample
    for s, sample in enumerate(group_info["samples"]):
        # Open data summary
        df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
        df = df.drop(columns=["Unnamed: 0"], axis=1)

        # Load data
        data = df[data_column]
        temperature = df["Temperature (C)"]
        real_days = df["Real Days"]
        real_weeks = [d/7 for d in real_days] # convert to weeks
        
        # Set any infinite values to nan and any negative values to zero
        data = data.replace([np.inf], np.nan)
        data = data.clip(lower=0)

        # Find initial value for normalization
        data_0 = float('nan')
        for value in data:
            if not math.isnan(value):
                data_0 = value
                break

        # Calculate normalized data
        if data_0 == 0: # if first value is zero, use nans
            data_norm = [float('nan')] * len(data)
        else:
            data_norm = [d / data_0 for d in data] # normalized data

        # Set last values
        for value in reversed(data):
            if not math.isnan(value):
                data_last[s] = value
                break

        for value in reversed(data_norm):
            if not math.isnan(value):
                data_norm_last[s] = value
                break

        # Set reference time and save data for averaging
        if t_ref is None:
            t_ref = np.asarray(real_weeks, dtype=float)
            n_ref = len(t_ref)
            all_data.append(data)
            all_data_norm.append(data_norm)
        else:
            # use nearest indeces in time for each time point
            idx = np.array([np.argmin(np.abs(t_ref - d)) for d in real_weeks])
            idx = np.clip(idx, 0, n_ref - 1)

            data_aligned = np.full(n_ref, np.nan)
            data_norm_aligned = np.full(n_ref, np.nan)

            for i_idx, val in zip(idx, data):
                if np.isnan(data_aligned[i_idx]):
                    data_aligned[i_idx] = val
                else:
                    pass
            for i_idx, val in zip(idx, data_norm):
                if np.isnan(data_norm_aligned[i_idx]):
                    data_norm_aligned[i_idx] = val
                else:
                    pass

            all_data.append(data_aligned)
            all_data_norm.append(data_norm_aligned)

        # Remove nans for plotting
        non_nans = [i for i, x in enumerate(data.tolist()) if not math.isnan(x)]
        real_weeks = [real_weeks[i] for i in non_nans]
        data = [data[i] for i in non_nans]
        data_norm = [data_norm[i] for i in non_nans]

        # Plot data
        fig.add_trace(
            go.Scatter(
                # x=accel_days,
                x=real_weeks,
                y=data,
                mode="lines",
                line=dict(width=1, color=shades[s]),
                name=sample,
                legendgroup=sample,
                showlegend=True
            ),
            row=1,
            col=1,
        )

        # Plot cic_norm
        fig.add_trace(
            go.Scatter(
                x=real_weeks,
                y=data_norm,
                mode="lines",
                line=dict(width=1, color=shades[s]),
                name=sample,
                legendgroup=sample,
                showlegend=False
            ),
            row=1,
            col=2,
        )

    # Once we're through all samples, calculate the average(s)
    DATA = np.vstack(all_data)
    DATA_NORM = np.vstack(all_data_norm)

    # If data for any samples is all nans, suppress the warning (common for broken electrodes - it won't plot that norm data for that sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mean_data = np.nanmean(DATA, axis=0)
        std_data = np.nanstd(DATA, axis=0)

        mean_data_norm = np.nanmean(DATA_NORM, axis=0)
        std_data_norm = np.nanstd(DATA_NORM, axis=0)

    mask = ~np.isnan(mean_data)

    t = t_ref[mask]

    mean_data = mean_data[mask]
    std_data = std_data[mask]
    meanplussd_data = [m + s for m, s in zip(mean_data, std_data)]
    meanminussd_data = [m - s for m, s in zip(mean_data, std_data)]
    meanminussd_data = [max(0, c) for c in meanminussd_data] # don't go below zero

    mean_data_norm = mean_data_norm[mask]
    std_data_norm = std_data_norm[mask]
    meanplussd_data_norm = [m + s for m, s in zip(mean_data_norm, std_data_norm)]
    meanminussd_data_norm = [m - s for m, s in zip(mean_data_norm, std_data_norm)]
    meanminussd_data_norm = [max(0, c) for c in meanminussd_data_norm] # don't go below zero
        
    # Then plot mean +/- SD
    shade = pc.sample_colorscale(BASE_COLORS[0], [1.0])[0]

    fig.add_trace(
        go.Scatter(
            x=t,
            y=mean_data,
            mode="lines",
            line=dict(width=1, color=shade),
            name=f"Average",
            legendgroup=f"Average",
            showlegend=True
        ),
        row=2,
        col=1,
    )
        
    fig.add_trace(
        go.Scatter(
            x=t,
            y=meanplussd_data,
            mode="lines",
            line=dict(width=1, color='lightgrey', dash='dot'),
            name=f"Average ± 1 SD",
            legendgroup=f"Average ± 1 SD",
            showlegend=True
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=meanminussd_data,
            mode="lines",
            line=dict(width=1, color='lightgrey', dash='dot'),
            name=f"Average ± 1 SD",
            legendgroup=f"Average ± 1 SD",
            showlegend=False
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=mean_data_norm,
            mode="lines",
            line=dict(width=1, color=shade),
            name=f"Normalized Average",
            legendgroup=f"Normalized Average",
            showlegend=True
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=meanplussd_data_norm,
            mode="lines",
            line=dict(width=1, color='lightgrey', dash='dot'),
            name=f"Normalized Average ± 1 SD",
            legendgroup=f"Normalized Average ± 1 SD",
            showlegend=True
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=meanminussd_data_norm,
            mode="lines",
            line=dict(width=1, color='lightgrey', dash='dot'),
            name=f"{group} NormalizedAverage ± 1 SD",
            legendgroup=f"Normalized Average ± 1 SD",
            showlegend=False
        ),
        row=2,
        col=2,
    )
        
    # Plot flagged dates
    if plot_flags:
        # Need to re-load data to calculate accel_days
        t_temp = df["Temperature (C)"]
        rd_temp = df["Real Days"]
        ad_temp = calculate_accel_days(pd.DataFrame(rd_temp), pd.DataFrame(t_temp))

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
                accel_days_flag = calculate_accel_days_single(days, rd_temp, ad_temp)

                add_vert_line(fig, 2, 2, accel_days_flag, f"{flag_g} ({sample} @ {round(accel_days_flag)} days)", shade, True)

    # Calculate total accelerated time and average temperature from the last set of processed data
    temperature = df["Temperature (C)"]
    real_days = df["Real Days"]
    accel_days = calculate_accel_days(pd.DataFrame(real_days), pd.DataFrame(temperature))

    total_accel_days = max(accel_days)
    total_accel_weeks = total_accel_days / 7
    total_accel_years = total_accel_days / 365.25

    total_real_days = max(real_days)
    total_real_weeks = total_real_days / 7 
    total_real_years = total_real_days / 365.25

    temperature = np.asarray(temperature)

    mask = np.isfinite(real_days) & np.isfinite(temperature)
    avg_temp = (
        np.trapezoid(temperature[mask], x=real_days[mask])
        / (real_days[mask].max() - real_days[mask].min())
    )

    # If we're less than 10 weeks, use 1 decimal, otherwise no decimals
    if total_real_weeks < 10:
        print('rounding')
        total_accel_weeks = round(total_accel_weeks, 1)
        total_real_weeks = round(total_real_weeks, 1)
    else:
        total_accel_weeks = round(total_accel_weeks)
        total_real_weeks = round(total_real_weeks)

    avg_temp = round(avg_temp, 2)
    
    # Update labels for real time
    # If we're under 10 weeks, use 1 decimal
    if total_real_weeks < 10:
        total_real_weeks = round(total_real_weeks, 1)
        real_label = f"{total_real_weeks} weeks"
    # If we're under a year, use no decimals
    elif total_real_weeks < 52:
        total_real_weeks = round(total_real_weeks)
        real_label = f"{total_real_weeks} weeks"
    # If we're over a year, use years with 1 decimal
    else:
        total_real_years = round(total_real_years, 1)
        real_label = f"{total_real_years} years"

    # Update labels for accelerated time
    # If we're under 10 weeks, use 1 decimal
    if total_accel_weeks < 10:
        total_accel_weeks = round(total_accel_weeks, 1)
        accel_label = f"{total_accel_weeks} weeks"
    # If we're under a year, use no decimals
    elif total_accel_weeks < 52:
        total_accel_weeks = round(total_accel_weeks)
        accel_label = f"{total_accel_weeks} weeks"
    # If we're over a year, use years with 1 decimal
    else:
        total_accel_years = round(total_accel_years, 1)
        accel_label = f"{total_accel_years} years"

    # Update plot title
    fig.update_layout(title_text=f"{title} - {real_label} at average {avg_temp} \u00b0C, equivalent to {accel_label} at 37 \u00b0C")

    # Find axis limits, excluding extraneous outliers
    y = DATA.ravel()
    y_norm = DATA_NORM.ravel()

    # For all plot types, use percentile to calculate the maximum range for absolute values, and the full range for normalized values
    y_max = np.nanpercentile(y, 100-PERCENTAGE_FOR_SCALING_ELECTRODE_DATA)
    y_norm_min = np.nanpercentile(y_norm, PERCENTAGE_FOR_SCALING_ELECTRODE_DATA)
    y_norm_max = np.nanpercentile(y_norm, 100-PERCENTAGE_FOR_SCALING_ELECTRODE_DATA)

    # For linear plots, set the minimum for absolute data to zero
    if y_axis_type == "linear":
        y_min = 0
    # For log plots, set the minimum for absolute data from the percentile calculation, and convert these limits to log scale
    elif y_axis_type == "log":
        y_min = np.nanpercentile(y, PERCENTAGE_FOR_SCALING_ELECTRODE_DATA)

        y_min = math.log10(y_min) if y_min > 0 else -3
        y_max = math.log10(y_max) if y_max > 0 else -3
    
    # For all normalized plots (which are log plots), convert limits to log scale
    y_norm_max = math.log10(y_norm_max) if y_norm_max > 0 else -3
    y_norm_min = math.log10(y_norm_min) if y_norm_min > 0 else -3

    # Update y axes
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    fig.update_yaxes(range=[y_min, y_max], row=2, col=1)
    fig.update_yaxes(range=[y_norm_min, y_norm_max], row=1, col=2)
    fig.update_yaxes(range=[y_norm_min, y_norm_max], row=2, col=2)

    # remove colon from title to produce filename
    file_title = title.replace(":", "")
    fig.write_html(f"{plot_path}/{file_title}.html")

    if plot_on:
        fig.show()

    return data_last, data_norm_last, total_accel_days

def plot_rh(group, data_path, sample_info_path, plot_path, title, plot_flags=False):
    fig_rh = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            "Relative Humidity",
            "Temperature"
        )
    )

    fig_rh.update_layout(
            title_text=title,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    fig_rh.update_xaxes(
            title_text="Time (weeks)",
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
    
    # Open group info
    with open(f"{sample_info_path}/{group}.json", 'r') as f:
        group_info = json.load(f)

    # Sample evenly spaced color shades from base color
    shades = pc.sample_colorscale(BASE_COLORS[0], list(np.linspace(0.4, 0.9, len(group_info["samples"]))))

    # Create array to save the last value for each sample and smallest last accel days (only updates if processing one group)
    rh_last = [float('nan')] * len(group_info["samples"])

    # Loop through each sample
    for s, sample in enumerate(group_info["samples"]):
        # Open data summary
        df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
        df = df.drop(columns=["Unnamed: 0"], axis=1)

        # Pull data
        rh = df["Relative Humidity (%)"]
        temperature = df["Temperature (C)"]
        real_days = df["Real Days"]
        real_weeks = [d/7 for d in real_days] # convert to weeks

        # Set any infinite values to nan and any negative values to zero
        rh = rh.replace([np.inf], np.nan)
        rh = rh.clip(lower=0)

        # Save the last rh value
        for value in reversed(rh):
            if not math.isnan(value):
                rh_last[s] = value
                break

        # Remove nans for plotting
        non_nans = [i for i, x in enumerate(rh.tolist()) if not math.isnan(x)]
        real_weeks = [real_weeks[i] for i in non_nans]
        rh = [rh[i] for i in non_nans]

        # Plot
        fig_rh.add_trace(
            go.Scatter(
                x=real_weeks,
                y=rh,
                mode="lines",
                line=dict(width=1, color=shades[s]),
                name=sample,
                legendgroup=sample,
                connectgaps=True
            ),
            row=1,
            col=1,
        )

        fig_rh.add_trace(
            go.Scatter(
                x=real_weeks,
                y=temperature,
                mode="lines",
                line=dict(width=1, color=shades[s]),
                name=sample,
                legendgroup=sample,
                connectgaps=True,
                showlegend=False
            ),
            row=2,
            col=1,
        )

        # Save rh data and max temp
        all_rh = np.concatenate([all_rh, rh])
        if max(temperature) > max_temp:
            max_temp = max(temperature)
    
        # Plot flagged dates
        if plot_flags:
            # Need to re-load data to calculate accel_days
            t_temp = df["Temperature (C)"]
            rd_temp = df["Real Days"]
            ad_temp = calculate_accel_days(pd.DataFrame(rd_temp), pd.DataFrame(t_temp))

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
                    accel_days_flag = calculate_accel_days_single(days, rd_temp, ad_temp)

                    add_vert_line(fig_rh, 2, 1, accel_days_flag, f"{flag_g} ({sample} @ {round(accel_days_flag)} days)", shades[s], True)

    # Calculate total accelerated time and average temperature from the last set of processed data
    temperature = df["Temperature (C)"]
    real_days = df["Real Days"]
    accel_days = calculate_accel_days(pd.DataFrame(real_days), pd.DataFrame(temperature))

    total_accel_days = max(accel_days)
    total_accel_weeks = total_accel_days / 7
    total_accel_years = total_accel_days / 365.25

    total_real_days = max(real_days)
    total_real_weeks = total_real_days / 7 
    total_real_years = total_real_days / 365.25

    temperature = np.asarray(temperature)

    mask = np.isfinite(real_days) & np.isfinite(temperature)
    avg_temp = (
        np.trapezoid(temperature[mask], x=real_days[mask])
        / (real_days[mask].max() - real_days[mask].min())
    )

    # If we're less than 10 weeks, use 1 decimal, otherwise no decimals
    if total_real_weeks < 10:
        total_accel_weeks = round(total_accel_weeks, 1)
        total_real_weeks = round(total_real_weeks, 1)
    else:
        total_accel_weeks = round(total_accel_weeks)
        total_real_weeks = round(total_real_weeks)

    avg_temp = round(avg_temp, 2)
    
    # Update labels for real time
    # If we're under 10 weeks, use 1 decimal
    if total_real_weeks < 10:
        total_real_weeks = round(total_real_weeks, 1)
        real_label = f"{total_real_weeks} weeks"
    # If we're under a year, use no decimals
    elif total_real_weeks < 52:
        total_real_weeks = round(total_real_weeks)
        real_label = f"{total_real_weeks} weeks"
    # If we're over a year, use years with 1 decimal
    else:
        total_real_years = round(total_real_years, 1)
        real_label = f"{total_real_years} years"

    # Update labels for accelerated time
    # If we're under 10 weeks, use 1 decimal
    if total_accel_weeks < 10:
        total_accel_weeks = round(total_accel_weeks, 1)
        accel_label = f"{total_accel_weeks} weeks"
    # If we're under a year, use no decimals
    elif total_accel_weeks < 52:
        total_accel_weeks = round(total_accel_weeks)
        accel_label = f"{total_accel_weeks} weeks"
    # If we're over a year, use years with 1 decimal
    else:
        total_accel_years = round(total_accel_years, 1)
        accel_label = f"{total_accel_years} years"

    # Update plot title
    fig_rh.update_layout(title_text=f"{title} - {real_label} at average {avg_temp} \u00b0C, equivalent to {accel_label} at 37 \u00b0C")

    # Find axis limits, excluding extraneous outliers
    # For RH, use percentile to calculate maximum, minimum is zero
    rh_y_max = np.nanpercentile(all_rh, 100-PERCENTAGE_FOR_SCALING_RH_DATA)
    rh_y_min = 0

    # For temp, use percentile to calculate maximum and minimum
    t_y_max = np.nanpercentile(temperature, 100-PERCENTAGE_FOR_SCALING_RH_DATA)
    t_y_min = np.nanpercentile(temperature, PERCENTAGE_FOR_SCALING_RH_DATA)

    # Update y axes
    fig_rh.update_yaxes(range=[rh_y_min, rh_y_max], row=1, col=1)
    fig_rh.update_yaxes(range=[t_y_min, t_y_max], row=2, col=1)

    # remove colon from title to produce filename
    file_title = title.replace(":", "")
    fig_rh.write_html(f"{plot_path}/{file_title}.html")

    if plot_on:
        fig_rh.show()

    return rh_last, total_accel_days
    
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