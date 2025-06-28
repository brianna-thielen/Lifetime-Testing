import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import numpy as np
import datetime

from support_functions.support_functions import calculate_accel_days

# Base color options - add more if you ever want to plot more than 5 groups in one plot
BASE_COLORS = [
    pc.sequential.Blues,
    pc.sequential.Reds,
    pc.sequential.Greens,
    pc.sequential.Oranges,
    pc.sequential.Purples,
]

# Turning plots on for debugging
plot_on = True

def plot_z(groups, data_path, sample_info_path, plot_path, title, norm=False):
    # If you're plotting normalized data, include second column
    if norm:
        fig_z = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Individual Electrodes",
                "Individual Electrodes (normalized to t=0)",
                "Average",
                "Average (normalized to t=0)"
            ),
            x_title="Accelerated Time (days)",
		    y_title="Impedance Magnitude (ohms)"
        )

        fig_z.update_yaxes(type="log", tick0=0, dtick=1, range=[-3, 3], row=1, col=2)
        fig_z.update_yaxes(type="log", tick0=0, dtick=1, range=[-3, 3], row=2, col=2)
    else:
         fig_z = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Individual Electrodes"
                "Average"
            ),
            x_title="Accelerated Time (days)",
            y_title="Impedance Magnitude (ohms)"
        )

    fig_z.update_layout(title_text=title)
    fig_z.update_yaxes(type="log", tick0=100, dtick=1, range=[2, 8], row=1, col=1)
    fig_z.update_yaxes(type="log", tick0=100, dtick=1, range=[2, 8], row=2, col=1)

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
        min_accel_days = float('inf')

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Plot z and save to z_average, z_last
            z = df["Impedance Magnitude at 1000 Hz (ohms)"]
            z_last[s] = z[-1]
            accel_days = df["Accelerated Days"]

            if len(groups) == 1:
                z_last[s] = z[-1]
                if min_accel_days > max(accel_days): # want to save the smallest of last accel days so we don't oversell progress
                    min_accel_days = max(accel_days)

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

            if len(z_average) == 0:
                z_average = z
            else:
                z_average = [a + b for a, b in zip(z_average, z)]

            # Plot z_norm and save to z_norm_average, z_norm_last
            if norm:
                # Normalize impedance to t=0
                z_norm = [r / z[0] for r in z]
                z_norm_last[s] = z_norm[-1]

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

                if len(z_norm_average) == 0:
                    z_norm_average = z_norm
                else:
                    z_norm_average = [a + b for a, b in zip(z_norm_average, z_norm)]

                    

        # Once we're through all samples, calculate the average(s)
        z_average = z_average / len(group_info["samples"])
        if norm:
            z_norm_average = z_norm_average / len(group_info["samples"])

        # Then plot them in the darkest color of the same color scale
        shade = pc.sample_colorscale(BASE_COLORS[g], [1.0])[0]

        fig_z.add_trace(
            go.Scatter(
                x=accel_days, #This will take accelerated days from the last sample - need to update this later to account for different accel days between samples within a group
                y=z_average,
                mode="lines",
                line=dict(width=1, color=shades[s]),
                name=f"{group} Average"
            ),
            row=2,
            col=1,
        )

        if norm:
            fig_z.add_trace(
            go.Scatter(
                x=accel_days, #This will take accelerated days from the last sample - need to update this later to account for different accel days between samples within a group
                y=z_norm_average,
                mode="lines",
                line=dict(width=1, color=shades[s]),
                name=f"{group} Average",
                showlegend=False
            ),
            row=2,
            col=2,
        )
        
        # Plot flagged dates
        flagged_dates_group = group_info["flagged_dates"]
        for flag_g, date_g in flagged_dates_group.items():
            # Convert to datetime
            date_g = datetime.datetime.strptime(date_g, "%Y-%m-%d %H:%M")

            # Calculate real days
            first_day = group_info["start_date"]
            first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

            real_days = date_g - first_day
            real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days

            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")

            # Find closest measurement to flagged date
            offset = abs(df["Real Days"] - real_days)
            ind_closest = offset.idxmin()
            temperature = float(df.loc[ind_closest, "Temperature (C)"])

            # Calculate accelerated days
            accel_days = calculate_accel_days(real_days, temperature, df)

            add_vert_line(fig_z, 2, 1, accel_days, f"{flag_g} ({sample} @ {accel_days} days)")


    fig_z.write_html(f"{plot_path}/{title}.html")

    if plot_on:
        fig_z.show()

    return z_last, z_norm_last, min_accel_days

def plot_cic(groups, data_path, sample_info_path, plot_path, title, norm=False):
    # If you're plotting normalized data, include second column
    if norm:
        fig_cic = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Individual Electrodes",
                "Individual Electrodes (normalized to t=0)",
                "Average",
                "Average (normalized to t=0)"
            ),
            x_title="Accelerated Time (days)",
		    y_title="Charge Injection Capacity @ 1000 us (uC/cm^2)"
        )

    else:
         fig_cic = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Individual Electrodes"
                "Average"
            ),
            x_title="Accelerated Time (days)",
            y_title="Charge Injection Capacity @ 1000 us (uC/cm^2)"
        )

    fig_cic.update_layout(title_text=title)

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
        min_accel_days = float('inf')

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Plot cic and save to cic_average, cic_last
            cic = df["Charge Injection Capacity @ 1000 us (uC/cm^2)"]
            cic_last[s] = cic[-1]
            accel_days = df["Accelerated Days"]

            if len(groups) == 1:
                cic_last[s] = cic[-1]
                if min_accel_days > max(accel_days): # want to save the smallest of last accel days so we don't oversell progress
                    min_accel_days = max(accel_days)

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

            if len(cic_average) == 0:
                cic_average = cic
            else:
                cic_average = [a + b for a, b in zip(cic_average, cic)]

            # Plot cic_norm and save to cic_norm_average, cic_norm_last
            if norm:
                # Normalize cic to t=0
                cic_norm = [r / cic[0] for r in cic]
                cic_norm_last[s] = cic_norm[-1]

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

                if len(cic_norm_average) == 0:
                    cic_norm_average = cic_norm
                else:
                    cic_norm_average = [a + b for a, b in zip(cic_norm_average, cic_norm)]
        
        # Once we're through all samples, calculate the average(s)
        cic_average = cic_average / len(group_info["samples"])
        if norm:
            cic_norm_average = cic_norm_average / len(group_info["samples"])

        # Then plot them in the darkest color of the same color scale
        shade = pc.sample_colorscale(BASE_COLORS[g], [1.0])[0]

        fig_cic.add_trace(
            go.Scatter(
                x=accel_days, #This will take accelerated days from the last sample - need to update this later to account for different accel days between samples within a group
                y=cic_average,
                mode="lines",
                line=dict(width=1, color=shades[s]),
                name=f"{group} Average"
            ),
            row=2,
            col=1,
        )

        if norm:
            fig_cic.add_trace(
            go.Scatter(
                x=accel_days, #This will take accelerated days from the last sample - need to update this later to account for different accel days between samples within a group
                y=cic_norm_average,
                mode="lines",
                line=dict(width=1, color=shades[s]),
                name=f"{group} Average",
                showlegend=False
            ),
            row=2,
            col=2,
        )
            
        # Plot flagged dates
        flagged_dates_group = group_info["flagged_dates"]
        for flag_g, date_g in flagged_dates_group.items():
            # Convert to datetime
            date_g = datetime.datetime.strptime(date_g, "%Y-%m-%d %H:%M")

            # Calculate real days
            first_day = group_info["start_date"]
            first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

            real_days = date_g - first_day
            real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days

            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")

            # Find closest measurement to flagged date
            offset = abs(df["Real Days"] - real_days)
            ind_closest = offset.idxmin()
            temperature = float(df.loc[ind_closest, "Temperature (C)"])

            # Calculate accelerated days
            accel_days = calculate_accel_days(real_days, temperature, df)

            add_vert_line(fig_cic, 1, 1, accel_days, f"{flag_g} ({sample} @ {accel_days} days)")

    fig_cic.write_html(f"{plot_path}/{title}.html")

    if plot_on:
        fig_cic.show()

    return cic_last, cic_norm_last, min_accel_days

def plot_rh(groups, data_path, sample_info_path, plot_path, title):
    fig_rh = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3]
    )

    fig_rh.update_layout(title_text=title)
    fig_rh.update_xaxes(title_text="Accelerated Time (days)")
    fig_rh.update_yaxes(title_text="Relative Humidity (%)", row=1, col=1)
    fig_rh.update_yaxes(title_text="Temperature (C)", row=2, col=1)

    # Loop through groups
    for g, group in enumerate(groups):
        # Open group info
        with open(f"{sample_info_path}/{group}.json", 'r') as f:
            group_info = json.load(f)

        # Sample evenly spaced color shades from base color
        shades = pc.sample_colorscale(BASE_COLORS[g], list(np.linspace(0.2, 0.9, len(group_info["samples"]))))

        # Create array to save the last value for each sample and smallest last accel days (only updates if processing one group)
        rh_last = []
        min_accel_days = float('inf')

        # Loop through each sample
        for s, sample in enumerate(group_info["samples"]):
            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)

            # Plot
            rh = df["Relative Humidity (%)"]
            temperature = df["Temperature (C)"]
            accel_days = df["Accelerated Days"]

            if len(groups) == 1:
                rh_last[s] = rh[-1]
                if min_accel_days > max(accel_days): # want to save the smallest of last accel days so we don't oversell progress
                    min_accel_days = max(accel_days)

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
                    y=temperature,
                    mode="lines",
                    line=dict(width=1, color=shades[s]),
                    name=sample,
                    connectgaps=True
                ),
                row=2,
                col=1,
            )
        
        # Plot flagged dates
        flagged_dates_group = group_info["flagged_dates"]
        for flag_g, date_g in flagged_dates_group.items():
            # Convert to datetime
            date_g = datetime.datetime.strptime(date_g, "%Y-%m-%d %H:%M")

            # Calculate real days
            first_day = group_info["start_date"]
            first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d %H:%M")

            real_days = date_g - first_day
            real_days = real_days.total_seconds() / 24 / 60 / 60 # convert to days

            # Open data summary
            df = pd.read_csv(f"{data_path}/{group}/{sample}_data_summary.csv")

            # Find closest measurement to flagged date
            offset = abs(df["Real Days"] - real_days)
            ind_closest = offset.idxmin()
            temperature = float(df.loc[ind_closest, "Temperature (C)"])

            # Calculate accelerated days
            accel_days = calculate_accel_days(real_days, temperature, df)

            add_vert_line(fig_rh, 2, 1, accel_days, f"{flag_g} ({sample} @ {accel_days} days)")


    fig_rh.write_html(f"{plot_path}/{title}.html")

    if plot_on:
        fig_rh.show()

    return rh_last, min_accel_days
    
def add_vert_line(fig, rows, cols, xline, label):
	for r in range(1, rows+1):
		for c in range(1, cols+1):
			# draw the vertical line in subplot (r,c):
			fig.add_vline(
				x=xline,
				line_width=1,
				line_dash="dash",
				line_color="black",
				row=r,                 # target row
				col=c,                 # target column
			)
			# draw the vertical text next to it, rotated 90°:
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