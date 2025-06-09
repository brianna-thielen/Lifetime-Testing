import pandas as pd
import os
import math
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_PATH = "./data/LCP Pt Grids"
PLOT_PATH = "./data/Plots"

# Electrode and test information
_ELECTRODE_DATA = pd.DataFrame({
	"Channel Name": ["G1X1-1", "G1X1-2", "G3X3S-1", "G3X3S-2", "G2X2S-1", "G2X2S-2", "G2X2L-1", "G2X2L-2"], 
	"Geometric Surface Area (mm^2)": [4, 4, 3.9204, 3.9204, 3.9601, 3.9601, 4, 4],
	"Pulse Amplitude (uA)": [400, 400, 400, 400, 400, 400, 400, 400],
})
_STIM_FREQUENCY = 50 #Hz
_START_DATE = datetime.datetime(2025, 4, 4, 4, 00)
_ELECTRODES_TO_IGNORE = pd.DataFrame({
    "Channel Name": [],
    "Ignore From": []
})


# Plotting constants
_MIN_COLOR = 0
_MAX_COLOR = 200
_NUM_EIS_ROWS = 2
_NUM_EIS_COLS = 4


def process_lcp_pt_grids_soak_data(plot_on=False):
	"""
	This is the main function that processes lifetime testing data for the LCP grids collected from the Intan
	"""

	print("Processing LCP Pt Grids soak test data...")

	# Initiate plots
	# fig_eis (subplots defined by _NUM_EIS_COLS and _NUM_EIS_ROWS): EIS vs time for each electrode individually
	fig_eis = make_subplots(
		rows=_NUM_EIS_ROWS, 
		cols=_NUM_EIS_COLS,
		specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}], 
		 	   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
		subplot_titles=(
			"2x2 mm footprint, 1x1 Grid (1)", "2x2 mm footprint, 1x1 Grid (2)", "2x2 mm footprint, 3x3 Grid (1)", "2x2 mm footprint, 3x3 Grid (2)", 
			"2x2 mm footprint, 2x2 Grid (1)", "2x2 mm footprint, 2x2 Grid (2)", "2.1x2.1 mm footprint, 2x2 Grid (1)", "2.1x2.1 mm footprint, 2x2 Grid (1)"
			),
		x_title="Frequency (Hz)",
		y_title="Left/Solid: Impedance Magnitude (Ohms)\nRight/Dashed: Impedance Phase (deg)"
	)
	fig_eis.update_layout(title_text="EIS Curve vs Time (light=day 0, dark=most recent)")

	# fig_impedance: 1k impedance vs time 
	fig_impedance = make_subplots(
		rows=1,
		cols=1,
		x_title="Accelerated Time (days)",
		y_title="Impedance Magnitude (ohms)"
	)
	fig_impedance.update_layout(title_text="1 kHz Impedance Magnitude vs Time")

	# fig_cic: CIC vs time
	fig_cic = make_subplots(
		rows=1,
		cols=1,
		x_title="Accelerated Time (days)",
		y_title="Charge Injection Capacity (uC/cm^2)"
	)
	fig_cic.update_layout(title_text="Charge Injection Capacity vs Time (1000 us pulse)")

	df_experiment, df_z, df_cic1000 = perform_data_analysis(DATA_PATH)

	# Plot data and save
	plot_eis(fig_eis, df_experiment, f"{DATA_PATH}//intan-eis")
	plot_impedance(fig_impedance, df_experiment, df_z)

	plot_cic(fig_cic, df_experiment, df_cic1000)

	fig_eis.write_html(f"{PLOT_PATH}/LCP-Pt-Grids_EIS-vs-Time.html")
	fig_impedance.write_html(f"{PLOT_PATH}/LCP-Pt-Grids_Impedance-vs-Time.html")

	fig_cic.write_html(f"{PLOT_PATH}/LCP-Pt-Grids_CIC-vs-Time.html")

	if plot_on:
		fig_eis.show()
		fig_impedance.show()

		fig_cic.show()

	# Extract most recent values of CIC and impedance (intan measurements only)
	mask = [col for col in df_z.columns if col.startswith('G')]
	
	z = df_z[mask].iloc[-1]

	cic = df_cic1000[mask].iloc[-1]

	# Extract days from df_experiment
	days = df_experiment['Accelerated Days'].iloc[-1]

	return days, cic, z

def perform_data_analysis(path):
	# Create dataframes for processed data
	df_experiment = pd.DataFrame({
		'Measurement Datetime': None,
		'Pulsing On': None,
		'Temperature (C)': [],
		'Accelerated Days': [],
		'Real Days': [],
		'Pulses': []
	})

	channel_names = _ELECTRODE_DATA["Channel Name"].tolist()

	df_z = pd.DataFrame(columns=["Measurement Datetime", "Real Days"] + channel_names)

	df_cic1000 = pd.DataFrame(columns=["Measurement Datetime", "Real Days"] + channel_names)

	# First, process spreadsheet (this contains early temperature)
	df_experiment = process_vt_spreadsheet_data(f"{path}//Soak Testing - LCP Grids (Active).csv", df_experiment)

	# Then, process intan measurements
	df_experiment, df_cic1000, df_z = process_intan_data(f"{path}//", df_experiment, df_cic1000, df_z)

	df_experiment = calc_accel_days_and_pulses(df_experiment)

	return df_experiment, df_z, df_cic1000

def process_vt_spreadsheet_data(filepath, df_experiment):
	"""
	Processes data from VT spreadsheet
	
	Returns updated df_experiment with temperature, accelerated days, real days, and pulsing info
	"""

	df = pd.read_csv(filepath)
	# Set row 1 to header titles
	df.columns = df.iloc[1]

	# Remove unwanted headers
	df = df[6:]

	# Loop through each time point and save data (if any)
	for idx, row in df.iterrows():
		# Pull data from spreadsheet row
		# for accel_days, we need to strip any commas first
		accel_days = row["Accel. Days (Total)"]
		accel_days = accel_days.replace(",", "")
		accel_days = float(accel_days)
		# the others aren't formatted with commas, so can be converted directly
		temp = float(row["Temp (deg C)"])
		test_date = row["Date"]
		test_time = row["Time"]
		
		# For some reason strptime isn't taking the AM or PM values with %p, so I put that in manually
		test_datetime = datetime.datetime.strptime(f"{test_date} {test_time[:-3]}", '%m/%d/%y %H:%M')
		if "12" not in test_time and "PM" in test_time:
			test_datetime += datetime.timedelta(hours=12)
		elif "12" in test_time and "AM" in test_time:
			test_datetime -= datetime.timedelta(hours=12)
		
		pulses = float(row["# Pulses"])
		stim_on = row["Stim On/Off"]
		if stim_on == "ON":
			stim_on = True
		else:
			stim_on = False

		# Calculate real days from start date
		real_days = test_datetime - _START_DATE
		real_days = real_days.total_seconds() / 60 / 60 / 24

		# Save timestamp and pulsing info to df_experiment
		# Only need to save new values
		if len(df_experiment) == 0 or test_datetime not in df_experiment["Measurement Datetime"].values:
			ind = len(df_experiment)
			df_experiment.loc[ind, "Measurement Datetime"] = test_datetime
			df_experiment.loc[ind, "Temperature (C)"] = temp
			df_experiment.loc[ind, "Accelerated Days"] = accel_days
			df_experiment.loc[ind, "Real Days"] = real_days
			df_experiment.loc[ind, "Pulses"] = pulses
			df_experiment.loc[ind, "Pulsing On"] = stim_on

			# Sort by datetime
			df_experiment = df_experiment.sort_values(by="Measurement Datetime")

	return df_experiment

def process_intan_data(path, df_experiment, df_cic1000, df_z):

	files = os.listdir(path)

	# Loop through each file and pull data
	for file in files:
		# Skip files that are not .csv
		if not file.endswith(".csv"):
			continue

		# Skip the summary file
		if file == "Soak Testing - LCP Grids (Active).csv":
			continue

		# Find timestamp
		year = int(file[18:22])
		month = int(file[22:24])
		day = int(file[24:26])
		hour = int(file[26:28])
		minutes = int(file[28:30])
		seconds = int(file[30:32])

		test_end_datetime = datetime.datetime(year, month, day, hour, minutes, seconds)

		# Calculate real days from start date
		real_days_end = test_end_datetime - _START_DATE
		real_days_end = real_days_end.total_seconds() / 60 / 60 / 24

		df = pd.read_csv(f"{path}{file}")

		# If there was a CIC measurement, the test start is ~8 mins before save time, and we need to save to df_cic
		if not math.isnan(df.loc[0, "Charge Injection Capacity @ 1000 us (uC/cm^2)"]):
			test_start_datetime = datetime.datetime(year, month, day, hour, minutes, seconds) - datetime.timedelta(minutes=8)

			# Calculate real days from start date
			real_days_start = test_start_datetime - _START_DATE
			real_days_start = real_days_start.total_seconds() / 60 / 60 / 24

			# Save measurement time
			ind_cic = len(df_cic1000)
			df_cic1000.loc[ind_cic, "Measurement Datetime"] = test_start_datetime
			df_cic1000.loc[ind_cic, "Real Days"] = real_days_start

			ind_z = len(df_z)
			df_z.loc[ind_z, "Measurement Datetime"] = test_start_datetime
			df_z.loc[ind_z, "Real Days"] = real_days_start
		# If there was not a CIC measurement, the test start is ~10 seconds before save time, and we don't need to save df_cic_intan
		else:
			test_start_datetime = datetime.datetime(year, month, day, hour, minutes, seconds) - datetime.timedelta(seconds=10)

			# Calculate real days from start date
			real_days_start = test_start_datetime - _START_DATE
			real_days_start = real_days_start.total_seconds() / 60 / 60 / 24

			# Save measurement time
			ind_z = len(df_z)
			df_z.loc[ind_z, "Measurement Datetime"] = test_start_datetime
			df_z.loc[ind_z, "Real Days"] = real_days_start

		# Temperature is the same for all samples - pull temp
		# Calculate accelerated days at the end in case files are processed out of order
		temp = df.loc[0, 'Temperature (C)']

		# Save measurement times to df_experiment
		ind = len(df_experiment)
		df_experiment.loc[ind, "Measurement Datetime"] = test_start_datetime
		df_experiment.loc[ind, "Real Days"] = real_days_start
		df_experiment.loc[ind, "Temperature (C)"] = temp
		df_experiment.loc[ind, "Pulsing On"] = False

		ind = len(df_experiment)
		df_experiment.loc[ind, "Measurement Datetime"] = test_end_datetime
		df_experiment.loc[ind, "Real Days"] = real_days_end
		df_experiment.loc[ind, "Temperature (C)"] = temp
		df_experiment.loc[ind, "Pulsing On"] = True

		# Pull impedance and CIC data from imported df
		for i in range(len(df)):
			sample = df.loc[i, "Channel Name"]
			impedance = df.loc[i, "Impedance Magnitude at 1000 Hz (ohms)"]
			cic = df.loc[i, "Charge Injection Capacity @ 1000 us (uC/cm^2)"]

			if cic < 0:
				cic = 0

			# Save to df_z_intan and df_cic1000_intan (if data exists)
			df_z.loc[ind_z, sample] = impedance

			if not math.isnan(cic):
				df_cic1000.loc[ind_cic, sample] = cic

		# Sort by datetime
		df_experiment = df_experiment.sort_values(by="Measurement Datetime")
		df_cic1000 = df_cic1000.sort_values(by="Measurement Datetime")
		df_z = df_z.sort_values(by="Measurement Datetime")

	return df_experiment, df_cic1000, df_z
			
def calc_accel_days_and_pulses(df_experiment):
	"""
	Calculates the number of accelerated days and cumulative pulses at each time step
	Temperature and datetime saved in df_experiment
	"""

	# Start counters for cumulative accelerated days and pulses
	cumulative_accel_days = 0.0
	cumulative_pulses = 0

	# Set starting datetime as "last test" for summative measurements
	last_date = df_experiment["Measurement Datetime"].iloc[0]

	# Pulsing on/off is the setting which pulsing was set/changed to at that time stamp
	# Therefore, calculations require the previous pulse setting
	last_stim = False

	# Loop through each row
	# df_experiment is sorted, so this will be in order
	for idx, row in df_experiment.iterrows():
		date = row["Measurement Datetime"]
		temp = row["Temperature (C)"]
		stim_on = row["Pulsing On"]
		test_pulses = row["Pulses"]
		test_days = row["Accelerated Days"]

		# Calculate number of days from previous measurement
		real_days = date - last_date
		real_days = real_days.total_seconds() / 60 / 60 / 24

		# Calculate accelerated days since previous measurement and add to cumulative count
		accel_days = real_days * 2 ** ((temp - 37)/10)
		cumulative_accel_days += accel_days

		# Calculate pulses since previous measurement and add to cumulative count, if stim is on
		if last_stim:
			pulses = _STIM_FREQUENCY * real_days * 24 * 60 * 60
			cumulative_pulses += pulses

		# Update settings for next measurement
		last_date = date
		last_stim = stim_on
		
		# Save calculated values to df_experiment
		df_experiment.loc[idx, "Accelerated Days"] = cumulative_accel_days
		df_experiment.loc[idx, "Pulses"] = cumulative_pulses
	
	return df_experiment
	
def plot_eis(fig_eis, df_experiment, path):
	"""
	Plots eis curves in their respective subplot of fig_eis
	"""

	files = os.listdir(path)

	# Find most recent date and number of days for plotting color purposes
	dates = [file[:-4] for file in files]
	dates = [file[-14:] for file in dates]
	dates = [int(file) for file in dates]
	latest = datetime.datetime.strptime(str(max(dates)), '%Y%m%d%H%M%S')
	latest_days = (latest - _START_DATE).total_seconds() / 60 / 60 / 24

	# Loop through each file, pull data, and plot
	for file in files:
		# Find timestamp
		i = len(file)
		year = int(file[i-18:i-14])
		month = int(file[i-14:i-12])
		day = int(file[i-12:i-10])
		hour = int(file[i-10:i-8])
		minutes = int(file[i-8:i-6])
		seconds = int(file[i-6:i-4])

		test_datetime = datetime.datetime(year, month, day, hour, minutes, seconds)

		# Calculate real days from start
		real_days = (test_datetime - _START_DATE).total_seconds() / 60 / 60 / 24

		# Set plot color based on age
		colort = int(((latest_days)-real_days)*(_MAX_COLOR-_MIN_COLOR)/(latest_days))

		# Find accel days and number of pulses from df_experiment
		# There may not be an exact match, so find the closest
		df_experiment["Time Difference"] = abs(df_experiment["Real Days"] - real_days)
		ind_closest = df_experiment["Time Difference"].idxmin()
		accel_days = df_experiment.loc[ind_closest, "Accelerated Days"]
		pulses = df_experiment.loc[ind_closest, "Pulses"]

		# Find sample number
		for n, s in enumerate(_ELECTRODE_DATA["Channel Name"]):
			if s in file:
				sample = s
				sample_num = n
				break

		# Check if the current file is in the ignore list
		if sample in _ELECTRODES_TO_IGNORE["Channel Name"].tolist():
			ignore_date = _ELECTRODES_TO_IGNORE.loc[_ELECTRODES_TO_IGNORE["Channel Name"] == sample, "Ignore From"].item()
			if ignore_date < test_datetime:
				continue

		# Define row and col based on constants
		plot_row = sample_num // _NUM_EIS_COLS + 1
		plot_col = sample_num % _NUM_EIS_COLS + 1

		# Import data
		df = pd.read_csv(f"{path}//{file}")

		# fig_eis: EIS vs time for each electrode individually
		# First, magnitude in solid line
		fig_eis.add_trace(
			go.Scatter(
				x=df["Frequency"],
				y=df["Impedance"],
				mode="lines",
				line=dict(width=1, dash='solid', color=f"rgb({colort},{colort},{colort})"),
				name=f"day {accel_days}",
				showlegend=False
			),
			row=plot_row,
			col=plot_col,
			secondary_y=False,
		)
		
		# Next, phase in dashed line
		fig_eis.add_trace(
			go.Scatter(
				x=df["Frequency"],
				y=df["Phase Angle"],
				mode="lines",
				line=dict(width=1, dash='dash', color=f"rgb({colort},{colort},{colort})"),
				name=f"day {accel_days}",
			),
			row=plot_row,
			col=plot_col,
			secondary_y=True,
		)
		
		fig_eis.update_xaxes(type="log", range=[1,4], tick0=10, dtick=1)
		fig_eis.update_yaxes(secondary_y=True, range=[-90,0], tick0=0, dtick=30)
		fig_eis.update_yaxes(type="log", secondary_y=False, tick0=10, dtick=1)

def plot_impedance(fig_impedance, df_experiment, df_z):
	"""
	Plots 1k impedance vs time
	Individual traces in narrow colors
	Average traces in thick black
	"""

	# Create dataframes for calculating average values
	df_avg = pd.DataFrame()

	# Loop through each sample
	for sample in _ELECTRODE_DATA["Channel Name"]:
		# Sample Number
		sample_num = int(sample[-2:])

		# Pull Z for current channel
		z = df_z[sample]
		z.dropna(inplace=True)

		ind = z.index.to_list()

		# Pull real days and datetime for current channel
		real_days = df_z["Real Days"]
		real_days = real_days.loc[ind]
		test_dates = df_z["Measurement Datetime"]
		test_dates = test_dates.loc[ind]

		# Check if the current file is in the ignore list
		if sample in _ELECTRODES_TO_IGNORE["Channel Name"].tolist():
			ignore_date = _ELECTRODES_TO_IGNORE.loc[_ELECTRODES_TO_IGNORE["Channel Name"] == sample, "Ignore From"].item()
			# Remove values at test dates >= ignore date
			ind_ignore = test_dates >= ignore_date
			z[ind_ignore] = float('nan')
			real_days[ind_ignore] = float('nan')

		# Find accel days and number of pulses from df_experiment
		# Need to loop through each time point for this since we have a list
		accel_days = []
		pulses = []

		for rd in real_days:
			# First, check if it's nan (i.e. a value to ignore)
			if math.isnan(rd):
				ad = float('nan')
				p = float('nan')
			else:
				# There may not be an exact match, so find the closest
				df_experiment["Time Difference"] = abs(df_experiment["Real Days"] - rd)
				ind_closest = df_experiment["Time Difference"].idxmin()
				ad = df_experiment.loc[ind_closest, "Accelerated Days"].item()
				p = df_experiment.loc[ind_closest, "Pulses"].item()

			accel_days.append(ad)
			pulses.append(p)

		# Find stim level and gsa, define label
		gsa = _ELECTRODE_DATA.loc[_ELECTRODE_DATA["Channel Name"] == sample, "Geometric Surface Area (mm^2)"].astype(float)
		gsa = gsa.values[0]

		stim = _ELECTRODE_DATA.loc[_ELECTRODE_DATA["Channel Name"] == sample, "Pulse Amplitude (uA)"].astype(float)
		stim = stim.values[0]

		if stim == 0:
			stim = "no stim"
		else:
			stim = str(stim)

		trace_label = f"{sample} ({stim} uA/ {gsa} mm^2)"

		# Normalize impedance to GSA
		z = z * gsa
		z = z.tolist()

		# Add data to dataframe for average plot
		df_avg[sample] = z

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'
		
		# Plot individual traces
		fig_impedance.add_trace(
			go.Scatter(
				x=accel_days,
				y=z,
				mode="lines",
				line=dict(width=1, dash=linetype),
				name=trace_label
			),
			row=1,
			col=1,
		)

	# Calculate averages
	average = df_avg.mean(axis=1)
	sd = df_avg.std(axis=1)

	# Plot pt average, left is lcr, right is intan
	fig_impedance.add_trace(
		go.Scatter(
			x=accel_days,
			y=average,
			mode="lines",
			line=dict(width=2, color=f"rgb(0,0,0)"),
			name="Average",
			# error_y=dict(type='data', array=sd, thickness=1, width=4, visible=True)
		),
		row=1,
		col=1,
	)

	fig_impedance.update_yaxes(type="log", tick0=100, dtick=1, range=[1, 6])
	fig_impedance.update_xaxes(range=[0, max(accel_days)])

def plot_cic(fig_cic, df_experiment, df_cic1000):
	"""
	Plots CIC vs time
	"""

	# Create dataframes for calculating average values
	df_avg = pd.DataFrame()

	# Loop through each sample
	for sample in _ELECTRODE_DATA["Channel Name"]:
		# Sample Number
		sample_num = int(sample[-2:])

		# Pull CIC for current channel
		cic1000 = df_cic1000[sample]
		cic1000.dropna(inplace=True)

		ind = cic1000.index.to_list()

		# Pull real days and datetime for current channel
		real_days = df_cic1000["Real Days"]
		real_days = real_days.loc[ind]
		test_dates = df_cic1000["Measurement Datetime"]
		test_dates = test_dates.loc[ind]

		# Check if the current file is in the ignore list
		if sample in _ELECTRODES_TO_IGNORE["Channel Name"].tolist():
			ignore_date = _ELECTRODES_TO_IGNORE.loc[_ELECTRODES_TO_IGNORE["Channel Name"] == sample, "Ignore From"].item()
			# Remove values at test dates >= ignore date
			ind_ignore = test_dates >= ignore_date
			cic1000[ind_ignore] = float('nan')
			real_days[ind_ignore] = float('nan')

		# Find accel days and number of pulses from df_experiment
		# Need to loop through each time point for this since we have a list
		accel_days = []
		pulses = []
		for rd in real_days:
			# First, check if it's nan (i.e. a value to ignore)
			if math.isnan(rd):
				ad = float('nan')
				p = float('nan')
			else:
				# There may not be an exact match, so find the closest
				df_experiment["Time Difference"] = abs(df_experiment["Real Days"] - rd)
				ind_closest = df_experiment["Time Difference"].idxmin()
				ad = df_experiment.loc[ind_closest, "Accelerated Days"].item()
				p = df_experiment.loc[ind_closest, "Pulses"].item()

			accel_days.append(ad)
			pulses.append(p)

		# Find stim level and gsa, define label
		gsa = _ELECTRODE_DATA.loc[_ELECTRODE_DATA["Channel Name"] == sample, "Geometric Surface Area (mm^2)"].astype(float)
		gsa = gsa.values[0]

		stim = _ELECTRODE_DATA.loc[_ELECTRODE_DATA["Channel Name"] == sample, "Pulse Amplitude (uA)"].astype(float)
		stim = stim.values[0]

		if stim == 0:
			stim = "no stim"
		else:
			stim = str(stim)

		trace_label = f"{sample} ({stim} uA/ {gsa} mm^2)"

		# Convert CIC to list
		cic1000 = cic1000.tolist()

		# Add to dataframe for average plot
		df_avg[sample] = cic1000

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'

		# Plot individual traces
		fig_cic.add_trace(
			go.Scatter(
				x=accel_days,
				y=cic1000,
				mode="lines",
				line=dict(width=1, dash=linetype),
				name=trace_label,
			),
			row=1,
			col=1,
		)

	# Calculate averages
	average = df_avg.mean(axis=1)
	sd = df_avg.std(axis=1)

	# Plot average
	fig_cic.add_trace(
		go.Scatter(
			x=accel_days,
			y=average,
			mode="lines",
			line=dict(width=2, color=f"rgb(0,0,0)"),
			name="Average",
			# error_y=dict(type='data', array=sd, visible=True)
		),
		row=1,
		col=1,
	)

	# fig_cic.update_yaxes(range=[0, 1000])
	fig_cic.update_xaxes(range=[0, max(accel_days)])
	
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