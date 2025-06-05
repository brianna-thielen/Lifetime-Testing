import math
import pandas as pd
import os
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Processes passive soak test data (EIS only)
DATA_PATH = "./data/LCP IDEs"
PLOT_PATH = "./data/Plots"

# Min and Max values for grayscale plotting
_MIN_COLOR = 0
_MAX_COLOR = 200

_PLOT_OFFSET = 5

# Temperature readings from spreadsheet
_START_DATE = datetime.datetime(2024, 11, 15, 15, 20)
_SPREADSHEET_DATA = pd.DataFrame({
	"Measurement Datetime": [datetime.datetime(2024,11,15,15,32),
						  datetime.datetime(2024,11,18,9,31),
						  datetime.datetime(2024,11,21,9,40),
						  datetime.datetime(2024,11,25,10,6),
						  datetime.datetime(2024,12,2,10,15),
						  datetime.datetime(2024,12,9,10,15),
						  datetime.datetime(2024,12,17,17,33),
						  datetime.datetime(2024,12,23,12,41)], 
	"Real Days": [0, 2.749, 5.756, 9.774, 16.780, 23.780, 32.084, 37.881],
	"Temperature (C)": [24.0, 62.9, 63.0, 63.6, 63.8, 63.1, 66.9, 66.5]
})
_SAMPLES = ["IDE-100-1", "IDE-100-2", "IDE-100-3", "IDE-100-4", "IDE-100-5", "IDE-100-6", "IDE-100-7", "IDE-100-8", 
			"IDE-25-1", "IDE-25-2", "IDE-25-3", "IDE-25-4", "IDE-25-5", "IDE-25-6", "IDE-25-7", "IDE-25-8"]

_ELECTRODES_TO_IGNORE = pd.DataFrame({
    "Sample": [],
    "Ignore From": []
})

def process_ide_soak_data(plot_on=False):
	"""
	Main function to process IDE soak test data.
	"""

	print("Processing LCP IDE soak test data...")

	# Initiate plots
	# fig_eis (4x4 subplots): EIS vs time for each electrode individually
	fig_eis = make_subplots(
		rows=4, 
		cols=4,
		specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}], 
		 	   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
			   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
		 	   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
		subplot_titles=(
			"100 um Sample 1", "100 um Sample 2", "100 um Sample 3", "100 um Sample 4", "100 um Sample 5", "100 um Sample 6", "100 um Sample 7", "100 um Sample 8",
			"25 um Sample 1", "25 um Sample 2", "25 um Sample 3", "25 um Sample 4", "25 um Sample 5", "25 um Sample 6", "25 um Sample 7", "25 um Sample 8"
		),
		x_title="Frequency (Hz)",
		y_title="Left/Solid: Impedance Magnitude (Ohms)\nRight/Dashed: Impedance Phase (deg)"
	)
	fig_eis.update_layout(title_text="EIS Curve vs Time (light=day 0, dark=most recent)")

	# fig_impedance: 1k impedance vs time 
	# top: individual traces; bottom: average; left: impedance; right: normalized impedance to t=0
	fig_impedance = make_subplots(
		rows=2,
		cols=2,
		subplot_titles=(
			"Individual Electrodes",
			"Individual Electrodes (normalized to t=0)",
			"Average",
			"Average (normalized)"
		),
		x_title="Accelerated Time (days)",
		y_title="Impedance Magnitude (ohms)"
	)
	fig_impedance.update_layout(title_text="1 kHz Impedance Magnitude vs Time")

	# 3: EIS for day 0 for all electrodes (individual traces)
	fig_eis_day0 = make_subplots(
		rows=1, 
		cols=1,
		specs=[[{"secondary_y": True}]],
	)
	fig_eis_day0.update_layout(title_text="Initial EIS curve for All Electrodes")

	# Pull data into df_z
	df_z = perform_data_analysis(DATA_PATH)

	# Calculate accelerated days from data
	df_z = calc_accel_days(df_z)

	# Plot data and save
	plot_eis(fig_eis, df_z, f"{DATA_PATH}")
	df_25, df_100 = plot_impedance(fig_impedance, df_z)

	fig_eis.write_html(f"{PLOT_PATH}/LCP-IDEs_EIS-vs-Time.html")
	fig_impedance.write_html(f"{PLOT_PATH}/LCP-IDEs_Impedance-vs-Time.html")


	if plot_on:
		fig_eis.show()
		fig_impedance.show()
		# fig_eis_day0.show()

	# Extract most recent values
	z_IDE_25 = df_25.filter(like="IDE-25").iloc[-1].values
	z_IDE_100 = df_100.filter(like="IDE-100").iloc[-1].values

	# Normalize to first measurement
	first_IDE_25 = df_25.filter(like="IDE-25").iloc[0].values
	first_IDE_100 = df_100.filter(like="IDE-100").iloc[0].values

	norm_IDE_25 = z_IDE_25 / first_IDE_25
	norm_IDE_100 = z_IDE_100 / first_IDE_100

	# Extract accelerated days
	days = df_z["Accelerated Days"].iloc[-1]

	# df_z.to_csv(f"{PLOT_PATH}/LCP_IDE_Soak_Data.csv", index=False)

	return days, z_IDE_25, z_IDE_100, norm_IDE_25, norm_IDE_100

def perform_data_analysis(path):
	# List of files
	files = os.listdir(f"{path}//")

	# Create dataframe for processed data
	df_z = pd.DataFrame({
		'Measurement Datetime': None,
		'Temperature (C)': [],
		'Accelerated Days': [],
		'Real Days': [],
		'IDE-25-1': [], 'IDE-25-2': [], 'IDE-25-3': [], 'IDE-25-4': [], 'IDE-25-5': [], 'IDE-25-6': [], 'IDE-25-7': [], 'IDE-25-8': [],
		'IDE-100-1': [], 'IDE-100-2': [], 'IDE-100-3': [], 'IDE-100-4': [], 'IDE-100-5': [], 'IDE-100-6': [], 'IDE-100-7': [], 'IDE-100-8': [],
	})
	
	# Loop through each folder
	for file in files:
		# If the file is an old plot, skip it
		if "html" in file:
			continue
		
		df_z = process_eis_data(df_z, file, path)

	# Sort by datetime
	df_z = df_z.sort_values(by="Measurement Datetime")

	return df_z


def process_eis_data(df_z, file, path):
	"""
	"""

	# Find timestamp
	i = len(file)
	year = int(file[i-18:i-14])
	month = int(file[i-14:i-12])
	day = int(file[i-12:i-10])
	hour = int(file[i-10:i-8])
	minutes = int(file[i-8:i-6])
	seconds = int(file[i-6:i-4])

	test_datetime = datetime.datetime(year, month, day, hour, minutes, seconds)

	# Calculate real days from start date
	real_days = test_datetime - _START_DATE
	real_days = real_days.total_seconds() / 60 / 60 / 24

	# Save timestamp
	ind = len(df_z)
	df_z.loc[ind, "Measurement Datetime"] = test_datetime
	df_z.loc[ind, "Real Days"] = real_days

	# Import file
	df = pd.read_csv(f"{path}//{file}")

	# Find 1k impedance and temperature
	index = (np.abs(df["Frequency"] - 1000)).idxmin()
	z_1k = df["Impedance"][index].item()
	# 3.5 degree offset between measurement and actual temperature
	temp = df["Temperature (Dry Bath)"][index].item() - 3.5

	# If temperature is not in the data (nan) or lower than expected (i.e. another sensor is plugged in), use the chart at the top to find closest value
	if math.isnan(temp) or temp < 50:
		_SPREADSHEET_DATA["Time Difference"] = abs(_SPREADSHEET_DATA["Real Days"] - real_days)
		ind_closest = _SPREADSHEET_DATA["Time Difference"].idxmin()
		temp = _SPREADSHEET_DATA.loc[ind_closest, "Temperature (C)"]

	# Determine sample number
	if "IDE-25" in file:
		sample = file[4:12]
	elif "IDE-100" in file:
		sample = sample = file[4:13]

	# Save 1k impedance to df_z_lcr
	df_z.loc[ind, sample] = z_1k
	df_z.loc[ind, "Temperature (C)"] = temp

	return df_z


def calc_accel_days(df_z):
	"""
	Calculates the number of accelerated days at each time step
	Temperature and datetime saved in df_z
	"""

	# Start counters for cumulative accelerated days
	cumulative_accel_days = 0.0

	# Set starting datetime as "last test" for summative measurements
	last_date = df_z["Measurement Datetime"].iloc[0]

	# Loop through each row
	# df_z is sorted, so this will be in order
	for idx, row in df_z.iterrows():
		date = row["Measurement Datetime"]
		temp = row["Temperature (C)"]

		# Calculate number of days from previous measurement
		real_days = date - last_date
		real_days = real_days.total_seconds() / 60 / 60 / 24

		# Calculate accelerated days since previous measurement and add to cumulative count
		accel_days = real_days * 2 ** ((temp - 37)/10)
		cumulative_accel_days += accel_days

		# Update settings for next measurement
		last_date = date
		
		# Save calculated values to df_experiment
		df_z.loc[idx, "Accelerated Days"] = cumulative_accel_days
		# df_z.loc[idx, "Real Days"] = real_days
	
	return df_z


def plot_eis(fig_eis, df_z, path):
	"""
	Plots eis curves in their respective subplot of fig_eis
	
	This has to loop back through all manual measurement files to find the EIS data
	"""

	files = os.listdir(path)
	numfiles = len(files)

	# Find most recent date and number of days for plotting color purposes
	dates = [file[:-4] for file in files]
	dates = [file[-14:] for file in dates]
	dates = [int(file) for file in dates]
	latest = datetime.datetime.strptime(str(max(dates)), '%Y%m%d%H%M%S')
	latest_days = (latest - _START_DATE).total_seconds() / 60 / 60 / 24

	# Save most recent plotted date
	last_plot_day_list = [-2*_PLOT_OFFSET] * 16 # 100-1 through 8, then 25-1 through 8

	# Loop through each file, pull data, and plot
	for n in range(numfiles):
		file = files[n]

		# Find sample number and latest plot point
		ind = file.index("IDE")
		if "IDE-25" in file:
			sample = file[ind:ind+8]
			sample_num = int(sample[-1:])
			last_plot_day = last_plot_day_list[sample_num + 7]
		elif "IDE-100" in file:
			sample = file[ind:ind+9]
			sample_num = int(sample[-1:])
			last_plot_day = last_plot_day_list[sample_num - 1]

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

		# If this test is within _PLOT_OFFSET days of latest_days, go to next file
		if real_days - last_plot_day < _PLOT_OFFSET:
			continue

		# Set plot color based on age
		colort = int(((latest_days)-real_days)*(_MAX_COLOR-_MIN_COLOR)/(latest_days))

		# Update latest_days
		if "IDE-25" in file:
			last_plot_day_list[sample_num + 7] = real_days
		elif "IDE-100" in file:
			last_plot_day_list[sample_num - 1] = real_days

		# Pull accel days from df_z
		accel_days = df_z[df_z["Real Days"] == real_days]["Accelerated Days"]

		# Rows 1 and 2 are 100 um; rows 3 and 4 are 25 um
		if "IDE-100" in file and sample_num < 5:
			plot_row = 1
			plot_col = sample_num
		elif "IDE-100" in file and sample_num >= 5:
			plot_row = 2
			plot_col = sample_num-4
		elif "IDE-25" in file and sample_num < 5:
			plot_row = 3
			plot_col = sample_num
		elif "IDE-25" in file and sample_num >= 5:
			plot_row = 4
			plot_col = sample_num-4

		# Import data
		df = pd.read_csv(f"{path}//{file}")

		# fig_eis (4x4 subplots): EIS vs time for each electrode individually
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
				showlegend=False
			),
			row=plot_row,
			col=plot_col,
			secondary_y=True,
		)
		
		fig_eis.update_xaxes(type="log", range=[1,4], tick0=10, dtick=1)
		fig_eis.update_yaxes(secondary_y=True, range=[-90,0], tick0=0, dtick=30)
		fig_eis.update_yaxes(type="log", secondary_y=False, tick0=10, dtick=1)


def plot_impedance(fig_impedance, df_z):
	"""
	Plots 1k impedance vs time
	Left is manual data, right is intan data
	"""

	# Create dataframes for calculating average values
	df_100 = pd.DataFrame()
	df_25 = pd.DataFrame()
	df_100_norm = pd.DataFrame()
	df_25_norm = pd.DataFrame()

	# Loop through each sample
	for sample in _SAMPLES:
		# Sample Number
		sample_num = int(sample[-1:])

		# Pull Z for current channel
		z = df_z[sample]
		z.dropna(inplace=True)
		
		ind = z.index.to_list()
		z = z.values.tolist()

		# Pull real days and datetime for current channel
		real_days = df_z["Real Days"]
		real_days = real_days.loc[ind]
		test_dates = df_z["Measurement Datetime"]
		test_dates = test_dates.loc[ind]

		# Check if the current file is in the ignore list
		if sample in _ELECTRODES_TO_IGNORE["Sample"].tolist():
			ignore_date = _ELECTRODES_TO_IGNORE.loc[_ELECTRODES_TO_IGNORE["Sample"] == sample, "Ignore From"].item()
			# Remove values at test dates >= ignore date
			ind_ignore = test_dates >= ignore_date
			z[ind_ignore] = float('nan')
			real_days[ind_ignore] = float('nan')

		# Find accel days from df_z
		# Need to loop through each time point for this since we have a list
		accel_days = []
		for rd in real_days:
			# First, check if it's nan (i.e. a value to ignore)
			if math.isnan(rd):
				ad = float('nan')
			else:
				# There may not be an exact match, so find the closest
				ad = df_z[df_z["Real Days"] == rd]["Accelerated Days"]

			accel_days.append(ad.iloc[0])

		# Normalize impedance to initial impedance
		z_norm = [r / z[0] for r in z]

		# Set plot color for current part number
		# Also add data to dataframe for average plot
		colorn = int((10-sample_num)*(_MAX_COLOR-_MIN_COLOR)/10)
		if "IDE-100" in sample:
			colorrgb = f"rgb({colorn},{colorn},255)"
			# df_100[sample] = z
			# df_100_norm[sample] = z_norm
		elif "IDE-25" in sample:
			colorrgb = f"rgb(255,{colorn},{colorn})"
			# df_25[sample] = z
			# df_25_norm[sample] = z_norm
		else:
			colorrgb = f"rgb({colorn},{colorn},{colorn})"
		
		# Plot individual traces, left is impedance, right is normalized impedance
		fig_impedance.add_trace(
			go.Scatter(
				x=accel_days,
				y=z,
				mode="lines",
				line=dict(width=1, color=colorrgb),
				name=sample
			),
			row=1,
			col=1,
		)
		fig_impedance.add_trace(
			go.Scatter(
				x=accel_days,
				y=z_norm,
				mode="lines",
				line=dict(width=1, color=colorrgb),
				name=sample,
				showlegend=False
			),
			row=1,
			col=2,
		)

	# Bottom subplot - average data
	# New method
	# Select columns for IDE-25 and IDE-100
	ide_100_cols = [col for col in df_z.columns if "IDE-100" in col]
	ide_25_cols = [col for col in df_z.columns if "IDE-25" in col]

	df_100 = df_z[["Accelerated Days"] + ide_100_cols].copy()
	df_25 = df_z[["Accelerated Days"] + ide_25_cols].copy()

	# Round Accelerated days to nearest integer
	df_100["Accelerated Days"] = df_100["Accelerated Days"].round(1).astype(int)
	df_25["Accelerated Days"] = df_25["Accelerated Days"].round(1).astype(int)

	# Group by “Accelerated Days” and compute the mean over all IDE columns (yields df whose index is accelerated days, columns are individual devices)
	df_100_grouped = (
		df_100
		.groupby("Accelerated Days")[ide_100_cols]
		.mean()
		.sort_index()
	)
	df_25_grouped = (
		df_25
		.groupby("Accelerated Days")[ide_25_cols]
		.mean()
		.sort_index()
	)

	# Average across all devices for each day (index = accelerated days, value = mean)
	df_100_mean = df_100_grouped.mean(axis=1)
	df_25_mean = df_25_grouped.mean(axis=1)

	# Normalize values to the first index in the series (i.e. starting data)
	first_test_100 = df_100_mean.index.min()
	first_test_25  = df_25_mean.index.min()

	baseline_100 = df_100_mean.loc[first_test_100]
	baseline_25  = df_25_mean.loc[first_test_25]

	df_100_norm = df_100_mean / baseline_100
	df_25_norm  = df_25_mean  / baseline_25

	df_100_norm = df_100_norm.reset_index(name="Normalized Mean")
	df_25_norm  = df_25_norm.reset_index(name="Normalized Mean")

	# Plot averages, left is average, right is normalized average
	fig_impedance.add_trace(
		go.Scatter(
			x=df_100_mean.index,
			y=df_100_mean.values,
			mode="lines",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="100 um Average",
			# error_y=dict(type='data', array=sd_100, visible=True)
		),
		row=2,
		col=1,
	)
	fig_impedance.add_trace(
		go.Scatter(
			x=df_100_norm["Accelerated Days"],
			y=df_100_norm["Normalized Mean"],
			mode="lines",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="100 um (normalized) Average",
			# error_y=dict(type='data', array=sd_100_norm, visible=True),
			showlegend=False
		),
		row=2,
		col=2,
	)

	fig_impedance.add_trace(
		go.Scatter(
			x=df_25_mean.index,
			y=df_25_mean.values,
			mode="lines",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="25 um Average",
			# error_y=dict(type='data', array=sd_25, visible=True)
		),
		row=2,
		col=1,
	)
	fig_impedance.add_trace(
		go.Scatter(
			x=df_25_norm["Accelerated Days"],
			y=df_25_norm["Normalized Mean"],
			mode="lines",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="25 um (normalized) Average",
			# error_y=dict(type='data', array=sd_25_norm, visible=True),
			showlegend=False
		),
		row=2,
		col=2,
	)

	fig_impedance.update_yaxes(type="log", tick0=100, dtick=1, range=[2, 8], row=1, col=1)
	fig_impedance.update_yaxes(type="log", tick0=100, dtick=1, range=[2, 8], row=2, col=1)
	fig_impedance.update_yaxes(type="log", tick0=0, dtick=1, range=[-3, 3], row=1, col=2)
	fig_impedance.update_yaxes(type="log", tick0=0, dtick=1, range=[-3, 3], row=2, col=2)
	fig_impedance.update_xaxes(range=[0, max(accel_days)])

	add_vert_line(fig_impedance, 2, 2, 785.07, "PBS Replaced")

	return df_25_grouped, df_100_grouped

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