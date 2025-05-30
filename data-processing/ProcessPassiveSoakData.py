import math
import argparse
import pandas as pd
import os
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Processes passive soak test data (EIS only)
# All data should be stored in a single folder, with:
# //[subfolder] for each test point, labeled as accelerated days (eg "12" for 12 days in accelerated soak)
# Data for all devices are in one folder, including label and test number in title (eg "EIS_[days]_IDE-25-1_[timestamp]")

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

def parse_args():
	"""
	Parses the command line arguments.
	"""

	parser = argparse.ArgumentParser(description="Echem Data Processing")
	parser.add_argument(
		"-p",
		"--path",
		type=str,
		default="data//IDE",
		help="Folder where all data is stored."
	)
	return parser.parse_args()

def main():
	"""
	This is the main function that processes the following lifetime testing data:
		EIS: processes all data and plots trends of 1 kHz impedance magnitude
	"""

	args = parse_args()

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
	# top: individual traces; bottom: average normalized to t=0; left: impedance; right: normalized impedance to t=0
	fig_impedance = make_subplots(
		rows=2,
		cols=2,
		subplot_titles=(
			"Individual Electrodes",
			"Individual Electrodes (normalized to t=0)",
			"Average +/- SD",
			"Average (normalized) +/- SD"
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
	df_z = perform_data_analysis(args.path)

	# Calculate accelerated days from data
	df_z = calc_accel_days(df_z)

	# Save dataframe
	dataframe_path = f"{args.path}//saved-dataframes//"
	df_z.to_csv(f"{dataframe_path}df_z.csv", index=False)

	# Create plots
	plot_eis(fig_eis, df_z, f"{args.path}//manual-measurements")
	plot_eis(fig_eis, df_z, f"{args.path}//automated-eis")
	plot_impedance(fig_impedance, df_z)

	fig_eis.show()
	fig_impedance.show()
	# fig_eis_day0.show()

def perform_data_analysis(path):
	# List of subfolders
	folders = os.listdir(f"{path}//")

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
	for folder in folders:
		# If it's the saved dataframes or archived measurements, skip it
		if folder == "saved-dataframes" or folder == "archived-measurements":
			continue

		else:
			print(f"Processing data from '{folder}'...")
			files = os.listdir(f"{path}//{folder}")
			# Loop through each file, and process data
			for file in files:
				df_z = process_eis_data(df_z, file, f"{path}//{folder}")

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
	
	print(f"Plotting EIS curves vs time...")
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

	print("Plotting 1k impedance vs time...")

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
			df_100[sample] = z
			df_100_norm[sample] = z_norm
		elif "IDE-25" in sample:
			colorrgb = f"rgb(255,{colorn},{colorn})"
			df_25[sample] = z
			df_25_norm[sample] = z_norm
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
	# Calculate averages
	average_100 = df_100.mean(axis=1)
	sd_100 = df_100.std(axis=1)
	average_25 = df_25.mean(axis=1)
	sd_25 = df_25.std(axis=1)

	average_100_norm = df_100_norm.mean(axis=1)
	sd_100_norm = df_100_norm.std(axis=1)
	average_25_norm = df_25_norm.mean(axis=1)
	sd_25_norm = df_25_norm.std(axis=1)

	# Plot averages, left is average, right is normalized average
	fig_impedance.add_trace(
		go.Scatter(
			x=accel_days,
			y=average_100,
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
			x=accel_days,
			y=average_100_norm,
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
			x=accel_days,
			y=average_25,
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
			x=accel_days,
			y=average_25_norm,
			mode="lines",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="25 um (normalized) Average",
			# error_y=dict(type='data', array=sd_25_norm, visible=True),
			showlegend=False
		),
		row=2,
		col=2,
	)

	fig_impedance.update_yaxes(type="log", tick0=100, dtick=1, range=[3, 8], row=1, col=1)
	fig_impedance.update_yaxes(type="log", tick0=100, dtick=1, range=[3, 8], row=2, col=1)
	fig_impedance.update_yaxes(type="log", tick0=0, dtick=1, range=[-3, 1], row=1, col=2)
	fig_impedance.update_yaxes(type="log", tick0=0, dtick=1, range=[-3, 1], row=2, col=2)
	fig_impedance.update_xaxes(range=[0, max(accel_days)])


if __name__ == "__main__":
	main()