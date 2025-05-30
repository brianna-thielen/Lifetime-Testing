import argparse
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import re

# Processes encapsulation soak test data (RH)
# All data should be stored in a single folder

_START_DATE = datetime.datetime(2024, 11, 4, 15, 25)

def parse_args():
	"""
	Parses the command line arguments.
	"""

	parser = argparse.ArgumentParser(description="Echem Data Processing")
	parser.add_argument(
		"-p",
		"--path",
		type=str,
		default="data//encapsulation",
		help="Folder where all data is stored."
	)
	return parser.parse_args()

def main():
	args = parse_args()

	# Initiate plot
	# 1: RH vs time for all encapsulation samples (individual traces)
	fig_rhtemp = make_subplots(
		rows=2,
		cols=1,
		row_heights=[0.7, 0.3],
	)

	# Process data
	df_encap = process_rh_data(args.path)

	# Plot data
	plot_rh_data(df_encap, fig_rhtemp)
	fig_rhtemp.show()


def process_rh_data(path):
	"""
	Process RH data for encapsulation samples
	"""

	# Start by processing initial data from soak testing spreadsheet
	df_encap = process_spreadsheet_data(f"{path}//Soak Testing - LCP Encapsulation (Passive).csv")

	# Next, process arduino data (capacitive sensors)
	# This needs to go first because the resistive sensors do not capture temperature data
	df_encap = process_arduino_data(f"{path}//", df_encap)
	
	# Finally, process LCR data (resistive sensors)
	df_encap = process_lcr_data(f"{path}//", df_encap)

	return df_encap

def process_spreadsheet_data(filepath):
	"""
	Processes data from the initial spreadsheet and saves it to df_encap
	"""

	df_encap = pd.read_csv(filepath)
	print("Reading initial data...")

	# Delete irrelevant columns
	df_encap.drop(columns=df_encap.columns[[2,6,7,8,9,10,11,12,14,16,18,20]], inplace=True)

	# Set row 1 to header titles
	df_encap.columns = df_encap.iloc[1]
	
	# Rename columns with lost info
	df_encap.columns.values[5] = "RH: 25 um LCP (C2) (%)"
	df_encap.columns.values[6] = "RH: 100 um LCP (R1) (%)"
	df_encap.columns.values[7] = "RH: 100 um LCP (R2) (%)"
	df_encap.columns.values[8] = "RH: 100 um LCP (C1) (%)"
	df_encap.columns.values[9] = "RH: 100 um LCP (C2) (%)"

	# Data starts at row 9
	df_encap = df_encap[3:]

	# Sort through and convert to datetime
	df_encap["Datetime"] = df_encap["Date"] + " " + df_encap["Time"]
	df_encap["Datetime"] = pd.to_datetime(df_encap["Datetime"], format="%m/%d/%y %I:%M %p")

	# Delete date and time columns
	df_encap.drop(columns=["Date", "Time", "Real Days (Current Step)"], inplace=True)

	# Convert accel days column to float
	df_encap["Accel. Days (Total)"] = df_encap["Accel. Days (Total)"].astype(float)

	# Add real days column
	df_encap["Real Days"] = None
	for idx, row in df_encap.iterrows():
		date = row["Datetime"]
		days = (date - _START_DATE).total_seconds() / 60 / 60 / 24
		df_encap.loc[idx, "Real Days"] = days

	# Add ambient columns
	df_encap["RH: ambient"] = ""
	df_encap["Temp: ambient"] = ""

	# Sort df_encap by accel days
	df_encap = df_encap.sort_values(by="Accel. Days (Total)")

	return df_encap


def process_arduino_data(path, df_encap):
	"""
	Processes data from capacitive sensors connected via I2C bus on arduino
	"""

	print("Processing arduino data...")

	# List all files
	files = os.listdir(path)

	# Loop through files and process arduino data (capacitive sensors) first
	# we need temp data from here to calculate RH from resistive sensors
	for file in files:
		# Skip spreadsheet and LCR files (labeled with EIS)
		if "Soak Testing" in file or "EIS_" in file:
			continue

		# print(f"Processing {file}...")
		raw_data = pd.read_csv(f"{path}//{file}")

		header = str(raw_data.iloc[0])

		date_pattern = r'\d{4}[/.]\d{2}[/.]\d{2}[/ ]\d{2}[/:]\d{2}[/:]\d{2}'
		match = re.search(date_pattern, header)
		match = match.group()

		start_time = datetime.datetime.strptime(match, "%Y.%m.%d %H:%M:%S")

		# Loop through each time point and save data (if any)
		for idx, row in raw_data.iterrows():
			pd.set_option('display.max_colwidth', None)
			row = raw_data.iloc[idx,0]
			
			# Find minutes
			min_index = row.index("mins elapsed=") + 13
			mins = row[min_index:]
			min_index = mins.index(" ")
			mins = float(mins[:min_index])

			# Separate into strings for each device 
			ambient_index = row.index("ambient") + 9
			encap_c_100_1_index = row.index("ENCAP-C-100-1") + 14
			encap_c_100_2_index = row.index("ENCAP-C-100-2") + 14
			encap_c_25_2_index = row.index("ENCAP-C-25-2") + 13

			# Get info for ambient
			ambient = row[ambient_index:(encap_c_100_1_index-15)]
			rh_index = ambient.index("RH=") + 3
			temp_index = ambient.index("T=") + 2

			rh_ambient = float(ambient[rh_index:(temp_index-3)])
			temp_ambient = float(ambient[temp_index:])

			# Get info for ENCAP-C-100-1
			encap_c_100_1 = row[encap_c_100_1_index:(encap_c_100_2_index-15)]
			rh_index = encap_c_100_1.index("RH=") + 3
			temp_index = encap_c_100_1.index("T=") + 2

			rh_c_100_1 = float(encap_c_100_1[rh_index:(temp_index-3)])
			temp_c_100_1 = float(encap_c_100_1[temp_index:])

			# Get info for ENCAP-C-100-2
			encap_c_100_2 = row[encap_c_100_2_index:(encap_c_25_2_index-14)]
			rh_index = encap_c_100_2.index("RH=") + 3
			temp_index = encap_c_100_2.index("T=") + 2

			rh_c_100_2 = float(encap_c_100_2[rh_index:(temp_index-3)])
			temp_c_100_2 = float(encap_c_100_2[temp_index:])

			# Get info for ENCAP-C-25-2
			encap_c_25_2 = row[encap_c_25_2_index:]
			rh_index = encap_c_25_2.index("RH=") + 3
			temp_index = encap_c_25_2.index("T=") + 2

			rh_c_25_2 = float(encap_c_25_2[rh_index:(temp_index-3)])
			temp_c_25_2 = float(encap_c_25_2[temp_index:])

			# Find average temperature of soaked samples
			avg_temp = (temp_c_100_1 + temp_c_100_2 + temp_c_25_2) / 3

			# Convert minutes to datetime and calculate aging
			current_time = start_time + datetime.timedelta(minutes=mins)
			real_days = (current_time - _START_DATE).total_seconds() / 60 / 60 / 24

			previous_time = df_encap["Datetime"].iloc[-1]
			previous_accel_time = float(df_encap["Accel. Days (Total)"].iloc[-1])
			delta = current_time - previous_time
			delta = delta.total_seconds() / 60 / 60 / 24
			accel_time = delta * 2 ** ((avg_temp - 37)/10) + previous_accel_time
			
			# Create new row of dataframe
			new_row = pd.DataFrame(
				{
					"Temperature (C)": [avg_temp],
					"Accel. Days (Total)": [accel_time],
					"RH: 25 um LCP (C2) (%)": [rh_c_25_2],
					"RH: 100 um LCP (R1) (%)": [float("nan")],
					"RH: 100 um LCP (R2) (%)": [float("nan")],
					"RH: 100 um LCP (C1) (%)": [rh_c_100_1],
					"RH: 100 um LCP (C2) (%)": [rh_c_100_2],
					"Datetime": [current_time],
					"Real Days": [real_days],
					"RH: ambient": [rh_ambient],
					"Temp: ambient": [temp_ambient]
				},
				index=[df_encap.index.max()+1]
			)
			
			df_encap = pd.concat([df_encap, new_row])

	# Sort df_encap by accel days
	df_encap = df_encap.sort_values(by="Accel. Days (Total)")

	return df_encap


def process_lcr_data(path, df_encap):
	"""
	Processes data from resistive sensors connected via LCR and mux
	"""

	print("Processing LCR data...")
	
	# List all files
	files = os.listdir(path)

	# Loop through files, skipping any non-resistive data
	for file in files:
		# Skip spreadsheet and arduino files
		if "EIS_" not in file:
			continue

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

		# Find temp from df_encap
		# There may not be an exact match, so find the closest
		df_encap["Time Difference"] = abs(df_encap["Real Days"] - real_days)
		ind_closest = df_encap["Time Difference"].idxmin()
		temp = float(df_encap.loc[ind_closest, "Temperature (C)"])

		# Also calculate accelerated days from closest time point
		real_closest = float(df_encap.loc[ind_closest, "Real Days"])
		accel_closest = float(df_encap.loc[ind_closest, "Accel. Days (Total)"])
		accel_days = accel_closest + (real_days - real_closest) * 2 ** ((temp - 37) / 10)

		# Import data
		data = pd.read_csv(f"{path}//{file}")
		impedance = data["Impedance"].item() / 1000
		rh = (impedance / (1.92e22 * temp ** -7.57)) ** (1 / (-6.99 + 0.052 * temp - 0.000225 * temp ** 2))

		# Determine if data is from sensor R1 or R2
		if "R-100-1" in file:
			rh1 = rh
			rh2 = float("nan")
		elif "R-100-2" in file:
			rh1 = float("nan")
			rh2 = rh

		# Create new row of dataframe
		new_row = pd.DataFrame(
			{
				"Temperature (C)": [temp],
				"Accel. Days (Total)": [accel_days],
				"RH: 25 um LCP (C2) (%)": [float("nan")],
				"RH: 100 um LCP (R1) (%)": [rh1],
				"RH: 100 um LCP (R2) (%)": [rh2],
				"RH: 100 um LCP (C1) (%)": [float("nan")],
				"RH: 100 um LCP (C2) (%)": [float("nan")],
				"Datetime": [test_datetime],
				"Real Days": [real_days],
				"RH: ambient": [float("nan")],
				"Temp: ambient": [float("nan")]
			},
			index=[df_encap.index.max()+1]
		)
		
		df_encap = pd.concat([df_encap, new_row])

	# Sort df_encap by accel days
	df_encap = df_encap.sort_values(by="Accel. Days (Total)")

	return df_encap


def plot_rh_data(df_encap, fig_rhtemp):
	fig_rhtemp.add_trace(
		go.Scatter(
			x=df_encap["Accel. Days (Total)"],
			y=df_encap["RH: 25 um LCP (C2) (%)"],
			mode="lines",
			line=dict(width=1, color=f"rgb(255,100,100)"),
			name=f"RH: 25 um LCP (C2)",
			connectgaps=True
		),
		row=1,
		col=1,
	)

	fig_rhtemp.add_trace(
		go.Scatter(
			x=df_encap["Accel. Days (Total)"],
			y=df_encap["RH: 100 um LCP (C1) (%)"],
			mode="lines",
			line=dict(width=1, color=f"rgb(150,150,255)"),
			name=f"RH: 100 um LCP (C1)",
			connectgaps=True
		),
		row=1,
		col=1,
	)

	fig_rhtemp.add_trace(
		go.Scatter(
			x=df_encap["Accel. Days (Total)"],
			y=df_encap["RH: 100 um LCP (C2) (%)"],
			mode="lines",
			line=dict(width=1, color=f"rgb(100,100,255)"),
			name=f"RH: 100 um LCP (C2)",
			connectgaps=True
		),
		row=1,
		col=1,
	)

	fig_rhtemp.add_trace(
		go.Scatter(
			x=df_encap["Accel. Days (Total)"],
			y=df_encap["RH: 100 um LCP (R1) (%)"],
			mode="lines",
			line=dict(width=1, color=f"rgb(0,0,255)"),
			name=f"RH: 100 um LCP (R1)",
			connectgaps=True
		),
		row=1,
		col=1,
	)

	fig_rhtemp.add_trace(
		go.Scatter(
			x=df_encap["Accel. Days (Total)"],
			y=df_encap["RH: 100 um LCP (R2) (%)"],
			mode="lines",
			line=dict(width=1, color=f"rgb(0,0,200)"),
			name=f"RH: 100 um LCP (R2)",
			connectgaps=True
		),
		row=1,
		col=1,
	)

	fig_rhtemp.add_trace(
		go.Scatter(
			x=df_encap["Accel. Days (Total)"],
			y=df_encap["RH: ambient"],
			mode="lines",
			line=dict(width=1, color=f"rgb(200,200,200)"),
			name=f"RH: Ambient"
		),
		row=1,
		col=1,
	)

	# Temperature
	fig_rhtemp.add_trace(
		go.Scatter(
			x=df_encap["Accel. Days (Total)"],
			y=df_encap["Temperature (C)"],
			mode="lines",
			line=dict(width=1, dash='dash', color=f"rgb(170,120,200)"),
			name=f"Temperature: Sample Average",
			connectgaps=True
		),
		row=2,
		col=1,
	)

	fig_rhtemp.add_trace(
		go.Scatter(
			x=df_encap["Accel. Days (Total)"],
			y=df_encap["Temp: ambient"],
			mode="lines",
			line=dict(width=1, dash='dash', color=f"rgb(200,200,200)"),
			name=f"Temperature: Ambient",
			connectgaps=True
		),
		row=2,
		col=1,
	)

	fig_rhtemp.update_xaxes(title_text="Accelerated Time (days)")
	fig_rhtemp.update_yaxes(title_text="Relative Humidity (%)", row=1, col=1)
	fig_rhtemp.update_yaxes(title_text="Temperature (C)", row=2, col=1)

if __name__ == "__main__":
	main()