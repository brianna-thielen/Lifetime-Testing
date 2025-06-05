import argparse
import pandas as pd
import os
import math
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_PATH = "./data/SIROF vs Pt"
PLOT_PATH = "./data/Plots"

# Min and Max colors for plotting
_MIN_COLOR = 0
_MAX_COLOR = 200

# Electrode and test information
_ELECTRODE_DATA = pd.DataFrame({
	"Channel Name": ['IR01', 'IR02', 'IR03', 'IR04', 'IR05', 'IR06', 'IR07', 'IR08', 'IR09', 'IR10', 
				  'PT01', 'PT02', 'PT03', 'PT04', 'PT05', 'PT06', 'PT07', 'PT08', 'PT09', 'PT10'], 
	"Channel Number": ['A-024', 'A-025', 'A-026', 'A-027', 'A-011', 'A-023', 'A-022', 'A-021', 'A-020', 'A-004', 
					'A-028', 'A-029', 'A-030', 'A-031', 'A-015', 'A-019', 'A-018', 'A-017', 'A-016', 'A-000'], 
	"Geometric Surface Area (mm^2)": [1.7, 0.8, 1.2, 1.2, 1.4, 1.4, 0.6, 0.8, 1.0, 1.2,
								   3.2, 2.2, 2.1, 2.3, 2.1, 2.6, 2.1, 1.7, 1.6, 2.8],
	"Pulse Amplitude (uA)": [600, 600, 600, 600, 0, 400, 400, 400, 400, 0,
						  400, 400, 400, 400, 0, 200, 200, 200, 200, 0],
	"Pulse Frequency (Hz)": [50, 50, 50, 50, 0, 50, 50, 50, 50, 0,
						  50, 50, 50, 50, 0, 50, 50, 50, 50, 0],
})
_STIM_FREQUENCY = 50 #Hz
_START_DATE = datetime.datetime(2024, 11, 7, 7, 43)
_ELECTRODES_TO_IGNORE = pd.DataFrame({
    "Channel Name": ['IR07'],
    "Ignore From": [datetime.datetime(2024, 12, 11)]
})



def process_coating_soak_data(plot_on=False):
	"""
	This is the main function that processes the following lifetime testing data:
	CV: processes the last curve from each day tested and plots trends of area inside cv curve
	EIS: processes all data and plots trends of 1 kHz impedance magnitude
	VT: all data exists in a spreadsheet that must be downloaded from google drive
	"""

	print("Processing SIROF vs Pt soak test data...")

	# Initiate plots
	# fig_cv (4x5 subplots): CV vs time for each electrode individually
	fig_cv = make_subplots(
		rows=4, 
		cols=5,
		subplot_titles=(
			"Pt Sample 1 (400 uA / 3.2 mm^2)", "Pt Sample 2 (400 uA / 2.2 mm^2)", "Pt Sample 3 (400 uA / 2.1 mm^2)", "Pt Sample 4 (400 uA / 2.3 mm^2)", "Pt Sample 5 (no stim / 2.1 mm^2)", 
			"Pt Sample 6 (200 uA / 2.6 mm^2)", "Pt Sample 7 (200 uA / 2.1 mm^2)", "Pt Sample 8 (200 uA / 1.7 mm^2)", "Pt Sample 9 (200 uA / 1.6 mm^2)", "Pt Sample 10 (no stim / 2.8 mm^2)",
			"Ir Sample 1 (600 uA / 1.7 mm^2)", "Ir Sample 2 (600 uA / 0.8 mm^2)", "Ir Sample 3 (600 uA / 1.2 mm^2)", "Ir Sample 4 (600 uA / 1.2 mm^2)", "Ir Sample 5 (no stim / 1.4 mm^2)", 
			"Ir Sample 6 (400 uA / 1.4 mm^2)", "Ir Sample 7 (400 uA / 0.6 mm^2)", "Ir Sample 8 (400 uA / 0.8 mm^2)", "Ir Sample 9 (400 uA / 1.0 mm^2)", "Ir Sample 10 (no stim / 1.2 mm^2)"
			),
		x_title="Voltage (V)",
		y_title="Current Density (A/mm^2)"
	)
	fig_cv.update_layout(title_text="CV Curve vs Time (light=day 0, dark=most recent)")

	# fig_cvarea: Area vs time for all electrodes 
	# top: individual traces; bottom: average normalized to t=0
	fig_cvarea = make_subplots(
		rows=2,
		cols=1,
		subplot_titles=(
			"Individual Electrodes",
			"Average +/- SD (normalized to t=0)"
		)
	)
	fig_cvarea.update_layout(title_text="CV Area vs Time")

	# fig_eis (4x5 subplots): EIS vs time for each electrode individually
	fig_eis = make_subplots(
		rows=4, 
		cols=5,
		specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}], 
		 	   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
			   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
		 	   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
		subplot_titles=(
			"Pt Sample 1 (400 uA / 3.2 mm^2)", "Pt Sample 2 (400 uA / 2.2 mm^2)", "Pt Sample 3 (400 uA / 2.1 mm^2)", "Pt Sample 4 (400 uA / 2.3 mm^2)", "Pt Sample 5 (no stim / 2.1 mm^2)", 
			"Pt Sample 6 (200 uA / 2.6 mm^2)", "Pt Sample 7 (200 uA / 2.1 mm^2)", "Pt Sample 8 (200 uA / 1.7 mm^2)", "Pt Sample 9 (200 uA / 1.6 mm^2)", "Pt Sample 10 (no stim / 2.8 mm^2)",
			"Ir Sample 1 (600 uA / 1.7 mm^2)", "Ir Sample 2 (600 uA / 0.8 mm^2)", "Ir Sample 3 (600 uA / 1.2 mm^2)", "Ir Sample 4 (600 uA / 1.2 mm^2)", "Ir Sample 5 (no stim / 1.4 mm^2)", 
			"Ir Sample 6 (400 uA / 1.4 mm^2)", "Ir Sample 7 (400 uA / 0.6 mm^2)", "Ir Sample 8 (400 uA / 0.8 mm^2)", "Ir Sample 9 (400 uA / 1.0 mm^2)", "Ir Sample 10 (no stim / 1.2 mm^2)"
			),
		x_title="Frequency (Hz)",
		y_title="Left/Solid: Impedance Magnitude (Ohms)\nRight/Dashed: Impedance Phase (deg)"
	)
	fig_eis.update_layout(title_text="EIS Curve vs Time (light=day 0, dark=most recent)")

	# fig_impedance: 1k impedance vs time 
	# top: individual traces; bottom: average normalized to t=0; left: manual measurements (LCR); right: intan measurements
	fig_impedance = make_subplots(
		rows=2,
		cols=2,
		subplot_titles=(
			"Individual Electrodes (LCR)",
			"Individual Electrodes (Intan)",
			"Average +/- SD (LCR)",
			"Average (Intan)"
		),
		x_title="Accelerated Time (days)",
		y_title="Impedance Magnitude * GSA (ohms*mm^2)"
	)
	fig_impedance.update_layout(title_text="1 kHz Impedance Magnitude vs Time")

	# fig_cic: CIC vs time
	# top: individual traces; bottom: average normalized to t=0; left: manual measurements (scope, 500 us pulse); right: intan measurements (1000 us pulse)
	fig_cic = make_subplots(
		rows=2,
		cols=2,
		subplot_titles=(
			"Individual Electrodes w/ 500 us pulse (scope)",
			"Individual Electrodes w/ 1000 us pulse (Intan)",
			"Average +/- SD w/ 500 us pulse (scope)",
			"Average w/ 1000 us pulse (Intan)"
		),
		x_title="Accelerated Time (days)",
		y_title="Charge Injection Capacity (uC/cm^2)"
	)
	fig_cic.update_layout(title_text="Charge Injection Capacity vs Time")

	# fig_cv_day0: CV for day 0 for all electrodes (individual traces)
	fig_cv_day0 = go.Figure()
	fig_cv_day0.update_layout(title_text="Initial CV curve for All Electrodes")

	# fig_eis_day0: EIS for day 0 for all electrodes (individual traces)
	fig_eis_day0 = make_subplots(
		rows=2, 
		cols=1,
	)
	fig_eis_day0.update_layout(title_text="Initial EIS curve for All Electrodes")

	df_experiment, df_cvarea_smu, df_z_lcr, df_z_intan, df_cic500_scope, df_cic1000_intan = perform_data_analysis(DATA_PATH)

	# Plot data and save
	plot_cv(fig_cv, df_experiment, f"{DATA_PATH}//manual-measurements")
	plot_cvarea(fig_cvarea, df_experiment, df_cvarea_smu)
	# plot_cv_day0(fig_cv_day0, DATA_PATH)

	plot_eis(fig_eis, df_experiment, f"{DATA_PATH}//manual-measurements")
	plot_impedance(fig_impedance, df_experiment, df_z_lcr, df_z_intan)
	# plot_eis_day0(fig_eis_day0, DATA_PATH)

	plot_cic(fig_cic, df_experiment, df_cic500_scope, df_cic1000_intan)

	fig_cv.write_html(f"{PLOT_PATH}/SIROF-vs-Pt_CV-vs-Time.html")
	fig_cvarea.write_html(f"{PLOT_PATH}/SIROF-vs-Pt_CV-Area-vs-Time.html")

	fig_eis.write_html(f"{PLOT_PATH}/SIROF-vs-Pt_EIS-vs-Time.html")
	fig_impedance.write_html(f"{PLOT_PATH}/SIROF-vs-Pt_Impedance-vs-Time.html")

	fig_cic.write_html(f"{PLOT_PATH}/SIROF-vs-Pt_CIC-vs-Time.html")

	if plot_on:
		fig_cv.show()
		fig_cvarea.show()
		# fig_cv_day0.show()

		fig_eis.show()
		fig_impedance.show()
		# fig_eis_day0.show()

		fig_cic.show()

	# Extract most recent values of CIC and impedance (intan measurements only)
	pt_mask = [col for col in df_z_intan.columns if col.startswith('PT')]
	ir_mask = [col for col in df_z_intan.columns if col.startswith('IR')]
	
	z_pt = df_z_intan[pt_mask].iloc[-1]
	z_ir = df_z_intan[ir_mask].iloc[-1]

	cic_pt = df_cic1000_intan[pt_mask].iloc[-1]
	cic_ir = df_cic1000_intan[ir_mask].iloc[-1]

	# Extract days from df_experiment
	days = df_experiment['Accelerated Days'].iloc[-1]

	# df_experiment.to_csv(f"{PLOT_PATH}/SIROF-vs-Pt_Soak_Data.csv", index=False)

	return days, cic_pt, cic_ir, z_pt, z_ir

def perform_data_analysis(path):
	# List of test points (subfolders)
	folders = os.listdir(f"{path}//")

	# Create dataframes for processed data
	df_experiment = pd.DataFrame({
		'Measurement Datetime': None,
		'Pulsing On': None,
		'Temperature (C)': [],
		'Accelerated Days': [],
		'Real Days': [],
		'Pulses': []
	})
	df_cvarea_smu = pd.DataFrame({
		'Measurement Datetime': None,
		'Real Days': [],
		'IR01': [], 'IR02': [], 'IR03': [], 'IR04': [], 'IR05': [], 'IR06': [], 'IR07': [], 'IR08': [], 'IR09': [], 'IR10': [], 
		'PT01': [], 'PT02': [], 'PT03': [], 'PT04': [], 'PT05': [], 'PT06': [], 'PT07': [], 'PT08': [], 'PT09': [], 'PT10': []
	})
	df_z_lcr = pd.DataFrame({
		'Measurement Datetime': None,
		'Real Days': [],
		'IR01': [], 'IR02': [], 'IR03': [], 'IR04': [], 'IR05': [], 'IR06': [], 'IR07': [], 'IR08': [], 'IR09': [], 'IR10': [], 
		'PT01': [], 'PT02': [], 'PT03': [], 'PT04': [], 'PT05': [], 'PT06': [], 'PT07': [], 'PT08': [], 'PT09': [], 'PT10': []
	})
	df_z_intan = pd.DataFrame({
		'Measurement Datetime': None,
		'Real Days': [],
		'IR01': [], 'IR02': [], 'IR03': [], 'IR04': [], 'IR05': [], 'IR06': [], 'IR07': [], 'IR08': [], 'IR09': [], 'IR10': [], 
		'PT01': [], 'PT02': [], 'PT03': [], 'PT04': [], 'PT05': [], 'PT06': [], 'PT07': [], 'PT08': [], 'PT09': [], 'PT10': []
	})
	df_cic500_scope = pd.DataFrame({
		'Measurement Datetime': None,
		'Real Days': [],
		'IR01': [], 'IR02': [], 'IR03': [], 'IR04': [], 'IR05': [], 'IR06': [], 'IR07': [], 'IR08': [], 'IR09': [], 'IR10': [], 
		'PT01': [], 'PT02': [], 'PT03': [], 'PT04': [], 'PT05': [], 'PT06': [], 'PT07': [], 'PT08': [], 'PT09': [], 'PT10': []
	})
	df_cic1000_intan = pd.DataFrame({
		'Measurement Datetime': None,
		'Real Days': [],
		'IR01': [], 'IR02': [], 'IR03': [], 'IR04': [], 'IR05': [], 'IR06': [], 'IR07': [], 'IR08': [], 'IR09': [], 'IR10': [], 
		'PT01': [], 'PT02': [], 'PT03': [], 'PT04': [], 'PT05': [], 'PT06': [], 'PT07': [], 'PT08': [], 'PT09': [], 'PT10': []
	})

	# First, process VT spreadsheet (this contains early data)
	df_experiment, df_cic500_scope = process_vt_spreadsheet_data(f"{path}//Soak Testing - SIROF (Active).csv", df_experiment, df_cic500_scope)

	# Next, process manual measurements
	df_experiment, df_cvarea_smu, df_z_lcr = process_manual_data(f"{path}//manual-measurements//", df_experiment, df_cvarea_smu, df_z_lcr)

	# Finally, process intan measurements
	df_experiment, df_cic1000_intan, df_z_intan = process_intan_data(f"{path}//", df_experiment, df_cic1000_intan, df_z_intan)

	df_experiment = calc_accel_days_and_pulses(df_experiment)

	return df_experiment, df_cvarea_smu, df_z_lcr, df_z_intan, df_cic500_scope, df_cic1000_intan

def process_vt_spreadsheet_data(filepath, df_experiment, df_cic500_scope):
	"""
	Processes data from VT spreadsheet
	
	Returns dataframe of formated data
	"""

	df = pd.read_csv(filepath)
	# Set row 1 to header titles
	df.columns = df.iloc[1]

	# Data starts at row 10
	df = df[11:]

	# Loop through each time point and save data (if any)
	for idx, row in df.iterrows():
		# Pull data from spreadsheet row
		# for accel_days, we need to strip any commas first
		accel_days = row["Accel. Days (Total)"]
		accel_days = accel_days.replace(",", "")
		accel_days = float(accel_days)
		# the others aren't formatted with commas, so can be converted directly
		temp = float(row["Temp\n(deg C)"])
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

		# If there is VT data, process it
		if math.isnan(float(row["PT01"])) == False:
			# Loop through each electrode
			for x in range(20):
				# Pull CIC data for current electrode from row
				label = row.index[x+8]
				# Check for missing data
				try:
					max_current = float(row[label])
				except ValueError:
					max_current = 0.0

				# Pull data for current electrode from dataframe
				elec_ind = _ELECTRODE_DATA[_ELECTRODE_DATA["Channel Name"] == label.upper()].index[0]
				gsa = float(_ELECTRODE_DATA.loc[elec_ind, "Geometric Surface Area (mm^2)"])

				# Calculate CIC - data in CSV is max current w/ 500 us pulse
				cic = max_current * 50 / gsa

				# Append values to df_cic500_scope
				# For CIC, save it in the same row if it's the same time point but a new sample
				if len(df_cic500_scope) == 0:
					ind = 0
				elif test_datetime not in df_cic500_scope["Measurement Datetime"].values:
					ind = len(df_cic500_scope)
				else:
					ind = df_cic500_scope[df_cic500_scope["Measurement Datetime"] == test_datetime].index[0]

				df_cic500_scope.loc[ind, "Measurement Datetime"] = test_datetime
				df_cic500_scope.loc[ind, "Real Days"] = real_days
				df_cic500_scope.loc[ind, label] = cic

	return df_experiment, df_cic500_scope

def process_intan_data(path, df_experiment, df_cic1000_intan, df_z_intan):

	files = os.listdir(path)

	# Loop through each file and pull data
	for file in files:

		# Skip files that are not .csv
		if not file.endswith(".csv"):
			continue

		# Skip the summary file
		if file == "Soak Testing - SIROF (Active).csv":
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

		if len(df) < 20:
			continue

		# If there was a CIC measurement, the test start is ~8 mins before save time, and we need to save to df_cic_intan
		if not math.isnan(df.loc[0, "Charge Injection Capacity @ 1000 us (uC/cm^2)"]):
			test_start_datetime = datetime.datetime(year, month, day, hour, minutes, seconds) - datetime.timedelta(minutes=8)

			# Calculate real days from start date
			real_days_start = test_start_datetime - _START_DATE
			real_days_start = real_days_start.total_seconds() / 60 / 60 / 24

			# Save measurement time
			ind_cic = len(df_cic1000_intan)
			df_cic1000_intan.loc[ind_cic, "Measurement Datetime"] = test_start_datetime
			df_cic1000_intan.loc[ind_cic, "Real Days"] = real_days_start

			ind_z = len(df_z_intan)
			df_z_intan.loc[ind_z, "Measurement Datetime"] = test_start_datetime
			df_z_intan.loc[ind_z, "Real Days"] = real_days_start
		# If there was not a CIC measurement, the test start is ~10 seconds before save time, and we don't need to save df_cic_intan
		else:
			test_start_datetime = datetime.datetime(year, month, day, hour, minutes, seconds) - datetime.timedelta(seconds=10)

			# Calculate real days from start date
			real_days_start = test_start_datetime - _START_DATE
			real_days_start = real_days_start.total_seconds() / 60 / 60 / 24

			# Save measurement time
			ind_z = len(df_z_intan)
			df_z_intan.loc[ind_z, "Measurement Datetime"] = test_start_datetime
			df_z_intan.loc[ind_z, "Real Days"] = real_days_start

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
			df_z_intan.loc[ind_z, sample] = impedance

			if not math.isnan(cic):
				df_cic1000_intan.loc[ind_cic, sample] = cic

		# Sort by datetime
		df_experiment = df_experiment.sort_values(by="Measurement Datetime")
		df_cic1000_intan = df_cic1000_intan.sort_values(by="Measurement Datetime")
		df_z_intan = df_z_intan.sort_values(by="Measurement Datetime")

	return df_experiment, df_cic1000_intan, df_z_intan

def process_manual_data(path, df_experiment, df_cvarea_smu, df_z_lcr):
	files = os.listdir(path)
	numfiles = len(files)

	# Loop through each file, pull data, and plot
	for n in range(numfiles):
		file = files[n]

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

		# Temp data not saved for most files in this dataset
		# instead it'll correspond to the closest manual VT measurement
		# as such, don't save it in df_experiment
		# will save to df_cvarea_smu or df_z_lcr after test type is determined
		
		# Determine the test type
		# for CV, we only want to process cycle 2 (or cycle 30 for day 0 on 2024-11-05 and 06)
		if "CV" in file and (("cycle2_" in file and "2024110" not in file) or ("cycle30_" in file)):
			# Save timestamp
			ind_cv = len(df_cvarea_smu)
			df_cvarea_smu.loc[ind_cv, "Measurement Datetime"] = test_datetime
			df_cvarea_smu.loc[ind_cv, "Real Days"] = real_days

			# Import file
			df = pd.read_csv(f"{path}{file}")

			# Calculate area inside the curve
			# Uses the shoelace theorem (code from chat gpt, verified in excel)
			volts = df["Voltage (V)"]
			microamps = df["Current (A)"] * 10**6
			cv_area = 0.5 * np.abs(np.dot(volts, np.roll(microamps,1)) - np.dot(microamps, np.roll(volts,1)))

			# Determine sample number
			if "PT" in file:
				ind = file.index("PT")
			elif "IR" in file:
				ind = file.index("IR")
			sample = file[ind:ind+4]

			# Save CV area to df_cvarea_smu
			df_cvarea_smu.loc[ind_cv, sample] = cv_area

		elif "EIS" in file:
			# Save timestamp
			ind_eis = len(df_z_lcr)
			df_z_lcr.loc[ind_eis, "Measurement Datetime"] = test_datetime
			df_z_lcr.loc[ind_eis, "Real Days"] = real_days

			# Import file
			df = pd.read_csv(f"{path}{file}")

			# find 1k impedance
			index = (np.abs(df["Frequency"] - 1000)).idxmin()
			z_1k = df["Impedance"][index]

			# Determine sample number
			if "PT" in file:
				ind = file.index("PT")
			elif "IR" in file:
				ind = file.index("IR")
			sample = file[ind:ind+4]

			# Save 1k impedance to df_z_lcr
			df_z_lcr.loc[ind_eis, sample] = z_1k

		# Sort by datetime
		df_experiment = df_experiment.sort_values(by="Measurement Datetime")
		df_cvarea_smu = df_cvarea_smu.sort_values(by="Measurement Datetime")
		df_z_lcr = df_z_lcr.sort_values(by="Measurement Datetime")

	return df_experiment, df_cvarea_smu, df_z_lcr
			
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

def plot_cv(fig_cv, df_experiment, path):
	"""
	Plots CV curves in their respective subplot of fig_cv
	
	This has to loop back through all manual measurement files to find the CV curves
	"""
	
	files = os.listdir(path)
	numfiles = len(files)

	# Find most recent date and number of days for plotting color purposes
	dates = [file[:-4] for file in files]
	dates = [file[-14:] for file in dates]
	dates = [int(file) for file in dates]
	latest = datetime.datetime.strptime(str(max(dates)), '%Y%m%d%H%M%S')
	latest_days = (latest - _START_DATE).total_seconds() / 60 / 60 / 24

	# Loop through each file, pull data, and plot
	for n in range(numfiles):
		file = files[n]

		# We only want to plot cycle 2 (or cycle 30 for day 0 on 2024-11-05 and 06)
		if not ("CV" in file and (("cycle2_" in file and "2024110" not in file) or ("cycle30_" in file))):
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

		# Set plot color based on age
		colort = int(((latest_days)-real_days)*(_MAX_COLOR-_MIN_COLOR)/(latest_days))

		# Find accel days and number of pulses from df_experiment
		# There may not be an exact match, so find the closest
		df_experiment["Time Difference"] = abs(df_experiment["Real Days"] - real_days)
		ind_closest = df_experiment["Time Difference"].idxmin()
		accel_days = df_experiment.loc[ind_closest, "Accelerated Days"]
		pulses = df_experiment.loc[ind_closest, "Pulses"]

		# Find material and sample number
		if "PT" in file:
			ind = file.index("PT")
			material = "PT"
		elif "IR" in file:
			ind = file.index("IR")
			material = "IR"
		sample = file[ind:ind+4]
		sample_num = int(file[ind+2:ind+4])

		# Check if the current file is in the ignore list
		if sample in _ELECTRODES_TO_IGNORE["Channel Name"].tolist():
			ignore_date = _ELECTRODES_TO_IGNORE.loc[_ELECTRODES_TO_IGNORE["Channel Name"] == sample, "Ignore From"].item()
			if ignore_date < test_datetime:
				continue

		# Rows 1 and 2 are Pt; rows 3 and 4 are Ir
		if material == "PT" and sample_num < 6:
			plot_row = 1
			plot_col = sample_num
		elif material == "PT" and sample_num >= 6:
			plot_row = 2
			plot_col = sample_num-5
		elif material == "IR" and sample_num < 6:
			plot_row = 3
			plot_col = sample_num
		elif material == "IR" and sample_num >= 6:
			plot_row = 4
			plot_col = sample_num-5

		gsa = _ELECTRODE_DATA.loc[_ELECTRODE_DATA["Channel Name"] == sample, "Geometric Surface Area (mm^2)"].astype(float)
		gsa = gsa.values[0]

		# Import data
		df = pd.read_csv(f"{path}//{file}")

		# fig_cv (4x5 subplots): CV vs time for each electrode individually
		df_normalized = df["Current (A)"] / gsa

		fig_cv.add_trace(
			go.Scatter(
				x=df["Voltage (V)"],
				y=df_normalized,
				mode="lines",
				line=dict(width=1, color=f"rgb({colort},{colort},{colort})"),
				name=f"day {accel_days}",
				showlegend=False
			),
			row=plot_row,
			col=plot_col,
		)

		fig_cv.update_xaxes(range=[-0.6,0.8], tick0=0, dtick=0.4)
		# fig_cv.update_yaxes(range=[-5e-6, 10e-6])
	
def plot_eis(fig_eis, df_experiment, path):
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

	# Loop through each file, pull data, and plot
	for n in range(numfiles):
		file = files[n]

		# We only want to plot EIS data
		if not "EIS" in file:
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

		# Set plot color based on age
		colort = int(((latest_days)-real_days)*(_MAX_COLOR-_MIN_COLOR)/(latest_days))

		# Find accel days and number of pulses from df_experiment
		# There may not be an exact match, so find the closest
		df_experiment["Time Difference"] = abs(df_experiment["Real Days"] - real_days)
		ind_closest = df_experiment["Time Difference"].idxmin()
		accel_days = df_experiment.loc[ind_closest, "Accelerated Days"]
		pulses = df_experiment.loc[ind_closest, "Pulses"]

		# Find material and sample number
		if "PT" in file:
			ind = file.index("PT")
			material = "PT"
		elif "IR" in file:
			ind = file.index("IR")
			material = "IR"
		sample = file[ind:ind+4]
		sample_num = int(file[ind+2:ind+4])

		# Check if the current file is in the ignore list
		if sample in _ELECTRODES_TO_IGNORE["Channel Name"].tolist():
			ignore_date = _ELECTRODES_TO_IGNORE.loc[_ELECTRODES_TO_IGNORE["Channel Name"] == sample, "Ignore From"].item()
			if ignore_date < test_datetime:
				continue

		# Rows 1 and 2 are Pt; rows 3 and 4 are Ir
		if material == "PT" and sample_num < 6:
			plot_row = 1
			plot_col = sample_num
		elif material == "PT" and sample_num >= 6:
			plot_row = 2
			plot_col = sample_num-5
		elif material == "IR" and sample_num < 6:
			plot_row = 3
			plot_col = sample_num
		elif material == "IR" and sample_num >= 6:
			plot_row = 4
			plot_col = sample_num-5

		# Import data
		df = pd.read_csv(f"{path}//{file}")

		# fig_eis (4x5 subplots): EIS vs time for each electrode individually
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

def plot_cvarea(fig_cvarea, df_experiment, df_cvarea_smu):
	"""
	Plots area inside the CV curve vs time
	"""

	# Create dataframes for calculating average values
	df_pt = pd.DataFrame()
	df_ir = pd.DataFrame()

	# Loop through each sample
	for sample in _ELECTRODE_DATA["Channel Name"]:
		# Sample Number
		sample_num = int(sample[-2:])

		# Pull CV area for current channel
		cv_area = df_cvarea_smu[sample]
		cv_area.dropna(inplace=True)

		ind_cv = cv_area.index.to_list()

		# Pull real days and datetime for current channel
		real_days = df_cvarea_smu["Real Days"]
		real_days = real_days.loc[ind_cv]

		test_dates = df_cvarea_smu["Measurement Datetime"]
		test_dates = test_dates.loc[ind_cv]

		# Check if the current file is in the ignore list
		if sample in _ELECTRODES_TO_IGNORE["Channel Name"].tolist():
			ignore_date = _ELECTRODES_TO_IGNORE.loc[_ELECTRODES_TO_IGNORE["Channel Name"] == sample, "Ignore From"].item()
			# Remove values at test dates >= ignore date
			ind_ignore = test_dates >= ignore_date
			cv_area[ind_ignore] = float('nan')
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

		# Normalize CV area to GSA
		cv_area = cv_area / gsa
		cv_area = cv_area.tolist()

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'

		# Set plot color for current part number
		# Also add data to dataframe for average plot
		colorn = int((10-sample_num)*(_MAX_COLOR-_MIN_COLOR)/10)
		if "PT" in sample:
			colorrgb = f"rgb({colorn},{colorn},255)"
			df_pt[sample] = cv_area
		elif "IR" in sample:
			colorrgb = f"rgb(255,{colorn},{colorn})"
			if len(cv_area) < len(df_ir):
				# If the cv_area is shorter than df_ir, fill with NaN
				cv_area += [float('nan')] * (len(df_ir) - len(cv_area))
			df_ir[sample] = cv_area
		else:
			colorrgb = f"rgb({colorn},{colorn},{colorn})"
		
		# Plot individual trace
		fig_cvarea.add_trace(
			go.Scatter(
				x=accel_days,
				y=cv_area,
				mode="lines+markers",
				line=dict(width=1, dash=linetype, color=colorrgb),
				name=trace_label
			),
			row=1,
			col=1,
		)

	# Bottom subplot - average of normalized data
	# Calculate averages
	average_pt = df_pt.mean(axis=1)
	sd_pt = df_pt.std(axis=1)
	
	average_ir = df_ir.mean(axis=1)
	sd_ir = df_ir.std(axis=1)

	# Plot pt average
	fig_cvarea.add_trace(
		go.Scatter(
			x=accel_days,
			y=average_pt,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="Pt Average",
			error_y=dict(type='data', array=sd_pt, visible=True)
		),
		row=2,
		col=1,
	)

	# Plot ir average
	fig_cvarea.add_trace(
		go.Scatter(
			x=accel_days,
			y=average_ir,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="Ir Average",
			error_y=dict(type='data', array=sd_ir, visible=True)
		),
		row=2,
		col=1,
	)

	fig_cvarea.update_yaxes(title_text="CV Area (W/mm^2)")
	fig_cvarea.update_xaxes(title_text="Accelerated Time (days)", range=[0, max(accel_days)])

	# Add vertical line with label where I increased stim and replaced the saline
	add_vert_line(fig_cvarea, 2, 1, 423.62, "Stim Increased")
	add_vert_line(fig_cvarea, 2, 1, 982.98, "PBS Replaced")

def plot_impedance(fig_impedance, df_experiment, df_z_lcr, df_z_intan):
	"""
	Plots 1k impedance vs time
	Left is manual data, right is intan data
	"""

	# Create dataframes for calculating average values
	df_pt_lcr = pd.DataFrame()
	df_ir_lcr = pd.DataFrame()
	df_pt_intan = pd.DataFrame()
	df_ir_intan = pd.DataFrame()

	# Loop through each sample
	for sample in _ELECTRODE_DATA["Channel Name"]:
		# Sample Number
		sample_num = int(sample[-2:])

		# Pull Z for current channel
		z_lcr = df_z_lcr[sample]
		z_lcr.dropna(inplace=True)

		z_intan = df_z_intan[sample]
		z_intan.dropna(inplace=True)

		ind_lcr = z_lcr.index.to_list()
		ind_intan = z_intan.index.to_list()

		# Pull real days and datetime for current channel
		real_days_lcr = df_z_lcr["Real Days"]
		real_days_lcr = real_days_lcr.loc[ind_lcr]
		test_dates_lcr = df_z_lcr["Measurement Datetime"]
		test_dates_lcr = test_dates_lcr.loc[ind_lcr]

		real_days_intan = df_z_intan["Real Days"]
		real_days_intan = real_days_intan.loc[ind_intan]
		test_dates_intan = df_z_intan["Measurement Datetime"]
		test_dates_intan = test_dates_intan.loc[ind_intan]

		# Check if the current file is in the ignore list
		if sample in _ELECTRODES_TO_IGNORE["Channel Name"].tolist():
			ignore_date = _ELECTRODES_TO_IGNORE.loc[_ELECTRODES_TO_IGNORE["Channel Name"] == sample, "Ignore From"].item()
			# Remove values at test dates >= ignore date
			ind_ignore_lcr = test_dates_lcr >= ignore_date
			ind_ignore_intan = test_dates_intan >= ignore_date
			z_lcr[ind_ignore_lcr] = float('nan')
			z_intan[ind_ignore_intan] = float('nan')
			real_days_lcr[ind_ignore_lcr] = float('nan')
			real_days_intan[ind_ignore_intan] = float('nan')

		# Find accel days and number of pulses from df_experiment
		# Need to loop through each time point for this since we have a list
		# Start with LCR data
		accel_days_lcr = []
		pulses_lcr = []
		for rd in real_days_lcr:
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

			accel_days_lcr.append(ad)
			pulses_lcr.append(p)
		
		# Then, Intan
		accel_days_intan = []
		pulses_intan = []
		for rd in real_days_intan:
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

			accel_days_intan.append(ad)
			pulses_intan.append(p)

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
		z_lcr = z_lcr * gsa
		z_lcr = z_lcr.tolist()

		z_intan = z_intan * gsa
		z_intan = z_intan.tolist()

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'

		# Set plot color for current part number
		# Also add data to dataframe for average plot
		colorn = int((10-sample_num)*(_MAX_COLOR-_MIN_COLOR)/10)
		if "PT" in sample:
			colorrgb = f"rgb({colorn},{colorn},255)"
			df_pt_lcr[sample] = z_lcr
			df_pt_intan[sample] = z_intan
		elif "IR" in sample:
			colorrgb = f"rgb(255,{colorn},{colorn})"
			df_ir_lcr[sample] = z_lcr
			df_ir_intan[sample] = z_intan
		else:
			colorrgb = f"rgb({colorn},{colorn},{colorn})"
		
		# Plot individual traces, left is lcr, right is intan
		fig_impedance.add_trace(
			go.Scatter(
				x=accel_days_lcr,
				y=z_lcr,
				mode="lines+markers",
				line=dict(width=1, dash=linetype, color=colorrgb),
				name=trace_label
			),
			row=1,
			col=1,
		)
		fig_impedance.add_trace(
			go.Scatter(
				x=accel_days_intan,
				y=z_intan,
				mode="lines",
				line=dict(width=1, dash=linetype, color=colorrgb),
				name=trace_label,
				showlegend=False
			),
			row=1,
			col=2,
		)

	# Bottom subplot - average of normalized data
	# Calculate averages
	average_pt_lcr = df_pt_lcr.mean(axis=1)
	sd_pt_lcr = df_pt_lcr.std(axis=1)
	average_ir_lcr = df_ir_lcr.mean(axis=1)
	sd_ir_lcr = df_ir_lcr.std(axis=1)

	average_pt_intan = df_pt_intan.mean(axis=1)
	sd_pt_intan = df_pt_intan.std(axis=1)
	average_ir_intan = df_ir_intan.mean(axis=1)
	sd_ir_intan = df_ir_intan.std(axis=1)

	# Plot pt average, left is lcr, right is intan
	fig_impedance.add_trace(
		go.Scatter(
			x=accel_days_lcr,
			y=average_pt_lcr,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="Pt Average",
			error_y=dict(type='data', array=sd_pt_lcr, visible=True)
		),
		row=2,
		col=1,
	)
	fig_impedance.add_trace(
		go.Scatter(
			x=accel_days_intan,
			y=average_pt_intan,
			mode="lines",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="Pt Average",
			# error_y=dict(type='data', array=sd_pt_intan, visible=True),
			showlegend=False
		),
		row=2,
		col=2,
	)

	# Plot ir average, left is lcr, right is intan
	fig_impedance.add_trace(
		go.Scatter(
			x=accel_days_lcr,
			y=average_ir_lcr,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="Ir Average",
			error_y=dict(type='data', array=sd_ir_lcr, visible=True)
		),
		row=2,
		col=1,
	)
	fig_impedance.add_trace(
		go.Scatter(
			x=accel_days_intan,
			y=average_ir_intan,
			mode="lines",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="Ir Average",
			# error_y=dict(type='data', array=sd_ir_intan, visible=True),
			showlegend=False
		),
		row=2,
		col=2,
	)

	fig_impedance.update_yaxes(type="log", tick0=100, dtick=1, range=[1, 6])
	fig_impedance.update_xaxes(range=[0, max(accel_days_intan+accel_days_lcr)])

	# Add vertical line with label where I increased stim and replaced the saline
	add_vert_line(fig_impedance, 2, 2, 423.62, "Stim Increased")
	add_vert_line(fig_impedance, 2, 2, 982.98, "PBS Replaced")

def plot_cic(fig_cic, df_experiment, df_cic500_scope, df_cic1000_intan):
	"""
	Plots CIC vs time
	Left is manual data @ 500 us, right is intan data @ 1000 us
	"""

	# Create dataframes for calculating average values
	df_pt_scope = pd.DataFrame()
	df_ir_scope = pd.DataFrame()
	df_pt_intan = pd.DataFrame()
	df_ir_intan = pd.DataFrame()

	# Loop through each sample
	for sample in _ELECTRODE_DATA["Channel Name"]:
		# Sample Number
		sample_num = int(sample[-2:])

		# Pull Z for current channel
		cic500_scope = df_cic500_scope[sample]
		cic500_scope.dropna(inplace=True)

		cic1000_intan = df_cic1000_intan[sample]
		cic1000_intan.dropna(inplace=True)

		ind_scope = cic500_scope.index.to_list()
		ind_intan = cic1000_intan.index.to_list()

		# Pull real days and datetime for current channel
		real_days_scope = df_cic500_scope["Real Days"]
		real_days_scope = real_days_scope.loc[ind_scope]
		test_dates_scope = df_cic500_scope["Measurement Datetime"]
		test_dates_scope = test_dates_scope.loc[ind_scope]

		real_days_intan = df_cic1000_intan["Real Days"]
		real_days_intan = real_days_intan.loc[ind_intan]
		test_dates_intan = df_cic1000_intan["Measurement Datetime"]
		test_dates_intan = test_dates_intan.loc[ind_intan]

		# Check if the current file is in the ignore list
		if sample in _ELECTRODES_TO_IGNORE["Channel Name"].tolist():
			ignore_date = _ELECTRODES_TO_IGNORE.loc[_ELECTRODES_TO_IGNORE["Channel Name"] == sample, "Ignore From"].item()
			# Remove values at test dates >= ignore date
			ind_ignore_scope = test_dates_scope >= ignore_date
			ind_ignore_intan = test_dates_intan >= ignore_date
			cic500_scope[ind_ignore_scope] = float('nan')
			cic1000_intan[ind_ignore_intan] = float('nan')
			real_days_scope[ind_ignore_scope] = float('nan')
			real_days_intan[ind_ignore_intan] = float('nan')

		# Find accel days and number of pulses from df_experiment
		# Need to loop through each time point for this since we have a list
		# Start with LCR data
		accel_days_scope = []
		pulses_scope = []
		for rd in real_days_scope:
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

			accel_days_scope.append(ad)
			pulses_scope.append(p)
		
		# Then, Intan
		accel_days_intan = []
		pulses_intan = []
		for rd in real_days_intan:
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

			accel_days_intan.append(ad)
			pulses_intan.append(p)

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
		cic500_scope = cic500_scope.tolist()
		cic1000_intan = cic1000_intan.tolist()

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'

		# Set plot color for current part number
		# Also add data to dataframe for average plot
		colorn = int((10-sample_num)*(_MAX_COLOR-_MIN_COLOR)/10)
		if "PT" in sample:
			colorrgb = f"rgb({colorn},{colorn},255)"
			df_pt_scope[sample] = cic500_scope
			df_pt_intan[sample] = cic1000_intan
		elif "IR" in sample:
			colorrgb = f"rgb(255,{colorn},{colorn})"
			df_ir_scope[sample] = cic500_scope
			df_ir_intan[sample] = cic1000_intan
		else:
			colorrgb = f"rgb({colorn},{colorn},{colorn})"
		
		# Plot individual traces, left is lcr, right is intan
		fig_cic.add_trace(
			go.Scatter(
				x=accel_days_scope,
				y=cic500_scope,
				mode="lines+markers",
				line=dict(width=1, dash=linetype, color=colorrgb),
				name=trace_label
			),
			row=1,
			col=1,
		)
		fig_cic.add_trace(
			go.Scatter(
				x=accel_days_intan,
				y=cic1000_intan,
				mode="lines",
				line=dict(width=1, dash=linetype, color=colorrgb),
				name=trace_label,
				showlegend=False
			),
			row=1,
			col=2,
		)

	# Bottom subplot - average of normalized data
	# Calculate averages
	average_pt_scope = df_pt_scope.mean(axis=1)
	sd_pt_scope = df_pt_scope.std(axis=1)
	average_ir_scope = df_ir_scope.mean(axis=1)
	sd_ir_scope = df_ir_scope.std(axis=1)

	average_pt_intan = df_pt_intan.mean(axis=1)
	sd_pt_intan = df_pt_intan.std(axis=1)
	average_ir_intan = df_ir_intan.mean(axis=1)
	sd_ir_intan = df_ir_intan.std(axis=1)

	# Plot pt average, left is lcr, right is intan
	fig_cic.add_trace(
		go.Scatter(
			x=accel_days_scope,
			y=average_pt_scope,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="Pt Average",
			error_y=dict(type='data', array=sd_pt_scope, visible=True)
		),
		row=2,
		col=1,
	)
	fig_cic.add_trace(
		go.Scatter(
			x=accel_days_intan,
			y=average_pt_intan,
			mode="lines",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="Pt Average",
			# error_y=dict(type='data', array=sd_pt_intan, visible=True),
			showlegend=False
		),
		row=2,
		col=2,
	)

	# Plot ir average, left is lcr, right is intan
	fig_cic.add_trace(
		go.Scatter(
			x=accel_days_scope,
			y=average_ir_scope,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="Ir Average",
			error_y=dict(type='data', array=sd_ir_scope, visible=True)
		),
		row=2,
		col=1,
	)
	fig_cic.add_trace(
		go.Scatter(
			x=accel_days_intan,
			y=average_ir_intan,
			mode="lines",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="Ir Average",
			# error_y=dict(type='data', array=sd_ir_intan, visible=True),
			showlegend=False
		),
		row=2,
		col=2,
	)

	fig_cic.update_yaxes(range=[0, 1000])
	fig_cic.update_xaxes(range=[0, max(accel_days_intan+accel_days_scope)])

	# Add vertical line with label where I increased stim and replaced the saline
	add_vert_line(fig_cic, 2, 2, 423.62, "Stim Increased")
	add_vert_line(fig_cic, 2, 2, 982.98, "PBS Replaced")
	

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