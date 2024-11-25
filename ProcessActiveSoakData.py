import argparse
import pandas as pd
import os
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# All data should be saved in /data/coatings
# /[subfolders] are for each test point, labeled as accelerated days (eg 0012)
# Data for all devices are in one folder, including label and test number in title

# True normalizes all graphs to GSA
_NORM_TO_AREA = True

# Min and Max values for grayscale plotting
_MIN_COLOR = 0
_MAX_COLOR = 0.8

# Geometric surface areas
_GSA = pd.DataFrame({"PT": [3.2,2.2,2.1,2.3,2.1,2.6,2.1,1.7,1.6,2.8],
                     "IR": [1.7,0.8,1.2,1.2,1.4,1.4,0.6,0.8,1.0,1.2]})

# Stim parameters
_STIM = pd.DataFrame({"PT": ["400 uA","400 uA","400 uA","400 uA","no stim","200 uA","200 uA","200 uA","200 uA","no stim"],
                      "IR": ["600 uA","600 uA","600 uA","600 uA","no stim","400 uA","400 uA","400 uA","400 uA","no stim"]})

def parse_args():
	"""
	Parses the command line arguments.
	"""

	parser = argparse.ArgumentParser(description="Echem Data Processing")
	parser.add_argument(
		"-p",
		"--path",
		type=str,
		default="data//coatings",
		help="Folder where all data is stored."
	)
	return parser.parse_args()

def main():
	"""
	This is the main function that processes the following lifetime testing data:
	CV: processes the last curve from each day tested and plots trends of area inside cv curve
	EIS: processes all data and plots trends of 1 kHz impedance magnitude
	VT: all data exists in a spreadsheet that must be downloaded from google drive
	"""

	args = parse_args()

	# Initiate plots
	# 1 (4x5 subplots): CV vs time for each electrode individually
	fig1 = make_subplots(
		rows=4, 
		cols=5,
		subplot_titles=(
			"Pt Sample 1 (400 uA / 3.2 mm^2)", "Pt Sample 2 (400 uA / 2.2 mm^2)", "Pt Sample 3 (400 uA / 2.1 mm^2)", "Pt Sample 4 (400 uA / 2.3 mm^2)", "Pt Sample 5 (no stim / 2.1 mm^2)", 
			"Pt Sample 6 (200 uA / 2.6 mm^2)", "Pt Sample 7 (200 uA / 2.1 mm^2)", "Pt Sample 8 (200 uA / 1.7 mm^2)", "Pt Sample 9 (200 uA / 1.6 mm^2)", "Pt Sample 10 (no stim / 2.8 mm^2)",
			"Ir Sample 1 (600 uA / 1.7 mm^2)", "Ir Sample 2 (600 uA / 0.8 mm^2)", "Ir Sample 3 (600 uA / 1.2 mm^2)", "Ir Sample 4 (600 uA / 1.2 mm^2)", "Ir Sample 5 (no stim / 1.4 mm^2)", 
			"Ir Sample 6 (400 uA / 1.4 mm^2)", "Ir Sample 7 (400 uA / 0.6 mm^2)", "Ir Sample 8 (400 uA / 0.8 mm^2)", "Ir Sample 9 (400 uA / 1.0 mm^2)", "Ir Sample 10 (no stim / 1.2 mm^2)"
			)
	)
	fig1.update_layout(title_text="CV Curve vs Time (light=day 0, dark=most recent)")

	# 2 (subplot top): Area vs time for all electrodes (individual traces)
	# 2 (subplot bottom): Area vs time for all electrodes (average, normalized to t=0)
	fig2 = make_subplots(
		rows=2,
		cols=1,
		subplot_titles=(
			"Individual Electrodes",
			"Average +/- SD (normalized to t=0)"
		)
	)
	fig2.update_layout(title_text="CV Area vs Time")

	# 3 (4x5 subplots): EIS vs time for each electrode individually
	fig3 = make_subplots(
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
			)
	)
	fig3.update_layout(title_text="EIS Curve vs Time (light=day 0, dark=most recent)")

	# 4 (subplot top): 1k impedance vs time for all electrodes (individual traces)
	# 4 (subplot bottom): 1k impedance vs time for all electrodes (average, normalized to t=0)
	fig4 = make_subplots(
		rows=2,
		cols=1,
		subplot_titles=(
			"Individual Electrodes",
			"Average +/- SD (normalized to t=0)"
		)
	)
	fig4.update_layout(title_text="1 kHz Impedance Magnitude vs Time")

	# 5 (subplot top): CIC vs time for all electrodes (individual traces)
	# 5 (subplot bottom): CIC vs time for all electrodes (average, normalized to t=0)
	fig5 = make_subplots(
		rows=2,
		cols=1,
		subplot_titles=(
			"Individual Electrodes",
			"Average +/- SD"
		)
	)
	fig5.update_layout(title_text="Charge Injection Capacity vs Time")

	# 6: CV for day 0 for all electrodes (individual traces)
	fig6 = go.Figure()
	fig6.update_layout(title_text="Initial CV curve for All Electrodes")

	# 7: EIS for day 0 for all electrodes (individual traces)
	fig7 = make_subplots(
		rows=2, 
		cols=1,
	)
	fig7.update_layout(title_text="Initial EIS curve for All Electrodes")

	df_pt_cv, df_pt_eis, df_pt_vt, df_ir_cv, df_ir_eis, df_ir_vt = perform_data_analysis(args.path, fig1, fig3, fig6, fig7)

	plot_summarized_cv_data(df_pt_cv, df_ir_cv, fig2)
	plot_summarized_eis_data(df_pt_eis, df_ir_eis, fig4)
	plot_summarized_vt_data(df_pt_vt, df_ir_vt, fig5)

	fig1.show()
	fig2.show()
	fig3.show()
	fig4.show()
	fig5.show()
	fig6.show()
	fig7.show()

def perform_data_analysis(path, fig1, fig3, fig6, fig7):
	# List of test points (subfolders)
	folders = os.listdir(f"{path}//")
	numfolders = len(folders)

	# Create dataframe for processed data
	# Rows are the sample number (minus 1, since part labels start at 1 not 0), columns are the accelerated days
	df_pt_cv = pd.DataFrame()
	df_pt_eis = pd.DataFrame()
	df_pt_vt = pd.DataFrame()
	df_ir_cv = pd.DataFrame()
	df_ir_eis = pd.DataFrame()
	df_ir_vt = pd.DataFrame()

	# Loop through each folder
	for f in range(numfolders):
		# Set plot color for current time step
		# One of the folders is actually the vt file set, so use (numfolders-1)
		colort = int(255*((numfolders-1)-f)*(_MAX_COLOR-_MIN_COLOR)/(numfolders-1))

		# List of CSVs to read
		folder = folders[f] #folder is also the number of accelerated days

		# If it's the vt data, open it
		if folder == "Soak Testing - SIROF (Active).csv":
			print("Processing VT data")
			df = pd.read_csv(f"{path}//{folder}")
			# Set row 1 to header titles
			df.columns = df.iloc[1]
			# Save GSA
			# gsa_ir = df.loc[4].iloc[8:18]
			# gsa_ir = gsa_ir.astype(float)
			# gsa_pt = df.loc[4].iloc[18:28]
			# gsa_pt = gsa_pt.astype(float)
			# Data starts at row 9
			df = df[9:]
			
			# Loop through each time point and save data (if any)
			for idx, row in df.iterrows():
				# If there is data, process it
				if math.isnan(float(row["PT01"])) == False:
					days = str(row.iloc[6])

					# For each dataset, need to calculate CIC (data in the csv is max current)
					# CIC = [(max current - mA) * (1 A / 1000 mA) * (500 us)] / [(GSA - mm2) * (1 cm / 10 mm)^2]
					
					# Columns X through Y are Ir data
					# print(row)
					irdata = row.iloc[8:18]
					irdata = irdata.astype(float)
					irdata = irdata.rename(index={"IR01": 0, "IR02": 1, "IR03": 2, "IR04": 3, "IR05": 4, "IR06": 5, "IR07": 6, "IR08": 7, "IR09": 8, "IR10": 9})
					irdata = (irdata*50).div(_GSA["IR"])
					irdata = irdata.T
					df_ir_vt[days] = irdata
					
					# Columns Y+1 through Z are Pt data
					ptdata = row.iloc[18:28]
					ptdata = ptdata.astype(float)
					ptdata = ptdata.rename(index={"PT01": 0, "PT02": 1, "PT03": 2, "PT04": 3, "PT05": 4, "PT06": 5, "PT07": 6, "PT08": 7, "PT09": 8, "PT10": 9})
					ptdata = (ptdata*50).div(_GSA["PT"])
					ptdata = ptdata.T
					df_pt_vt[days] = ptdata

			# Data in the csv is maximum current - we need to calculate CIC
			# CIC = [(max current - mA) * (1 A / 1000 mA) * (500 us)] / [(GSA - mm2) * (1 cm / 10 mm)^2]
			# gsa = 

		# If it's a real folder, say what you're processing and open it
		else:
			print(f"Processing Data from Day {folder}")
			fpath = f"{path}//{folder}//"
			files = os.listdir(fpath)
			numfiles = len(files)

			# Loop through each file and plot
			for n in range(numfiles):
				file = files[n]
				
				# Determine the test type
				# for CV, we only want to process cycle 2 (or cycle 30 for day 0)
				if "CV" in file and (("cycle2" in file and folder != "0000") or ("cycle30" in file and folder == "0000")):
					df = pd.read_csv(f"{fpath}{file}")

					# Calculate area inside the curve
					# Uses the shoelace theorem (code from chat gpt, verified in excel)
					volts = df["Voltage (V)"]
					microamps = df["Current (A)"] * 10**6
					cv_area = 0.5 * np.abs(np.dot(volts, np.roll(microamps,1)) - np.dot(microamps, np.roll(volts,1)))

					# Determine if the sample is platinum or iridium, save area, and plot
					if "PT" in file:
						index = file.index("PT")
						sample_num = int(file[index+2:index+4])
						df_pt_cv.loc[(sample_num-1), folder] = cv_area
						plot_cv_curve("PT", sample_num, df, fig1, colort, folder)
					elif "IR" in file:
						index = file.index("IR")
						sample_num = int(file[index+2:index+4])
						df_ir_cv.loc[(sample_num-1), folder] = cv_area
						plot_cv_curve("IR", sample_num, df, fig1, colort, folder)

					# Set plot color for current part number
					colorn = int(255*(10-sample_num)*(_MAX_COLOR-_MIN_COLOR)/10)

					# Separate block for plotting day 0 CV
					if folder == "0000":
						# 6: CV for day 0 for all electrodes (individual traces)
						# Set color, find stim level and gsa, define label
						if "IR" in file:
							c = f"rgb(255,{colorn},{colorn})"
							stim = _STIM["IR"].iloc[sample_num-1]
							gsa = str(_GSA["IR"].iloc[sample_num-1]) + " mm^2"
							trace_label = f"IR {sample_num} ({stim} / {gsa})"
							if _NORM_TO_AREA:
								norm = _GSA["IR"].iloc[sample_num-1]
							else:
								norm = 1
						else:
							c = f"rgb({colorn},{colorn},255)"
							stim = _STIM["PT"].iloc[sample_num-1]
							gsa = str(_GSA["PT"].iloc[sample_num-1]) + " mm^2"
							trace_label = f"PT {sample_num} ({stim} / {gsa})"
							if _NORM_TO_AREA:
								norm = _GSA["PT"].iloc[sample_num-1]
							else:
								norm = 1

						# Set plot type (non-stim = dashed lines)
						if stim == "no stim":
							linetype = 'dash'
						else:
							linetype = 'solid'

						fig6.add_trace(
							go.Scatter(
								x=df["Voltage (V)"],
								y=df["Current (A)"] / norm,
								mode="lines",
								line=dict(width=1, dash=linetype, color=c),
								name=trace_label
							),
						)

						fig6.update_xaxes(title_text="Voltage (V)")
						fig6.update_yaxes(title_text="Current Density (A/mm^2)")

				elif "EIS" in file:
					df = pd.read_csv(f"{fpath}{file}")

					# find 1k impedance
					index = (np.abs(df["Frequency"] - 1000)).idxmin()
					z_1k = df["Impedance"][index]

					# Determine if the sample is platinum or iridium, save 1k impedance, and plot
					if "PT" in file:
						index = file.index("PT")
						sample_num = int(file[index+2:index+4])
						df_pt_eis.loc[(sample_num-1), folder] = z_1k
						plot_eis_curve("PT", sample_num, df, fig3, colort, folder)
					elif "IR" in file:
						index = file.index("IR")
						sample_num = int(file[index+2:index+4])
						df_ir_eis.loc[(sample_num-1), folder] = z_1k
						plot_eis_curve("IR", sample_num, df, fig3, colort, folder)
					
					# Set plot color for current part number
					colorn = int(255*(10-sample_num)*(_MAX_COLOR-_MIN_COLOR)/10)

					# Separate block for plotting day 0 EIS
					if folder == "0000":
						# 7: EIS for day 0 for all electrodes (individual traces)
						# Set color, find stim level and gsa, define label
						if "IR" in file:
							c = f"rgb(255,{colorn},{colorn})"
							stim = _STIM["IR"].iloc[sample_num-1]
							gsa = str(_GSA["IR"].iloc[sample_num-1]) + " mm^2"
							trace_label = f"IR {sample_num} ({stim} / {gsa})"
							if _NORM_TO_AREA:
								norm = _GSA["IR"].iloc[sample_num-1]
							else:
								norm = 1
						else:
							c = f"rgb({colorn},{colorn},255)"
							stim = _STIM["PT"].iloc[sample_num-1]
							gsa = str(_GSA["PT"].iloc[sample_num-1]) + " mm^2"
							trace_label = f"PT {sample_num} ({stim} / {gsa})"
							if _NORM_TO_AREA:
								norm = _GSA["PT"].iloc[sample_num-1]
							else:
								norm = 1

						# Set plot type (non-stim = dashed lines)
						if stim == "no stim":
							linetype = 'dash'
						else:
							linetype = 'solid'

						# First, magnitude in top plot
						fig7.add_trace(
							go.Scatter(
								x=df["Frequency"],
								y=df["Impedance"] * norm,
								mode="lines",
								line=dict(width=1, dash=linetype, color=c),
								name=f"{trace_label}",
							),
							row=1,
							col=1,
						)

						# Next, phase in bottom plot
						fig7.add_trace(
							go.Scatter(
								x=df["Frequency"],
								y=df["Phase Angle"],
								mode="lines",
								line=dict(width=1, dash=linetype, color=c),
								showlegend=False
							),
							row=2,
							col=1,
						)

						fig7.update_xaxes(type="log", title_text="Frequency (Hz)")
						fig7.update_yaxes(type="log", title_text="Z Magnitude * GSA (ohm*mm^2)", row=1, col=1)
						fig7.update_yaxes(title_text="Z Phase (deg)", row=2, col=1)

								   
	df_pt_cv = df_pt_cv.sort_index(axis=1)
	df_pt_eis = df_pt_eis.sort_index(axis=1)
	df_pt_vt = df_pt_vt.sort_index(axis=1)
	df_ir_cv = df_ir_cv.sort_index(axis=1)
	df_ir_eis = df_ir_eis.sort_index(axis=1)
	df_ir_vt = df_ir_vt.sort_index(axis=1)

	return df_pt_cv, df_pt_eis, df_pt_vt, df_ir_cv, df_ir_eis, df_ir_vt

def plot_cv_curve(material, sample_num, df, fig1, colort, folder):
	"""
	Plots CV curves for Pt and Ir electrodes in their respective subplot of fig1
	   
	Inputs: 
		material: "PT" or "IR"
		sample_num: sample number
		df: CV data
		fig1: the figure to add the plot (setup for fig1, could theoretically be expanded later)
		colort: the color for the plot
	"""
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

	gsa = _GSA[material].iloc[sample_num-1]

	# fig1 (4x5 subplots): CV vs time for each electrode individually
	df_normalized = df["Current (A)"] / gsa

	fig1.add_trace(
		go.Scatter(
			x=df["Voltage (V)"],
			y=df_normalized,
			mode="lines",
			line=dict(width=1, color=f"rgb({colort},{colort},{colort})"),
			name=f"day {folder}",
			showlegend=False
		),
		row=plot_row,
		col=plot_col,
	)

	fig1.update_xaxes(range=[-0.6,0.8], title_text="Voltage (V)")
	fig1.update_yaxes(range=[-5e-6, 10e-6], title_text="Current Density (A/mm^2)")
	
def plot_eis_curve(material, sample_num, df, fig3, colort, folder):
	"""
	Plots EIS curves for Pt and Ir electrodes in their respective subplot of fig3
	   
	Inputs: 
		material: "PT" or "IR"
		sample_num: sample number
		df: EIS data
		fig1: the figure to add the plot (setup for fig3, could theoretically be expanded later)
		colort: the color for the plot
		folder: the folder being processed, also the number of days
	"""
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
	
	# 3 (4x5 subplots): EIS vs time for each electrode individually
	# First, magnitude in solid line
	fig3.add_trace(
		go.Scatter(
			x=df["Frequency"],
			y=df["Impedance"],
			mode="lines",
			line=dict(width=1, dash='solid', color=f"rgb({colort},{colort},{colort})"),
			name=f"day {folder}",
			showlegend=False
		),
		row=plot_row,
		col=plot_col,
		secondary_y=False,
	)

	# Next, phase in dashed line
	fig3.add_trace(
		go.Scatter(
			x=df["Frequency"],
			y=df["Phase Angle"],
			mode="lines",
			line=dict(width=1, dash='dash', color=f"rgb({colort},{colort},{colort})"),
			name=f"day {folder}",
			showlegend=False
		),
		row=plot_row,
		col=plot_col,
		secondary_y=True,
	)

	fig3.update_xaxes(type="log", range=[1,4], title_text="Frequency (Hz)")
	fig3.update_yaxes(secondary_y=True, range=[-90,0], title_text="Z Phase (deg)")
	fig3.update_yaxes(type="log", secondary_y=False, range=[2,6], title_text="Z Mag (Ohms)")

def plot_summarized_cv_data(df_pt_cv, df_ir_cv, fig2):
	"""
	Plots area inside the CV curve vs time

	Inputs:
		df_pt_cv: summarized CV areas for pt
		df_ir_cv: summarized CV areas for ir
		fig2: figure to plot (top subplot is individual traces, bottom is average for all traces of area normalized to t=0)
	"""

	# Find time points from df labels
	time = list(df_pt_cv)
	time = [int(t) for t in time]

	# Top subplot - individual traces
	# Loop through each Pt sample
	for i in range(len(df_pt_cv)):
		# Set label
		if i < 9:
			sample = f"PT0{i+1}"
		else:
			sample = f"PT{i+1}"
		areas = df_pt_cv.iloc[i].tolist()

		# Find stim level and gsa, define label
		stim = _STIM["PT"].iloc[i]
		gsa = str(_GSA["PT"].iloc[i]) + " mm^2"
		trace_label = f"{sample} ({stim} / {gsa})"

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'

		# Set plot color for current part number
		colorn = int(255*(10-i)*(_MAX_COLOR-_MIN_COLOR)/10)

		if _NORM_TO_AREA:
			norm = _GSA["PT"].iloc[i]
		else:
			norm = 1
		
		# Plot individual trace
		fig2.add_trace(
			go.Scatter(
				x=time,
				y=[ar / norm for ar in areas],
				mode="lines+markers",
				line=dict(width=1, dash=linetype, color=f"rgb({colorn},{colorn},255)"),
				name=trace_label
			),
			row=1,
			col=1,
		)

	# Loop through each Ir sample
	for i in range(len(df_ir_cv)):
		# Set label
		if i < 9:
			sample = f"IR0{i+1}"
		else:
			sample = f"IR{i+1}"
		areas = df_ir_cv.iloc[i].tolist()

		# Find stim level and gsa, define label
		stim = _STIM["IR"].iloc[i]
		gsa = str(_GSA["IR"].iloc[i]) + " mm^2"
		trace_label = f"{sample} ({stim} / {gsa})"

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'
		
		if _NORM_TO_AREA:
			norm = _GSA["IR"].iloc[i]
		else:
			norm = 1

		# Plot individual trace
		fig2.add_trace(
			go.Scatter(
				x=time,
				y=[ar / norm for ar in areas],
				mode="lines+markers",
				line=dict(width=1, dash=linetype, color=f"rgb(255,{colorn},{colorn})"),
				name=trace_label
			),
			row=1,
			col=1,
		)

	# Bottom subplot - average of normalized data
	# Calculate and plot Pt average
	df_normalized = df_pt_cv.div(_GSA["PT"], axis=0)
	averages = df_normalized.mean(axis=0)
	sds = df_normalized.std(axis=0)
	
	fig2.add_trace(
		go.Scatter(
			x=time,
			y=averages,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="Pt Average",
			error_y=dict(type='data', array=sds, visible=True)
		),
		row=2,
		col=1,
	)

	# Calculate and plot Ir average
	df_normalized = df_ir_cv.div(_GSA["IR"], axis=0)
	averages = df_normalized.mean(axis=0)
	sds = df_normalized.std(axis=0)
	
	fig2.add_trace(
		go.Scatter(
			x=time,
			y=averages,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="Ir Average",
			error_y=dict(type='data', array=sds, visible=True)
		),
		row=2,
		col=1,
	)

	fig2.update_yaxes(title_text="CV Area (W/mm^2)")
	fig2.update_xaxes(title_text="Accelerated Time (days)", range=[0, max(time)])

def plot_summarized_eis_data(df_pt_eis, df_ir_eis, fig4):
	"""
	Plots 1 kHz impedance magnitude vs time

	Inputs:
		df_pt_eis: summarized 1k impedance for pt
		df_ir_eis: summarized 1k impedance for ir
		fig4: figure to plot (top subplot is individual traces, bottom is average for all traces of impedance normalized to t=0)
	"""

	# Find time points from df labels
	time = list(df_pt_eis)
	time = [int(t) for t in time]

	# Top subplot - individual traces
	# Loop through each Pt sample
	for i in range(len(df_pt_eis)):
		# Set label
		if i < 9:
			sample = f"PT0{i+1}"
		else:
			sample = f"PT{i+1}"
		impedances = df_pt_eis.iloc[i].tolist()

		# Find stim level and gsa, define label
		stim = _STIM["PT"].iloc[i]
		gsa = str(_GSA["PT"].iloc[i]) + " mm^2"
		trace_label = f"{sample} ({stim} / {gsa})"

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'
		
		# Set plot color for current part number
		colorn = int(255*(10-i)*(_MAX_COLOR-_MIN_COLOR)/10)
		
		if _NORM_TO_AREA:
			norm = _GSA["PT"].iloc[i]
		else:
			norm = 1

		# Plot individual trace
		fig4.add_trace(
			go.Scatter(
				x=time,
				y=[im * norm for im in impedances],
				mode="lines+markers",
				line=dict(width=1, dash=linetype, color=f"rgb({colorn},{colorn},255)"),
				name=trace_label
			),
			row=1,
			col=1,
		)

	# Loop through each Ir sample
	for i in range(len(df_ir_eis)):
		# Set label
		if i < 9:
			sample = f"IR0{i+1}"
		else:
			sample = f"IR{i+1}"
		impedances = df_ir_eis.iloc[i].tolist()

		# Find stim level and gsa, define label
		stim = _STIM["IR"].iloc[i]
		gsa = str(_GSA["IR"].iloc[i]) + " mm^2"
		trace_label = f"{sample} ({stim} / {gsa})"

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'

		# Set plot color for current part number
		colorn = int(255*(10-i)*(_MAX_COLOR-_MIN_COLOR)/10)

		if _NORM_TO_AREA:
			norm = _GSA["IR"].iloc[i]
		else:
			norm = 1
		
		# Plot individual trace
		fig4.add_trace(
			go.Scatter(
				x=time,
				y=[im * norm for im in impedances],
				mode="lines+markers",
				line=dict(width=1, dash=linetype, color=f"rgb(255,{colorn},{colorn})"),
				name=trace_label
			),
			row=1,
			col=1,
		)

	# Bottom subplot - average of normalized data
	# Calculate and plot Pt average
	df_normalized = df_pt_eis.mul(_GSA["PT"], axis=0)
	averages = df_normalized.mean(axis=0)
	sds = df_normalized.std(axis=0)

	fig4.add_trace(
		go.Scatter(
			x=time,
			y=averages,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="Pt Average",
			error_y=dict(type='data', array=sds, visible=True)
		),
		row=2,
		col=1,
	)

	# Calculate and plot Ir average
	df_normalized = df_ir_eis.mul(_GSA["IR"], axis=0)
	averages = df_normalized.mean(axis=0)
	sds = df_normalized.std(axis=0)

	print(averages)
	
	fig4.add_trace(
		go.Scatter(
			x=time,
			y=averages,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="Ir Average",
			error_y=dict(type='data', array=sds, visible=True)
		),
		row=2,
		col=1,
	)

	fig4.update_yaxes(type="log", title_text="Z Magnitude * GSA (ohms*mm^2)")
	fig4.update_xaxes(title_text="Accelerated Time (days)", range=[0, max(time)])

def plot_summarized_vt_data(df_pt_vt, df_ir_vt, fig5):
	"""
	Plots CIC vs time

	Inputs:
		df_pt_vt: summarized CIC for pt
		df_ir_vt: summarized CIC for ir
		fig5: figure to plot (top subplot is individual traces, bottom is average for all traces)
	"""

	# Find time points from df labels
	time = list(df_pt_vt)
	time = [float(t) for t in time]

	# Top subplot - individual traces
	# Loop through each Pt sample
	for i in range(len(df_pt_vt)):
		# Set sample
		if i < 9:
			sample = f"PT0{i+1}"
		else:
			sample = f"PT{i+1}"
		cics = df_pt_vt.iloc[i].tolist()

		# Find stim level and gsa, define label
		# print(df_pt_vt)
		# print(_STIM)
		# print(i)
		stim = _STIM["PT"].iloc[i]
		gsa = str(_GSA["PT"].iloc[i]) + " mm^2"
		trace_label = f"{sample} ({stim} / {gsa})"

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'

		# Set plot color for current part number
		colorn = int(255*(10-i)*(_MAX_COLOR-_MIN_COLOR)/10)
		
		# Plot individual trace
		fig5.add_trace(
			go.Scatter(
				x=time,
				y=cics,
				mode="lines+markers",
				line=dict(width=1, dash=linetype, color=f"rgb({colorn},{colorn},255)"),
				name=trace_label
			),
			row=1,
			col=1,
		)

	# Loop through each Ir sample
	for i in range(len(df_ir_vt)):
		# Set sample
		if i < 9:
			sample = f"IR0{i+1}"
		else:
			sample = f"IR{i+1}"
		cics = df_ir_vt.iloc[i].tolist()

		# Find stim level and gsa, define label
		stim = _STIM["IR"].iloc[i]
		gsa = str(_GSA["IR"].iloc[i]) + " mm^2"
		trace_label = f"{sample} ({stim} / {gsa})"

		# Set plot type (non-stim = dashed lines)
		if stim == "no stim":
			linetype = 'dash'
		else:
			linetype = 'solid'
		
		# Plot individual trace
		fig5.add_trace(
			go.Scatter(
				x=time,
				y=cics,
				mode="lines+markers",
				line=dict(width=1, dash=linetype, color=f"rgb(255,{colorn},{colorn})"),
				name=trace_label
			),
			row=1,
			col=1,
		)

	# Bottom subplot - average of normalized data
	# Calculate and plot Pt average
	averages = df_pt_vt.mean(axis=0)
	sds = df_pt_vt.std(axis=0)
	
	fig5.add_trace(
		go.Scatter(
			x=time,
			y=averages,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="Pt Average",
			error_y=dict(type='data', array=sds, visible=True)
		),
		row=2,
		col=1,
	)

	# Calculate and plot Ir average
	averages = df_ir_vt.mean(axis=0)
	sds = df_ir_vt.std(axis=0)
	
	fig5.add_trace(
		go.Scatter(
			x=time,
			y=averages,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="Ir Average",
			error_y=dict(type='data', array=sds, visible=True)
		),
		row=2,
		col=1,
	)

	fig5.update_yaxes(title_text="Charge Injection Capacity (uC/cm2)")
	fig5.update_xaxes(title_text="Accelerated Time (days)", range=[0, max(time)])

if __name__ == "__main__":
	main()