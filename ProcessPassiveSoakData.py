import argparse
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Processes passive soak test data (EIS only)
# All data should be stored in a single folder, with:
# //[subfolder] for each test point, labeled as accelerated days (eg "12" for 12 days in accelerated soak)
# Data for all devices are in one folder, including label and test number in title (eg "EIS_[days]_IDE-25-1_[timestamp]")

# Min and Max values for grayscale plotting
_MIN_COLOR = 0
_MAX_COLOR = 0.8

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
	# 1 (4x4 subplots): EIS vs time for each electrode individually
	fig1 = make_subplots(
		rows=4, 
		cols=4,
		specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}], 
		 	   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
			   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
		 	   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
		subplot_titles=(
			"100 um Sample 1", "100 um Sample 2", "100 um Sample 3", "100 um Sample 4", "100 um Sample 5", "100 um Sample 6", "100 um Sample 7", "100 um Sample 8",
			"25 um Sample 1", "25 um Sample 2", "25 um Sample 3", "25 um Sample 4", "25 um Sample 5", "25 um Sample 6", "25 um Sample 7", "25 um Sample 8"
		)
	)
	fig1.update_layout(title_text="EIS Curve vs Time (light=day 0, dark=most recent)")

	# 2 (subplot top): 1k impedance vs time for all electrodes (individual traces)
	# 2 (subplot bottom): 1k impedance vs time for all electrodes (average)
	fig2 = make_subplots(
		rows=2,
		cols=1,
		subplot_titles=(
			"Individual Electrodes",
			"Average +/- SD"
		)
	)
	fig2.update_layout(title_text="1 kHz Impedance Magnitude vs Time")

	# 3: EIS for day 0 for all electrodes (individual traces)
	fig3 = make_subplots(
		rows=1, 
		cols=1,
		specs=[[{"secondary_y": True}]],
	)
	fig3.update_layout(title_text="Initial EIS curve for All Electrodes")

	df_100_eis, df_25_eis = perform_data_analysis(args.path, fig1, fig3)

	plot_summarized_eis_data(df_100_eis, df_25_eis, fig2)

	fig1.show()
	fig2.show()
	fig3.show()

def perform_data_analysis(path, fig1, fig3):
	# List of test points (subfolders)
	folders = os.listdir(f"{path}//")
	numfolders = len(folders)

	# Create dataframe for processed data
	# Rows are the sample number (minus 1, since part labels start at 1 not 0), columns are the accelerated days
	df_100_eis = pd.DataFrame()
	df_25_eis = pd.DataFrame()

	# Loop through each folder
	for f in range(numfolders):
		# Set plot color for current time step
		colort = int(255*(numfolders-f)*(_MAX_COLOR-_MIN_COLOR)/numfolders)

		# List of CSVs to read
		folder = folders[f] #folder is also the number of accelerated days
		print(f"Processing Data from Day {folder}")
		fpath = f"{path}//{folder}//"
		files = os.listdir(fpath)
		numfiles = len(files)

		# Loop through each file and plot
		for n in range(numfiles):
			file = files[n]
			df = pd.read_csv(f"{fpath}{file}")

			# find 1k impedance
			index = (np.abs(df["Frequency"] - 1000)).idxmin()
			z_1k = df["Impedance"][index]

			# Determine if the sample is 25 um or 100 um, save 1k impedance, and plot
			if "IDE-100" in file:
				index = file.index("IDE")
				sample_num = int(file[index+8:index+9])
				df_100_eis.loc[(sample_num-1), folder] = z_1k
				plot_eis_curve("IDE-100", sample_num, df, fig1, colort)
			elif "IDE-25" in file:
				index = file.index("IDE")
				sample_num = int(file[index+7:index+8])
				df_25_eis.loc[(sample_num-1), folder] = z_1k
				plot_eis_curve("IDE-25", sample_num, df, fig1, colort)
			
			# Set plot color for current part number
			colorn = int(255*(10-sample_num)*(_MAX_COLOR-_MIN_COLOR)/10)

			# Separate block for plotting day 0 EIS
			if folder == "0000":
				# 7: EIS for day 0 for all electrodes (individual traces)
				if "IDE-25" in file:
					c = f"rgb(255,{colorn},{colorn})"
					trace_label = f"IDE-25-{sample_num}"
				else:
					c = f"rgb({colorn},{colorn},255)"
					trace_label = f"IDE-100-{sample_num}"
				# First, magnitude in solid line
				fig3.add_trace(
					go.Scatter(
						x=df["Frequency"],
						y=df["Impedance"],
						mode="lines",
						line=dict(width=1, dash='solid', color=c),
						name=trace_label,
					),
					row=1,
					col=1,
					secondary_y=False,
				)

				# Next, phase in dashed line
				fig3.add_trace(
					go.Scatter(
						x=df["Frequency"],
						y=df["Phase Angle"],
						mode="lines",
						line=dict(width=1, dash='dash', color=c),
						showlegend=False
					),
					row=1,
					col=1,
					secondary_y=True,
				)

				fig3.update_xaxes(type="log", title_text="Frequency (Hz)")
				fig3.update_yaxes(secondary_y=True, title_text="Impedance Phase (deg)")
				fig3.update_yaxes(type="log", secondary_y=False, range=[2,7.7], title_text="Impedance Magnitude (Ohms)")
				

	df_100_eis = df_100_eis.sort_index()
	df_25_eis = df_25_eis.sort_index()

	return df_100_eis, df_25_eis
	
def plot_eis_curve(thickness, sample_num, df, fig1, colort):
	"""
	Plots EIS curves for 100 and 25 um IDEs in their respective subplot of fig1
	   
	Inputs: 
		thickness: "IDE-100" or "IDE-25"
		sample_num: sample number
		df: EIS data
		fig1: the figure to add the plot (setup for fig1, could theoretically be expanded later)
		colort: the color for the plot
	"""

	# Rows 1 and 2 are 100 um; rows 3 and 4 are 25 um
	if thickness == "IDE-100" and sample_num < 5:
		plot_row = 1
		plot_col = sample_num
	elif thickness == "IDE-100" and sample_num >= 5:
		plot_row = 2
		plot_col = sample_num-4
	elif thickness == "IDE-25" and sample_num < 5:
		plot_row = 3
		plot_col = sample_num
	elif thickness == "IDE-25" and sample_num >= 5:
		plot_row = 4
		plot_col = sample_num-4

	# 3 (4x5 subplots): EIS vs time for each electrode individually
	# First, magnitude in solid line
	fig1.add_trace(
		go.Scatter(
			x=df["Frequency"],
			y=df["Impedance"],
			mode="lines",
			line=dict(width=1, dash='solid', color=f"rgb({colort},{colort},{colort})"),
			showlegend=False
		),
		row=plot_row,
		col=plot_col,
		secondary_y=False,
	)

	# Next, phase in dashed line
	fig1.add_trace(
		go.Scatter(
			x=df["Frequency"],
			y=df["Phase Angle"],
			mode="lines",
			line=dict(width=1, dash='dash', color=f"rgb({colort},{colort},{colort})"),
			showlegend=False
		),
		row=plot_row,
		col=plot_col,
		secondary_y=True,
	)

	fig1.update_xaxes(type="log", range=[1,4], title_text="Frequency (Hz)")
	fig1.update_yaxes(secondary_y=True, title_text="Z Phase (deg)")
	fig1.update_yaxes(type="log", secondary_y=False, range=[2,7.7], title_text="Z Mag (Ohms)")
	
def plot_summarized_eis_data(df_100_eis, df_25_eis, fig2):
	"""
	Plots 1 kHz impedance magnitude vs time

	Inputs:
		df_100_eis: summarized 1k impedance for 100 um IDEs
		df_25_eis: summarized 1k impedance for 25 um IDEs
		fig2: figure to plot (top subplot is individual traces, bottom is average for all traces of impedance)
	"""

	# Find time points from df labels
	time = list(df_100_eis)
	time = [int(t) for t in time]

	# Top subplot - individual traces
	# Loop through each 10 um sample
	for i in range(len(df_100_eis)):
		if i < 9:
			sample = f"IDE-100-{i+1}"
		else:
			sample = f"IDE-100-{i+1}"
		impedances = df_100_eis.iloc[i].tolist()
		
		# Set plot color for current part number
		colorn = int(255*(10-i)*(_MAX_COLOR-_MIN_COLOR)/10)
		
		# Plot individual trace
		fig2.add_trace(
			go.Scatter(
				x=time,
				y=impedances,
				mode="lines+markers",
				line=dict(width=1, color=f"rgb({colorn},{colorn},255)"),
				name=sample
			),
			row=1,
			col=1,
		)

	# Loop through each 25 um sample
	for i in range(len(df_25_eis)):
		if i < 9:
			sample = f"IDE-25-{i+1}"
		else:
			sample = f"IDE-25-{i+1}"
		impedances = df_25_eis.iloc[i].tolist()

		# Set plot color for current part number
		colorn = int(255*(10-i)*(_MAX_COLOR-_MIN_COLOR)/10)
		
		# Plot individual trace
		fig2.add_trace(
			go.Scatter(
				x=time,
				y=impedances,
				mode="lines+markers",
				line=dict(width=1, color=f"rgb(255,{colorn},{colorn})"),
				name=sample
			),
			row=1,
			col=1,
		)

	# Bottom subplot - average data
	# Calculate and plot 100 um IDE average
	averages = df_100_eis.mean(axis=0)
	sds = df_100_eis.std(axis=0)

	fig2.add_trace(
		go.Scatter(
			x=time,
			y=averages,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(0,0,155)"),
			name="100 um IDE Average",
			error_y=dict(type='data', array=sds, visible=True)
		),
		row=2,
		col=1,
	)

	# Calculate and plot 25 um IDE average
	averages = df_25_eis.mean(axis=0)
	sds = df_25_eis.std(axis=0)
	
	fig2.add_trace(
		go.Scatter(
			x=time,
			y=averages,
			mode="lines+markers",
			line=dict(width=1, color=f"rgb(155,0,0)"),
			name="25 um IDE Average",
			error_y=dict(type='data', array=sds, visible=True)
		),
		row=2,
		col=1,
	)

	fig2.update_yaxes(type="log", range=[2,7.7], title_text="Impedance Magnitude (Ohms)")
	fig2.update_xaxes(title_text="Accelerated Time (days)", range=[0, max(time)])

if __name__ == "__main__":
	main()