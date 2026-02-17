import json
import re
import statistics
import pandas as pd
import numpy as np

print(np.__version__)
# SAMPLE_INFORMATION_PATH = './test_information/samples'
# # done with: encap cap, encap res, ide25, ide100, grids, pt, ir

# f1 = f"data/SIROF on Pt Foil/IR01_data_summary.csv"
# f2 = f"data/SIROF on Pt Foil/IR02_data_summary.csv"
# f3 = f"data/SIROF on Pt Foil/IR03_data_summary.csv"
# f4 = f"data/SIROF on Pt Foil/IR04_data_summary.csv"
# f5 = f"data/SIROF on Pt Foil/IR05_data_summary.csv"
# f6 = f"data/SIROF on Pt Foil/IR06_data_summary.csv"
# f7 = f"data/SIROF on Pt Foil/IR07_data_summary.csv"
# f8 = f"data/SIROF on Pt Foil/IR08_data_summary.csv"
# f9 = f"data/SIROF on Pt Foil/IR09_data_summary.csv"
# f10 = f"data/SIROF on Pt Foil/IR10_data_summary.csv"

# df1 = pd.read_csv(f1)
# df1 = df1.drop(columns=["Unnamed: 0"], axis=1)
# df1 = df1.sort_values("Real Days").reset_index(drop=True)

# df2 = pd.read_csv(f2)
# df2 = df2.drop(columns=["Unnamed: 0"], axis=1)
# df2 = df2.sort_values("Real Days").reset_index(drop=True)

# df3 = pd.read_csv(f3)
# df3 = df3.drop(columns=["Unnamed: 0"], axis=1)
# df3 = df3.sort_values("Real Days").reset_index(drop=True)

# df4 = pd.read_csv(f4)
# df4 = df4.drop(columns=["Unnamed: 0"], axis=1)
# df4 = df4.sort_values("Real Days").reset_index(drop=True)

# df5 = pd.read_csv(f5)
# df5 = df5.drop(columns=["Unnamed: 0"], axis=1)
# df5 = df5.sort_values("Real Days").reset_index(drop=True)

# df6 = pd.read_csv(f6)
# df6 = df6.drop(columns=["Unnamed: 0"], axis=1)
# df6 = df6.sort_values("Real Days").reset_index(drop=True)

# df7 = pd.read_csv(f7)
# df7 = df7.drop(columns=["Unnamed: 0"], axis=1)
# df7 = df7.sort_values("Real Days").reset_index(drop=True)

# df8 = pd.read_csv(f8)
# df8 = df8.drop(columns=["Unnamed: 0"], axis=1)
# df8 = df8.sort_values("Real Days").reset_index(drop=True)

# df9 = pd.read_csv(f9)
# df9 = df9.drop(columns=["Unnamed: 0"], axis=1)
# df9 = df9.sort_values("Real Days").reset_index(drop=True)

# df10 = pd.read_csv(f10)
# df10 = df10.drop(columns=["Unnamed: 0"], axis=1)
# df10 = df10.sort_values("Real Days").reset_index(drop=True)

# out = [df.copy() for df in [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]]
# anchor_df = df1.copy()
# anchor_tbl = anchor_df[["Real Days"]].rename(columns={"Real Days": "anchor_day"}).reset_index(drop=True)
# anchor_tbl["anchor_id"] = np.arange(len(anchor_tbl), dtype=int)

# match_tables = []
# for i, df in enumerate(out):
#     left = df[["Real Days", "Temperature (C)"]].copy().reset_index().rename(columns={"index": "row_idx"})
#     right = anchor_tbl.sort_values("anchor_day")

#     m = pd.merge_asof(
#         left.sort_values("Real Days"),
#         right,
#         left_on="Real Days",
#         right_on="anchor_day",
#         direction="nearest",
#         tolerance=None
#     )

#     m["df_i"] = i
#     match_tables.append(m)

# matches = pd.concat(match_tables, ignore_index=True)

# matches = matches.dropna(subset=["anchor_id"]).copy()
# matches["anchor_id"] = matches["anchor_id"].astype(int)

# avg_by_anchor = (
#     matches.groupby("anchor_id")["Temperature (C)"]
#     .mean()
#     .rename("avg_temp")
#     .reset_index()
# )

# matches = matches.merge(avg_by_anchor, on="anchor_id", how="left")

# for i in range(len(out)):
#     sub = matches[matches["df_i"] == i]
#     # Update only the matched rows
#     out[i].loc[sub["row_idx"].to_numpy(), "Temperature (C)"] = sub["avg_temp"].to_numpy()

# out[0].to_csv(f1)
# out[1].to_csv(f2)
# out[2].to_csv(f3)
# out[3].to_csv(f4)
# out[4].to_csv(f5)
# out[5].to_csv(f6)
# out[6].to_csv(f7)
# out[7].to_csv(f8)
# out[8].to_csv(f9)
# out[9].to_csv(f10)