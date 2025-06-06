from data_processing.lcp_encapsulation_data_processing import process_encapsulation_soak_data
from data_processing.lcp_ide_data_processing import process_ide_soak_data
from data_processing.sirof_vs_pt_data_processing import process_coating_soak_data

import pandas as pd

GROUPS = {
    "SIROF vs Pt": ["IR01", "IR02", "IR03", "IR04", "IR05", "IR06", "IR07", "IR08", "IR09", "IR10", "PT01", "PT02", "PT03", "PT04", "PT05", "PT06", "PT07", "PT08", "PT09", "PT10"],
    "LCP Pt Grids": ["G1X1-1", "G1X1-2", "G3X3S-1", "G3X3S-2", "G2X2S-1", "G2X2S-2", "G2X2L-1", "G2X2L-2"],
    "LCP IDEs": ["IDE-25-1", "IDE-25-2", "IDE-25-3", "IDE-25-4", "IDE-25-5", "IDE-25-6", "IDE-25-7", "IDE-25-8", "IDE-100-1", "IDE-100-2", "IDE-100-3", "IDE-100-4", "IDE-100-5", "IDE-100-6", "IDE-100-7", "IDE-100-8"],
    "LCP Encapsulation": ["ENCAP-R-100-1", "ENCAP-R-100-2", "ENCAP-C-25-2", "ENCAP-C-100-1", "ENCAP-C-100-2"],
}

FLAGS = {
    "SIROF (vs Pt) - Z": 10000, # ohms
    "SIROF (vs Pt) - CIC": 30, # uC/cm2
    "Pt (vs SIROF) - Z": 10000, # ohms
    "Pt (vs SIROF) - CIC": 30, # uC/cm2
    "LCP Pt Grids - Z": 10000, # ohms
    "LCP Pt Grids - CIC": 30, # uC/cm2
    "LCP IDEs - value": 10000, # ohms
    "LCP IDEs - change": 0.8, # difference from start
    "LCP Encapsulation - Cap": 20, # %RH
    "LCP Encapsulation - Res": 40, # %RH
}


# days_encap, RH_cap_sensors, RH_res_sensors = process_encapsulation_soak_data()

# flagged_indices_r = [i for i, rh in enumerate(RH_res_sensors) if rh > FLAGS["LCP Encapsulation - Res"]]
# flagged_indices_c = [i for i, rh in enumerate(RH_cap_sensors) if rh > FLAGS["LCP Encapsulation - Cap"]]

# if len(flagged_indices_r) + len(flagged_indices_c) == 0:
#     summary_encap = f"Encapsulation test at {round(days_encap/365.25, 1)} accelerated years, all parts within expected range."
# else:
#     rh_sensors_r = GROUPS["LCP Encapsulation"][0:2]
#     rh_sensors_c = GROUPS["LCP Encapsulation"][2:5]
#     failing_devices = [rh_sensors_r[i] for i in flagged_indices_r] + [rh_sensors_c[i] for i in flagged_indices_c]
#     summary_encap = f"Encapsulation test at {round(days_encap/365.25, 1)} accelerated years, {', '.join(failing_devices)} above expected range."

# print(summary_encap)

# days_ide, z_IDE_25, z_IDE_100, norm_IDE_25, norm_IDE_100 = process_ide_soak_data()

# flagged_indices_z25 = [i for i, z in enumerate(z_IDE_25) if z < FLAGS["LCP IDEs - value"]]
# flagged_indices_z100 = [i for i, z in enumerate(z_IDE_100) if z < FLAGS["LCP IDEs - value"]]
# flagged_indices_norm25 = [i for i, n in enumerate(norm_IDE_25) if n < FLAGS["LCP IDEs - change"]]
# flagged_indices_norm100 = [i for i, n in enumerate(norm_IDE_100) if n < FLAGS["LCP IDEs - change"]]

# flagged_indices_25 = [x for x in flagged_indices_z25 if x in flagged_indices_norm25]
# flagged_indices_100 = [x for x in flagged_indices_z100 if x in flagged_indices_norm100]

# if len(flagged_indices_25) + len(flagged_indices_100) == 0:
#     summary_ide = f"LCP IDE test at {round(days_ide/365.25, 1)} accelerated years, all parts within expected range."
# else:
#     ides_25 = GROUPS["LCP IDEs"][0:8]
#     ides_100 = GROUPS["LCP IDEs"][8:16]

#     failing_devices = [ides_25[i] for i in flagged_indices_25] + [ides_100[i] for i in flagged_indices_100]
#     summary_ide = f"LCP IDE test at {round(days_ide/365.25, 1)} accelerated years, {', '.join(failing_devices)} below expected range."

# print(summary_ide)

days_sirof, cic_pt, cic_ir, z_pt, z_ir = process_coating_soak_data()

flagged_indices_cicpt = [i for i, c in enumerate(cic_pt) if c < FLAGS["Pt (vs SIROF) - CIC"]]
flagged_indices_cicir = [i for i, c in enumerate(cic_ir) if c < FLAGS["SIROF (vs Pt) - CIC"]]
flagged_indices_zpt = [i for i, z in enumerate(z_pt) if z > FLAGS["Pt (vs SIROF) - Z"]]
flagged_indices_zir = [i for i, z in enumerate(z_ir) if z > FLAGS["SIROF (vs Pt) - Z"]]

if len(flagged_indices_cicpt) + len(flagged_indices_cicir) + len(flagged_indices_zpt) + len(flagged_indices_zir) == 0:
    summary_sirof = f"SIROF vs Pt test at {round(days_sirof/365.25, 1)} accelerated years, all parts within expected range."
else:
    sirof = GROUPS["SIROF vs Pt"][0:10]
    pt = GROUPS["SIROF vs Pt"][10:20]

    failing_devices_cic = [pt[i] for i in flagged_indices_cicpt] + [sirof[i] for i in flagged_indices_cicir]
    failing_devices_z = [pt[i] for i in flagged_indices_zpt] + [sirof[i] for i in flagged_indices_zir]

    if len(failing_devices_cic) == 0:
        summary_sirof = f"LCP IDE test at {round(days_sirof/365.25, 1)} accelerated years, {', '.join(failing_devices_z)} above expected Z range."
    elif len(failing_devices_z) == 0:
        summary_sirof = f"LCP IDE test at {round(days_sirof/365.25, 1)} accelerated years, {', '.join(failing_devices_cic)} below expected CIC range."
    else:
        summary_sirof = f"LCP IDE test at {round(days_sirof/365.25, 1)} accelerated years, {', '.join(failing_devices_cic)} below expected CIC range, {', '.join(failing_devices_z)} above expected Z range."

    print(summary_sirof)