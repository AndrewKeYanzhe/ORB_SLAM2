

# frame_id,state,num_keyframes,num_mappoints,mnTracked,mnTrackedVO,num_orb_features
# 6,1,0,0,0,0,7538
# 6,1,0,0,0,0,7538
# 6,1,0,0,0,0,7538
# 6,1,0,0,0,0,7538
# 7,1,0,0,0,0,7522
# 7,1,0,0,0,0,7522
# 7,1,0,0,0,0,7522
# 7,1,0,0,0,0,7522


# SYSTEM_NOT_READY=-1,
# NO_IMAGES_YET=0,
# NOT_INITIALIZED=1,
# OK=2,
# LOST=3

# quirk in my data logging code: when ORBSLAM2 is restarted, it will continue appending to the csv file


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# csv_1 = 'slam_stats_sdr_log_V1_lost_incomplete.csv'
# csv_1 = 'slam_stats_hdr_log_V1_4000orb_lost_complete.csv'

# this is a good comparison
csv_1 = 'slam_stats_hdr_log_V5_ThFast20_7_lost.csv'
# csv_2 = 'slam_stats_hdr_log_V3_ThFast3_3_partial.csv'
csv_2 = 'slam_stats_hdr_log_V2_lost_partial.csv'
csv_3 = 'slam_stats_hdr_log_V4_full.csv'


def check_frame_id_decrements(data):
    """
    Checks for decreases in the 'frame_id' column of the given DataFrame.
    Prints a warning if any decrement is found and a message if no decrements occur.
    
    Parameters:
    data (pd.DataFrame): The DataFrame to check.
    """
    decrement_warning = False

    for i in range(1, len(data)):
        if data.iloc[i]["frame_id"] < data.iloc[i - 1]["frame_id"]:
            print(f"Warning: frame_id decreased from {data.iloc[i - 1]['frame_id']} to {data.iloc[i]['frame_id']} at index {i}.")
            decrement_warning = True

    if not decrement_warning:
        print("No decrement in 'frame_id' for the entire DataFrame.")

# Load the CSV files and skip repeated lines
# pq_data = pd.read_csv("slam_stats_pq_4000orb.csv", header=0, names=["frame_id", "state", "num_keyframes", "num_mappoints", "mnTracked", "mnTrackedVO", "num_orb_features"])
# sdr_data = pd.read_csv("slam_stats_sdr_4000orb.csv", header=0, names=["frame_id", "state", "num_keyframes", "num_mappoints", "mnTracked", "mnTrackedVO", "num_orb_features"])
# Define the data types for each column
data_types = {
    "frame_id": int,
    "state": int,
    "num_keyframes": int,
    "num_mappoints": int,
    "mnTracked": int,
    "mnTrackedVO": int,
    "num_orb_features": int,
}

# Load the CSV files with specified data types
pq_data = pd.read_csv(csv_1, header=0, names=data_types.keys(), dtype=data_types)
sdr_data = pd.read_csv(csv_2, header=0, names=data_types.keys(), dtype=data_types)
data_3 = pd.read_csv(csv_3, header=0, names=data_types.keys(), dtype=data_types)

# Drop duplicate rows
pq_data = pq_data.drop_duplicates()
sdr_data = sdr_data.drop_duplicates()
data_3 = data_3.drop_duplicates()

print('pq_data')
check_frame_id_decrements(pq_data)
print('sdr_data')
check_frame_id_decrements(sdr_data)


# clean up data. otherwise mntracked always uses previous row's value if tracking is lost
pq_data.loc[pq_data["state"] == 3, "mnTracked"] = np.nan
sdr_data.loc[sdr_data["state"] == 3, "mnTracked"] = np.nan
data_3.loc[data_3["state"] == 3, "mnTracked"] = np.nan



# Subsample the data to 100 points for each trend
# pq_data = pq_data.iloc[::max(1, len(pq_data) // 100)]
# sdr_data = sdr_data.iloc[::max(1, len(sdr_data) // 100)]

# Extract frame_id and mnTracked for plotting
pq_frame_id = pq_data["frame_id"]
pq_mnTracked = pq_data["mnTracked"]



sdr_frame_id = sdr_data["frame_id"]
sdr_mnTracked = sdr_data["mnTracked"]

data_3_frame_id = data_3["frame_id"]
data_3_mnTracked = data_3["mnTracked"]

# print(pq_data)
# print(pq_data["frame_id"][0].dtypes)

labels = ["iniThFAST=20, minThFAST=7", "iniThFAST=9, minThFAST=3", "iniThFAST=3, minThFAST=3"]
colors = ["red", "#598eff", "green"]  # Light blue in hex


plt.figure(figsize=(4.5, 4.5))
plt.plot(pq_frame_id, pq_mnTracked, label=labels[0], color=colors[0])
plt.plot(sdr_frame_id, sdr_mnTracked, label=labels[1], color=colors[1])
plt.plot(data_3_frame_id, data_3_mnTracked, label=labels[2], color=colors[2])



# Add labels, title, and legend
plt.xlabel("frame number")
plt.ylabel("number of feature points matched")
plt.xlim(710, 750)
plt.ylim(0,200)
plt.title("Feature point matches (HDR-log)")
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.savefig("feature_point_matches_hdr-log.pdf")
plt.show()