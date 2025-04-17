

# frame_id,state,num_keyframes,num_mappoints,mnTracked,mnTrackedVO,num_orb_features
# 6,1,0,0,0,0,7538
# 6,1,0,0,0,0,7538
# 6,1,0,0,0,0,7538
# 6,1,0,0,0,0,7538
# 7,1,0,0,0,0,7522
# 7,1,0,0,0,0,7522
# 7,1,0,0,0,0,7522
# 7,1,0,0,0,0,7522
# 7,1,0,0,0,0,7522
# 7,1,0,0,0,0,7522
# 8,1,0,0,0,0,7525
# 8,1,0,0,0,0,7525
# 8,1,0,0,0,0,7525
# 8,1,0,0,0,0,7525
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files and skip repeated lines
pq_data = pd.read_csv("slam_stats_pq_4000orb.csv", header=0, names=["frame_id", "state", "num_keyframes", "num_mappoints", "mnTracked", "mnTrackedVO", "num_orb_features"])
sdr_data = pd.read_csv("slam_stats_sdr_4000orb.csv", header=0, names=["frame_id", "state", "num_keyframes", "num_mappoints", "mnTracked", "mnTrackedVO", "num_orb_features"])

# Convert all columns to integer type
pq_data = pq_data.astype(int)
sdr_data = sdr_data.astype(int)

# Drop duplicate rows
pq_data = pq_data.drop_duplicates()
sdr_data = sdr_data.drop_duplicates()

# Subsample the data to 100 points for each trend
# pq_data = pq_data.iloc[::max(1, len(pq_data) // 100)]
# sdr_data = sdr_data.iloc[::max(1, len(sdr_data) // 100)]

# Extract frame_id and mnTracked for plotting
pq_frame_id = pq_data["frame_id"]
pq_mnTracked = pq_data["mnTracked"]

sdr_frame_id = sdr_data["frame_id"]
sdr_mnTracked = sdr_data["mnTracked"]

print(pq_data)
# print(pq_data["frame_id"][0].dtypes)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(pq_frame_id, pq_mnTracked, label="PQ mnTracked", color="blue")
plt.plot(sdr_frame_id, sdr_mnTracked, label="SDR mnTracked", color="red")



# Add labels, title, and legend
plt.xlabel("Frame ID")
plt.ylabel("mnTracked")
plt.title("mnTracked Trends for PQ and SDR (Subsampled)")
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.savefig("mnTracked_trends_subsampled.pdf")
plt.show()