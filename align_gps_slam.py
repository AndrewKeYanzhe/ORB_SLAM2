from read_gps import read_gps_and_plot
from plot_slam_trajectory import read_slam_trajectory
import numpy as np

# 1 sample per second
gps_x, gps_y, gps_time = read_gps_and_plot(show_plot=False)


index=-1

print('\n index:',index)

print('\ntest gps data')
print(gps_time[index],gps_x[index],gps_y[index])

# about 3 keyframes per second
slam_time, slam_x_og, slam_y_og, slam_z_og = read_slam_trajectory("KeyFrameTrajectory_pq_4000orb.txt")

print('\ntest slam data')
print(slam_time[index],slam_x_og[index],slam_y_og[index],slam_z_og[index])

print('\nlength of gps data:',len(gps_time))
print('length of slam data:',len(slam_time))

slam_start_time = slam_time[0]
slam_end_time = slam_time[-1]

# Filter GPS data to include only entries between SLAM start and end times
filtered_gps_x = [x for i, x in enumerate(gps_x) if slam_start_time <= gps_time[i] <= slam_end_time]
filtered_gps_y = [y for i, y in enumerate(gps_y) if slam_start_time <= gps_time[i] <= slam_end_time]
filtered_gps_time = [t for t in gps_time if slam_start_time <= t <= slam_end_time]

# Test filtered GPS data
index = -1
print('index:', index)
print('\nFiltered GPS data')
print(filtered_gps_time[index], filtered_gps_x[index], filtered_gps_y[index])

gps_x = filtered_gps_x
gps_y = filtered_gps_y
gps_time = filtered_gps_time

print('\nlength of filtered gps data:', len(gps_time))


# Function to find the closest SLAM time index for a given GPS time
# input: gps_time value, slam_time list
def find_closest_index(target_time, time_list):
    return np.argmin(np.abs(np.array(time_list) - target_time))

# SLAM data is 3fps, GPS data is 1fps. this remaps SLAM XYZ to 1fps, aligned with GPS list
slam_x = [slam_x_og[find_closest_index(t, slam_time)] for t in filtered_gps_time]
slam_y = [slam_y_og[find_closest_index(t, slam_time)] for t in filtered_gps_time]
slam_z = [slam_z_og[find_closest_index(t, slam_time)] for t in filtered_gps_time]

print('length of slam_x:', len(slam_x))



