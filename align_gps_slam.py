from read_gps import read_gps_and_plot
from plot_slam_trajectory import read_slam_trajectory, plot_2d_trajectory
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





def horn_alignment(slam_points, gps_points):
    """
    Aligns SLAM points to GPS points using Horn's method. by copilot
    Horn alignment doesnt have scale.
    
    this is actually Umeyama's method
    1. Compute the centroids of both point sets  
    2. Subtract centroids to center the point sets  
    3. Compute the covariance matrix between the centered sets  
    4. Perform Singular Value Decomposition (SVD) on the covariance matrix  
    5. Compute the rotation matrix using the SVD result  
    6. Compute the scale factor (optional)  
    7. Compute the translation vector  

    

    Args:
        slam_points: Nx2 numpy array of SLAM trajectory points.
        gps_points: Nx2 numpy array of GPS trajectory points.
    Returns:
        aligned_slam_points: Nx2 numpy array of aligned SLAM points.
        transformation_matrix: 3x3 transformation matrix (scale, rotation, translation).
    """
    # Ensure the points are numpy arrays
    slam_points = np.asarray(slam_points)
    gps_points = np.asarray(gps_points)

    # Compute the centroids of both point sets
    slam_centroid = np.mean(slam_points, axis=0)
    gps_centroid = np.mean(gps_points, axis=0)

    # Center the points by subtracting the centroids
    slam_centered = slam_points - slam_centroid
    gps_centered = gps_points - gps_centroid

    # Compute the cross-covariance matrix
    W = np.dot(gps_centered.T, slam_centered)

    # Perform Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(W)

    # Compute the rotation matrix
    R = np.dot(U, Vt)

    # Ensure a proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)

    # Compute the scale factor
    scale = np.trace(np.dot(R.T, W)) / np.sum(slam_centered ** 2)

    # Compute the translation vector
    t = gps_centroid - scale * np.dot(slam_centroid, R.T)

    # Apply the transformation to SLAM points
    aligned_slam_points = scale * np.dot(slam_points, R.T) + t

    # Construct the transformation matrix
    transformation_matrix = np.eye(3)
    transformation_matrix[:2, :2] = scale * R
    transformation_matrix[:2, 2] = t

    return aligned_slam_points, transformation_matrix

# Prepare SLAM and GPS points for alignment
slam_points = np.column_stack((slam_x, slam_y))
gps_points = np.column_stack((gps_x, gps_y))

# Align SLAM points to GPS points using Horn's method
aligned_slam_points, transformation_matrix = horn_alignment(slam_points, gps_points)

# Extract aligned SLAM x and y coordinates
aligned_slam_x, aligned_slam_y = aligned_slam_points[:, 0], aligned_slam_points[:, 1]

# Print the transformation matrix
print("Horn's Method Transformation Matrix:")
print(transformation_matrix)

# Plot the aligned trajectories
plot_2d_trajectory(gps_x, gps_y, title1='GPS', x2=aligned_slam_x, y2=aligned_slam_y, title2='SLAM')