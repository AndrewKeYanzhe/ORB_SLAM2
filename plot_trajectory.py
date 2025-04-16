# https://github.com/raulmur/ORB_SLAM2/issues/467

# Every row has 8 entries containing timestamp (in seconds), position and orientation (as quaternion):
# timestamp x y z q_x q_y q_z q_w

import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def read_trajectory(file_path):
    """Reads trajectory data from the file and returns numpy arrays for x, y, z."""
    data = np.loadtxt(file_path, comments='#', usecols=(1, 2, 3))
    return data[:, 0], data[:, 1], data[:, 2]

def plot_3d_trajectory(file_path):
    """Plots the 3D trajectory using x, y, z coordinates."""
    x, y, z = read_trajectory(file_path)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap that transitions from green to red
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(x)))
    
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i])
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory")
    # ax.legend()
    # plt.show()
    plt.savefig('3d_plot.pdf')

def plot_2d_trajectory(file_path):
    """Plots the 2D trajectory using x, y coordinates."""
    x, y, _ = read_trajectory(file_path)
    
    # plt.figure(figsize=(4.5, 2.5))  # Set figure size for a two-column paper
    plt.figure(figsize=(4.5, 4.5))  # Set figure size for a two-column paper
    
    # Create a colormap that transitions from green to red
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(x)))
    
    for i in range(len(x) - 1):
        plt.plot(x[i:i+2], y[i:i+2], color=colors[i])
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Trajectory (XY Plane)")
    
    # # Set axis limits to ensure square scaling
    # x_min, x_max = min(x), max(x)
    # y_min, y_max = min(y), max(y)
    # axis_min = min(x_min, y_min)
    # axis_max = max(x_max, y_max)
    # plt.xlim(axis_min, axis_max)
    # plt.ylim(axis_min, axis_max)
    
    plt.axis("equal")  # Ensure equal scaling
    plt.gca().set_aspect('equal', adjustable='datalim')  # Ensure square chart area
    # 'box': Adjusts the box size of the plot area (the figure area stays fixed).
    # 'datalim': Would adjust the data limits instead.
    plt.tight_layout()
    plt.savefig('2d_plot.pdf')


trajectory_path = "KeyFrameTrajectory_sdr_4000orb.txt"
# trajectory_path = "KeyFrameTrajectory_pq_4000orb.txt"

plot_3d_trajectory(trajectory_path)
plot_2d_trajectory(trajectory_path)