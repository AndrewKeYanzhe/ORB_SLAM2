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
    
    plt.figure()
    
    # Create a colormap that transitions from green to red
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(x)))
    
    for i in range(len(x) - 1):
        plt.plot(x[i:i+2], y[i:i+2], color=colors[i])
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Trajectory (XY Plane)")
    # plt.legend()
    plt.axis("equal")  # Ensure equal scaling
    # plt.show()
    plt.savefig('2d_plot.pdf')


plot_3d_trajectory("KeyFrameTrajectory_360p_sdr_4000orb.txt")
plot_2d_trajectory("KeyFrameTrajectory_360p_sdr_4000orb.txt")