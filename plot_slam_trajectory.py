# https://github.com/raulmur/ORB_SLAM2/issues/467

# Every row has 8 entries containing timestamp (in seconds), position and orientation (as quaternion):
# timestamp x y z q_x q_y q_z q_w

import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def read_slam_trajectory(file_path):
    """Reads trajectory data from the file and returns numpy arrays for timestamp, x, y, z."""
    data = np.loadtxt(file_path, comments='#', usecols=(0, 1, 2, 3))  # Include the timestamp column
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]  # Return timestamp, x, y, z

def plot_3d_trajectory(file_path):
    """Plots the 3D trajectory using x, y, z coordinates."""
    _, x, y, z = read_slam_trajectory(file_path)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap that transitions from green to red
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(x)))
    
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i])
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D SLAM Trajectory")
    # ax.legend()
    # plt.show()
    plt.savefig('3d_plot_slam.pdf')

def plot_2d_trajectory(x,y, title1 = None, x2=None, y2=None, title2=None):
    """Plots the 2D trajectory using x, y coordinates."""
    
    
    # plt.figure(figsize=(4.5, 2.5))  # Set figure size for a two-column paper
    plt.figure(figsize=(4.5, 4.5))  # Set figure size for a two-column paper
    
    # Create a colormap that transitions from green to red
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(x)))
    colors2 = plt.cm.Reds(np.linspace(0.2, 1, len(x)))
    
    for i in range(len(x) - 1):
        if i == len(x) // 2:
            plt.plot(x[i:i+2], y[i:i+2], color=colors[i], label=title1)
        else:
            plt.plot(x[i:i+2], y[i:i+2], color=colors[i])
    
    if x2 is not None and y2 is not None:
        for i in range(len(x2) - 1):
            if i == len(x2) //2:
                plt.plot(x2[i:i+2], y2[i:i+2], color=colors2[i], label=title2)
            else:
                plt.plot(x2[i:i+2], y2[i:i+2], color=colors2[i])
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D SLAM Trajectory (XY Plane)")
    

    
    plt.axis("equal")  # Ensure equal scaling
    plt.gca().set_aspect('equal', adjustable='datalim')  # Ensure square chart area
    # 'box': Adjusts the box size of the plot area (the figure area stays fixed).
    # 'datalim': Would adjust the data limits instead.
    plt.legend()
    plt.tight_layout()
    plt.savefig('2d_plot_slam.pdf')
    plt.show()

if __name__ == "__main__":
    # Example usage

    # trajectory_path = "KeyFrameTrajectory_sdr_4000orb.txt"
    trajectory_path = "KeyFrameTrajectory_pq_4000orb.txt"

    # plot_3d_trajectory(trajectory_path)

    _, x, y, _ = read_slam_trajectory(trajectory_path)
    plot_2d_trajectory(x,y)