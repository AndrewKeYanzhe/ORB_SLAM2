import matplotlib.pyplot as plt
import folium
import re

# File path
gnss_log_file = "gnss_log_2025_02_28_20_25_16_HomLib.txt"
gnss_log_file = 'gnss_log_2025_02_28_21_08_40_toGym.txt'
gnss_log_file='gnss_log_2025_02_28_23_05_22_fromGym.txt'
gnss_log_file='gnss_log_2025_03_04_14_00_07_beechwoods.txt'

# Regex pattern to match relevant lines
pattern = re.compile(r"Fix,gps,([-\d.]+),([-\d.]+),")

# Location fix format:
#   Fix,Provider,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SpeedMps,AccuracyMeters,BearingDegrees,UnixTimeMillis,SpeedAccuracyMps,BearingAccuracyDegrees,elapsedRealtimeNanos,VerticalAccuracyMeters,MockLocation
# Fix,gps,52.17132169,0.1711431,86.537353515625,1.05,35.74975,324.2,1741096809398,1.2041595,43.867138,327904097670807,20.321562,0

start_time=1741097195803

# round down to nearest second
start_time== (start_time // 1000) * 1000

from pyproj import Proj, transform

def read_gps_and_plot(show_plot=True):
    """Reads GPS data from the file and plots the trajectory in meters (x, y) with the first point as the origin."""
    """Plots the trajectory in meters (x, y) instead of latitude and longitude, with the first point as the origin."""
    '''return x_coords, y_coords, unix_times in seconds'''
    # Lists to store extracted coordinates
    latitudes = []
    longitudes = []
    unix_times = []  # List to store corresponding unix_time_millis

    # Read and extract data from the file
    with open(gnss_log_file, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                fields = line.split(",")
                unix_time_millis = int(fields[8])  # 9th element (0-based index)

                if unix_time_millis >= start_time:
                    latitudes.append(float(match.group(1)))
                    longitudes.append(float(match.group(2)))
                    unix_times.append(unix_time_millis/1000)  # Append the unix_time_millis

    # Convert latitude and longitude to x, y in meters
    if latitudes and longitudes:
        # Define a projection (e.g., UTM or local projection)
        proj_wgs84 = Proj(proj="latlong", datum="WGS84")  # WGS84 (latitude/longitude)
        proj_utm = Proj(proj="utm", zone=31, datum="WGS84")  # UTM Zone 31 (adjust zone as needed)

        x_coords, y_coords = transform(proj_wgs84, proj_utm, longitudes, latitudes)

        # Set the first point as the origin
        x_origin, y_origin = x_coords[0], y_coords[0]
        x_coords = [x - x_origin for x in x_coords]
        y_coords = [y - y_origin for y in y_coords]

        if show_plot:
            # Plot the GPS coordinates in meters
            plt.figure(figsize=(4.5, 4.5))
            plt.plot(x_coords, y_coords, marker='o', markersize=1, linestyle='-', color='b', label="GPS Path")  # Reduced marker size

            plt.xlabel("X (meters)")
            plt.ylabel("Y (meters)")
            plt.title("GPS Coordinates (Origin at First Point)")
            plt.axis("equal")  # Ensure equal scaling
            plt.gca().set_aspect('equal', adjustable='datalim')  # Ensure 
            plt.legend()
            plt.grid()
            plt.savefig('gps_path_plot.pdf')
            plt.show()
    else:
        print("No valid GPS coordinates found.")

    
    return x_coords, y_coords, unix_times



def plot_trajectory_on_html_map():



    # Lists to store extracted coordinates
    coordinates = []

    # Read and extract data from the file
    with open(gnss_log_file, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                fields = line.split(",")
                unix_time_millis = int(fields[8])  # 9th element (0-based index)

                if unix_time_millis >= start_time:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    coordinates.append((lat, lon))

    # Create a folium map centered at the first GPS point
    if coordinates:
        start_lat, start_lon = coordinates[0]
        gps_map = folium.Map(location=[start_lat, start_lon], zoom_start=15,control_scale = True)

        # Add a polyline (GPS path)
        folium.PolyLine(coordinates, color="blue", weight=2.5, opacity=1).add_to(gps_map)

        # Add start and end markers
        folium.Marker(coordinates[0], popup="Start", icon=folium.Icon(color="green")).add_to(gps_map)
        folium.Marker(coordinates[-1], popup="End", icon=folium.Icon(color="red")).add_to(gps_map)

        # Save map as an HTML file and display it
        gps_map.save("gps_path_map.html")
        print("Map saved as gps_path_map.html. Open it in a browser to view the route.")
    else:
        print("No valid GPS coordinates found.")






if __name__ == "__main__":
    plot_trajectory_on_html_map()
    read_gps_and_plot()
