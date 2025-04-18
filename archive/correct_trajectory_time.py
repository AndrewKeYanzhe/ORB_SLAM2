import os

filename = "KeyFrameTrajectory_sdr_V1_4000orb.txt"  # Replace this with your actual file name
change = 1741097195.803000 - 1741737600.000000

# Create new filename
name, ext = os.path.splitext(filename)
new_filename = f"{name}_timeCorrected{ext}"

with open(filename, "r") as infile, open(new_filename, "w") as outfile:
    for line in infile:
        parts = line.strip().split()
        if parts:
            original_time = float(parts[0])
            corrected_time = original_time + change
            corrected_line = f"{corrected_time:.6f} " + " ".join(parts[1:]) + "\n"
            outfile.write(corrected_line)
