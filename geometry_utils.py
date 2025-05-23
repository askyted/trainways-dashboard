import math
import pickle

import pyproj
from shapely.geometry import Point
from shapely.ops import transform

# Load the LineString geometry from the pickle file
with open("toulouse_narbonne.pkl", "rb") as f:
    train_line = pickle.load(f)

# Prepare coordinate transformers (WGS84 to UTM and back)
# Assuming UTM zone 31N for Toulouse-Narbonne region
transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
transformer_to_latlon = pyproj.Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)

def to_utm(x, y, z=None):
    return transformer_to_utm.transform(x, y)

def to_latlon(x, y, z=None):
    return transformer_to_latlon.transform(x, y)

# Convert train line to UTM coordinates for accurate distance measurements
line_utm = transform(to_utm, train_line)
line_length = line_utm.length  # total length in meters
# Divide the line into 500 m segments
num_segments = math.ceil(line_length / 500)
# Breakpoints along the line (every 500 m, plus the end)
points_utm = [line_utm.interpolate(min(i * 500, line_length)) for i in range(num_segments + 1)]
# Back to lat/lon for plotting
x_coords = [pt.x for pt in points_utm]
y_coords = [pt.y for pt in points_utm]
lon_coords, lat_coords = transformer_to_latlon.transform(x_coords, y_coords)
lat_coords = list(lat_coords)
lon_coords = list(lon_coords)
# Map center (centroid of the line in lat/lon)
line_latlon = transform(to_latlon, line_utm)
center_point = line_latlon.centroid
center_lat = center_point.y
center_lon = center_point.x
