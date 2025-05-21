import pandas as pd
import pickle
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj
import numpy as np
import math
import plotly.graph_objects as go
import gradio as gr
from dateparser import parse
from datetime import datetime, timedelta

# 1. Load the data and parse timestamps, load the LineString
df_all = pd.read_csv("df_trainways.csv")
df_all.columns = df_all.columns.str.strip().str.lower()

# Parse UNIX timestamp (float) to datetime

df_all["timestamp"] = df_all["timestamp"].apply(
    lambda x: parse(str(x), languages=['fr'], settings={"TIMEZONE": "Europe/Paris", "RETURN_AS_TIMEZONE_AWARE": False})
)
df_all["ts_unix"] = df_all["timestamp"].apply(lambda x: x.timestamp())

df_all["datetime"] = pd.to_datetime(df_all["timestamp"], unit="s")
# Load the LineString geometry from the pickle file
with open("toulouse_narbonne.pkl", "rb") as f:
    train_line = pickle.load(f)

# Prepare coordinate transformers (WGS84 to UTM and back)
# Assuming UTM zone 31N for Toulouse-Narbonne region
transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
transformer_to_latlon = pyproj.Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
# Define helper functions for shapely.transform (to handle optional z coordinate)
def to_utm(x, y, z=None):
    return transformer_to_utm.transform(x, y)
def to_latlon(x, y, z=None):
    return transformer_to_latlon.transform(x, y)

# Convert train line to UTM coordinates for accurate distance measurements
line_utm = transform(to_utm, train_line)
line_length = line_utm.length  # total length in meters
# 4. Divide the line into 500 m segments
num_segments = math.ceil(line_length / 500)
# Compute segment breakpoints along the line (every 500 m, plus the end of the line)
points_utm = [line_utm.interpolate(min(i * 500, line_length)) for i in range(num_segments + 1)]
# Transform those breakpoints back to lat/lon for plotting
x_coords = [pt.x for pt in points_utm]
y_coords = [pt.y for pt in points_utm]
lon_coords, lat_coords = transformer_to_latlon.transform(x_coords, y_coords)
lat_coords = list(lat_coords)
lon_coords = list(lon_coords)
# Calculate map center (centroid of the line in lat/lon) for initial view
line_latlon = transform(to_latlon, line_utm)
center_point = line_latlon.centroid
center_lat = center_point.y
center_lon = center_point.x

# 2 & 3. Define the function to filter data and project points onto the line
def update_map(operator, trajet, metric, start_ts, end_ts):
    # 🔁 Convertir les timestamps UNIX en datetime (GMT+2)
    start_dt = datetime.fromtimestamp(start_ts) - timedelta(hours=2)
    end_dt = datetime.fromtimestamp(end_ts) - timedelta(hours=2)

    metric = metric.strip().lower()


    mask = (
        (df_all["operateur"].str.strip().str.lower() == operator.strip().lower()) &
        (df_all["trajet"].str.strip().str.lower() == trajet.strip().lower()) &
        (df_all["timestamp"] >= start_dt) &
        (df_all["timestamp"] <= end_dt)
    )

    df_filtered = df_all[mask].copy()
    if df_filtered.empty:
        gr.Warning("⚠️ Aucun point pour cette sélection")
        return go.Figure(), go.Figure()

    # 🔁 Projection des points sur la ligne
    utm_x, utm_y = transformer_to_utm.transform(df_filtered["longitude"].values, df_filtered["latitude"].values)
    df_filtered["distance"] = [line_utm.project(Point(x, y)) for x, y in zip(utm_x, utm_y)]

    def get_color(val):
        if pd.isna(val):
            return "red"
        if metric == "connectmbs":
            if val >= 4:
                return "green"
            elif val >= 1:
                return "orange"
            return "red"
        elif metric == "rsrp":
            if val >= -90:
                return "green"
            elif val >= -110:
                return "orange"
            return "red"
        elif metric == "rsrq":
            if val >= -10:
                return "green"
            elif val >= -15:
                return "orange"
            return "red"
        else:
            return "red"

    df_filtered["point_color"] = df_filtered[metric].apply(get_color)
    df_filtered["segment_index"] = (df_filtered["distance"] // 500).astype("Int64")

    # 🔁 Coloration des segments
    segment_colors = ["red"] * num_segments
    for idx, group in df_filtered.groupby("segment_index"):
        segment_colors[idx] = group["point_color"].mode().iloc[0]

    # 📍 FOCUS sur les distances utiles
    min_dist = df_filtered["distance"].min()
    max_dist = df_filtered["distance"].max()

    focus_coords = []
    for i, (pt1, pt2) in enumerate(zip(points_utm[:-1], points_utm[1:])):
        seg_dist = line_utm.project(pt1)
        if seg_dist >= min_dist - 500 and seg_dist <= max_dist + 500:
            focus_coords.append(i)

    # ➕ Construction de la carte
    fig = go.Figure()
    color_segments = {col: {"lat": [], "lon": []} for col in ["green", "orange", "red", "gray"]}

    g = o = r = 0
    for i in focus_coords:
        col = segment_colors[i]
        if col == "green": g += 1
        elif col == "orange": o += 1
        elif col == "red": r += 1
        color_segments[col]["lat"].extend([lat_coords[i], lat_coords[i + 1], None])
        color_segments[col]["lon"].extend([lon_coords[i], lon_coords[i + 1], None])

    for col, coords in color_segments.items():
        if coords["lat"]:
            fig.add_trace(go.Scattermapbox(
                lat=coords["lat"], lon=coords["lon"],
                mode="lines", line=dict(color=col, width=5),
                hoverinfo="none", name=col
            ))

    segment_lats = [lat_coords[i] for i in focus_coords] + [lat_coords[i + 1] for i in focus_coords]
    segment_lons = [lon_coords[i] for i in focus_coords] + [lon_coords[i + 1] for i in focus_coords]

    if segment_lats and segment_lons:
        center_lat = np.mean(segment_lats)
        center_lon = np.mean(segment_lons)
    else:
        center_lat = line_latlon.centroid.y
        center_lon = line_latlon.centroid.x

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=8.5,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False
    )

    # ➕ Ajouter le graphique circulaire
    nombre_seg = g + o + r
    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=["Bonne", "Moyenne", "Mauvaise"],
                values=[g, o, r],
                marker=dict(colors=["green", "orange", "red"]),
                hole=0.3,
                textinfo="label+percent"
            )
        ]
    )
    fig_pie.update_layout(
        title=f"Qualité de connectivité sur {nombre_seg} segments",
        height=300,
        margin={"t": 40, "b": 0, "l": 0, "r": 0}
    )

    return fig, fig_pie


def update_display(ts_start, ts_end):
    readable_start = datetime.fromtimestamp(ts_start).strftime("%d/%m %H:%M")
    readable_end = datetime.fromtimestamp(ts_end).strftime("%d/%m %H:%M")
    return readable_start, readable_end

# 6. Create the Gradio interface with interactive controls
with gr.Blocks() as demo:
    gr.Markdown("## Carte de connectivité sur le trajet Toulouse-Narbonne")
    with gr.Row():
        operator_input = gr.Dropdown(label="Opérateur", choices=["Orange", "SFR", "Transatel", "Stellar"], value="Orange")
        trip_input = gr.Radio(label="Trajet", choices=["aller", "retour"], value="aller")
        metric_input = gr.Dropdown(
            label="Métrique",
            choices=["connectMbs", "rsrp", "rsrq"],
            value="connectMbs",
        )
        start_input = gr.Slider(
            label="Début",
            minimum=df_all["ts_unix"].min(),
            maximum=df_all["ts_unix"].max(),
            value=df_all["ts_unix"].min(),
            step=60,
            elem_classes="smaller"
        )

        end_input = gr.Slider(
            label="Fin",
            minimum=df_all["ts_unix"].min(),
            maximum=df_all["ts_unix"].max(),
            value=df_all["ts_unix"].max(),
            step=60,
            elem_classes="smaller"
        )


    show_button = gr.Button("Afficher la carte")
    map_output = gr.Plot(label="Carte")
    pie_output = gr.Plot(label="Qualité de la portion")

    # Bind the update function to the button click
    show_button.click(
        fn=update_map,
        inputs=[operator_input, trip_input, metric_input, start_input, end_input],
        outputs=[map_output, pie_output]
    )
    # Optionally, load the initial view on app launch
    demo.load(
        fn=update_map,
        inputs=[operator_input, trip_input, metric_input, start_input, end_input],
        outputs=[map_output, pie_output],
    )



# Launch the dashboard
demo.queue()  # enable queued processing for Warning messages
if __name__ == "__main__":
    demo.launch()
