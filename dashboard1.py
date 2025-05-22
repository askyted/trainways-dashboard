"""Interactive dashboard to explore mobile connectivity along the train line."""

import math
from datetime import datetime, timedelta
import pickle

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyproj
from dateparser import parse
from shapely.geometry import Point
from shapely.ops import transform


def load_dataset(path: str) -> pd.DataFrame:
    """Load and normalize the CSV dataset."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["timestamp"] = df["timestamp"].apply(
        lambda x: parse(
            str(x),
            languages=["fr"],
            settings={"TIMEZONE": "Europe/Paris", "RETURN_AS_TIMEZONE_AWARE": False},
        )
    )
    df["ts_unix"] = df["timestamp"].apply(lambda x: x.timestamp())
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


df_all = load_dataset("df_trainways.csv")


def get_color(value: float, metric: str) -> str:
    """Return a color for the given metric value."""
    if pd.isna(value):
        return "red"
    metric = metric.lower()
    if metric == "connectmbs":
        if value >= 4:
            return "green"
        if value >= 1:
            return "orange"
        return "red"
    if metric == "rsrp":
        if value >= -90:
            return "green"
        if value >= -110:
            return "orange"
        return "red"
    if metric == "rsrq":
        if value >= -10:
            return "green"
        if value >= -15:
            return "orange"
        return "red"
    return "red"


def clip_connectmbs(val: float) -> float:
    """Return a sanitized connectmbs value."""
    try:
        if pd.isna(val):
            return 0
        val = float(val)
        return 6 if val > 5 else val
    except Exception:
        return 0


stations_data = {
    "Gare": [
        "Labège-Innopole", "Escalquens", "Baziège", "Villenouvelle",
        "Villefranche-de-Lauragais", "Avignonet-Lauragais", "Castelnaudary",
        "Bram", "Carcassonne", "Lézignan-Corbières", "Narbonne"
    ],
    "Latitude": [
        43.5469, 43.5167, 43.4833, 43.4667,
        43.4000, 43.3833, 43.3167, 43.2500,
        43.2175, 43.2000, 43.1875
    ],
    "Longitude": [
        1.5133, 1.5667, 1.6167, 1.6667,
        1.7333, 1.8000, 1.9500, 2.1167,
        2.3518, 2.7500, 3.0037
    ]
}

# Création du DataFrame
df_gares = pd.DataFrame(stations_data)


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
    """Return map, pie chart and distance chart for the selected period."""
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

    df_filtered["point_color"] = df_filtered[metric].apply(lambda v: get_color(v, metric))
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
   # ➕ Lissage : regrouper par tranche de 100 mètres

    df_filtered["connectmbs"] = df_filtered["connectmbs"].apply(clip_connectmbs)

    df_filtered["distance_bin"] = (df_filtered["distance"] // 100)*100
    df_grouped = df_filtered.groupby("distance_bin").agg(
        mean_connectmbs=("connectmbs", "mean"),  # 🔁 ici c'est bien minuscule
        count=("connectmbs", "count")
    ).reset_index()

    # 🔧 Recalculer les distances des gares si pas déjà fait
    utm_gare_x, utm_gare_y = transformer_to_utm.transform(df_gares["Longitude"].values, df_gares["Latitude"].values)
    df_gares["distance"] = [line_utm.project(Point(x, y)) for x, y in zip(utm_gare_x, utm_gare_y)]

    # ➕ Nouveau graphique : connectmbs en fonction de la distance
    fig_dist = go.Figure()

    # Courbe lissée
    fig_dist.add_trace(go.Scatter(
        x=df_grouped["distance_bin"],
        y=df_grouped["mean_connectmbs"],
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(size=6),
        name="connectmbs (moyenne)"
    ))

    # Lignes verticales + noms des gares
    for _, row in df_gares.iterrows():
        fig_dist.add_shape(
            type="line",
            x0=row["distance"],
            x1=row["distance"],
            y0=df_grouped["mean_connectmbs"].min(),
            y1=df_grouped["mean_connectmbs"].max(),
            line=dict(color="blue")
        )
        fig_dist.add_annotation(
            x=row["distance"],
            y=df_grouped["mean_connectmbs"].max(),
            text=row["Gare"],
            showarrow=False,
            yanchor="bottom",
            textangle=-90,
            font=dict(size=12, color="blue")

        )
    tick_vals = np.linspace(df_grouped["distance_bin"].min(), df_grouped["distance_bin"].max(), num=10)

    tick_text = [f"{int(val/1000)}" for val in tick_vals]
    fig_dist.update_layout(
        xaxis=dict(
            title="Distance le long du trajet (km)",
            tickvals=tick_vals,
            ticktext=tick_text
        ),
        title="Connectivité (connectmbs) en fonction de la distance",
        xaxis_title="Distance le long du trajet (km)",
        yaxis_title="connectmbs (moyenne)",
        height=400,
        margin={"t": 40, "b": 40, "l": 40, "r": 40}

    )
    # ➕ Lignes de référence horizontales
    fig_dist.add_shape(
        type="line",
        x0=df_grouped["distance_bin"].min(),
        x1=df_grouped["distance_bin"].max(),
        y0=5,
        y1=5,
        line=dict(color="green", width=2, dash="dash"),
        name="Seuil Excellent"
    )

    fig_dist.add_shape(
        type="line",
        x0=df_grouped["distance_bin"].min(),
        x1=df_grouped["distance_bin"].max(),
        y0=2,
        y1=2,
        line=dict(color="orange", width=2, dash="dash"),
        name="Seuil Moyen"
    )

    fig_dist.add_shape(
        type="line",
        x0=df_grouped["distance_bin"].min(),
        x1=df_grouped["distance_bin"].max(),
        y0=0,
        y1=0,
        line=dict(color="red", width=2, dash="dash"),
        name="Seuil Faible"
    )

    # Temporal chart: raw connectmbs values over time
    fig_time = go.Figure()
    fig_time.add_trace(
        go.Scatter(
            x=df_filtered["timestamp"],
            y=df_filtered["connectmbs"],
            mode="markers",
            marker=dict(color=df_filtered["point_color"], size=5),
            name="connectmbs"
        )
    )
    for _, row in df_gares.iterrows():
        idx = (df_filtered["distance"] - row["distance"]).abs().idxmin()
        ts_gare = df_filtered.loc[idx, "timestamp"]
        fig_time.add_shape(
            type="line",
            x0=ts_gare,
            x1=ts_gare,
            y0=df_filtered["connectmbs"].min(),
            y1=df_filtered["connectmbs"].max(),
            line=dict(color="blue")
        )
        fig_time.add_annotation(
            x=ts_gare,
            y=df_filtered["connectmbs"].max(),
            text=row["Gare"],
            showarrow=False,
            yanchor="bottom",
            textangle=-90,
            font=dict(size=12, color="blue")
        )
    fig_time.update_layout(
        title="Connectivité (connectmbs) en fonction du temps",
        xaxis_title="Horodatage",
        yaxis_title="connectmbs",
        height=400,
        margin={"t": 40, "b": 40, "l": 40, "r": 40}
)



    return fig, fig_pie, fig_dist, fig_time


def update_display(ts_start, ts_end):
    """Format slider timestamps for display."""
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
    dist_output = gr.Plot(label="Connectivité en fonction de la distance")
    time_output = gr.Plot(label="Connectivité en fonction du temps")


    # Bind the update function to the button click
    show_button.click(
        fn=update_map,
        inputs=[operator_input, trip_input, metric_input, start_input, end_input],
        outputs=[map_output, pie_output, dist_output, time_output]
    )
    # Optionally, load the initial view on app launch
    demo.load(
        fn=update_map,
        inputs=[operator_input, trip_input, metric_input, start_input, end_input],
        outputs=[map_output, pie_output, dist_output, time_output],
    )



# Launch the dashboard
demo.queue()  # enable queued processing for Warning messages
if __name__ == "__main__":
    demo.launch()
