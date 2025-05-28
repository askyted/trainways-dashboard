from datetime import datetime, timedelta
from typing import Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point

from data_utils import (
    df_gares,
    get_color,
    clip_connectmbs,
    get_current_dataset,
)
from geometry_utils import (
    line_utm,
    points_utm,
    num_segments,
    lon_coords,
    lat_coords,
    transformer_to_utm,
    transformer_to_latlon,
)


def filter_data(start_dt: datetime, end_dt: datetime) -> "pd.DataFrame | None":
    """Return portion of the current dataset within the given time range."""
    import pandas as pd

    df = get_current_dataset()
    mask = (df["prev_received_at"] >= start_dt) & (df["prev_received_at"] <= end_dt)
    df_filtered = df[mask].copy()
    if df_filtered.empty:
        gr.Warning("⚠️ Aucun point pour cette sélection")
        return None
    return df_filtered


def prepare_timeline(df_filtered):
    import pandas as pd

    df_time = df_filtered[["prev_received_at", "connectmbs", "point_color"]].sort_values("prev_received_at")
    missing = []
    previous_ts = None
    for ts in df_time["prev_received_at"]:
        if previous_ts is not None:
            gap = (ts - previous_ts).total_seconds()
            while gap > 10:
                previous_ts += timedelta(seconds=10)
                missing.append({"prev_received_at": previous_ts, "connectmbs": 0, "point_color": "red"})
                gap -= 10
        previous_ts = ts
    if missing:
        df_time = pd.concat([df_time, pd.DataFrame(missing)], ignore_index=True)
        df_time = df_time.sort_values("prev_received_at")
    return df_time


def longest_duration_bar(df_time) -> go.Figure:
    df_time["run"] = df_time["point_color"].ne(df_time["point_color"].shift()).cumsum()
    durations = df_time.groupby(["point_color", "run"]).size() * 10 / 60
    longest = durations.groupby("point_color").max()
    bar_vals = [longest.get(col, 0) for col in ["green", "orange", "red"]]
    fig_bar = go.Figure(data=[go.Bar(x=["Vert", "Orange", "Rouge"], y=bar_vals, marker_color=["green", "orange", "red"])] )
    fig_bar.update_layout(title="Durée max en continu (minutes)", height=300, margin={"t": 40, "b": 0, "l": 0, "r": 0})
    return fig_bar


def segment_stats(df_filtered) -> Tuple[list, list, int, int, int]:
    segment_colors = ["red"] * num_segments
    for idx, group in df_filtered.groupby("segment_index"):
        segment_colors[idx] = group["point_color"].mode().iloc[0]

    min_dist = df_filtered["distance"].min()
    max_dist = df_filtered["distance"].max()

    focus_coords = []
    for i, pt in enumerate(points_utm[:-1]):
        seg_dist = line_utm.project(pt)
        if seg_dist >= min_dist - 500 and seg_dist <= max_dist + 500:
            focus_coords.append(i)

    g = o = r = 0
    for i in focus_coords:
        col = segment_colors[i]
        if col == "green":
            g += 1
        elif col == "orange":
            o += 1
        elif col == "red":
            r += 1
    return focus_coords, segment_colors, g, o, r


def build_map(focus_coords, segment_colors) -> go.Figure:
    fig = go.Figure()
    color_segments = {col: {"lat": [], "lon": []} for col in ["green", "orange", "red", "gray"]}

    for i in focus_coords:
        col = segment_colors[i]
        color_segments[col]["lat"].extend([lat_coords[i], lat_coords[i + 1], None])
        color_segments[col]["lon"].extend([lon_coords[i], lon_coords[i + 1], None])

    for col, coords in color_segments.items():
        if coords["lat"]:
            fig.add_trace(
                go.Scattermapbox(
                    lat=coords["lat"],
                    lon=coords["lon"],
                    mode="lines",
                    line=dict(color=col, width=5),
                    hoverinfo="none",
                    name=col,
                )
            )
    if focus_coords:
        segment_lats = [lat_coords[i] for i in focus_coords] + [lat_coords[i + 1] for i in focus_coords]
        segment_lons = [lon_coords[i] for i in focus_coords] + [lon_coords[i + 1] for i in focus_coords]
        center_lat = float(np.mean(segment_lats))
        center_lon = float(np.mean(segment_lons))
    else:
        from geometry_utils import center_lat, center_lon
        center_lat = float(center_lat)
        center_lon = float(center_lon)

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=8.5,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
    )
    return fig


def pie_chart(g: int, o: int, r: int) -> go.Figure:
    nombre_seg = g + o + r
    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=["Bonne", "Moyenne", "Mauvaise"],
                values=[g, o, r],
                marker=dict(colors=["green", "orange", "red"]),
                hole=0.3,
                textinfo="label+percent",
            )
        ]
    )
    fig_pie.update_layout(
        title=f"Qualité de connectivité sur {nombre_seg} segments",
        height=300,
        margin={"t": 40, "b": 0, "l": 0, "r": 0},
    )
    return fig_pie


def distance_chart(df_filtered) -> go.Figure:
    df_filtered["connectmbs"] = df_filtered["connectmbs"].apply(clip_connectmbs)
    df_filtered["distance_bin"] = (df_filtered["distance"] // 100) * 100
    df_grouped = (
        df_filtered.groupby("distance_bin").agg(mean_connectmbs=("connectmbs", "mean"), count=("connectmbs", "count")).reset_index()
    )

    utm_gare_x, utm_gare_y = transformer_to_utm.transform(df_gares["Longitude"].values, df_gares["Latitude"].values)
    df_gares["distance"] = [line_utm.project(Point(x, y)) for x, y in zip(utm_gare_x, utm_gare_y)]

    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Scatter(
            x=df_grouped["distance_bin"],
            y=df_grouped["mean_connectmbs"],
            mode="lines+markers",
            line=dict(color="black", width=2),
            marker=dict(size=6),
            name="connectmbs (moyenne)",
        )
    )
    for _, row in df_gares.iterrows():
        fig_dist.add_shape(
            type="line",
            x0=row["distance"],
            x1=row["distance"],
            y0=df_grouped["mean_connectmbs"].min(),
            y1=df_grouped["mean_connectmbs"].max(),
            line=dict(color="blue"),
        )
        fig_dist.add_annotation(
            x=row["distance"],
            y=df_grouped["mean_connectmbs"].max(),
            text=row["Gare"],
            showarrow=False,
            yanchor="bottom",
            textangle=-90,
            font=dict(size=12, color="blue"),
        )
    tick_vals = np.linspace(df_grouped["distance_bin"].min(), df_grouped["distance_bin"].max(), num=10)
    tick_text = [f"{int(val/1000)}" for val in tick_vals]
    fig_dist.update_layout(
        xaxis=dict(title="Distance le long du trajet (km)", tickvals=tick_vals, ticktext=tick_text),
        title="Connectivité (connectmbs) en fonction de la distance",
        xaxis_title="Distance le long du trajet (km)",
        yaxis_title="connectmbs (moyenne)",
        height=400,
        margin={"t": 40, "b": 40, "l": 40, "r": 40},
    )
    fig_dist.add_shape(
        type="line",
        x0=df_grouped["distance_bin"].min(),
        x1=df_grouped["distance_bin"].max(),
        y0=5,
        y1=5,
        line=dict(color="green", width=2, dash="dash"),
        name="Seuil Excellent",
    )
    fig_dist.add_shape(
        type="line",
        x0=df_grouped["distance_bin"].min(),
        x1=df_grouped["distance_bin"].max(),
        y0=2,
        y1=2,
        line=dict(color="orange", width=2, dash="dash"),
        name="Seuil Moyen",
    )
    fig_dist.add_shape(
        type="line",
        x0=df_grouped["distance_bin"].min(),
        x1=df_grouped["distance_bin"].max(),
        y0=0,
        y1=0,
        line=dict(color="red", width=2, dash="dash"),
        name="Seuil Faible",
    )
    return fig_dist


def time_chart(df_filtered, df_time) -> go.Figure:
    fig_time = go.Figure()
    fig_time.add_trace(
        go.Scatter(
            x=df_time["prev_received_at"],
            y=df_time["connectmbs"],
            mode="markers",
            marker=dict(color=df_time["point_color"], size=5),
            name="connectmbs",
        )
    )
    for _, row in df_gares.iterrows():
        idx = (df_filtered["distance"] - row["distance"]).abs().idxmin()
        ts_gare = df_filtered.loc[idx, "prev_received_at"]
        fig_time.add_shape(
            type="line",
            x0=ts_gare,
            x1=ts_gare,
            y0=df_filtered["connectmbs"].min(),
            y1=df_filtered["connectmbs"].max(),
            line=dict(color="blue"),
        )
        fig_time.add_annotation(
            x=ts_gare,
            y=df_filtered["connectmbs"].max(),
            text=row["Gare"],
            showarrow=False,
            yanchor="bottom",
            textangle=-90,
            font=dict(size=12, color="blue"),
        )
    fig_time.update_layout(
        title="Connectivité (connectmbs) en fonction du temps",
        xaxis_title="Horodatage",
        yaxis_title="connectmbs",
        height=400,
        margin={"t": 40, "b": 40, "l": 40, "r": 40},
    )
    return fig_time


def update_map(metric, start_ts, end_ts, chart_choice):
    """Update all charts for Dashboard 1 using the current dataset."""
    start_dt = datetime.fromtimestamp(start_ts) - timedelta(hours=2)
    end_dt = datetime.fromtimestamp(end_ts) - timedelta(hours=2)

    metric = metric.strip().lower()

    df_filtered = filter_data(start_dt, end_dt)
    if df_filtered is None:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()

    utm_x, utm_y = transformer_to_utm.transform(df_filtered["longitude"].values, df_filtered["latitude"].values)
    df_filtered["distance"] = [line_utm.project(Point(x, y)) for x, y in zip(utm_x, utm_y)]
    df_filtered["point_color"] = df_filtered[metric].apply(lambda v: get_color(v, metric))
    df_filtered["segment_index"] = (df_filtered["distance"] // 500).astype("Int64")

    df_time = prepare_timeline(df_filtered)
    fig_bar = longest_duration_bar(df_time)
    focus_coords, segment_colors, g, o, r = segment_stats(df_filtered)
    fig = build_map(focus_coords, segment_colors)
    fig_pie = pie_chart(g, o, r)
    fig_dist = distance_chart(df_filtered)
    fig_time = time_chart(df_filtered, df_time)
    chart = fig_dist if chart_choice == "distance" else fig_time
    return fig, fig_pie, fig_bar, chart


def format_ts(ts_start, ts_end):
    readable_start = datetime.fromtimestamp(ts_start).strftime("%d/%m %H:%M:%S")
    readable_end = datetime.fromtimestamp(ts_end).strftime("%d/%m %H:%M:%S")
    return readable_start, readable_end
