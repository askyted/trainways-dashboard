import pandas as pd
import plotly.graph_objects as go

# Load and preprocess Zoom dataset
_df_zoom = pd.read_csv("df_zoom_retour.csv")
_df_zoom.columns = _df_zoom.columns.str.strip()
_df_zoom["timestamp"] = pd.to_datetime(_df_zoom["timeStamp"])
_df_zoom["ts_unix"] = _df_zoom["timestamp"].view("int64") // 10**9

METRICS = [
    "wifi_rssi",
    "audio.recvInfo.jitter",
    "audio.recvInfo.latency",
    "video.recvInfo.jitter",
    "video.recvInfo.latency",
]


def plot_metrics(metrics, start_ts, end_ts):
    df = _df_zoom[(_df_zoom["ts_unix"] >= start_ts) & (_df_zoom["ts_unix"] <= end_ts)]
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    for metric in metrics:
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df[metric], mode="lines", name=metric)
            )
    fig.update_layout(title="Zoom Metrics", xaxis_title="Time", yaxis_title="Value")
    return fig


def dataset_range():
    return float(_df_zoom["ts_unix"].min()), float(_df_zoom["ts_unix"].max())
