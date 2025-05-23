import os
import pandas as pd
import plotly.express as px
import gradio as gr


DATA_PATH = "df_zoom_retour.csv"


def load_zoom_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load Zoom call statistics or return an example if missing."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        gr.Warning(f"⚠️ Fichier {path} introuvable - utilisation d'un exemple")
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-05-15 10:00", periods=10, freq="T"),
            "jitter": [2, 3, 2, 1, 3, 5, 4, 3, 2, 1],
            "latency": [30, 32, 28, 31, 33, 35, 34, 32, 29, 30],
            "packet_loss": [0, 1, 0, 0, 2, 3, 1, 0, 0, 0],
        })
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


with gr.Blocks() as demo:
    gr.Markdown("## Dashboard 2 - Zoom")
    df_zoom = load_zoom_data()

    with gr.Accordion("Données brutes", open=False):
        gr.Dataframe(df_zoom)

    jitter_plot = gr.Plot(label="Jitter")
    latency_plot = gr.Plot(label="Latence")
    loss_plot = gr.Plot(label="Perte de paquets")

    def make_plots():
        fig_jitter = px.line(df_zoom, x="timestamp", y="jitter", title="Jitter")
        fig_latency = px.line(df_zoom, x="timestamp", y="latency", title="Latence")
        fig_loss = px.line(df_zoom, x="timestamp", y="packet_loss", title="Perte de paquets")
        return fig_jitter, fig_latency, fig_loss

    demo.load(fn=make_plots, outputs=[jitter_plot, latency_plot, loss_plot])


demo.queue()
if __name__ == "__main__":
    demo.launch()
