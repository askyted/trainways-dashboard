import gradio as gr

from dashboard1 import demo as dashboard1_app
try:
    from dashboard2 import demo as dashboard2_app
except Exception:
    dashboard2_app = gr.Blocks()
    with dashboard2_app:
        gr.Markdown("Dashboard 2 (Zoom) placeholder")
try:
    from dashboard3 import demo as dashboard3_app
except Exception:
    dashboard3_app = gr.Blocks()
    with dashboard3_app:
        gr.Markdown("Dashboard 3 (GNetTrack) placeholder")

app = gr.TabbedInterface(
    [dashboard1_app, dashboard2_app, dashboard3_app],
    [
        "Dashboard 1 - Trainways",
        "Dashboard 2 - Zoom",
        "Dashboard 3 - GNetTrack",
    ],
)

if __name__ == "__main__":
    app.launch()
