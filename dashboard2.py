"""Interactive dashboard for Zoom call metrics"""

import gradio as gr

from zoom_utils import METRICS, plot_metrics, dataset_range

start_ts, end_ts = dataset_range()

with gr.Blocks() as demo:
    gr.Markdown("## Dashboard 2 - Zoom metrics")
    with gr.Row():
        metric_input = gr.CheckboxGroup(label="Métriques", choices=METRICS, value=[METRICS[0]])
    with gr.Row():
        start_input = gr.Slider(label="Début", minimum=start_ts, maximum=end_ts, value=start_ts, step=1)
        end_input = gr.Slider(label="Fin", minimum=start_ts, maximum=end_ts, value=end_ts, step=1)
    show_button = gr.Button("Afficher")
    plot_output = gr.Plot(label="Graphique")

    show_button.click(plot_metrics, inputs=[metric_input, start_input, end_input], outputs=plot_output)
    demo.load(plot_metrics, inputs=[metric_input, start_input, end_input], outputs=plot_output)

if __name__ == "__main__":
    demo.launch()
