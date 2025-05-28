"""Interactive dashboard to explore mobile connectivity along the train line."""

import gradio as gr

from data_utils import get_current_dataset
from charts import update_map


with gr.Blocks(
    css="""
    .small-plot {
        max-width: 350px;
        margin-left: auto;
        margin-right: auto;
    }
    """,
) as demo:
    gr.Markdown("## Carte de connectivité sur le trajet Toulouse-Narbonne")
    with gr.Row():
        metric_input = gr.Dropdown(
            label="Métrique",
            choices=["connectMbs", "rsrp", "rsrq"],
            value="connectMbs",
        )
        graph_choice = gr.Radio(
            label="Graphique",
            choices=["distance", "timestamp"],
            value="distance",
        )
        start_input = gr.Slider(
            label="Début",
            minimum=get_current_dataset()["prev_ts_unix"].min(),
            maximum=get_current_dataset()["prev_ts_unix"].max(),
            value=get_current_dataset()["prev_ts_unix"].min(),
            step=1,
            elem_classes="smaller",
        )
        end_input = gr.Slider(
            label="Fin",
            minimum=get_current_dataset()["prev_ts_unix"].min(),
            maximum=get_current_dataset()["prev_ts_unix"].max(),
            value=get_current_dataset()["prev_ts_unix"].max(),
            step=1,
            elem_classes="smaller",
        )

    show_button = gr.Button("Afficher la carte")
    map_output = gr.Plot(label="Carte")
    with gr.Row():
        pie_output = gr.Plot(
            label="Qualité de la portion",
            elem_classes="small-plot",
            scale=1,
            min_width=200,
        )
        bar_output = gr.Plot(
            label="Durée continue par couleur",
            elem_classes="small-plot",
            scale=1,
            min_width=200,
        )
    chart_output = gr.Plot(label="Graphique choisi")

    show_button.click(
        fn=update_map,
        inputs=[metric_input, start_input, end_input, graph_choice],
        outputs=[map_output, pie_output, bar_output, chart_output],
    )
    demo.load(
        fn=update_map,
        inputs=[metric_input, start_input, end_input, graph_choice],
        outputs=[map_output, pie_output, bar_output, chart_output],
    )


demo.queue()
if __name__ == "__main__":
    demo.launch()
