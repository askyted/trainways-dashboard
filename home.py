import gradio as gr
from extract_trainways import extract_sans_date, pinData_to_df, timestamp_to_date
from data_utils import set_current_dataset

search_results = []


def fetch_trips(os_choice, depart, arrivee):
    """Retrieve trips from Firebase and present them."""
    ios = os_choice.lower() == "ios"
    try:
        df = extract_sans_date(ios, depart.strip(), arrivee.strip())
    except Exception as e:
        return gr.update(choices=[]), f"Erreur : {e}"

    if df.empty:
        return gr.update(choices=[]), "Aucun trajet trouvé."

    global search_results
    search_results = []
    options = []
    for i, row in df.iterrows():
        try:
            n_points = len(pinData_to_df(row.get("pinData", "")))
        except Exception:
            n_points = 0
        operator = row.get("operateur", "N/A")
        date_str = timestamp_to_date(row.get("timestamp", 0))
        label = f"Trajet {i+1} - {operator} - {date_str} - {n_points} points"
        options.append((label, str(i)))
        search_results.append(row)
    labels = [lbl for lbl, _ in options]
    return gr.update(choices=labels, value=None), "Choisissez un trajet dans la liste"


def select_trip(choice):
    if not choice:
        return "Aucun trajet sélectionné"
    idx = int(choice.split()[1]) - 1 if choice[0].isdigit() else int(choice)
    row = search_results[idx]
    try:
        df = pinData_to_df(row.get("pinData", ""))
    except Exception:
        return "Impossible de parser les données"
    set_current_dataset(df)
    return "Trajet chargé. Ouvrez l'onglet Dashboard 1."


with gr.Blocks() as demo:
    gr.Markdown("## Sélection du trajet depuis Firebase")
    with gr.Row():
        os_input = gr.Radio(["iOS", "Android"], label="Plateforme", value="iOS")
        depart_input = gr.Textbox(label="Ville de départ")
        arrivee_input = gr.Textbox(label="Ville d'arrivée")
    fetch_button = gr.Button("Rechercher")
    trip_list = gr.Dropdown(label="Trajets disponibles")
    info = gr.Markdown()
    choose_button = gr.Button("Utiliser ce trajet")
    message = gr.Markdown()

    fetch_button.click(fetch_trips, inputs=[os_input, depart_input, arrivee_input], outputs=[trip_list, info])
    choose_button.click(select_trip, inputs=trip_list, outputs=message)

if __name__ == "__main__":
    demo.launch()

