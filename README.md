# Trainways Dashboard

This project provides an interactive Gradio dashboard to explore connectivity quality along the Toulouse–Narbonne railway line.

## Usage

To access all available dashboards, run:

```bash
python3 app.py
```

The home page lets you switch between:

- **Dashboard 1** – the original Trainways connectivity dashboard
- **Dashboard 2** – explore metrics collected during a Zoom call
- **Dashboard 3** – placeholder for upcoming GNetTrack data

The second dashboard relies on data from `df_zoom_retour.csv` and lets you
visualize several call quality metrics recorded during a Zoom session.

The first dashboard lets you choose the operator, direction, metric and a time range. Timestamps are selected with sliders but the chosen start and end times are displayed as readable dates just below the controls.

Alongside the connectivity map and pie chart, the dashboard also includes:

- A bar chart indicating the longest continuous time (10 s steps) spent in the green, orange and red zones.
- A filter to display either the distance or timestamp chart.

The selected chart shows either connectivity averaged by distance or the raw timeline with station arrival markers.

Dependencies are listed in `requirements.txt`.
