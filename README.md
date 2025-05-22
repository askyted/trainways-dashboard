# Trainways Dashboard

This project provides an interactive Gradio dashboard to explore connectivity quality along the Toulouse–Narbonne railway line.

## Usage

Run the dashboard with:

```bash
python3 dashboard1.py
```

The interface lets you choose the operator, direction, metric and a time range. Timestamps are selected with sliders but the chosen start and end times are displayed as readable dates just below the controls.

Alongside the connectivity map and pie chart, the dashboard now also includes:

- A bar chart showing the total time spent in green, orange and red zones.
- A text indicator of the longest continuous duration in the green zone.
- A timeline chart showing raw connectMbs values with station arrival markers.

Dependencies are listed in `requirements.txt`.
