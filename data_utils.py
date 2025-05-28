import pandas as pd
from dateparser import parse


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

    df["prev_received_at"] = df["previousserverreceivedat"].apply(
        lambda x: parse(
            str(x),
            languages=["en"],
            settings={"TIMEZONE": "Europe/Paris", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        if pd.notna(x) and str(x).strip() != "" else None
    )
    df["prev_received_at"] = df["prev_received_at"].fillna(df["timestamp"])

    df["ts_unix"] = df["timestamp"].apply(lambda x: x.timestamp())
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["prev_ts_unix"] = df["prev_received_at"].apply(lambda x: x.timestamp())
    return df


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


# Stations information
stations_data = {
    "Gare": [
        "Labège-Innopole",
        "Escalquens",
        "Baziège",
        "Villenouvelle",
        "Villefranche-de-Lauragais",
        "Avignonet-Lauragais",
        "Castelnaudary",
        "Bram",
        "Carcassonne",
        "Lézignan-Corbières",
        "Narbonne",
    ],
    "Latitude": [
        43.5469,
        43.5167,
        43.4833,
        43.4667,
        43.4000,
        43.3833,
        43.3167,
        43.2500,
        43.2175,
        43.2000,
        43.1875,
    ],
    "Longitude": [
        1.5133,
        1.5667,
        1.6167,
        1.6667,
        1.7333,
        1.8000,
        1.9500,
        2.1167,
        2.3518,
        2.7500,
        3.0037,
    ],
}

df_gares = pd.DataFrame(stations_data)

# Global dataset
df_all = load_dataset("df_trainways.csv")

# Current dataset used by the dashboards. This can be replaced at runtime
# from the home page.
df_current = df_all.copy()


def current_dataset_range() -> tuple[float, float]:
    """Return the (min_ts, max_ts) range of the current dataset."""
    return (
        df_current["prev_ts_unix"].min(),
        df_current["prev_ts_unix"].max(),
    )


def set_current_dataset(df: pd.DataFrame) -> None:
    """Set the dataset to be used in the dashboards."""
    global df_current
    df_current = df


def get_current_dataset() -> pd.DataFrame:
    """Return the dataset currently in use."""
    return df_current
