"""Load interaction events and items. Plain pandas, readable top to bottom."""
import pandas as pd


def load_events(path: str = "data/events.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_items(path: str = "data/items.csv") -> pd.DataFrame:
    return pd.read_csv(path)


# Weight event types: a purchase signals more interest than a view.
EVENT_WEIGHTS = {"view": 1.0, "click": 2.0, "purchase": 4.0}


def build_user_item_matrix(events: pd.DataFrame):
    """Return a user x item weighted-interaction matrix (pandas DataFrame)."""
    ev = events.copy()
    ev["weight"] = ev["event_type"].map(EVENT_WEIGHTS).fillna(1.0)
    matrix = ev.pivot_table(index="user_id", columns="item_id",
                            values="weight", aggfunc="sum", fill_value=0.0)
    return matrix
