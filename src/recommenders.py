"""Three honest recommendation approaches, from simple to less simple.
All are small and explainable — this is the core of the project."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def popularity_recommender(events: pd.DataFrame, k: int = 5):
    """Approach 1: recommend the globally most-interacted items. The baseline."""
    top = events["item_id"].value_counts().head(k).index.tolist()
    return top


def item_cf_recommender(matrix: pd.DataFrame, user_id: int, k: int = 5):
    """Approach 2: item-item collaborative filtering via cosine similarity.
    Score each item by similarity to what the user already interacted with."""
    if user_id not in matrix.index:
        return []
    item_sim = cosine_similarity(matrix.T)  # items x items
    item_ids = matrix.columns.to_numpy()
    user_vec = matrix.loc[user_id].to_numpy()
    scores = item_sim @ user_vec               # weighted by user's history
    seen = user_vec > 0
    scores[seen] = -np.inf                      # don't recommend already-seen items
    top_idx = np.argsort(scores)[::-1][:k]
    return [int(item_ids[i]) for i in top_idx if np.isfinite(scores[i])]


def category_recommender(events: pd.DataFrame, items: pd.DataFrame, user_id: int, k: int = 5):
    """Approach 3: content-based — recommend popular items from the user's
    most-interacted category."""
    merged = events.merge(items, on="item_id")
    user_ev = merged[merged["user_id"] == user_id]
    if user_ev.empty:
        return popularity_recommender(events, k)
    fav_cat = user_ev["category"].value_counts().idxmax()
    seen = set(user_ev["item_id"])
    pool = merged[(merged["category"] == fav_cat) & (~merged["item_id"].isin(seen))]
    return pool["item_id"].value_counts().head(k).index.tolist()
