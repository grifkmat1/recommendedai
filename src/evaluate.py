"""Offline evaluation with a leave-last-out protocol:
hold out each user's LAST interaction, recommend k items from the rest,
and measure hit-rate@k (did the held-out item appear in the recommendations?).
This is a standard, honest way to compare recommenders offline."""
import pandas as pd
from src.data import load_events, load_items, build_user_item_matrix
from src.recommenders import popularity_recommender, item_cf_recommender, category_recommender


def leave_last_out(events: pd.DataFrame):
    """Split: for each user, last row = test, the rest = train."""
    events = events.reset_index(drop=True)
    test_idx = events.groupby("user_id").tail(1).index
    test = events.loc[test_idx]
    train = events.drop(test_idx)
    return train, test


def hit_rate_at_k(recommend_fn, train, test, k=5):
    hits = 0
    users = test["user_id"].unique()
    for u in users:
        held_out = test[test["user_id"] == u]["item_id"].iloc[0]
        recs = recommend_fn(u, k)
        if held_out in recs:
            hits += 1
    return hits / len(users)


def run():
    events, items = load_events(), load_items()
    train, test = leave_last_out(events)
    matrix = build_user_item_matrix(train)

    pop = lambda u, k: popularity_recommender(train, k)
    cf  = lambda u, k: item_cf_recommender(matrix, u, k)
    cat = lambda u, k: category_recommender(train, items, u, k)

    print(f"Users evaluated: {test['user_id'].nunique()}   (hit-rate@5)")
    results = {}
    for name, fn in [("Popularity", pop), ("Item-CF", cf), ("Category", cat)]:
        hr = hit_rate_at_k(fn, train, test, k=5)
        results[name] = hr
        print(f"  {name:<12} {hr:.3f}")
    return results


if __name__ == "__main__":
    run()
