"""Tests: matrix shape, recommenders return valid items, eval runs."""
from src.data import load_events, load_items, build_user_item_matrix
from src.recommenders import popularity_recommender, item_cf_recommender, category_recommender
from src.evaluate import run, leave_last_out


def test_matrix_is_users_by_items():
    events = load_events()
    m = build_user_item_matrix(events)
    assert m.shape[0] > 0 and m.shape[1] > 0


def test_popularity_returns_k():
    events = load_events()
    assert len(popularity_recommender(events, 5)) == 5


def test_item_cf_excludes_seen():
    events = load_events()
    m = build_user_item_matrix(events)
    user = m.index[0]
    recs = item_cf_recommender(m, user, 5)
    seen = set(events[events["user_id"] == user]["item_id"])
    assert all(r not in seen for r in recs)


def test_eval_runs_and_cf_beats_popularity():
    results = run()
    assert results["Item-CF"] >= results["Popularity"]
