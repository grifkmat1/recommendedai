"""Minimal FastAPI recommendation service.
Run with: uvicorn src.api:app --reload
GET /recommend/{user_id} returns item-CF recommendations."""
from fastapi import FastAPI
from src.data import load_events, load_items, build_user_item_matrix
from src.recommenders import item_cf_recommender, popularity_recommender

app = FastAPI(title="RecommendedAI API", version="0.1.0")

_events = load_events()
_items = load_items()
_matrix = build_user_item_matrix(_events)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 5):
    recs = item_cf_recommender(_matrix, user_id, k)
    if not recs:  # cold-start fallback for unknown users
        recs = popularity_recommender(_events, k)
        return {"user_id": user_id, "items": recs, "strategy": "popularity_fallback"}
    return {"user_id": user_id, "items": recs, "strategy": "item_cf"}
