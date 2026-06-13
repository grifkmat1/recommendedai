# RecommendedAI

A focused recommendation engine: build a weighted user–item interaction matrix and compare three recommendation approaches using a rigorous offline evaluation protocol.

Built to be readable and explainable end-to-end.

---

## What it does

- Loads interaction events (`view`, `click`, `purchase`) with weighted signals — purchases count more than views
- Builds a **weighted user × item matrix**
- Implements three recommendation approaches:
  1. **Popularity** — globally most-interacted items (the baseline)
  2. **Item-CF** — item–item collaborative filtering via cosine similarity
  3. **Category** — content-based, recommending popular items from the user's most-engaged category
- Evaluates all three with **leave-last-out hit-rate@k** — the standard honest offline protocol
- Serves recommendations via a **FastAPI** endpoint with cold-start fallback

---

## Project structure

```
recommendedai/
├── data/
│   ├── events.csv             # synthetic interactions: user_id, item_id, event_type
│   └── items.csv              # item catalog: item_id, category
├── src/
│   ├── data.py                # load events/items, build weighted user × item matrix
│   ├── recommenders.py        # popularity, item-CF, category implementations
│   ├── evaluate.py            # leave-last-out hit-rate@k evaluation
│   └── api.py                 # FastAPI: GET /recommend/{user_id}, GET /health
├── tests/
│   └── test_recommenders.py   # pytest
├── scripts_gen_data.py        # regenerate synthetic dataset
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run evaluation (prints hit-rate@5 for all three approaches)
PYTHONPATH=. python src/evaluate.py

# Run tests
PYTHONPATH=. pytest

# Start the API (docs at localhost:8000/docs)
uvicorn src.api:app --reload
```

### Docker

```bash
docker-compose up
```

### Example API call

```bash
curl http://localhost:8000/recommend/42
# {"user_id": 42, "items": [17, 5, 63, 28, 41], "strategy": "item_cf"}
```

---

## Note on the dataset

`data/events.csv` and `data/items.csv` are **synthetic** (200 users, 80 items, ~2,860 events) generated with seeded category preferences so the recommenders have real signal to discover. Not real user data. Numbers are illustrative; swap in real interaction logs and re-run `src/evaluate.py` for an honest result.

---

## Key design decisions

| Decision | Why |
|---|---|
| **Leave-last-out evaluation** | Hold out each user's final interaction, then check if the model would have recommended it — a standard, non-leaky offline protocol |
| **Event weighting (4/2/1)** | Purchase > click > view; a purchase signals much stronger intent |
| **Cold-start fallback** | Unknown users get popularity recommendations — a real, named tradeoff rather than a silent failure |
| **Item-CF over User-CF** | Item–item similarities are more stable over time; new interactions don't require rebuilding the full similarity matrix |

---

## Future work

- Real interaction data + re-evaluation
- Precision/recall@k and NDCG metrics
- Matrix factorization (implicit ALS)
- Next.js frontend + PostgreSQL event store + AWS deployment
