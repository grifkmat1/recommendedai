"""Regenerate the synthetic interaction dataset. Documented as synthetic."""
import csv, random
random.seed(7)
N_USERS, N_ITEMS = 200, 80
CATEGORIES = ["electronics", "books", "home", "sports", "toys"]
items = [{"item_id": i, "category": random.choice(CATEGORIES)} for i in range(N_ITEMS)]
events = []
for u in range(N_USERS):
    pref = random.choice(CATEGORIES)
    for _ in range(random.randint(5, 25)):
        pool = [it for it in items if it["category"] == pref] if random.random() < 0.75 else items
        it = random.choice(pool)
        events.append((u, it["item_id"], random.choice(["view","view","view","click","purchase"])))
with open("data/items.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["item_id","category"])
    for it in items: w.writerow([it["item_id"], it["category"]])
with open("data/events.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["user_id","item_id","event_type"]); w.writerows(events)
print(f"users={N_USERS} items={N_ITEMS} events={len(events)}")
