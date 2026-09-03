"""
baseline.py
-----------
The model everyone else will build: a per-row tabular classifier on node
features (RandomForest). It has no idea accounts are connected. It will catch
some obvious fraud but MISS the coordinated rings — which is exactly the gap
the graph model closes.

Also holds the shared evaluation harness (evaluate) used by both models so the
comparison is apples-to-apples.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Cost assumptions (put these on a slide — "honest metrics incl. FP cost")
COST_MISSED_FRAUD = 5000      # avg INR lost when a ring member slips through
COST_FALSE_ALARM = 800        # cost of wrongly freezing/reviewing a good account


def load():
    d = np.load("data/graph.npz", allow_pickle=True)
    return d["x"], d["y"], d["edge_index"], d["feature_cols"]


def split_mask(y, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(len(y) * (1 - test_frac))
    train = np.zeros(len(y), bool); train[idx[:cut]] = True
    test = ~train
    return train, test


def evaluate(name, y_true, y_pred, ring_types=None):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    cost = fn * COST_MISSED_FRAUD + fp * COST_FALSE_ALARM
    print(f"\n[{name}]  precision={p:.3f}  recall={r:.3f}  f1={f:.3f}")
    print(f"         TP={tp}  FP={fp}  FN={fn}   est. net cost = INR {cost:,}")
    # per-ring recall: where does each model win/lose?
    if ring_types is not None:
        df = pd.DataFrame({"y": y_true, "pred": y_pred, "ring": ring_types})
        for rt in ["cycle", "fanout", "collusion", "device"]:
            sub = df[(df.ring == rt)]
            if len(sub):
                rec = (sub.pred == 1).mean()
                print(f"         recall on {rt:<10} = {rec:.2f}  (n={len(sub)})")
    return {"precision": p, "recall": r, "f1": f, "cost": cost}


def run_baseline():
    x, y, _, _ = load()
    nodes = pd.read_csv("data/nodes.csv")
    train, test = split_mask(y)
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=0)
    clf.fit(x[train], y[train])
    pred = clf.predict(x[test])
    return evaluate("Tabular baseline (RandomForest)", y[test], pred,
                    ring_types=nodes["ring_type"].to_numpy()[test])


if __name__ == "__main__":
    run_baseline()
