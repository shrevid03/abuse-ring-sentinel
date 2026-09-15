"""Canonical twin evaluation over multiple seeds — mean±std + pooled per-ring recall.
Held-out unseen rings, inductive GNN.
Run: PYTHONPATH=src python src/twin_eval_seeds.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import twin_eval_canonical as C

RING_TYPES = ["cycle", "fanout", "collusion", "device"]

COST_MISSED_FRAUD = 5000
COST_FALSE_ALARM = 800


def prf(yt, yp):
    p, r, f, _ = precision_recall_fscore_support(
        yt, yp, average="binary", zero_division=0
    )
    return p, r, f


def main():
    x, y, ei = C.load()
    rt = pd.read_csv("data/nodes.csv")["ring_type"].to_numpy()

    xs = C.structural_features(x, ei, len(y))

    N_SEEDS = 5

    names = [
        "RF (per-row)",
        "RF + structural",
        "GraphSAGE (inductive)"
    ]

    scores = {n: [] for n in names}
    costs = {n: [] for n in names}

    pooled = {
        n: {rt_: [0, 0] for rt_ in RING_TYPES}
        for n in names
    }

    for s in range(N_SEEDS):

        tr, te, _ = C.holdout_ring_masks(
            y, ei, 0.3, seed=s
        )

        preds = {
            "RF (per-row)":
                C.rf(x[tr], y[tr], x)[te],

            "RF + structural":
                C.rf(xs[tr], y[tr], xs)[te],

            "GraphSAGE (inductive)":
                C.gnn_inductive(x, y, ei, tr)[te],
        }

        yte = y[te]
        rte = rt[te]

        for n, pr in preds.items():

            # precision / recall / F1
            scores[n].append(prf(yte, pr))

            # false positives + false negatives
            fp = int(((pr == 1) & (yte == 0)).sum())
            fn = int(((pr == 0) & (yte == 1)).sum())

            cost = (
                fn * COST_MISSED_FRAUD
                + fp * COST_FALSE_ALARM
            )

            costs[n].append(cost)

            # pooled recall for each fraud-ring type
            for i in range(len(yte)):
                if yte[i] == 1:
                    pooled[n][rte[i]][1] += 1

                    if pr[i] == 1:
                        pooled[n][rte[i]][0] += 1

    print("=" * 78)
    print(
        f"CANONICAL TWIN — held-out unseen rings (inductive), "
        f"mean±std over {N_SEEDS} seeds"
    )
    print("=" * 78)

    print(
        f"{'model':<24}"
        f"{'precision':>16}"
        f"{'recall':>16}"
        f"{'F1':>16}"
    )

    for n in names:
        a = np.array(scores[n])

        print(
            f"{n:<24}"
            f"{a[:,0].mean():>8.3f}±{a[:,0].std():<6.3f}"
            f"{a[:,1].mean():>8.3f}±{a[:,1].std():<6.3f}"
            f"{a[:,2].mean():>8.3f}±{a[:,2].std():<6.3f}"
        )

    print("\nPer-ring recall (pooled across all seeds):")

    print(
        f"{'model':<24}"
        + "".join(f"{t:>12}" for t in RING_TYPES)
    )

    for n in names:
        row = ""

        for t in RING_TYPES:
            caught, total = pooled[n][t]

            if total:
                row += f"{caught / total:>12.2f}"
            else:
                row += f"{'-':>12}"

        print(f"{n:<24}{row}")

    print("\nEstimated operational cost across seeds:")
    print(
        f"(missed fraud = INR {COST_MISSED_FRAUD:,}, "
        f"false alarm = INR {COST_FALSE_ALARM:,})"
    )

    for n in names:
        c = np.array(costs[n])

        print(
            f"{n:<24}"
            f"INR {c.mean():,.0f} ± {c.std():,.0f}"
            f"   seeds={c.tolist()}"
        )


if __name__ == "__main__":
    main()
