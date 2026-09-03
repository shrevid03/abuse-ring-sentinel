"""
coevolve.py — CO-EVOLVING adversarial red-team (DEFENSIVE research).
Attacker vs detector arms race over rounds, with convergence tracking.
Run:  python coevolve.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from train_gnn import SAGE
from baseline import split_mask

torch.manual_seed(0)
ROUNDS = 8


def load():
    d = np.load("data/graph.npz", allow_pickle=True)
    x = d["x"].astype(np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x, d["y"].astype(np.int64), d["edge_index"].astype(np.int64)


def undirected(ei_np):
    ei = torch.tensor(ei_np)
    return torch.cat([ei, ei.flip(0)], dim=1)


def train(x, ei, y, mask, epochs=140):
    model = SAGE(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    xt, yt, m = torch.tensor(x), torch.tensor(y), torch.tensor(mask)
    w = torch.tensor([1.0, float((y == 0).sum()) / max(int((y == 1).sum()), 1)],
                     dtype=torch.float32)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(xt, ei)[m], yt[m], weight=w)
        loss.backward(); opt.step()
    return model


def confidences(model, x, ei):
    model.eval()
    with torch.no_grad():
        return F.softmax(model(torch.tensor(x), ei), dim=1)[:, 1].numpy()


def ring_recall(model, x, ei, y, test):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x), ei).argmax(1).numpy()
    m = (y == 1) & test
    return float((pred[m] == 1).mean())


def greedy_attack(ei_np, y, conf, budget, seed=0):
    """Confidence-guided, budget-limited edge flips against the current detector.
       Removals: intra-ring edges with highest mutual support (conf[u]+conf[v]).
       Additions: ring node -> low-confidence normal (conf[u]-conf[w] highest)."""
    r = np.random.default_rng(seed)
    ring = set(np.where(y == 1)[0].tolist())
    normal = np.where(y == 0)[0]

    removals = [(u, v, conf[u] + conf[v]) for u, v in ei_np.T.tolist()
                if u in ring and v in ring]
    removals.sort(key=lambda t: t[2], reverse=True)

    additions = []
    for u in ring:
        for w in r.choice(normal, size=min(6, len(normal)), replace=False):
            additions.append((u, int(w), conf[u] - conf[int(w)]))
    additions.sort(key=lambda t: t[2], reverse=True)

    n_rem = budget // 2
    n_add = budget - n_rem
    remove_set = set((u, v) for u, v, _ in removals[:n_rem])
    add_list = [(u, w) for u, w, _ in additions[:n_add]]

    kept = [(u, v) for u, v in ei_np.T.tolist() if (u, v) not in remove_set]
    kept += add_list
    return np.array(kept, dtype=np.int64).T


def main():
    x, y, ei_clean = load()
    train_m, test_m = split_mask(y)
    n_ring = int((y == 1).sum())
    budget = 3 * n_ring

    model = train(x, undirected(ei_clean), y, train_m)
    r0 = ring_recall(model, x, undirected(ei_clean), y, test_m)

    print("=" * 66)
    print("CO-EVOLVING RED-TEAM  (attacker vs detector, over rounds)")
    print("=" * 66)
    print(f"clean detector ring-recall: {r0:.2f}   attack budget: {budget} edges\n")
    print(f"{'round':>5} {'robustness_before':>18} {'robustness_after':>18}")

    before, after = [], []
    for rd in range(1, ROUNDS + 1):
        conf = confidences(model, x, undirected(ei_clean))
        ei_atk = greedy_attack(ei_clean, y, conf, budget, seed=rd)
        rb = ring_recall(model, x, undirected(ei_atk), y, test_m)
        model = train(x, undirected(ei_atk), y, train_m)
        ra = ring_recall(model, x, undirected(ei_atk), y, test_m)
        before.append(rb); after.append(ra)
        print(f"{rd:>5} {rb:>18.2f} {ra:>18.2f}")

    print("-" * 66)
    conv = np.std(before[-3:])
    trend = before[-1] - before[0]
    print(f"attacker damage trend (round1 -> last): {before[0]:.2f} -> {before[-1]:.2f}  ({trend:+.2f})")
    print(f"last-3-round stability (std): {conv:.3f}  "
          f"({'converging' if conv < 0.06 else 'still oscillating'})")
    print("defense-only: sandbox arms race, structural edits only")

    _chart(r0, before, after)
    return before, after


def _chart(r0, before, after):
    try:
        import matplotlib, os
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rounds = list(range(1, len(before) + 1))
        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        fig.patch.set_facecolor("#0a1428"); ax.set_facecolor("#0a1428")
        ax.axhline(r0, color="#8aa2c6", ls="--", lw=1, label=f"clean recall ({r0:.2f})")
        ax.plot(rounds, before, "-o", color="#ff5c78", lw=2, label="under fresh attack (robustness)")
        ax.plot(rounds, after, "-o", color="#46f08a", lw=2, label="after hardening")
        ax.set_xlabel("co-evolution round", color="#8aa2c6")
        ax.set_ylabel("ring recall", color="#8aa2c6")
        ax.set_ylim(0, 1.08)
        ax.set_title("Co-evolving red-team: does the arms race converge?",
                     color="#e8f1ff", fontsize=12, fontweight="bold", pad=12)
        for s in ax.spines.values(): s.set_color("#20365f")
        ax.tick_params(colors="#8aa2c6")
        ax.legend(facecolor="#0f1d38", edgecolor="#20365f", labelcolor="#e8f1ff", fontsize=9)
        plt.tight_layout()
        os.makedirs("out", exist_ok=True)
        plt.savefig("out/coevolve_convergence.png", facecolor="#0a1428")
        print("chart -> out/coevolve_convergence.png")
    except Exception as e:
        print(f"(chart skipped: {e})")


if __name__ == "__main__":
    main()