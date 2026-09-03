import numpy as np
import torch
import torch.nn.functional as F
from train_gnn import SAGE
from baseline import split_mask

torch.manual_seed(0)


def load():
    d = np.load("data/graph.npz", allow_pickle=True)
    x = d["x"].astype(np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x, d["y"].astype(np.int64), d["edge_index"].astype(np.int64)


def attack(edge_index, y, drop_frac=0.7, cover_k=4, seed=0):
    """Adaptive adversary: thin intra-ring edges + add cover traffic to normals."""
    r = np.random.default_rng(seed)
    ring = set(np.where(y == 1)[0].tolist())
    normal = np.where(y == 0)[0]
    kept = []
    for u, v in edge_index.T.tolist():
        if u in ring and v in ring and r.random() < drop_frac:
            continue
        kept.append((u, v))
    for u in ring:
        for _ in range(cover_k):
            kept.append((u, int(r.choice(normal))))
    return np.array(kept, dtype=np.int64).T


def to_undirected(ei_np):
    ei = torch.tensor(ei_np)
    return torch.cat([ei, ei.flip(0)], dim=1)


def train(x, ei, y, mask, epochs=150):
    model = SAGE(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yt = torch.tensor(y); m = torch.tensor(mask)
    w = torch.tensor([1.0, float((y == 0).sum()) / max(int((y == 1).sum()), 1)],
                     dtype=torch.float32)
    xt = torch.tensor(x)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(xt, ei)[m], yt[m], weight=w)
        loss.backward(); opt.step()
    return model


def ring_recall(model, x, ei, y, test):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x), ei).argmax(1).numpy()
    m = (y == 1) & test
    return float((pred[m] == 1).mean())


def main():
    x, y, ei_clean_np = load()
    train_m, test_m = split_mask(y)
    ei_clean = to_undirected(ei_clean_np)

    model = train(x, ei_clean, y, train_m)
    r_clean = ring_recall(model, x, ei_clean, y, test_m)

    ei_attacked = to_undirected(attack(ei_clean_np, y, seed=1))
    r_attacked = ring_recall(model, x, ei_attacked, y, test_m)

    model_hard = train(x, to_undirected(attack(ei_clean_np, y, seed=2)), y, train_m)
    ei_eval = to_undirected(attack(ei_clean_np, y, seed=3))
    r_hardened = ring_recall(model_hard, x, ei_eval, y, test_m)

    print("=" * 60)
    print("RED-TEAM: robustness of the ring detector (ring-recall)")
    print("=" * 60)
    print(f"  clean graph              : {r_clean:.2f}")
    print(f"  under adaptive attack    : {r_attacked:.2f}   (delta {r_attacked - r_clean:+.2f})")
    print(f"  after adversarial harden : {r_hardened:.2f}   (recovered {r_hardened - r_attacked:+.2f})")
    print("-" * 60)
    print("attack = thin 70% of intra-ring edges + 4 cover edges/ring node")
    print("defense-only: sandbox stress test, no operational fraud guidance")

    try:
        import matplotlib, os
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=140)
        fig.patch.set_facecolor("#0a1428"); ax.set_facecolor("#0a1428")
        vals = [r_clean, r_attacked, r_hardened]
        cols = ["#26e0d4", "#ff5c78", "#46f08a"]
        bars = ax.bar(["Clean", "Under attack", "Hardened"], vals, color=cols, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}",
                    ha="center", color="#e8f1ff", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.08); ax.set_ylabel("ring recall", color="#8aa2c6")
        ax.set_title("Detector robustness under adaptive attack", color="#e8f1ff",
                     fontsize=13, fontweight="bold", pad=12)
        for s in ax.spines.values(): s.set_color("#20365f")
        ax.tick_params(colors="#8aa2c6"); plt.tight_layout()
        os.makedirs("out", exist_ok=True)
        plt.savefig("out/redteam_robustness.png", facecolor="#0a1428")
        print("chart -> out/redteam_robustness.png")
    except Exception as e:
        print(f"(chart skipped: {e})")


if __name__ == "__main__":
    main()