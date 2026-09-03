"""
train_gnn_real.py  (offline version + threshold tuning)
GraphSAGE on the REAL Elliptic Bitcoin graph, read from local CSVs.
Place the 3 CSVs in data_real/elliptic/ then run:
    PYTHONPATH=src python src/train_gnn_real.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from train_gnn import SAGE
from baseline import evaluate

torch.manual_seed(0)
ROOT = "data_real/elliptic"


def _prf(y_true, pred):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    return p, r, f


def tune_threshold(y_true, prob, metric="f1"):
    """Pick the cutoff that maximizes F1 — chosen on TRAIN predictions only, then
    applied to held-out test. Honest tuning, not test-set peeking."""
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        _, _, f = _prf(y_true, (prob >= t).astype(int))
        if f > best:
            best, best_t = f, float(t)
    return best_t


def _need(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"\nMissing file: {path}\n"
            "Download from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n"
            "and place the three CSVs in data_real/elliptic/\n")


def load_elliptic(root=ROOT):
    f_feat = os.path.join(root, "elliptic_txs_features.csv")
    f_cls  = os.path.join(root, "elliptic_txs_classes.csv")
    f_edge = os.path.join(root, "elliptic_txs_edgelist.csv")
    for p in (f_feat, f_cls, f_edge):
        _need(p)

    feats = pd.read_csv(f_feat, header=None)
    tx_id     = feats.iloc[:, 0].to_numpy()
    time_step = feats.iloc[:, 1].to_numpy().astype(int)
    X         = feats.iloc[:, 2:].to_numpy(dtype=np.float32)

    cls = pd.read_csv(f_cls)
    cls.columns = ["txId", "class"]
    cmap = {"1": 1, "2": 0, "unknown": -1, 1: 1, 2: 0}
    lab = dict(zip(cls["txId"], cls["class"].map(lambda c: cmap.get(c, -1))))
    y = np.array([lab.get(t, -1) for t in tx_id], dtype=np.int64)

    id_map = {t: i for i, t in enumerate(tx_id)}

    edges = pd.read_csv(f_edge)
    edges.columns = ["src", "dst"]
    src = edges["src"].map(id_map); dst = edges["dst"].map(id_map)
    keep = src.notna() & dst.notna()
    ei = np.array([src[keep].to_numpy(), dst[keep].to_numpy()], dtype=np.int64)

    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    labelled = y != -1
    train = labelled & (time_step <= 34)
    test  = labelled & (time_step >= 35)
    if train.sum() == 0 or test.sum() == 0:
        rng = np.random.default_rng(0)
        idx = np.where(labelled)[0]; rng.shuffle(idx)
        cut = int(len(idx) * 0.7)
        train = np.zeros(len(y), bool); train[idx[:cut]] = True
        test  = np.zeros(len(y), bool); test[idx[cut:]] = True

    return X, y, ei, train, test


def main():
    X, y, ei_np, train, test = load_elliptic()
    x = torch.tensor(X)
    y_t = torch.tensor(np.where(y < 0, 0, y))
    ei = torch.tensor(ei_np); ei = torch.cat([ei, ei.flip(0)], dim=1)
    tr = torch.tensor(train)

    print(f"nodes={len(y)}  labelled={int((y!=-1).sum())}  "
          f"illicit={int((y==1).sum())}  edges={ei_np.shape[1]}")
    print(f"train nodes={int(train.sum())}  test nodes={int(test.sum())}")

    model = SAGE(x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    n_pos = int((y_t[tr] == 1).sum()); n_neg = int((y_t[tr] == 0).sum())
    w = torch.tensor([1.0, max(n_neg / max(n_pos, 1), 1.0)], dtype=torch.float32)

    for epoch in range(1, 201):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(x, ei)[tr], y_t[tr], weight=w)
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(x, ei), dim=1)[:, 1].numpy()

    y_np = y_t.numpy()
    tr_np, te_np = train, test

    evaluate("GraphSAGE REAL Elliptic — default threshold 0.50",
             y_np[te_np], (prob[te_np] >= 0.5).astype(int))

    best_t = tune_threshold(y_np[tr_np], prob[tr_np], metric="f1")
    evaluate(f"GraphSAGE REAL Elliptic — tuned threshold {best_t:.2f} (chosen on train)",
             y_np[te_np], (prob[te_np] >= best_t).astype(int))

    print("\nThreshold sweep on held-out test:")
    print(f"{'thr':>5} {'prec':>6} {'rec':>6} {'f1':>6}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p, r, f = _prf(y_np[te_np], (prob[te_np] >= t).astype(int))
        print(f"{t:>5.2f} {p:>6.3f} {r:>6.3f} {f:>6.3f}")


if __name__ == "__main__":
    main()