"""
twin_eval_canonical.py — CANONICAL twin evaluation — held-out unseen rings; test-ring edges excluded from GNN training message passing.

Split: HELD-OUT UNSEEN RINGS (inductive). Whole rings are recovered as connected
components of the fraud-only subgraph and split train/test, so the test rings —
and, for the GNN, their edges — never participate in training. This is stricter
than a random node split and tests generalization to held-out fraud components.

All three models are evaluated on the SAME test rings:
  - RandomForest (per-row features)
  - RandomForest + hand-engineered structural features (pagerank, clustering, triangles, k-core)
  - GraphSAGE (INDUCTIVE: trained on train-only edges, inference on the full graph)

Run:  PYTHONPATH=src python src/twin_eval_canonical.py
"""
import numpy as np, pandas as pd, networkx as nx
import torch, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from train_gnn import SAGE
from baseline import evaluate

torch.manual_seed(0)


def load():
    d = np.load("data/graph.npz", allow_pickle=True)
    return d["x"].astype(np.float32), d["y"].astype(np.int64), d["edge_index"].astype(np.int64)


def holdout_ring_masks(y, ei, test_frac=0.3, seed=7):
    rng = np.random.default_rng(seed)
    N = len(y)
    ctime = rng.random(N)
    G = nx.Graph(); G.add_nodes_from(range(N))
    for a, b in ei.T:
        if y[a] == 1 and y[b] == 1:
            G.add_edge(int(a), int(b))
    fraud = [i for i in range(N) if y[i] == 1]
    n_rings = 0
    for comp in nx.connected_components(G.subgraph(fraud)):
        if len(comp) > 1:
            t = rng.random(); n_rings += 1
            for n in comp:
                ctime[n] = t
    thr = np.quantile(ctime, 1 - test_frac)
    train = ctime <= thr; test = ctime > thr
    return train, test, n_rings


def train_only_edges(ei, train_mask):
    """Keep only edges where BOTH endpoints are train nodes."""
    src, dst = ei
    keep = train_mask[src] & train_mask[dst]
    return ei[:, keep]


def norm_train(x, train_mask):
    """Fit normalization statistics on training nodes only."""
    mu = x[train_mask].mean(0)
    sd = x[train_mask].std(0)
    return (x - mu) / (sd + 1e-6)


def gnn_inductive(x, y, ei, train):
    xt = torch.tensor(
        norm_train(x, train),
        dtype=torch.float32
    )
    yt = torch.tensor(y)
    tr_ei = train_only_edges(ei, train)                 # message passing: train edges only
    tr_und = torch.cat([torch.tensor(tr_ei), torch.tensor(tr_ei).flip(0)], 1)
    full_und = torch.cat([torch.tensor(ei), torch.tensor(ei).flip(0)], 1)
    m = torch.tensor(train)
    w = torch.tensor([1.0, float((y[train] == 0).sum()) / max(int((y[train] == 1).sum()), 1)],
                     dtype=torch.float32)
    model = SAGE(xt.size(1)); opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(xt, tr_und)[m], yt[m], weight=w)  # train on train-only graph
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(xt, full_und).argmax(1).numpy()                   # infer on full graph


def structural_features(x, ei, N):
    G = nx.Graph(); G.add_nodes_from(range(N))
    for a, b in ei.T: G.add_edge(int(a), int(b))
    pr = nx.pagerank(G, alpha=0.85); clu = nx.clustering(G)
    tri = nx.triangles(G); core = nx.core_number(G)
    extra = np.array([[pr.get(i, 0), clu.get(i, 0), tri.get(i, 0), core.get(i, 0)] for i in range(N)],
                     dtype=np.float32)
    return np.hstack([x, extra])


def rf(xtr, ytr, xall):
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0)
    clf.fit(xtr, ytr); return clf.predict(xall)


def main():
    x, y, ei = load()
    rt = pd.read_csv("data/nodes.csv")["ring_type"].to_numpy()
    train, test, n_rings = holdout_ring_masks(y, ei, 0.3)
    print("=" * 74)
    print("CANONICAL TWIN EVALUATION — held-out unseen rings (inductive GNN)")
    print("=" * 74)
    print(f"rings {n_rings} | train {int(train.sum())} test {int(test.sum())} | fraud in test {int(y[test].sum())}")

    xs = structural_features(x, ei, len(y))
    evaluate("RandomForest (per-row)",
             y[test], rf(x[train], y[train], x)[test], ring_types=rt[test])
    evaluate("RandomForest + structural features",
             y[test], rf(xs[train], y[train], xs)[test], ring_types=rt[test])
    evaluate("GraphSAGE (inductive, held-out rings)",
             y[test], gnn_inductive(x, y, ei, train)[test], ring_types=rt[test])


if __name__ == "__main__":
    main()
