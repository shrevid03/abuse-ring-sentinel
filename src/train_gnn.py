"""
train_gnn.py
------------
GraphSAGE node classifier — the productionized detector. Instead of us
hand-crafting graph features, message passing LEARNS relational structure,
including the dense bipartite collusion that hand-crafted metrics miss.

Also implements the two things Razorpay's bar cares about beyond accuracy:
  1. A TRIAGE GATE (auto-clear / human-review / auto-flag) so no account is
     silently frozen — the "one failure handled gracefully" requirement.
  2. An AUDIT TRAIL: every flagged node logged with score, band, and the
     suspicious neighbours that drove the decision (explainability).

Run:  PYTHONPATH=src python3 src/train_gnn.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from baseline import split_mask, evaluate

torch.manual_seed(0)

# review band: scores between LOW and HIGH go to a human, not auto-actioned
BAND_LOW, BAND_HIGH = 0.35, 0.75


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hid)
        self.c2 = SAGEConv(hid, hid)
        self.lin = torch.nn.Linear(hid, 2)

    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, 0.3, self.training)
        x = F.relu(self.c2(x, ei))
        return self.lin(x)


def load_data():
    d = np.load("data/graph.npz", allow_pickle=True)
    x = torch.tensor(d["x"])
    # normalize features (helps convergence)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = torch.tensor(d["y"])
    ei = torch.tensor(d["edge_index"])
    ei = torch.cat([ei, ei.flip(0)], dim=1)          # make undirected for msg passing
    return Data(x=x, y=y, edge_index=ei)


def triage_and_audit(scores, data, nodes, edges):
    """Turn probabilities into bounded actions + an explainable audit log."""
    band = np.where(scores >= BAND_HIGH, "auto_flag",
             np.where(scores >= BAND_LOW, "human_review", "auto_clear"))
    # explanation: for flagged nodes, list the highest-risk neighbours
    score_map = dict(zip(range(len(scores)), scores))
    src, dst = edges["src"].to_numpy(), edges["dst"].to_numpy()
    rows = []
    flagged = np.where(band != "auto_clear")[0]
    for n in flagged:
        nbrs = list(dst[src == n]) + list(src[dst == n])
        risky = sorted(nbrs, key=lambda m: score_map.get(m, 0), reverse=True)[:3]
        rows.append({
            "node_id": int(nodes.iloc[n]["node_id"]),
            "risk_score": round(float(scores[n]), 3),
            "action": band[n],
            "true_label": int(nodes.iloc[n]["label"]),
            "ring_type": nodes.iloc[n]["ring_type"],
            "top_risky_neighbours": risky,
        })
    audit = pd.DataFrame(rows).sort_values("risk_score", ascending=False)
    audit.to_csv("out/audit_trail.csv", index=False)
    return band, audit


def main():
    data = load_data()
    nodes = pd.read_csv("data/nodes.csv")
    edges = pd.read_csv("data/edges.csv")
    train, test = split_mask(data.y.numpy())
    tr = torch.tensor(train); te = torch.tensor(test)

    model = SAGE(data.x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    w = torch.tensor([1.0, (train & (data.y.numpy() == 0)).sum() /
                            max((train & (data.y.numpy() == 1)).sum(), 1)], dtype=torch.float32)

    for epoch in range(1, 201):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[tr], data.y[tr], weight=w)
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1].numpy()
    pred = (prob >= 0.5).astype(int)

    res = evaluate("GraphSAGE (learned message passing)", data.y.numpy()[test],
                   pred[test], ring_types=nodes["ring_type"].to_numpy()[test])

    band, audit = triage_and_audit(prob, data, nodes, edges)
    print("\nTriage gate (bounded actions, nothing auto-frozen blindly):")
    print(pd.Series(band).value_counts().to_string())
    print(f"\nAudit trail written to out/audit_trail.csv "
          f"({len(audit)} accounts queued for action/review)")
    print("Sample audit rows:")
    print(audit.head(6).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
