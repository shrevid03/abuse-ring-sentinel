"""
generate_graph.py
------------------
Builds a synthetic *money-movement* graph for the Razorpay Buildathon
"AI Risk Manager" track (abuse-ring sentinel).

Core idea being tested:
    A fraud RING is individually invisible, collectively obvious.
    Each ring member looks like a normal account on its own row of features,
    but the *structure* they form (cycles, fan-outs, dense collusion,
    shared devices) gives them away. A per-row tabular model misses this;
    a graph model catches it.

Output (saved to data/):
    - nodes.csv        node_id, node-level features, label, ring_type
    - edges.csv        src, dst, amount, timestamp, shared_device
    - graph.npz        arrays ready for PyTorch Geometric (x, y, edge_index)

Everything is synthetic. No real data, no PII. Defense-only by construction.
"""

import numpy as np
import pandas as pd
import networkx as nx

RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------
# 1. Normal population: ordinary accounts with plausible transactions
# --------------------------------------------------------------------------
def make_normal_population(n_accounts=1000, n_devices=700):
    """Accounts transact with a handful of counterparties. Nothing coordinated."""
    G = nx.DiGraph()
    for a in range(n_accounts):
        G.add_node(a,
                   label=0,
                   ring_type="none",
                   device=int(RNG.integers(0, n_devices)),   # mostly unique-ish
                   account_age_days=int(RNG.integers(30, 1500)))

    # each account sends money to a few random others (light, organic)
    ts = 0
    for a in range(n_accounts):
        k = RNG.integers(1, 6)
        partners = RNG.choice(n_accounts, size=k, replace=False)
        for p in partners:
            if p == a:
                continue
            amount = float(np.round(RNG.lognormal(mean=7.0, sigma=1.0), 2))  # ~ few thousand INR
            ts += int(RNG.integers(1, 400))
            G.add_edge(a, int(p), amount=amount, timestamp=ts,
                       shared_device=0)
    return G


# --------------------------------------------------------------------------
# 2. Ring injectors — four structurally distinct abuse patterns
#    (members are deliberately calibrated to look "normal" per-row)
# --------------------------------------------------------------------------
def _next_id(G):
    return max(G.nodes) + 1


def inject_cycle_ring(G, size=8):
    """Layering: money flows in a closed loop A->B->C->...->A."""
    start = _next_id(G)
    ids = list(range(start, start + size))
    dev = int(RNG.integers(9000, 9100))            # ring shares a device pool
    base_ts = int(RNG.integers(1, 500))
    for i, nid in enumerate(ids):
        G.add_node(nid, label=1, ring_type="cycle",
                   device=dev if RNG.random() < 0.6 else int(RNG.integers(0, 700)),
                   account_age_days=int(RNG.integers(30, 1500)))
    for i in range(size):
        a, b = ids[i], ids[(i + 1) % size]         # closes the loop
        amount = float(np.round(RNG.lognormal(7.0, 1.0), 2))  # near-normal amounts
        G.add_edge(a, b, amount=amount, timestamp=base_ts + i,
                   shared_device=1)
    return ids


def inject_fanout_ring(G, size=12):
    """Bust-out: one hub fans money out to many fresh mules in a tight window."""
    start = _next_id(G)
    hub = start
    mules = list(range(start + 1, start + size))
    dev = int(RNG.integers(9100, 9200))
    G.add_node(hub, label=1, ring_type="fanout",
               device=dev, account_age_days=int(RNG.integers(30, 1500)))
    base_ts = int(RNG.integers(1, 500))
    for m in mules:
        G.add_node(m, label=1, ring_type="fanout",
                   device=dev if RNG.random() < 0.5 else int(RNG.integers(0, 700)),
                   account_age_days=int(RNG.integers(30, 1500)))
        amount = float(np.round(RNG.lognormal(7.0, 1.0), 2))
        G.add_edge(hub, m, amount=amount, timestamp=base_ts + int(RNG.integers(0, 5)),
                   shared_device=1)
    return [hub] + mules


def inject_collusion_ring(G, size_a=6, size_b=6):
    """Dense bipartite burst: a cluster of buyers all hit a cluster of sellers,
    synchronized in time (fake demand / promo abuse)."""
    start = _next_id(G)
    buyers = list(range(start, start + size_a))
    sellers = list(range(start + size_a, start + size_a + size_b))
    dev = int(RNG.integers(9200, 9300))
    base_ts = int(RNG.integers(1, 500))
    for n in buyers + sellers:
        G.add_node(n, label=1, ring_type="collusion",
                   device=dev if RNG.random() < 0.4 else int(RNG.integers(0, 700)),
                   account_age_days=int(RNG.integers(30, 1500)))
    for b in buyers:
        for s in sellers:
            if RNG.random() < 0.8:                 # dense, not complete
                amount = float(np.round(RNG.lognormal(7.0, 1.0), 2))
                G.add_edge(b, s, amount=amount,
                           timestamp=base_ts + int(RNG.integers(0, 8)),
                           shared_device=1)
    return buyers + sellers


def inject_device_cluster(G, size=10):
    """Many otherwise-unrelated accounts share one device fingerprint."""
    start = _next_id(G)
    ids = list(range(start, start + size))
    dev = int(RNG.integers(9300, 9400))
    for nid in ids:
        G.add_node(nid, label=1, ring_type="device",
                   device=dev, account_age_days=int(RNG.integers(30, 1500)))
    # they also send small amounts to a shared collector
    collector = ids[0]
    for nid in ids[1:]:
        amount = float(np.round(RNG.lognormal(7.0, 1.0), 2))
        G.add_edge(nid, collector, amount=amount,
                   timestamp=int(RNG.integers(1, 500)), shared_device=1)
    return ids


# --------------------------------------------------------------------------
# 3. Node-level features (deliberately "flat" — rings hide here)
# --------------------------------------------------------------------------
def compute_node_features(G):
    """Per-account features a *tabular* model would use. On purpose, these do
    NOT encode ring structure well, so the baseline struggles on coordinated
    fraud — which is the whole point of the demo."""
    rows = []
    for n in G.nodes:
        out_edges = list(G.out_edges(n, data=True))
        in_edges = list(G.in_edges(n, data=True))
        out_amts = [d["amount"] for *_, d in out_edges] or [0.0]
        in_amts = [d["amount"] for *_, d in in_edges] or [0.0]
        rows.append({
            "node_id": n,
            "out_degree": len(out_edges),
            "in_degree": len(in_edges),
            "total_out": float(np.sum(out_amts)),
            "total_in": float(np.sum(in_amts)),
            "mean_amount": float(np.mean(out_amts + in_amts)),
            "std_amount": float(np.std(out_amts + in_amts)),
            "account_age_days": G.nodes[n]["account_age_days"],
            "n_counterparties": len(set([v for _, v, _ in out_edges] +
                                        [u for u, _, _ in in_edges])),
            "device": G.nodes[n]["device"],
            "label": G.nodes[n]["label"],
            "ring_type": G.nodes[n]["ring_type"],
        })
    return pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Assemble + export
# --------------------------------------------------------------------------
def build(n_normal=3000, n_cycles=6, n_fanouts=5, n_collusions=5, n_devices=5):
    G = make_normal_population(n_accounts=n_normal)
    for _ in range(n_cycles):     inject_cycle_ring(G, size=int(RNG.integers(6, 12)))
    for _ in range(n_fanouts):    inject_fanout_ring(G, size=int(RNG.integers(8, 16)))
    for _ in range(n_collusions): inject_collusion_ring(G)
    for _ in range(n_devices):    inject_device_cluster(G, size=int(RNG.integers(8, 14)))

    feats = compute_node_features(G)

    # edges table
    edge_rows = [{"src": u, "dst": v, "amount": d["amount"],
                  "timestamp": d["timestamp"], "shared_device": d["shared_device"]}
                 for u, v, d in G.edges(data=True)]
    edges = pd.DataFrame(edge_rows)

    # arrays for PyTorch Geometric
    feature_cols = ["out_degree", "in_degree", "total_out", "total_in",
                    "mean_amount", "std_amount", "account_age_days",
                    "n_counterparties"]
    # remap node ids to 0..N-1 contiguous
    id_map = {nid: i for i, nid in enumerate(feats["node_id"])}
    x = feats[feature_cols].to_numpy(dtype=np.float32)
    y = feats["label"].to_numpy(dtype=np.int64)
    ei = np.array([[id_map[u], id_map[v]] for u, v in G.edges()], dtype=np.int64).T

    np.savez("data/graph.npz", x=x, y=y, edge_index=ei,
             feature_cols=np.array(feature_cols))
    feats.to_csv("data/nodes.csv", index=False)
    edges.to_csv("data/edges.csv", index=False)

    n_fraud = int(y.sum())
    print(f"nodes={len(y)}  fraud={n_fraud} ({100*n_fraud/len(y):.1f}%)  edges={len(edges)}")
    print("ring breakdown:")
    print(feats[feats.label == 1]["ring_type"].value_counts().to_string())
    return G, feats, edges


if __name__ == "__main__":
    build()
