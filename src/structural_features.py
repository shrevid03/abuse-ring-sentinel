"""
structural_features.py
----------------------
The honest ablation. Same RandomForest, same rows — but we add GRAPH-derived
features (pagerank, clustering, triangles, k-core, shared-device group size).
If recall on the coordinated rings (cycle / collusion) jumps, we've proven the
missing signal was structural. The GNN then learns this automatically instead
of us hand-engineering it.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from baseline import split_mask, evaluate


def structural_node_features(nodes, edges):
    G = nx.from_pandas_edgelist(edges, "src", "dst", create_using=nx.DiGraph())
    Gu = G.to_undirected()
    for n in nodes["node_id"]:
        if n not in Gu:
            Gu.add_node(n); G.add_node(n)

    pr = nx.pagerank(G, alpha=0.85)
    clust = nx.clustering(Gu)
    tri = nx.triangles(Gu)
    core = nx.core_number(Gu)

    # shared-device group size: how many accounts share this account's device
    dev_group = nodes.groupby("device")["node_id"].transform("count")
    dev_size = dict(zip(nodes["node_id"], dev_group))

    feat = pd.DataFrame({"node_id": nodes["node_id"]})
    feat["pagerank"]      = feat.node_id.map(pr).fillna(0)
    feat["clustering"]    = feat.node_id.map(clust).fillna(0)
    feat["triangles"]     = feat.node_id.map(tri).fillna(0)
    feat["kcore"]         = feat.node_id.map(core).fillna(0)
    feat["device_group"]  = feat.node_id.map(dev_size).fillna(1)
    return feat


def run():
    nodes = pd.read_csv("data/nodes.csv")
    edges = pd.read_csv("data/edges.csv")
    y = nodes["label"].to_numpy()

    base_cols = ["out_degree", "in_degree", "total_out", "total_in",
                 "mean_amount", "std_amount", "account_age_days", "n_counterparties"]
    struct = structural_node_features(nodes, edges)
    X = pd.concat([nodes[base_cols].reset_index(drop=True),
                   struct.drop(columns="node_id").reset_index(drop=True)], axis=1).to_numpy(np.float32)

    train, test = split_mask(y)
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)
    clf.fit(X[train], y[train])
    pred = clf.predict(X[test])
    return evaluate("Baseline + structural features", y[test], pred,
                    ring_types=nodes["ring_type"].to_numpy()[test])


if __name__ == "__main__":
    run()
