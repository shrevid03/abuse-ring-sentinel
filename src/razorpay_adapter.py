"""
razorpay_adapter.py  —  Razorpay -> Sentinel INGESTION ADAPTER
================================================================

Scope (read this before pitching it):

  This file is the ADAPTER + a deterministic INTEGRATION SMOKE TEST.
  It is NOT the detector. The pipeline has three separate stages:

    1. ADAPTER  (this file)      Razorpay webhook event  ->  graph + evidence
    2. DETECTOR (train_gnn.py)   graph  ->  GraphSAGE structural-risk score
    3. TRIAGE   (train_gnn.py /  score + evidence  ->  clear / review / flag
                 review_layer.py)                       ->  analyst + audit

  What runs here is stage 1 plus a *transparent shared-cluster heuristic* used
  only as a smoke test, to prove the ingestion path end-to-end WITHOUT the
  trained model loaded. The FLAG/REVIEW labels this file prints come from that
  deterministic heuristic (connected shared-identifier clusters), NOT from
  GraphSAGE. In production the assembled graph (below) is handed to the trained
  GraphSAGE detector, which produces the risk score that triage acts on.

  Events below use the real Razorpay webhook envelope
  ( {"event": "...", "payload": {"payment": {"entity": {...}}}} ) so the adapter
  consumes Razorpay-compatible payloads, not an invented schema. They are
  test-mode-SHAPED synthetic events — no live webhook is wired in this demo.

Run:  python src/razorpay_adapter.py
"""

import networkx as nx
from collections import defaultdict

# --- Razorpay-compatible webhook payloads (real envelope, synthetic test data) ---
# Three "different" accounts quietly share ONE card fingerprint + ONE device
# (the hidden ring). The rest are ordinary, independent payments.
WEBHOOK_EVENTS = [
    {"event": "payment.captured", "payload": {"payment": {"entity": {
        "id": "pay_R1", "entity": "payment", "amount": 4800000, "currency": "INR",
        "status": "captured", "method": "card", "email": "arjun@example.com",
        "contact": "+919800000001", "order_id": "order_M1",
        "card": {"id": "card_x1", "fingerprint": "cf_SHARED9", "last4": "4417", "network": "Visa"},
        "notes": {"device_id": "dev_SHARED", "ip": "203.0.113.9"}}}}},
    {"event": "payment.captured", "payload": {"payment": {"entity": {
        "id": "pay_R2", "entity": "payment", "amount": 5100000, "currency": "INR",
        "status": "captured", "method": "card", "email": "meera@example.com",
        "contact": "+919800000002", "order_id": "order_M1",
        "card": {"id": "card_x2", "fingerprint": "cf_SHARED9", "last4": "4417", "network": "Visa"},
        "notes": {"device_id": "dev_SHARED", "ip": "203.0.113.9"}}}}},
    {"event": "payment.captured", "payload": {"payment": {"entity": {
        "id": "pay_R3", "entity": "payment", "amount": 4950000, "currency": "INR",
        "status": "captured", "method": "upi", "email": "rahul@example.com",
        "contact": "+919800000003", "order_id": "order_M1", "vpa": "rahul@okhdfc",
        "card": {"id": "card_x3", "fingerprint": "cf_SHARED9", "last4": "4417", "network": "Visa"},
        "notes": {"device_id": "dev_SHARED", "ip": "203.0.113.10"}}}}},
    {"event": "payment.captured", "payload": {"payment": {"entity": {
        "id": "pay_N1", "entity": "payment", "amount": 120000, "currency": "INR",
        "status": "captured", "method": "card", "email": "sara@example.com",
        "contact": "+919811111111", "order_id": "order_M2",
        "card": {"id": "card_a1", "fingerprint": "cf_A1", "last4": "1111", "network": "Mastercard"},
        "notes": {"device_id": "dev_A1", "ip": "198.51.100.4"}}}}},
    {"event": "payment.captured", "payload": {"payment": {"entity": {
        "id": "pay_N2", "entity": "payment", "amount": 340000, "currency": "INR",
        "status": "captured", "method": "upi", "email": "vikram@example.com",
        "contact": "+919822222222", "order_id": "order_M3", "vpa": "vikram@okaxis",
        "notes": {"device_id": "dev_A2", "ip": "198.51.100.7"}}}}},
    {"event": "payment.captured", "payload": {"payment": {"entity": {
        "id": "pay_N3", "entity": "payment", "amount": 89000, "currency": "INR",
        "status": "captured", "method": "card", "email": "neha@example.com",
        "contact": "+919833333333", "order_id": "order_M4",
        "card": {"id": "card_a3", "fingerprint": "cf_A3", "last4": "2222", "network": "Visa"},
        "notes": {"device_id": "dev_A3", "ip": "198.51.100.9"}}}}},
]

REVIEW_AT, FLAG_AT = 2, 3   # smoke-test cluster-size thresholds (NOT the GNN)


# ============================ STAGE 1: ADAPTER =============================
def unwrap(event):
    """Pull the payment entity out of a Razorpay webhook envelope."""
    return event["payload"]["payment"]["entity"]


def account_id(p):
    return p.get("email") or p.get("vpa") or p["id"]


def identifiers(p):
    """Shared identifiers a payment exposes — these link 'separate' accounts."""
    ids = {}
    if p.get("card", {}).get("fingerprint"): ids["card"] = p["card"]["fingerprint"]
    if p.get("notes", {}).get("device_id"):  ids["device"] = p["notes"]["device_id"]
    if p.get("notes", {}).get("ip"):         ids["ip"] = p["notes"]["ip"]
    if p.get("order_id"):                    ids["beneficiary"] = p["order_id"]
    return ids


def build_graph(events):
    """Razorpay webhook events -> typed entity graph with shared-identifier edges.
    THIS is the adapter's real job: event -> graph representation. The resulting
    graph is what the GraphSAGE detector consumes in production."""
    G = nx.Graph()
    shared = defaultdict(set)
    for ev in events:
        p = unwrap(ev)
        acc = account_id(p)
        G.add_node(acc, kind="account")
        for kind, val in identifiers(p).items():
            node = f"{kind}:{val}"
            G.add_node(node, kind=kind)
            G.add_edge(acc, node, via=p["id"])
            if kind in ("card", "device"):
                shared[val].add(acc)
    return G, shared


# ================= SMOKE TEST (deterministic — NOT the detector) ==============
def smoke_test_score(G, shared):
    """Deterministic integration smoke test: routes accounts by the size of their
    shared-identifier cluster. This exists ONLY to prove the ingestion path runs
    end-to-end without the trained model. The production detector is GraphSAGE
    (train_gnn.py); it would score the graph built above, not this heuristic."""
    reports = []
    for acc in [n for n, d in G.nodes(data=True) if d["kind"] == "account"]:
        cluster, evidence = {acc}, []
        for node in G.neighbors(acc):
            if G.nodes[node]["kind"] in ("card", "device"):
                mates = shared[node.split(":", 1)[1]]
                if len(mates) > 1:
                    cluster |= mates
                    evidence.append(f"shares {node.split(':',1)[0]} {node.split(':',1)[1]} "
                                    f"with {len(mates)-1} other account(s): "
                                    f"{sorted(mates - {acc})}")
        n = len(cluster)
        action = "FLAG" if n >= FLAG_AT else "REVIEW" if n >= REVIEW_AT else "CLEAR"
        reports.append((acc, action, n, evidence))
    return reports


def main():
    print("=" * 70)
    print("RAZORPAY -> SENTINEL  (ingestion adapter + integration smoke test)")
    print("=" * 70)
    print("stage 1 ADAPTER : Razorpay webhook event -> graph + evidence")
    print("stage 2 DETECTOR: graph -> GraphSAGE score        (train_gnn.py, prod)")
    print("stage 3 TRIAGE  : score + evidence -> clear/review/flag -> analyst\n")

    # STAGE 1 — adapter
    G, shared = build_graph(WEBHOOK_EVENTS)
    n_acc = sum(1 for _, d in G.nodes(data=True) if d["kind"] == "account")
    print(f"[adapter] ingested {len(WEBHOOK_EVENTS)} 'payment.captured' webhooks "
          f"-> {n_acc} accounts, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # SMOKE TEST — deterministic, clearly labelled as NOT GraphSAGE
    print("\n[smoke test — deterministic shared-cluster heuristic, NOT GraphSAGE]")
    for acc, action, n, evidence in sorted(smoke_test_score(G, shared), key=lambda r: -r[2]):
        tag = {"FLAG": "FLAG  ", "REVIEW": "REVIEW", "CLEAR": " clear"}[action]
        print(f"  [{tag}] {acc:<22} cluster={n}")
        for e in evidence:
            print(f"            - {e}")

    print("\nIn production: the graph above -> GraphSAGE -> structural-risk score ->")
    print("triage. FLAG/REVIEW cases go to the evidence-grounded copilot; no account")
    print("is auto-frozen (human sign-off required before any fraud classification).")


if __name__ == "__main__":
    main()