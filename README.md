# Abuse-Ring Sentinel

**Razorpay AI Buildathon — Track 02, AI Risk Manager**

Coordinated payment abuse is *individually invisible, collectively obvious*.
A ring of colluding accounts each looks like a normal customer on its own row of
data — a per-transaction fraud model waves them through. The fraud only becomes
visible in the **structure** they form together: circular money flows, bust-out
fan-outs, dense buyer↔seller collusion, and shared-device clusters.

This project detects those rings with a graph neural network, reports honest
precision/recall **including false-positive cost**, and never auto-freezes an
account without a bounded, auditable decision.

> Strictly defense-only: this system flags and explains suspected abuse for human
> review. It takes no offensive action and generates no attack capability.

## The result (held-out test set, synthetic data)

| Model | Precision | Recall | Collusion recall | Est. net cost |
|---|---|---|---|---|
| Tabular baseline (per-row RandomForest) | 0.77 | 0.67 | 0.52 | ₹115,400 |
| + hand-engineered structural features | 1.00 | 0.83 | 0.52 | ₹55,000 |
| **GraphSAGE (learned message passing)** | 0.99 | 1.00 | 1.00 | ₹800 |

The story in one line: hand-crafted graph features fix cycles and shared-device
rings, but **dense bipartite collusion is only caught once the model learns
relational structure end-to-end.** A per-row model cannot express it; fixed graph
metrics cannot express it; the GNN can.

## Architecture

```
synthetic transaction graph            detector                     bounded action
─────────────────────────────         ─────────────────           ──────────────────
accounts + money-movement edges  ──▶  GraphSAGE (2 layers)  ──▶  risk score per node
+ shared-device edges                 message passing over          │
+ 4 injected ring patterns            the transaction graph         ├─ ≥0.75  auto-flag
(cycle, fanout, collusion, device)                                  ├─ 0.35–0.75 human review
                                                                    └─ <0.35  auto-clear
                                                                          │
                                                        audit_trail.csv (score + band +
                                                        neighbours that drove the flag)
```

## Run it

```bash
pip install -r requirements.txt
python src/generate_graph.py        # build labelled synthetic graph -> data/
python src/baseline.py              # per-row baseline (the thing to beat)
python src/structural_features.py   # ablation: proves the signal is structural
python src/train_gnn.py             # GNN + triage gate + audit trail -> out/
```

## How this meets the track bar

- **Working detector, measured precision & recall on a held-out set** — three
  models compared on the same split, per-ring recall broken out.
- **Honest metrics including false-positive cost** — every result carries an
  INR net-cost figure (missed fraud + false alarms), not just F1.
- **Defense-only** — detection and human-routed review; no offensive capability.
- **Graceful handling** — a review band means borderline accounts go to a human,
  never a silent freeze.
- **Audit trail** — every flag is logged with its score and the specific risky
  neighbours behind the decision, so an analyst can see *why*.

## Honesty / limitations (say this in the pitch)

Numbers are on **synthetic** data with cleanly injected ring patterns, so they are
an upper bound — real rings are noisier and adversarial. The contribution is the
*approach and the honest evaluation harness*, not the specific decimals. Next steps
on real data: label scarcity (semi-supervised / positive-unlabelled), concept drift,
and adversaries who deliberately thin their structure to evade graph signals.
