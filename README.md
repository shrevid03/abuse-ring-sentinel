# Abuse-Ring Sentinel

**A structural-risk layer for Razorpay — catching coordinated fraud that per-transaction models miss.**

*Razorpay AI Buildathon · Track 02 (AI Risk Manager)*

---

Existing risk systems are good at asking *"does this one transaction look suspicious?"*. Abuse-Ring Sentinel asks a different, complementary question: *"do these accounts — each individually normal — become suspicious when you look at them together?"*

Coordinated payment abuse (money-laundering cycles, bust-out fan-outs, buyer–seller collusion, device farms) is **individually invisible, collectively obvious**. Each account looks fine on its own row of data, so a per-transaction model waves it through. The fraud only appears in the *structure* the group forms — so we detect it with a graph neural network, and wrap it in a human-in-the-loop, auditable review system.

## Where it fits in Razorpay

![Razorpay to Sentinel flow](out/razorpay_flow.png)

A Razorpay payment event exposes shared identifiers — card fingerprint, device, VPA, payout account. The adapter turns those into graph edges, so when many "separate" accounts quietly share them, the coordinated cluster becomes visible.

```
Razorpay payment.captured webhook
   → extract entities (account · card · device · beneficiary)
   → build risk graph (shared-identifier edges link "separate" accounts)
   → GraphSAGE scores structural risk
   → triage: clear / review / flag
   → analyst reviews the evidence (human sign-off required)
```

Try it: `python src/razorpay_adapter.py` — ingests Razorpay-shaped `payment.captured` webhooks and shows three accounts flagged for sharing a card + device, while independent payments stay clear.

## The system, in three parts

**1. Detect** — a GraphSAGE model scores each account using its *connections*, not just its own features, so it catches coordinated rings a per-row model can't.

**2. Stress-test (defensive)** — we evaluate the detector's robustness by applying structural perturbations to rings **in a closed synthetic sandbox, against our own detector**, then adversarially train against those failures to restore robustness. Fraud is adversarial, so evaluating only on clean data gives false confidence. This is defensive robustness evaluation — no attack tooling, nothing that operates outside our own test graph.

**3. Decide safely** — the model never auto-freezes anyone. Flags become evidence-cited cases; an LLM copilot converts model evidence into investigator-readable explanations and answers questions about a case (it never makes the risk decision); a human approves or overrides with a reason; and the model's recommendation and the human's decision are stored as **separate, append-only records** so accountability is always traceable.

## Results (measured, honestly)

**Controlled twin** (structure is the only signal — isolates the hypothesis):
| | collusion recall |
|---|---|
| Tabular baseline (per-row) | 0.52 |
| Hand-engineered structural features | 0.52 |
| **GraphSAGE (learned structure)** | **1.00** |

Only the GNN recovers the dense bipartite collusion signal — per-row features and hand-crafted graph metrics both miss it. (The twin's near-perfect overall scores are a *controlled experiment*, not a production claim: features are deliberately non-discriminative so structure is the only thing left to learn.)

**Real Elliptic Bitcoin graph** (203,769 nodes, temporal held-out split — the reality check):
| model | precision | recall | F1 |
|---|---|---|---|
| GraphSAGE | 0.75 | 0.59 | 0.66 |
| RandomForest | 0.99 | 0.69 | **0.81** |

RandomForest wins here — and that's expected: Elliptic's features already include aggregated neighbour statistics, so the graph signal is baked into the rows. The claim is **not** "GNNs beat tabular models everywhere." It's narrower and defensible: *when coordinated abuse is hidden at the transaction level and the signal lives in relationships, graph learning recovers what per-transaction models miss* — which the twin demonstrates cleanly. GNN F1 is stable at ~0.66 across all thresholds (an honest ceiling from distribution shift in the test period, not a tuning artifact).

**Defensive robustness evaluation** (ring recall, closed synthetic sandbox, our own detector):

`clean 1.00 → perturbed 0.03 → after adversarial hardening 0.94`

The unhardened detector is fragile under deliberate structural perturbation — which is exactly why we test for it. Adversarial training on those perturbations restores robustness. All robustness experiments operate only on our synthetic test graph, to evaluate and harden our own detector.


## False-positive cost (THE BAR: honest metrics including false-positive cost)

Track 02 explicitly asks for honest metrics including false-positive cost, so we
report operational cost alongside precision, recall and F1. The evaluation harness
assigns configurable costs to false alarms (legitimate accounts sent to review)
and to missed coordinated-abuse cases. Lower is better.

| model (twin) | precision | recall | est. net cost |
|---|---|---|---|
| Tabular baseline (per-row) | 0.77 | 0.67 | ₹115,400 |
| **GraphSAGE** | **0.99** | **1.00** | **₹800** |

Cost assumptions are explicit in `baseline.py` (`COST_MISSED_FRAUD`,
`COST_FALSE_ALARM`) so the trade-off is auditable, and the decision threshold
is a business dial: high-recall to catch more fraud, high-precision to spare
good customers. On real Elliptic data the same harness reports net cost for
both the GNN and the tabular baseline.

## Run it

From the project root, in an env with the deps (`pip install -r requirements.txt`), with `PYTHONPATH=src`:

```bash
# Razorpay integration adapter (ingestion + smoke test)
python src/razorpay_adapter.py

# synthetic-twin pipeline
python src/generate_graph.py
python src/baseline.py
python src/structural_features.py
PYTHONPATH=src python src/train_gnn.py

# review layer: investigator + precedent memory + parallel ledger
export GEMINI_API_KEY=<key>          # for LLM narration (falls back if unset)
PYTHONPATH=src python src/review_layer.py

# analyst copilot chatbot (grounded in one flagged case)
PYTHONPATH=src python src/chat_review.py

# defensive red-team
PYTHONPATH=src python src/red_team.py

# real Elliptic data (see below), then:
PYTHONPATH=src python src/train_gnn_real.py
PYTHONPATH=src python src/baseline_real.py

# additional robustness experiments (research)
cd redteam-lab && python coevolve.py
```

**Elliptic data:** download from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), unzip, and place the three CSVs in `data_real/elliptic/` (gitignored — the feature file exceeds GitHub's size limit).

## Compliance posture

The workflow is designed around human review, attribution, and auditability, **informed by** RBI's fraud-risk principles — in particular natural justice (an opportunity to be heard before an adverse classification, per the Rajesh Agarwal judgment). No account is auto-classified as fraud; a human signs off first, and both the recommendation and the decision are retained. This is a product design informed by regulatory principles, not a claim of formal regulatory compliance.

## Repository

```
src/
  razorpay_adapter.py     Razorpay webhook → graph (ingestion adapter + smoke test)
  generate_graph.py       synthetic digital twin with injected ring types
  baseline.py             per-row tabular baseline + eval harness
  structural_features.py  structural-feature ablation
  train_gnn.py            GraphSAGE detector + triage gate + audit trail
  train_gnn_real.py       detector on real Elliptic data (+ threshold tuning)
  baseline_real.py        tabular baseline on Elliptic
  review_layer.py         investigator agent, precedent memory, parallel ledger
  chat_review.py          analyst copilot (evidence-grounded LLM chat)
  red_team.py             defensive adversarial red-team
compliance_rules.yaml     RBI-grounded rules as config
index.html                animated project site (deploys on GitHub Pages)
redteam-lab/              additional robustness-evaluation experiments (research)
```

## Future work

- **Temporal GNNs** (EvolveGCN / TGN) — the real fix for the Elliptic distribution-shift ceiling.
- **Semi-supervised / PU learning** — exploit the ~77% unlabelled nodes; realistic for production.
- **Heterogeneous multi-relation graphs** — device / bank / IP / email edges scored together.
- **Broader defensive robustness** — evaluation across a wider range of synthetic structural perturbations, with cross-round memory for more stable hardening.
- **Live Razorpay webhook** — consume real test-mode events end to end.
