# Abuse-Ring Sentinel

**A structural-risk layer for Razorpay — detecting coordinated abuse that can be difficult to see transaction-by-transaction.**

*Razorpay AI Buildathon · Track 02 (AI Risk Manager)*

---

## Core idea

Traditional transaction-risk systems often ask:

> **Does this individual transaction look suspicious?**

Abuse-Ring Sentinel asks a complementary question:

> **Do accounts that look relatively normal individually become suspicious when their relationships are analyzed together?**

Coordinated abuse can create relational patterns such as:

- money-flow cycles / layering
- fan-out through mule-like accounts
- buyer–seller collusion
- shared-device / shared-identity farms

The central hypothesis is:

> **Sometimes the fraud signal is not in the row — it is in the edges.**

Abuse-Ring Sentinel models accounts and transactions as a graph, uses GraphSAGE
to assign structural-risk scores, stress-tests the detector against structural
perturbations, and routes suspicious accounts into an evidence-grounded
human-review workflow.

---

# End-to-end architecture

```text
Payment / transaction events
        ↓
Graph construction
        ↓
GraphSAGE structural-risk detector
        ↓
Risk score per account
        ↓
Triage
        ↓
Deterministic investigator
        ↓
Verified case evidence
        ↓
Gemini analyst copilot
        ↓
Human reviewer
        ↓
Separate model + human audit records
```

The system deliberately separates:

```text
DETECTION       → GraphSAGE
EVIDENCE        → deterministic NetworkX graph queries
EXPLANATION     → Gemini
FINAL DECISION  → human reviewer
```

Gemini does not decide whether an account is fraudulent.

---

# 1. Controlled synthetic payment-network testbed

The main experimental environment is a **controlled synthetic digital twin —
essentially a simulated payment network with known ground truth**.

It is not a strict real-time industrial digital twin because it does not
continuously synchronize with a live payment network.

The current generated graph contains approximately:

```text
3,229 accounts
9,236 directed transaction edges
229 fraud-labelled accounts
~7.1% fraud prevalence
```

Four coordinated patterns are injected:

```text
cycle
    closed money-flow loops / layering

fanout
    one hub distributing to multiple mule-like accounts

collusion
    dense buyer → seller bipartite blocks

device
    multiple accounts sharing one device identity
```

Transaction amounts and basic account attributes are sampled close to the normal
population so that the experiment focuses on whether relational structure
provides additional signal.

Because the environment is synthetic, the exact fraud membership and ring
structure are known.

The twin is therefore a **controlled mechanism testbed**, not a production
performance estimate.

---

# 2. Account features

Each account begins with eight account-level features:

```text
out_degree
in_degree
total_out
total_in
mean_amount
std_amount
account_age_days
n_counterparties
```

These features already contain some local structural information — particularly
degree and number of counterparties — so the basic Random Forest should not be
interpreted as completely graph-blind.

GraphSAGE adds learned neighborhood context on top of these features.

---

# 3. GraphSAGE detector

The detector is a two-layer GraphSAGE node classifier:

```text
8 account features
        ↓
SAGEConv(8 → 64)
        ↓
ReLU
        ↓
Dropout(0.30)
        ↓
SAGEConv(64 → 64)
        ↓
ReLU
        ↓
Linear(64 → 2)
        ↓
Softmax
        ↓
normal / fraud risk score
```

Training configuration:

```text
optimizer        Adam
learning rate    0.01
weight decay     5e-4
loss             weighted cross-entropy
epochs           200
hidden size      64
GraphSAGE layers 2
```

Each GraphSAGE layer performs approximately one-hop neighborhood aggregation.

Therefore:

```text
Layer 1 → roughly 1-hop context
Layer 2 → roughly 2-hop context
```

Conceptually, each layer combines a node's current representation with an
aggregation of its immediate neighbors.

The first layer maps the original 8-dimensional account representation into a
64-dimensional hidden representation.

The second layer keeps the representation at 64 dimensions while incorporating
information that has propagated through another neighborhood layer.

The resulting 64-dimensional vector is a learned latent representation of the
account and its graph context.

Individual embedding dimensions are not explicitly interpretable as concepts
such as "cycle score", "device score", or "collusion score".

The final classifier produces two raw **logits**:

```text
normal logit
fraud logit
```

Softmax converts them into two values summing to one.

The fraud-class softmax output is used as the model's **risk score**.

It should not be interpreted as a calibrated real-world probability of fraud.

---

## Message-passing direction

The underlying transaction graph is directed.

For GraphSAGE message passing, each transaction edge is mirrored:

```python
edge_index = torch.cat(
    [edge_index, edge_index.flip(0)],
    dim=1
)
```

So:

```text
A → B
```

effectively becomes:

```text
A ↔ B
```

for message passing.

Some direction information is still retained through account features such as:

```text
in_degree
out_degree
total_in
total_out
```

but the graph topology itself does not preserve transaction direction during
message passing.

A production extension could use directed or relation-aware graph modeling.

---

# 4. Canonical evaluation methodology

The canonical synthetic evaluation uses **held-out fraud components** rather
than randomly hiding individual fraud nodes.

Whole connected fraud components are assigned together to either training or
test.

During GraphSAGE training:

```text
only edges where BOTH endpoints are training nodes
are available for message passing
```

This prevents held-out fraud-component edges from participating in training
message passing.

Feature normalization statistics are also fitted only on the training partition
and then frozen.

At inference time, the learned GraphSAGE function is applied to the full
observed graph and metrics are calculated only on held-out test nodes.

This is best described as:

> **Inductive with respect to held-out fraud components during training.**

All models use the same test mask for each seed.

The reported results are mean ± standard deviation over **five different
held-out-component split seeds on the same generated graph**.

---

# 5. Baselines

Three models are compared.

```text
1. Random Forest

   eight account-level features


2. Structural Random Forest

   eight account-level features
   +
   PageRank
   clustering coefficient
   triangle count
   k-core number


3. GraphSAGE

   eight account-level features
   +
   learned neighborhood message passing
```

The structural Random Forest is especially important because it tests whether
learned message passing still adds value after explicit graph statistics have
already been engineered into a tabular model.

---

# 6. Controlled-twin results

## Overall performance — five split seeds

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Random Forest | 0.708 ± 0.050 | 0.750 ± 0.080 | 0.724 ± 0.039 |
| Structural Random Forest | 0.979 ± 0.014 | 0.957 ± 0.034 | 0.967 ± 0.020 |
| **GraphSAGE** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |

## Recall by coordinated pattern

| Model | Cycle | Fanout | Collusion | Device |
|---|---:|---:|---:|---:|
| Random Forest | 0.49 | 1.00 | 0.61 | 0.77 |
| Structural Random Forest | 0.99 | 1.00 | 0.82 | 1.00 |
| **GraphSAGE** | **1.00** | **1.00** | **1.00** | **1.00** |

The most interesting difference appears on the collusion pattern:

```text
Random Forest       0.61
Structural RF       0.82
GraphSAGE           1.00
```

In this controlled testbed, learned neighborhood message passing captures the
collusion structure more effectively than the tested handcrafted structural
features.

The perfect GraphSAGE result is **not presented as expected production
performance**.

The synthetic environment intentionally contains clearly learnable relational
motifs, so this result demonstrates the mechanism rather than proving
real-world accuracy.

---

# 7. Illustrative cost model

The evaluation code contains explicit scenario-level cost assumptions:

```text
missed fraud account   ₹5,000
false alarm              ₹800
```

These are illustrative development assumptions.

They are **not Razorpay loss estimates**.

Using those assumptions:

| Model | Estimated cost |
|---|---:|
| Random Forest | ₹94,720 ± ₹41,334 |
| Structural Random Forest | ₹14,960 ± ₹11,674 |
| GraphSAGE | ₹0 |

In a production system, threshold selection would instead depend on actual
monetary loss, false-positive cost, customer friction, analyst capacity and
probability calibration.

---

# 8. Defensive structural red-team evaluation

High clean accuracy does not guarantee robustness.

A coordinated group may attempt to reduce how obvious its graph structure looks.

The synthetic red-team experiment applies two bounded structural perturbations:

```text
DROP

remove a fraction of fraud-to-fraud edges
```

and:

```text
COVER

add connections from fraud nodes to normal nodes
```

For example:

```text
drop = 0.70
cover = 4
```

means approximately:

```text
70% of fraud-to-fraud edges are removed
+
4 normal cover edges are added per fraud node
```

This experiment runs only inside the controlled synthetic graph.

It is a **defensive structural-robustness stress test**, not a production
adversary.

---

# 9. Canonical five-seed robustness result

Using held-out fraud components and separate perturbation seeds:

```text
clean detector
1.000 ± 0.000 recall

        ↓
structural perturbation

unhardened detector
0.003 ± 0.007 recall

        ↓
adversarial hardening

hardened detector on fresh perturbations
0.931 ± 0.027 recall
```

Headline result:

```text
1.000  →  0.003  →  0.931
clean     attacked    hardened
```

The large collapse under attack is an important finding.

It shows that although the clean detector performs extremely well on the
controlled graph, it depends strongly on structural signals that can be
disrupted.

---

# 10. Adversarial hardening

Hardening does not require a different neural-network architecture.

The same GraphSAGE model is retrained using a deliberately perturbed training
graph:

```text
clean graph
    ↓
generate structural perturbation A
    ↓
train GraphSAGE on perturbed graph
```

Evaluation then uses a different random realization:

```text
generate fresh perturbation B
    ↓
evaluate hardened model
```

The model is therefore not evaluated on the exact corrupted graph used during
hardening.

Conceptually, this is graph-based **adversarial data augmentation**.

---

# 11. Hardening-severity ablation

Aggressive hardening can improve robustness while harming clean performance.

A five-seed development ablation compared:

```text
70% edge-drop hardening
vs
50% edge-drop hardening
```

while evaluating both against a fresh 70% structural attack.

| Training setup | Clean recall | Recall vs 70% attack |
|---|---:|---:|
| 70% hardening | 0.745 | **0.931** |
| **50% hardening** | **0.947** | 0.906 |

The 50% training perturbation preserved substantially more clean recall while
retaining strong robustness against the harsher 70% evaluation attack.

This does **not** establish 50% as universally optimal.

It demonstrates that robustness is an **operating-point trade-off** rather than
a free improvement.

---

# 12. Model-aware RED / BLUE co-evolution sandbox

The final live demo contains a separate bounded co-evolution experiment.

RED evaluates multiple allowed structural perturbation strategies against the
**current detector**:

```text
drop=.30  cover=2
drop=.50  cover=2
drop=.50  cover=4
drop=.70  cover=4
drop=.70  cover=6
```

For every candidate:

```text
apply candidate attack
        ↓
evaluate CURRENT GraphSAGE model
        ↓
measure held-out fraud recall
```

RED chooses the candidate producing the **lowest fraud recall**, because this is
the attack that currently damages the detector most.

BLUE then creates a fresh realization of that selected strategy and retrains the
detector.

The updated model becomes the defender in the next round.

Example live trajectory:

| Round | RED drop | Cover | Pre-defense recall | Clean recall after BLUE | Fresh attack after BLUE |
|---|---:|---:|---:|---:|---:|
| 1 | 0.50 | 4 | 0.019 | 0.981 | 0.849 |
| 2 | 0.70 | 6 | 0.811 | 0.453 | 0.962 |

Round 1 shows BLUE recovering strongly from the attack.

Round 2 demonstrates an important failure mode:

```text
adversarial robustness ↑
clean performance ↓
```

The detector becomes highly robust against the selected attack but
over-specializes enough that clean recall falls substantially.

This reinforces the need to optimize both clean and adversarial performance
instead of blindly maximizing robustness to one perturbation.

The final live demo defaults to **two co-evolution rounds** so the presentation
remains short.

A longer exploratory run can be launched with:

```bash
python run_full_demo.py --coevolve-rounds 8
```

---

## Confidence-guided co-evolution prototype

`redteam-lab/coevolve.py` contains an earlier experimental attacker that operates
at the individual-edge level.

The current GraphSAGE model first assigns fraud-confidence scores.

The attacker then prioritizes removing edges between accounts that the current
model considers highly suspicious:

```text
high-confidence fraud node
        ↔
high-confidence fraud node
```

The removal score is based on the combined current fraud confidence of the two
endpoints.

The attacker also prioritizes adding cover edges between:

```text
high-confidence fraud node
        ↔
low-confidence normal node
```

After BLUE retrains the detector, confidence scores change.

The next RED round can therefore target different edges.

This prototype uses an older random-split / full-normalization setup and is
treated as an experimental research sandbox rather than the source of the
canonical five-seed robustness numbers.

---

# 13. Triage

GraphSAGE produces a risk score for every account.

The current routing thresholds are:

```text
risk < 0.35
    → auto-clear
```

```text
0.35 ≤ risk < 0.75
    → human review
```

```text
risk ≥ 0.75
    → high-risk escalation
    → still requires human review before adverse classification
```

The values `0.35` and `0.75` are development/demo thresholds.

Because the softmax score has not been separately calibrated:

```text
risk score = 0.75
```

does **not** necessarily mean:

```text
75% real-world probability of fraud
```

Production thresholds should instead be selected using actual losses,
false-positive tolerance, analyst-review capacity and probability calibration.

---

# 14. Deterministic investigator

Accounts routed into review are passed to a deterministic investigator.

The investigator uses **NetworkX**, a Python graph-analysis library, to inspect
the selected account and its surrounding graph.

Evidence checks currently include:

```text
shared-device groups

closed transaction cycles

high-fanout hubs

radius-2 ego networks

dense / triangle-free collusion structure

GraphSAGE risk score
```

The investigator is **not GNNExplainer**.

It does not claim to expose the exact internal reasoning of GraphSAGE.

Instead, it independently gathers human-readable graph evidence around an
account that has already been escalated by the detector.

Its output is a JSON-compatible case object:

```python
{
    "recommendation": "review",
    "pattern": "device",
    "risk_score": 0.98,
    "reasoning_steps": [
        {
            "claim": "...",
            "evidence": ["..."],
            "source": "..."
        }
    ]
}
```

Each reasoning step contains:

```text
claim
evidence
source
```

so investigator statements remain tied to explicit graph records or model
output.

---

## Multiple detected patterns

An account may produce evidence for more than one pattern.

For example:

```text
shared device
+
fanout relationship
+
high GraphSAGE risk score
```

All supported evidence remains inside `reasoning_steps`.

The current implementation stores only one top-level primary:

```text
pattern
```

field.

A stronger production schema would explicitly support multiple patterns:

```python
{
    "primary_pattern": "device",
    "detected_patterns": [
        "device",
        "fanout"
    ]
}
```

---

# 15. Gemini analyst copilot

Gemini is used **after** deterministic evidence generation.

It does not discover graph evidence and does not calculate the risk score.

The pipeline is:

```text
GraphSAGE
    ↓
risk score

NetworkX investigator
    ↓
structured verified evidence

evidence_text(case)
    ↓
Gemini prompt
    ↓
analyst conversation
```

The investigator's evidence is converted into text and injected directly into
Gemini's prompt.

The chatbot is therefore best described as:

> **A case-grounded conversational copilot over fixed verified context.**

It is **not RAG**.

There is currently:

```text
no vector database

no embedding retrieval

no top-k document retrieval

no dynamic NetworkX tool invocation by Gemini
```

Gemini receives the case evidence that has already been generated by the
investigator.

The system prompt explicitly instructs Gemini to answer only from that evidence.

For example:

```text
"What city is this account in?"
```

is used as a grounding / hallucination test.

If city information is not present in the evidence, the expected response is
that the system does not have that information.

Gemini's role is therefore **usability and explanation**, not fraud detection or
final decision-making.

---

# 16. Human-in-the-loop decision

The human reviewer remains the final authority.

The model recommendation is stored first:

```json
{
  "type": "recommendation",
  "case_id": "...",
  "model_score": 0.98,
  "pattern": "device",
  "recommendation": "review"
}
```

The human decision is stored separately:

```json
{
  "type": "decision",
  "case_id": "...",
  "human_decision": "flag",
  "reason": "...",
  "reviewer": "..."
}
```

The human record does not overwrite the model recommendation.

This allows later comparison of:

```text
model recommendation
vs
human decision
```

and makes disagreement measurable.

The current JSONL ledger is append-only at the **application level**.

It is not claimed to be cryptographically immutable.

---

# 17. Precedent retrieval

The codebase contains an experimental precedent-retrieval function that can
search previously resolved cases using GraphSAGE-embedding similarity.

It is intentionally **not part of the main live demo**.

A previous human decision is not automatically ground truth.

Blindly surfacing previous decisions could:

```text
propagate reviewer mistakes

create confirmation bias

compare embeddings generated by different model versions
```

A production precedent system would therefore require outcome verification,
quality controls, version compatibility and provenance guarantees.

---

# 18. Razorpay ingestion boundary

The detector is currently trained and evaluated on the controlled synthetic
payment graph.

`src/razorpay_adapter.py` separately demonstrates how Razorpay-shaped:

```text
payment.captured
```

events can be converted into graph entities and relationships.

The synthetic webhook examples contain information such as:

```text
account identity

card fingerprint

device ID

IP address

order relationship
```

These become graph relationships such as:

```text
account
   ├── card fingerprint
   ├── device
   ├── IP
   └── order-linked entity
```

When multiple apparently separate accounts share identifiers, they become
connected through the entity graph.

Example:

```text
Account A ── shared_card_X

Account B ── shared_card_X

Account C ── shared_card_X
```

The relationship becomes visible even if each transaction appears ordinary
individually.

---

## Current Razorpay adapter boundary

The current Razorpay adapter does **not yet run the trained GraphSAGE detector**
on the generated entity graph.

Its printed:

```text
FLAG
REVIEW
CLEAR
```

labels come from a transparent shared-identifier cluster heuristic.

That heuristic exists only as an **integration smoke test** to demonstrate that
Razorpay-shaped events can be transformed into graph structure.

The intended production flow is:

```text
Razorpay webhook events
        ↓
persistent evolving graph
        ↓
detector-compatible account graph
        ↓
GraphSAGE
        ↓
risk score
        ↓
triage
        ↓
investigator
        ↓
Gemini
        ↓
human analyst
```

The Razorpay adapter demonstrates the ingestion boundary.

The synthetic payment network is where the GraphSAGE detector itself is
currently trained and evaluated.

---

# 19. Real-data reality check — Elliptic

The project is also evaluated on the public Elliptic Bitcoin transaction graph.

Dataset scale:

```text
203,769 transaction nodes

234,355 transaction edges

46,564 labelled transactions

4,545 illicit labelled transactions

166 numeric features
```

The 166 Elliptic features include:

```text
94 local transaction features

72 aggregated one-hop neighborhood features
```

This is important because the Random Forest is already receiving substantial
engineered neighborhood information.

Observed result:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| GraphSAGE | 0.749 | 0.588 | 0.659 |
| Random Forest | 0.988 | 0.687 | **0.810** |

This does not contradict the project hypothesis.

Instead, it demonstrates that when graph information has already been heavily
engineered into tabular features, a tree-based model can be extremely
competitive.

The claim is therefore **not**:

> GNNs beat tabular models everywhere.

The narrower claim is:

> **Graph learning becomes especially useful when important relational
> information is not already represented effectively in the input rows.**

---

## Elliptic split

The Elliptic experiment uses a **temporal label split**:

```text
training labels
time step ≤ 34

test labels
time step ≥ 35
```

However, GraphSAGE message passing still operates over the full static graph.

The Elliptic implementation also normalizes features using the full feature
matrix.

The experiment should therefore **not** be described as strict temporal graph
isolation.

It is a real-data reality check rather than the canonical controlled mechanism
experiment.

---

# 20. AI vs deterministic software

The system intentionally uses AI only where learning or language understanding
provides value.

```text
GraphSAGE
    → learned structural-risk representation
```

```text
Gemini
    → natural-language interface over verified evidence
```

Critical supporting components remain deterministic:

```text
synthetic graph generation

triage thresholds

NetworkX investigator

governance rules

audit logging

human final decision
```

This separation keeps the operational review path easier to reproduce and audit.

---

# 21. Main live demo

The final demo entry point is:

```bash
python run_full_demo.py
```

The demo follows this sequence:

```text
1. Controlled synthetic payment-network summary

2. Gemini preflight

3. Canonical five-seed robustness experiment

4. Hardening-severity ablation

5. Model-aware RED / BLUE co-evolution sandbox

6. Deterministic investigator on a flagged account

7. Grounded Gemini case summary

8. Interactive analyst chatbot over the SAME case

9. Human final decision

10. Separate model + human audit records

11. Optional Razorpay ingestion smoke test
```

Optional demo flags:

```bash
# recompute the hardening ablation live
python run_full_demo.py --recompute-ablation
```

```bash
# skip canonical red-team section
python run_full_demo.py --skip-redteam
```

```bash
# skip model-aware co-evolution
python run_full_demo.py --skip-coevolve
```

```bash
# longer exploratory co-evolution run
python run_full_demo.py --coevolve-rounds 8
```

The default co-evolution demo uses two rounds to keep the presentation concise.

---

# 22. Running individual experiments

Install dependencies:

```bash
pip install -r requirements.txt
```

Canonical twin evaluation:

```bash
PYTHONPATH=src python src/twin_eval_seeds.py
```

Canonical five-seed defensive robustness:

```bash
PYTHONPATH=src python src/red_team_seeds.py
```

Investigator + human-review layer:

```bash
export GEMINI_API_KEY="your_key"

PYTHONPATH=src python src/review_layer.py
```

Grounded analyst chatbot:

```bash
export GEMINI_API_KEY="your_key"

PYTHONPATH=src python src/chat_review.py
```

Razorpay-shaped ingestion smoke test:

```bash
python src/razorpay_adapter.py
```

Elliptic experiment:

```bash
PYTHONPATH=src python src/train_gnn_real.py

PYTHONPATH=src python src/baseline_real.py
```

---

# 23. Compliance posture

The workflow is designed around:

```text
human review

evidence attribution

auditability

separation between model recommendation and final decision
```

The design is **informed by** RBI fraud-risk and natural-justice principles:
adverse fraud classification is not silently applied by the AI system without a
human review step.

This is a product-design principle informed by regulatory requirements.

It is **not a claim of formal compliance**.

---

# 24. Repository structure

```text
run_full_demo.py
    final live-demo launcher


src/

  generate_graph.py
      controlled synthetic payment-network generator

  train_gnn.py
      two-layer GraphSAGE detector

  twin_eval_canonical.py
      held-out fraud-component evaluation logic

  twin_eval_seeds.py
      five-seed canonical model comparison

  structural_features.py
      engineered graph-feature Random Forest experiment

  red_team.py
      structural perturbation generator

  red_team_canonical.py
      inductive robustness helpers

  red_team_seeds.py
      five-seed canonical robustness evaluation

  review_layer.py
      deterministic investigator
      human-review workflow
      separate audit ledger
      experimental precedent function

  chat_review.py
      evidence-grounded Gemini analyst chatbot

  razorpay_adapter.py
      Razorpay-shaped webhook
      → entity graph
      → deterministic integration smoke test

  train_gnn_real.py
      GraphSAGE Elliptic reality check

  baseline_real.py
      Random Forest Elliptic baseline


redteam-lab/

  coevolve.py
      earlier confidence-guided red-team research prototype


compliance_rules.yaml
    governance configuration


index.html
    project site
```

---

# 25. Known limitations

This is a buildathon prototype rather than a production fraud platform.

Important current limitations include:

- The synthetic fraud motifs are cleaner than real adversarial behavior.

- The five canonical seeds use different holdout splits of the **same generated
  graph**, rather than five independently generated payment networks.

- Base account features such as degree and number of counterparties are computed
  from the static graph before train/test splitting.

- Structural-RF PageRank, clustering, triangle and k-core features are computed
  on the full static graph.

- GraphSAGE message passing is bidirectional, so transaction direction is not
  explicitly preserved in the GNN topology.

- Transaction amount and timestamp are not directly used as GraphSAGE edge
  attributes.

- The canonical red-team perturbation uses known synthetic fraud labels as an
  oracle.

- Node features are not recomputed after red-team topology perturbation, so the
  experiment primarily measures sensitivity to connectivity changes.

- The GraphSAGE detector has no explicit temporal or recurrent memory.

- The investigator generates independent graph evidence rather than a faithful
  attribution of GraphSAGE internals.

- The current investigator schema records one primary pattern even when multiple
  evidence patterns may exist.

- The current fanout evidence heuristic can be made more direction-aware.

- The JSONL audit ledger is application-level append-only rather than
  cryptographically immutable.

- The Razorpay adapter demonstrates ingestion but does not yet invoke the trained
  GraphSAGE detector.

- The Elliptic experiment uses a temporal label split but does not enforce strict
  temporal graph isolation.

---

# 26. Production extensions

Natural next steps include:

### Directed / heterogeneous graph modeling

Represent distinct relationship types such as:

```text
account
card
device
IP
bank account
beneficiary
merchant
```

and preserve edge direction explicitly.

### Temporal graph models

Capture how account relationships evolve rather than treating the network as
one static snapshot.

Possible approaches include TGN-style temporal graph models or temporal graph
features.

### Scalable neighborhood sampling

The prototype performs full-graph message passing.

At large payment-network scale, training and inference would require graph
partitioning and neighborhood sampling.

### Persistent streaming graph state

Update account features and relationships as new payment events arrive instead
of rebuilding the graph from scratch.

### Probability calibration

Calibrate raw classifier confidence so that operational risk scores better match
observed fraud frequencies.

### Clean + adversarial joint training

Instead of training only on heavily perturbed graphs, jointly optimize clean and
attacked objectives, for example:

```text
L =
λ × L_clean
+
(1 - λ) × L_adversarial
```

to preserve clean performance while improving robustness.

### Multi-pattern investigator schema

Expose all supported evidence patterns rather than forcing one top-level pattern
label.

### Reproducible model versioning

Store:

```text
model version
feature-pipeline version
graph snapshot / event cutoff
threshold configuration
embedding version
case evidence
```

so historical decisions can be reconstructed exactly.

### Historical payment-network backtesting

Evaluate incremental value against real confirmed outcomes and an existing risk
stack.

### Live Razorpay test-mode integration

Move from:

```text
synthetic Razorpay-shaped events
```

to:

```text
live test webhook
    ↓
persistent graph
    ↓
GraphSAGE
    ↓
investigator
    ↓
analyst
```

---

# Takeaway

Abuse-Ring Sentinel is built around three ideas.

> **1. Relational risk**
>
> Coordinated abuse may only become visible when accounts are analyzed together.

> **2. Robustness matters**
>
> Excellent clean performance can hide severe structural brittleness.

> **3. AI should not be the final authority**
>
> GraphSAGE identifies structural risk, deterministic graph analysis turns it
> into evidence, Gemini makes that evidence easier to interrogate, and a human
> makes the final decision.

The goal is not to replace an existing fraud stack.

It is to prototype a **complementary structural-risk layer** for detecting and
investigating coordinated behavior that is difficult to represent in isolated
transaction rows.
