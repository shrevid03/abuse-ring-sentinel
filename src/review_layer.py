"""
review_layer.py
---------------
The human-in-the-loop review + institutional-memory layer that sits AFTER the
GNN's triage gate. It never scores fraud (that's the deterministic GNN) — it
only handles the cases the gate routes to a human. Four capabilities:

  1. Investigator  – gathers EVIDENCE deterministically (graph queries) and emits
                     reasoning steps where every claim carries record pointers.
                     No evidence -> the step does not render. (An LLM may narrate
                     these verified facts; see narrate() — the evidence itself
                     never depends on the LLM.)
  2. Human override – a reviewer approves OR reverses the recommendation, with a
                     MANDATORY reason + identity. The human is the final authority.
  3. Parallel ledger – append-only. The agent's recommendation and the human's
                     decision are stored as SEPARATE records linked by case_id,
                     never overwriting each other. This is the accountability spine.
  4. Precedent      – for a new case, retrieve the most similar RESOLVED cases from
                     the ledger (GNN-embedding nearest neighbours) and surface them
                     WITH provenance. Informs the human; never auto-decides.

Run:  PYTHONPATH=src python3 src/review_layer.py
"""

import json, time, os
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from train_gnn import SAGE

torch.manual_seed(0)
LEDGER = "out/decision_ledger.jsonl"
BAND_LOW, BAND_HIGH = 0.35, 0.75


# ---------------------------------------------------------------- model + scores
def train_and_score():
    d = np.load("data/graph.npz", allow_pickle=True)
    x = torch.tensor(d["x"]); x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = torch.tensor(d["y"])
    ei = torch.tensor(d["edge_index"]); ei = torch.cat([ei, ei.flip(0)], 1)

    model = SAGE(x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    w = torch.tensor([1.0, float((y == 0).sum()) / max(int((y == 1).sum()), 1)],
                     dtype=torch.float32)
    for _ in range(150):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(x, ei), y, weight=w); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        h = F.relu(model.c1(x, ei)); h = F.relu(model.c2(h, ei))   # embeddings
        prob = F.softmax(model.lin(h), 1)[:, 1]
    return prob.numpy(), h.numpy(), y.numpy()


# ---------------------------------------------------------------- investigator
def _rbi_check(action):
    """RBI natural-justice rule (post Rajesh Agarwal): no fraud classification
    without a human due-process step. Returns a reasoning step with a rule id."""
    if action in ("auto_flag", "review"):
        return {"claim": "Account cannot be auto-classified as fraud; a human "
                         "sign-off is required before any adverse classification.",
                "evidence": ["RBI-NJ-01 (natural justice / opportunity to be heard)"],
                "source": "compliance_rules.yaml"}
    return None


def investigate(node, prob, G, nodes, scores_map):
    """Assemble evidence-cited reasoning steps + a recommendation for one node."""
    steps, pattern = [], "unclear"
    score = float(prob[node])
    nbrs = list(G.successors(node)) + list(G.predecessors(node))

    # --- device-sharing evidence ---
    dev = nodes.loc[nodes.node_id == node, "device"].iloc[0]
    shared = nodes[nodes.device == dev]["node_id"].tolist()
    if len(shared) >= 5:
        pattern = "device"
        steps.append({"claim": f"Shares one device fingerprint with {len(shared)-1} "
                               "other accounts — abnormal for independent users.",
                      "evidence": [f"device={int(dev)}", f"accounts={shared[:8]}"],
                      "source": "nodes.csv"})

    # --- cycle evidence (closed money loop through this node) ---
    try:
        cyc = nx.find_cycle(G, source=node)
        if 2 <= len(cyc) <= 8:
            pattern = "cycle" if pattern == "unclear" else pattern
            steps.append({"claim": f"Sits on a closed money-flow loop of length "
                                   f"{len(cyc)} (layering signature).",
                          "evidence": [f"cycle_edges={[(u,v) for u,v,*_ in cyc]}"],
                          "source": "edges.csv"})
    except nx.NetworkXNoCycle:
        pass

    # --- fanout evidence (a hub distributing to many) ---
    for h in nbrs:
        if G.out_degree(h) >= 8:
            pattern = "fanout" if pattern == "unclear" else pattern
            tgts = list(G.successors(h))
            steps.append({"claim": f"Receives from a hub that fans out to "
                                   f"{G.out_degree(h)} accounts (bust-out signature).",
                          "evidence": [f"hub={h}", f"sample_targets={tgts[:8]}"],
                          "source": "edges.csv"})
            break

    # --- collusion evidence (dense bipartite block, near-zero triangles) ---
    ego = nx.ego_graph(G.to_undirected(), node, radius=2)
    if ego.number_of_nodes() >= 8:
        tri = sum(nx.triangles(ego).values())
        dens = nx.density(ego)
        if tri == 0 and dens > 0.12:
            pattern = "collusion" if pattern == "unclear" else pattern
            steps.append({"claim": f"Embedded in a dense bipartite block "
                                   f"({ego.number_of_nodes()} accounts, density "
                                   f"{dens:.2f}, 0 triangles) — buyer/seller "
                                   "collusion signature invisible to per-row models.",
                          "evidence": [f"block_nodes={list(ego.nodes)[:10]}"],
                          "source": "graph structure"})

    steps.append({"claim": f"GNN risk score {score:.2f}.",
                  "evidence": [f"node={node}", f"score={score:.4f}"],
                  "source": "GraphSAGE"})

    action = ("auto_flag" if score >= BAND_HIGH else
              "review" if score >= BAND_LOW else "auto_clear")
    rbi = _rbi_check(action)
    if rbi:
        steps.append(rbi)
        action = "review"          # NJ rule forces a human step before flagging

    steps = [s for s in steps if s.get("evidence")]      # no evidence -> no step
    return {"recommendation": action, "pattern": pattern,
            "risk_score": round(score, 4), "reasoning_steps": steps}


def _deterministic_narrative(case):
    lines = [f"- {s['claim']}  [{', '.join(map(str, s['evidence']))}]"
             for s in case["reasoning_steps"]]
    return (f"Recommendation: {case['recommendation'].upper()} "
            f"(pattern: {case['pattern']})\n" + "\n".join(lines))


def narrate(case, use_llm=True):
    """Gemini Flash rephrases ONLY the verified evidence — invents nothing.
    Falls back to deterministic text if key/package/network missing."""
    if not use_llm or not os.environ.get("GEMINI_API_KEY"):
        return _deterministic_narrative(case)
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        evidence = "\n".join(
            f"- {s['claim']}  EVIDENCE[{', '.join(map(str, s['evidence']))}]"
            for s in case["reasoning_steps"])
        prompt = (
            "You are a fraud-analyst assistant writing a concise case summary for a "
            "human reviewer. Use ONLY the evidence provided — do not invent facts, "
            "accounts, amounts or names. Keep every statement tied to its EVIDENCE tag. "
            "Close with the recommended action and note that a human must sign off "
            "before any fraud classification.\n\n"
            f"Recommendation: {case['recommendation'].upper()}\n"
            f"Suspected pattern: {case['pattern']}\n"
            f"GNN risk score: {case['risk_score']}\n\n"
            f"Verified evidence:\n{evidence}")
        model = genai.GenerativeModel("gemini-3.6-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return _deterministic_narrative(case) + f"\n(LLM narration unavailable: {e})"
# ---------------------------------------------------------------- parallel ledger
def append(rec):
    os.makedirs("out", exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def open_case(case_id, node, case, embedding):
    """Append the AGENT record. Never edited afterwards."""
    append({"type": "recommendation", "case_id": case_id, "ts": time.time(),
            "node_id": int(node), "model_score": case["risk_score"],
            "pattern": case["pattern"], "recommendation": case["recommendation"],
            "reasoning_steps": case["reasoning_steps"],
            "embedding": [round(float(v), 4) for v in embedding]})


def resolve_case(case_id, node, decision, reason, reviewer):
    """Append the HUMAN record — SEPARATE row, parallel to the agent's. Reason
    is mandatory; the recommendation is preserved untouched for accountability."""
    if not reason or not reason.strip():
        raise ValueError("override/approval requires a reason (accountability).")
    append({"type": "decision", "case_id": case_id, "ts": time.time(),
            "node_id": int(node), "human_decision": decision,
            "reason": reason, "reviewer": reviewer})


def load_ledger():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]


def metrics():
    """Agreement / override rate by folding the parallel records on case_id."""
    recs = load_ledger()
    agent = {r["case_id"]: r for r in recs if r["type"] == "recommendation"}
    human = {r["case_id"]: r for r in recs if r["type"] == "decision"}
    both = [(agent[c], human[c]) for c in human if c in agent]
    if not both:
        return {}
    def canon(x): return "flag" if x in ("auto_flag", "review", "flag") else "clear"
    overrides = sum(canon(a["recommendation"]) != canon(h["human_decision"])
                    for a, h in both)
    return {"resolved": len(both), "override_rate": round(overrides / len(both), 2),
            "agreement_rate": round(1 - overrides / len(both), 2)}


# ---------------------------------------------------------------- precedent (memory)
def precedent(query_emb, exclude_node, k=3):
    """Retrieve most similar RESOLVED cases from the ledger, with provenance.
    Suggestion only — it is surfaced to the human, never auto-applied."""
    recs = load_ledger()
    agent = {r["case_id"]: r for r in recs if r["type"] == "recommendation"}
    resolved = [r for r in recs if r["type"] == "decision" and r["case_id"] in agent]
    q = np.array(query_emb); scored = []
    for h in resolved:
        a = agent[h["case_id"]]
        if a["node_id"] == exclude_node:
            continue
        e = np.array(a["embedding"])
        sim = float(q @ e / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-9))
        scored.append((sim, a, h))
    scored.sort(reverse=True, key=lambda t: t[0])
    return [{"similarity": round(s, 3), "past_node": a["node_id"],
             "past_pattern": a["pattern"], "human_decision": h["human_decision"],
             "reason": h["reason"], "reviewer": h["reviewer"], "case_id": h["case_id"]}
            for s, a, h in scored[:k]]


# ---------------------------------------------------------------- demo
def main():
    open(LEDGER, "w").close()                       # fresh ledger for the demo
    prob, emb, y = train_and_score()
    edges = pd.read_csv("data/edges.csv")
    nodes = pd.read_csv("data/nodes.csv")
    G = nx.from_pandas_edgelist(edges, "src", "dst", create_using=nx.DiGraph())
    for n in nodes.node_id:
        if n not in G: G.add_node(n)
    scores_map = dict(enumerate(prob))

    # cases the triage gate sends to review/flag, highest risk first
    flagged = [n for n in nodes.node_id if prob[n] >= BAND_LOW]
    flagged.sort(key=lambda n: prob[n], reverse=True)

    print("=" * 74)
    print("REVIEW-AND-MEMORY LAYER  (runs after the GNN triage gate)")
    print("=" * 74)

    # --- Case A: agent investigates, human APPROVES (confirms fraud) ---
    a = flagged[0]
    caseA = investigate(a, prob, G, nodes, scores_map)
    open_case("CASE-A", a, caseA, emb[a])
    print(f"\n[CASE-A] node {a} — investigator report:\n" + narrate(caseA))
    print(f"  precedent found: {precedent(emb[a], a)}  (none yet — cold start)")
    resolve_case("CASE-A", a, "flag", "Confirmed cycle + shared device; "
                 "matches known layering ring.", "analyst_shreya")
    print("  -> human APPROVED (flag). recommendation + decision stored in parallel.")

    # --- Case B: a SIMILAR case later — precedent should now surface CASE-A ---
    b = next(n for n in flagged[1:]
             if caseA["pattern"] != "unclear"
             and investigate(n, prob, G, nodes, scores_map)["pattern"] == caseA["pattern"])
    caseB = investigate(b, prob, G, nodes, scores_map)
    open_case("CASE-B", b, caseB, emb[b])
    print(f"\n[CASE-B] node {b} — investigator report:\n" + narrate(caseB))
    prec = precedent(emb[b], b)
    print("  precedent surfaced to the human:")
    for p in prec:
        print(f"    ~{p['similarity']} similar -> {p['reviewer']} decided "
              f"'{p['human_decision']}' because: {p['reason']}")
    resolve_case("CASE-B", b, "flag", "Same pattern as CASE-A precedent; "
                 "confirmed.", "analyst_shreya")
    print("  -> human decided WITH precedent in hand (faster, consistent).")

    # --- Case C: human OVERRIDES the agent (recommend flag, human clears) ---
    c = flagged[2]
    caseC = investigate(c, prob, G, nodes, scores_map)
    open_case("CASE-C", c, caseC, emb[c])
    print(f"\n[CASE-C] node {c} — agent recommended: {caseC['recommendation'].upper()}")
    resolve_case("CASE-C", c, "clear",
                 "Legit group-buying merchant cluster; false positive.", "analyst_ravi")
    print("  -> human OVERRODE to CLEAR (reason mandatory). agent record preserved.")

    print("\n" + "-" * 74)
    print("ACCOUNTABILITY METRICS (folded from parallel records):", metrics())
    print("\nSample ledger entries (agent + human kept side by side):")
    for r in load_ledger()[:2] + load_ledger()[-1:]:
        print("  ", json.dumps({k: r[k] for k in list(r)[:6]}))


if __name__ == "__main__":
    main()
