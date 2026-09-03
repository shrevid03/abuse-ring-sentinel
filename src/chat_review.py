"""
chat_review.py — analyst copilot. Ask questions about a flagged case;
Gemini answers grounded ONLY in that case's real evidence.
Run:  PYTHONPATH=src python src/chat_review.py
"""
import os
import google.generativeai as genai
from review_layer import train_and_score, investigate, precedent
import numpy as np, pandas as pd, networkx as nx

MODEL = "gemini-3.6-flash"   # the one that worked for you

def build_case():
    prob, emb, y = train_and_score()
    nodes = pd.read_csv("data/nodes.csv")
    edges = pd.read_csv("data/edges.csv")
    G = nx.from_pandas_edgelist(edges, "src", "dst", create_using=nx.DiGraph())
    for n in nodes.node_id:
        if n not in G: G.add_node(n)
    flagged = [n for n in nodes.node_id if prob[n] >= 0.35]
    flagged.sort(key=lambda n: prob[n], reverse=True)
    node = flagged[0]
    case = investigate(node, prob, G, nodes, {i: p for i, p in enumerate(prob)})
    return node, case

def evidence_text(case):
    return "\n".join(f"- {s['claim']}  EVIDENCE[{', '.join(map(str, s['evidence']))}]"
                     for s in case["reasoning_steps"])

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY first:  export GEMINI_API_KEY=your_key")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    node, case = build_case()
    ev = evidence_text(case)
    print(f"\n=== Analyst copilot — reviewing flagged account {node} ===")
    print(f"(recommendation: {case['recommendation'].upper()}, pattern: {case['pattern']})")
    print("Ask about this case. Type 'quit' to exit.\n")

    system = (
        "You are a fraud-analyst copilot helping a human reviewer decide on ONE "
        "flagged account. Answer ONLY from the evidence below. If asked something "
        "the evidence doesn't cover, say you don't have that information. Never invent "
        "accounts, amounts, or facts. Be concise.\n\n"
        f"FLAGGED ACCOUNT: node {node}\n"
        f"RECOMMENDATION: {case['recommendation'].upper()} (pattern: {case['pattern']})\n"
        f"EVIDENCE:\n{ev}\n"
    )
    model = genai.GenerativeModel(MODEL, system_instruction=system)
    chat = model.start_chat(history=[])

    while True:
        q = input("you > ").strip()
        if q.lower() in ("quit", "exit", "q", ""):
            print("ending review."); break
        try:
            r = chat.send_message(q)
            print("copilot >", r.text.strip(), "\n")
        except Exception as e:
            print("copilot > (error:", e, ")\n")

if __name__ == "__main__":
    main()