"""Canonical defensive red-team — inductive + held-out unseen rings (consistent
with twin_eval_canonical). ring-recall: clean -> attack -> hardened.
Run: PYTHONPATH=src python src/red_team_canonical.py"""
import numpy as np, torch, torch.nn.functional as F
import twin_eval_canonical as C
import red_team as RT
from train_gnn import SAGE

torch.manual_seed(0)

def train_inductive(xt, ei_np, y, train):
    tr_ei = C.train_only_edges(ei_np, train)
    tr_und = torch.cat([torch.tensor(tr_ei), torch.tensor(tr_ei).flip(0)], 1)
    m = torch.tensor(train)
    w = torch.tensor([1.0, float((y[train]==0).sum())/max(int((y[train]==1).sum()),1)], dtype=torch.float32)
    model = SAGE(xt.size(1)); opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(xt, tr_und)[m], torch.tensor(y)[m], weight=w); loss.backward(); opt.step()
    return model

def ring_recall(model, xt, ei_np, y, test):
    und = torch.cat([torch.tensor(ei_np), torch.tensor(ei_np).flip(0)], 1)
    model.eval()
    with torch.no_grad(): pred = model(xt, und).argmax(1).numpy()
    m = (y==1) & test
    return float((pred[m]==1).mean())

def main():
    x, y, ei = C.load()
    xt = torch.tensor(C.norm(x))
    train, test, n = C.holdout_ring_masks(y, ei, 0.3, seed=7)
    model = train_inductive(xt, ei, y, train)
    r_clean = ring_recall(model, xt, ei, y, test)
    r_atk = ring_recall(model, xt, RT.attack(ei, y, seed=1), y, test)
    model_h = train_inductive(xt, RT.attack(ei, y, seed=2), y, train)
    r_hard = ring_recall(model_h, xt, RT.attack(ei, y, seed=3), y, test)
    print("="*64)
    print("RED-TEAM (canonical: inductive, held-out unseen rings) — ring-recall")
    print("="*64)
    print(f"  clean graph              : {r_clean:.2f}")
    print(f"  under structural perturbation    : {r_atk:.2f}   (delta {r_atk-r_clean:+.2f})")
    print(f"  after adversarial harden : {r_hard:.2f}   (recovered {r_hard-r_atk:+.2f})")
    print("-"*64)
    print("defense-only: closed sandbox, own detector, structural edits only")
    return r_clean, r_atk, r_hard

if __name__ == "__main__":
    main()
