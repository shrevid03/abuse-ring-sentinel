"""
5-seed canonical red-team evaluation.

For each run:
1. Hold out whole unseen rings.
2. Train GraphSAGE inductively on train-only edges.
3. Measure clean held-out ring recall.
4. Attack with a fresh structural perturbation.
5. Adversarially train on a DIFFERENT perturbation.
6. Evaluate hardened model on another fresh perturbation.

Run:
PYTHONPATH=src python src/red_team_seeds.py
"""

import numpy as np
import torch

import twin_eval_canonical as C
import red_team as RT
import red_team_canonical as RC


N_SEEDS = 5


def main():
    x, y, ei = C.load()
    xt = torch.tensor(C.norm(x))

    clean_scores = []
    attacked_scores = []
    hardened_scores = []

    print("=" * 78)
    print("CANONICAL RED-TEAM — inductive held-out rings, 5 seeds")
    print("=" * 78)

    for s in range(N_SEEDS):

        # Different held-out rings each run
        train, test, n_rings = C.holdout_ring_masks(
            y, ei, test_frac=0.3, seed=s
        )

        # ---------------- CLEAN MODEL ----------------
        torch.manual_seed(1000 + s)

        clean_model = RC.train_inductive(
            xt, ei, y, train
        )

        r_clean = RC.ring_recall(
            clean_model, xt, ei, y, test
        )

        # ---------------- ATTACK CLEAN MODEL ----------------
        # Fresh attack for evaluation
        attack_eval_seed = 100 + s

        ei_attacked = RT.attack(
            ei, y, seed=attack_eval_seed
        )

        r_attacked = RC.ring_recall(
            clean_model, xt, ei_attacked, y, test
        )

        # ---------------- ADVERSARIAL HARDENING ----------------
        # Different attack from evaluation attack
        attack_train_seed = 200 + s

        ei_hard_train = RT.attack(
            ei, y, seed=attack_train_seed
        )

        torch.manual_seed(2000 + s)

        hardened_model = RC.train_inductive(
            xt, ei_hard_train, y, train
        )

        # ---------------- FRESH ATTACK ----------------
        # Never evaluate on the exact perturbation used for hardening
        attack_test_seed = 300 + s

        ei_hard_eval = RT.attack(
            ei, y, seed=attack_test_seed
        )

        r_hardened = RC.ring_recall(
            hardened_model, xt, ei_hard_eval, y, test
        )

        clean_scores.append(r_clean)
        attacked_scores.append(r_attacked)
        hardened_scores.append(r_hardened)

        print(
            f"seed {s}: "
            f"clean={r_clean:.3f}  "
            f"attacked={r_attacked:.3f}  "
            f"hardened={r_hardened:.3f}  "
            f"| test fraud={(y[test] == 1).sum()}"
        )

    clean_scores = np.array(clean_scores)
    attacked_scores = np.array(attacked_scores)
    hardened_scores = np.array(hardened_scores)

    print("\n" + "-" * 78)
    print("MEAN ± STD")
    print("-" * 78)

    print(
        f"clean     : "
        f"{clean_scores.mean():.3f} ± {clean_scores.std():.3f}"
    )

    print(
        f"attacked  : "
        f"{attacked_scores.mean():.3f} ± {attacked_scores.std():.3f}"
    )

    print(
        f"hardened  : "
        f"{hardened_scores.mean():.3f} ± {hardened_scores.std():.3f}"
    )

    print("\nSummary:")
    print(
        f"{clean_scores.mean():.2f} "
        f"→ {attacked_scores.mean():.2f} "
        f"→ {hardened_scores.mean():.2f}"
    )


if __name__ == "__main__":
    main()
