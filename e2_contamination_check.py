#!/usr/bin/env python3
"""
E2 — Contamination-rate check for SMSR Component 2 (zero API cost).

Backs the paper's claim (Sec. VII-C) that the uniform-independent-sampling
premise of Theorem 2 holds: because each ablation run samples k of m candidates
uniformly without replacement, the probability a run is contaminated is exactly
1 - p_clean = 1 - C(m-t,k)/C(m,k), INDEPENDENT of the similarity rank of the
adversarial entries. This script replicates the sampler in smsr.py
(SMSRDefence.retrieve) and compares the empirical contaminated-run fraction to
the theoretical value across the evaluated pools.

Run:  python3 e2_contamination_check.py
Writes e2_contamination.json and prints a table.
"""
from math import comb
import json
import numpy as np

K = 5
N_TRIALS = 300_000   # Monte-Carlo queries per (store, t, n_runs)


def p_clean(m, t, k=K):
    return comb(m - t, k) / comb(m, k) if m - t >= k else 0.0


def simulate(m, t, n_runs, k=K, n_trials=N_TRIALS, seed=0):
    """Replicates smsr.py: per run, rng.choice(m, size=k, replace=False);
    a run is contaminated iff it contains >=1 of the t adversarial entries
    (indices 0..t-1). Returns (empirical_contam_fraction, theoretical 1-p_clean)."""
    rng = np.random.default_rng(seed)
    contaminated = 0
    total = n_runs * n_trials
    for _ in range(n_trials):
        for _ in range(n_runs):
            idx = rng.choice(m, size=k, replace=False)
            contaminated += int(np.any(idx < t))
    return contaminated / total, 1.0 - p_clean(m, t, k)


def main():
    # Evaluated pools: small-store eval (m'=10+t) and production Tier-1 (m=20)
    configs = (
        [("eval", 10 + t, t, 5) for t in (1, 2, 3)] +
        [("tier1", 20, t, 5) for t in (1, 2, 3)]
    )
    rows = []
    print(f"{'store':<7}{'m':>4}{'t':>3}{'n_runs':>7}{'  emp_contam':>13}"
          f"{'  1-p_clean':>12}{'  |dev|':>9}")
    print("-" * 56)
    max_dev = 0.0
    for store, m, t, n_runs in configs:
        emp, thy = simulate(m, t, n_runs)
        dev = abs(emp - thy)
        max_dev = max(max_dev, dev)
        rows.append(dict(store=store, m=m, t=t, n_runs=n_runs,
                         empirical_contam=emp, theoretical=thy, abs_dev=dev))
        print(f"{store:<7}{m:>4}{t:>3}{n_runs:>7}{emp:>13.4f}{thy:>12.4f}{dev:>9.4f}")
    print("-" * 56)
    print(f"max |deviation| = {max_dev:.4f}  (paper reports <= 0.0006)")
    json.dump({"n_trials": N_TRIALS, "k": K, "max_abs_dev": max_dev, "rows": rows},
              open("e2_contamination.json", "w"), indent=2)
    print("wrote e2_contamination.json")


if __name__ == "__main__":
    main()
