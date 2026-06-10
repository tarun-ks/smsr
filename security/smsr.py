"""
smsr.py
=======
Signed Memory with Smoothed Retrieval (SMSR) — the certified defence.

Two components:
  Component 1 — Provenance filter (hard boundary)
    At retrieval time, filter all memories to HMAC-verified entries only.
    An adversary who cannot forge HMAC tags cannot inject retrievable memories.
    Certificate: forging one tag succeeds with prob ≤ 2^{-256} (HMAC-SHA256).

  Component 2 — Randomised ablation (authenticated-adversary robustness)
    For the case where the attacker is a legitimate user who CAN write signed
    memories: retrieve top-m verified candidates, randomly sample k < m,
    run the LLM n_runs times with different random subsets, return majority vote.

    Certificate (hypergeometric):
      Let t = number of adversarial (but signed) entries in top-m candidates.
      P(≥ r adversarial in one sample of k from m) = 1 - ∑_{j=0}^{r-1} H(j; m, t, k)
      where H is the hypergeometric PMF.
      Over n_runs majority-vote runs the clean-majority probability is computable.
"""

import numpy as np
from math import comb
from collections import Counter


def hypergeometric_cdf(m: int, t: int, k: int, r: int) -> float:
    """P(≤ r adversarial in sample of k from population of m with t adversarial)."""
    total = comb(m, k)
    if total == 0:
        return 1.0
    return sum(
        comb(t, j) * comb(m - t, k - j)
        for j in range(min(r, t, k) + 1)
        if k - j <= m - t
    ) / total


def compute_certificate(m: int, t: int, k: int, r: int, n_runs: int) -> dict:
    """
    Compute the (t, delta) certificate for SMSR Component 2.

    Returns dict with:
      p_clean_single   : P(< r adversarial in one run's k-sample)
      p_wrong_majority : upper bound on P(majority vote is wrong)
      delta            : the certified delta (= p_wrong_majority)
    """
    p_clean = hypergeometric_cdf(m, t, k, r - 1)          # P(< r adversarial)
    p_dirty = 1.0 - p_clean                                # P(≥ r adversarial)
    # P(majority wrong) ≤ P(majority of n_runs runs have ≥ r adversarial)
    # By binomial: P(X ≥ ceil(n_runs/2)) where X ~ Bin(n_runs, p_dirty)
    from math import floor
    majority_threshold = n_runs // 2 + 1
    p_wrong_majority = sum(
        comb(n_runs, x) * (p_dirty ** x) * ((1 - p_dirty) ** (n_runs - x))
        for x in range(majority_threshold, n_runs + 1)
    )
    return {
        "m": m, "t": t, "k": k, "r": r, "n_runs": n_runs,
        "p_clean_single"  : p_clean,
        "p_dirty_single"  : p_dirty,
        "p_wrong_majority": p_wrong_majority,
        "delta"           : p_wrong_majority,
    }


class SMSRDefence:
    """
    Wraps a MemoryStore + HMACProvenance to provide certified retrieval.

    Modes:
      "none"        — no defence (undefended baseline)
      "c1"          — Component 1 only: verified memories only
      "c1c2"        — Component 1 + Component 2: verified + randomised ablation
    """

    def __init__(
        self,
        provenance,
        mode: str = "c1c2",
        m: int = 20,    # over-fetch candidates
        k: int = 5,     # memories shown to LLM per run
        n_runs: int = 5 # number of ablation runs for majority vote
    ):
        assert mode in ("none", "c1", "c1c2"), f"Unknown mode: {mode}"
        self.provenance = provenance
        self.mode       = mode
        self.m          = m
        self.k          = k
        self.n_runs     = n_runs

    def retrieve(self, store, query: str) -> list[list[dict]]:
        """
        Returns a list of memory lists — one per LLM run.
        For mode "none" / "c1": returns a single list (one run).
        For mode "c1c2": returns n_runs lists (randomised ablation).
        """
        if self.mode == "none":
            mems = store.retrieve(query, k=self.k, verified_only=False)
            return [mems]

        if self.mode == "c1":
            mems = store.retrieve(query, k=self.k, verified_only=True)
            return [mems]

        # c1c2: retrieve top-m verified, then sample k n_runs times
        candidates = store.retrieve(query, k=self.m, verified_only=True)
        if len(candidates) <= self.k:
            # not enough candidates to ablate — return as-is
            return [candidates] * self.n_runs

        rng = np.random.default_rng()
        runs = []
        for _ in range(self.n_runs):
            idx = rng.choice(len(candidates), size=self.k, replace=False)
            runs.append([candidates[i] for i in sorted(idx)])
        return runs

    def certificate(self, t_adversarial: int, r: int = 1) -> dict:
        """
        Compute the formal (t, delta) certificate for Component 2.
        t_adversarial: assumed number of signed adversarial entries in top-m.
        r: minimum adversarial entries in sample needed for attack to succeed.
        """
        if self.mode != "c1c2":
            return {"mode": self.mode, "note": "certificate only for c1c2 mode"}
        return compute_certificate(self.m, t_adversarial, self.k, r, self.n_runs)
