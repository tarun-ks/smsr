#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMSR certificate recomputation, sanity checks, and figure regeneration.

What this script does
---------------------
1. Recomputes p_clean and the Theorem-2 certificate delta for every (t, m) in
   Table I, and prints them next to the values currently printed in the paper so
   the mismatches are visible.
2. Re-derives the parameter-selection numbers in Sec VII-F and Fig 3 captions,
   and reports the SMALLEST m that actually reaches a target delta for each t
   (this is where the "m=10 gives delta<=0.10 at t=1" claim breaks).
3. Regenerates Fig 3(a) [delta vs adversary budget t] and Fig 3(b) [delta vs
   pool size m] as vector PDFs you can drop straight into the paper.
4. Provides a Monte-Carlo simulator of the 3-way verdict vote. The Theorem-2
   bound is the worst case (LLM always takes the bait on a contaminated run).
   If Table I was produced by simulation with an implicit "take rate" < 1, this
   lets you (a) see what take-rate reproduces each printed value and (b) decide
   whether to report the *bound* (Thm 2) or a *simulated* delta -- but you must
   then label them distinctly. Run with --reconcile to print this.

Run:
    python3 smsr_certificate.py                 # table + figures
    python3 smsr_certificate.py --reconcile     # also explain Table I numbers
    python3 smsr_certificate.py --latex         # also emit a LaTeX table body

Dependencies: numpy, matplotlib (scipy optional, not required).
"""

from math import comb, ceil
import argparse

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False


# ----------------------------------------------------------------------------- 
# Default SMSR parameters (match the paper)
# -----------------------------------------------------------------------------
M_DEFAULT = 20      # over-fetch pool size
K_DEFAULT = 5       # entries sampled per ablation run
N_RUNS_DEFAULT = 7  # ablation runs

# Values exactly as printed in Table I of the current draft (for comparison only).
# Format: t -> (p_clean_printed, delta_printed)
PAPER_TABLE = {
    1:  (0.750, 0.071),
    2:  (0.579, 0.294),
    3:  (0.399, 0.584),
    5:  (0.194, 0.934),
    10: (0.016, 1.000),
}

# Empirical ASR points you currently have (single run per scenario).
# REPLACE these with the mean +/- CI values once Experiment A (repetitions) is done.
# t -> empirical absolute ASR (fraction)
EMPIRICAL_ASR = {
    1: 0.133,   # authenticated, direct, t=1  (2/15)
    3: 0.933,   # authenticated, direct, t=3  (14/15)
}


# -----------------------------------------------------------------------------
# Core certificate math
# -----------------------------------------------------------------------------
def p_clean(m: int, k: int, t: int) -> float:
    """Probability a single uniform k-sample (without replacement) from m
    candidates, of which t are adversarial, contains ZERO adversarial entries.
    Hypergeometric: C(m-t, k) / C(m, k)."""
    t = min(t, m)
    if k > m:
        raise ValueError("k cannot exceed m")
    if m - t < k:
        return 0.0
    return comb(m - t, k) / comb(m, k)


def delta_bound(m: int, k: int, n_runs: int, t: int) -> float:
    """Theorem-2 certificate: probability the majority verdict is malicious,
    upper-bounded by P(X >= ceil(n_runs/2)) for X ~ Binomial(n_runs, 1 - p_clean).
    This is the WORST CASE: every contaminated run is assumed to yield 'malicious'."""
    p = p_clean(m, k, t)
    q = 1.0 - p
    thr = ceil(n_runs / 2)
    return float(sum(comb(n_runs, i) * (q ** i) * (p ** (n_runs - i))
                     for i in range(thr, n_runs + 1)))


def smallest_m_for_target(k: int, n_runs: int, t: int, target: float,
                          m_max: int = 200) -> int | None:
    """Smallest pool size m (>k) achieving delta_bound <= target for given t."""
    for m in range(max(k + 1, t + k), m_max + 1):
        if delta_bound(m, k, n_runs, t) <= target:
            return m
    return None


# -----------------------------------------------------------------------------
# Monte-Carlo of the 3-way verdict vote (for reconciling Table I)
# -----------------------------------------------------------------------------
def simulate_delta(m: int, k: int, n_runs: int, t: int,
                   p_take: float = 1.0, p_correct_clean: float = 1.0,
                   n_sims: int = 200_000, seed: int = 0) -> float:
    """Monte-Carlo estimate of P(plurality verdict == malicious). Vectorised.

    Per ablation run we draw k of m candidates (t adversarial):
      * if >=1 adversarial sampled: verdict = MALICIOUS w.p. p_take, else NEITHER
      * if 0 adversarial sampled  : verdict = CORRECT  w.p. p_correct_clean, else NEITHER
    Attack 'succeeds' if MALICIOUS is the unique plurality verdict across runs.

    Whether a run is contaminated is Bernoulli(1 - p_clean) -- exact, since only
    '>=1 adversarial sampled' matters. p_take = p_correct_clean = 1.0 reproduces
    the Theorem-2 worst-case bound. Lowering p_take models an LLM that does not
    always take the bait and yields a *smaller* simulated delta -- one candidate
    explanation for the paper's lower-than-bound Table I values.
    """
    rng = np.random.default_rng(seed)
    q = 1.0 - p_clean(m, k, t)                       # P(run contaminated)
    contaminated = rng.random((n_sims, n_runs)) < q
    u = rng.random((n_sims, n_runs))
    mal = contaminated & (u < p_take)
    cor = (~contaminated) & (u < p_correct_clean)
    c_mal = mal.sum(axis=1)
    c_cor = cor.sum(axis=1)
    c_nei = n_runs - c_mal - c_cor
    succ = (c_mal > c_cor) & (c_mal > c_nei)
    return float(succ.mean())


def take_rate_reproducing(target_delta: float, m: int, k: int, n_runs: int, t: int,
                          n_sims: int = 120_000) -> float | None:
    """Find p_take in [0,1] whose simulated delta ~ target_delta (coarse search).
    Helps explain a printed Table I value if it was simulated."""
    best, best_gap = None, 1e9
    for p_take in np.linspace(0.0, 1.0, 21):
        d = simulate_delta(m, k, n_runs, t, p_take=p_take, n_sims=n_sims, seed=1)
        gap = abs(d - target_delta)
        if gap < best_gap:
            best, best_gap = float(p_take), gap
    return best if best_gap < 0.05 else None


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_table(m=M_DEFAULT, k=K_DEFAULT, n_runs=N_RUNS_DEFAULT):
    print(f"\n=== Corrected Table I  (m={m}, k={k}, n_runs={n_runs}) ===")
    print(f"{'t':>3} | {'p_clean':>8} {'delta':>8} || {'paper p':>8} {'paper d':>8} | flags")
    print("-" * 64)
    for t, (pp, pd) in PAPER_TABLE.items():
        p = p_clean(m, k, t)
        d = delta_bound(m, k, n_runs, t)
        fp = "" if abs(p - pp) < 2e-3 else "p!"
        fd = "" if abs(d - pd) < 1e-2 else "delta!"
        flag = " ".join(x for x in (fp, fd) if x)
        print(f"{t:>3} | {p:>8.3f} {d:>8.3f} || {pp:>8.3f} {pd:>8.3f} | {flag}")
    print("\nLegend: 'p!' = p_clean mismatch vs paper; 'delta!' = delta mismatch vs paper.")


def print_param_guide(k=K_DEFAULT, n_runs=N_RUNS_DEFAULT):
    print(f"\n=== Parameter-selection re-derivation (k={k}, n_runs={n_runs}) ===")
    for t in (1, 2, 3):
        m10 = delta_bound(10, k, n_runs, t)
        m20 = delta_bound(20, k, n_runs, t)
        m35 = delta_bound(35, k, n_runs, t)
        m_for_010 = smallest_m_for_target(k, n_runs, t, 0.10)
        m_for_005 = smallest_m_for_target(k, n_runs, t, 0.05)
        print(f"t={t}: delta(m=10)={m10:.3f}  delta(m=20)={m20:.3f}  delta(m=35)={m35:.3f}"
              f"  | smallest m for delta<=0.10: {m_for_010}"
              f"  | for delta<=0.05: {m_for_005}")
    print("\nNote: the draft claims 'for t=1, m=10 achieves delta<=0.10'. As shown,"
          "\n      m=10 gives delta=0.500 at t=1; you actually need m>=18 (and you use m=20).")


def emit_latex(m=M_DEFAULT, k=K_DEFAULT, n_runs=N_RUNS_DEFAULT):
    print(f"\n=== LaTeX table body (m={m}, k={k}, n_runs={n_runs}) ===")
    print("% t & p_clean & delta & Empirical ASR")
    for t in PAPER_TABLE:
        p = p_clean(m, k, t)
        d = delta_bound(m, k, n_runs, t)
        emp = EMPIRICAL_ASR.get(t)
        emp_s = f"{emp*100:.1f}\\%" if emp is not None else "---"
        print(f"{t} & {p:.3f} & {d:.3f} & {emp_s} \\\\")


def reconcile_table(m=M_DEFAULT, k=K_DEFAULT, n_runs=N_RUNS_DEFAULT):
    print(f"\n=== Reconciling Table I via 3-way-vote Monte-Carlo (m={m}, k={k}, n={n_runs}) ===")
    print("Worst-case (p_take=1.0) should equal the Theorem-2 bound:")
    for t in PAPER_TABLE:
        d_bound = delta_bound(m, k, n_runs, t)
        d_sim = simulate_delta(m, k, n_runs, t, p_take=1.0, n_sims=120_000, seed=2)
        print(f"  t={t}: bound={d_bound:.3f}  sim(p_take=1.0)={d_sim:.3f}")
    print("\nIf the paper's (lower) delta came from simulation, here is the implied"
          "\n'take rate' p_take that reproduces each printed value (None = no good fit):")
    for t, (_, pd) in PAPER_TABLE.items():
        if pd >= 0.999:
            continue
        pt = take_rate_reproducing(pd, m, k, n_runs, t)
        print(f"  t={t}: printed delta={pd:.3f}  -> implied p_take ~ {pt}")
    print("\nTakeaway: either report the Theorem-2 BOUND in Table I (recommended for a"
          "\n'certified' paper), or, if you report a simulated value, label the column"
          "\n'simulated delta (p_take=...)' and state the bound separately. Do not mix them.")


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def make_fig3a(out_pdf="fig3a_certificate_vs_budget.pdf",
               out_png="fig3a_certificate_vs_budget.png",
               k=K_DEFAULT, n_runs=N_RUNS_DEFAULT):
    if not HAVE_MPL:
        print("matplotlib unavailable; skipping fig3a")
        return
    ts = list(range(0, 15))
    d20 = [delta_bound(20, k, n_runs, t) for t in ts]
    d15 = [delta_bound(15, k, n_runs, t) for t in ts]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(ts, d20, marker="o", ms=3, label="m=20 pool")
    ax.plot(ts, d15, marker="s", ms=3, label="m=15 pool")
    if EMPIRICAL_ASR:
        ex = sorted(EMPIRICAL_ASR)
        ey = [EMPIRICAL_ASR[t] for t in ex]
        ax.scatter(ex, ey, color="red", zorder=5, label="Empirical ASR (single run)")
        for t in ex:
            ax.annotate(f"{EMPIRICAL_ASR[t]*100:.0f}%", (t, EMPIRICAL_ASR[t]),
                        textcoords="offset points", xytext=(5, 4), color="red", fontsize=8)
    ax.axhline(0.10, ls="--", lw=1, color="gray", label=r"$\delta=0.10$ target")
    ax.set_xlabel("t (adversarial signed injections)")
    ax.set_ylabel(r"$\delta$ (certified wrong-output prob.)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(a) Certificate bound vs. adversary budget", fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"wrote {out_pdf} / {out_png}")


def make_fig3b(out_pdf="fig3b_effect_of_m.pdf",
               out_png="fig3b_effect_of_m.png",
               k=K_DEFAULT, n_runs=N_RUNS_DEFAULT):
    if not HAVE_MPL:
        print("matplotlib unavailable; skipping fig3b")
        return
    ms = list(range(k + 1, 41))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for t, style in zip((1, 2, 3), ("-", "--", ":")):
        dd = [delta_bound(m, k, n_runs, t) for m in ms]
        ax.plot(ms, dd, style, label=f"t={t}")
    ax.axhline(0.10, ls="--", lw=1, color="gray", label=r"$\delta=0.10$")
    # mark the true crossing for t=1 (m=18), contradicting the draft's m=10 claim
    m1 = smallest_m_for_target(k, n_runs, 1, 0.10)
    if m1:
        ax.axvline(m1, color="C0", lw=0.8, alpha=0.5)
        ax.annotate(f"t=1 reaches 0.10 at m={m1}", (m1, 0.5),
                    textcoords="offset points", xytext=(4, 0), fontsize=7, color="C0")
    ax.set_xlabel("m (over-fetch pool size)")
    ax.set_ylabel(r"$\delta$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b) Effect of pool size m (k=5, n=7)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"wrote {out_pdf} / {out_png}")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SMSR certificate recompute + figures")
    ap.add_argument("--reconcile", action="store_true",
                    help="explain Table I via 3-way-vote Monte-Carlo")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table body")
    ap.add_argument("--no-figs", action="store_true", help="skip figure generation")
    args = ap.parse_args()

    print_table()
    print_param_guide()

    # Corollary 1 sanity
    d1 = delta_bound(M_DEFAULT, K_DEFAULT, N_RUNS_DEFAULT, 1)
    emp1 = EMPIRICAL_ASR.get(1)
    print("\n=== Corollary 1 sanity ===")
    print(f"t=1 certificate delta = {d1*100:.1f}%  (per-query upper bound)")
    if emp1 is not None:
        print(f"empirical absolute ASR = {emp1*100:.1f}%  ({round(emp1*15)}/15 scenarios)")
        print(f"  is {emp1*100:.1f}% <= {d1*100:.1f}% ?  {emp1 <= d1}  "
              f"(the draft says '13.3% is below 7.1%', which is FALSE)")
        print("  Correct framing: 2/15 is statistically consistent with a per-query")
        print("  rate of 7.1% within the (wide) CI for n=15; do NOT claim it is 'below' delta.")

    if args.latex:
        emit_latex()
    if args.reconcile:
        reconcile_table()
    if not args.no_figs:
        make_fig3a()
        make_fig3b()


if __name__ == "__main__":
    main()
