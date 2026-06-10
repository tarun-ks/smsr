#!/usr/bin/env python3
"""
run_all_experiments.py
======================
Master runner — executes all remaining pre-submission experiments:

  E4   A-MemGuard-style consensus baseline          (~$1,  ~20 min)
  E6   Judge reliability (Haiku vs Sonnet kappa)    (~$1,  ~10 min)
  E7   Second LLM generality (Sonnet as agent)      (~$1,  ~10 min)
  E1   Remaining c1c2 reps: flooding/t=1,           (~$14, ~2h)
       direct/t=2, direct/t=3, flooding/t=2, flooding/t=3
  T1   Tier-1 large-store: 20 seeds → m=20,         (~$15, ~2.5h)
       re-run auth/direct/t=1/c1c2 to validate
       the production certificate δ=0.071

Skips any config that already has a result saved to disk.
Loads ANTHROPIC_API_KEY from a .env file or the environment.
"""

import sys, json, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(override=False)  # loads .env from the current directory if present

from agent.agent          import _get_client, _format_memories, SYSTEM_PROMPT
from security.provenance  import HMACProvenance
from security.smsr        import SMSRDefence
from attacks.minja        import ATTACK_SCENARIOS, AuthenticatedAttack
from eval.llm_judge       import judge as llm_judge
from run_final_experiments import (run_e4, run_e6, run_e7,
                                    majority_verdict_from_runs, wilson_ci,
                                    OUT_DIR)
from run_experiments      import build_seed_store

SCENARIOS = ATTACK_SCENARIOS[:15]

# ── helpers ───────────────────────────────────────────────────────────────────
def delta_bound(m, k, n, t):
    from math import comb, ceil
    if m - t < k: return 1.0
    pc = comb(m-t, k) / comb(m, k)
    q  = 1.0 - pc
    thr = ceil(n/2)
    return sum(comb(n, i)*(q**i)*(pc**(n-i)) for i in range(thr, n+1))


# ── E1 remaining ─────────────────────────────────────────────────────────────
def run_e1_remaining(client, n_reps=30):
    e1_path = OUT_DIR / "e1_repetitions.json"
    results = json.loads(e1_path.read_text()) if e1_path.exists() else {}
    print(f"\nE1 remaining — already have: {list(results.keys())}", flush=True)

    for t_adv in [1, 2, 3]:
        for variant in ["direct", "flooding"]:
            label = f"auth/{variant}/t={t_adv}/c1c2"
            if label in results:
                print(f"  SKIP {label} (done)", flush=True)
                continue

            print(f"\n  [{label}]", flush=True)
            provenance = HMACProvenance()
            store      = build_seed_store(provenance)
            attack     = AuthenticatedAttack(store, provenance)
            defence    = SMSRDefence(provenance, mode="c1c2", m=20, k=5, n_runs=7)
            per_scenario = {}

            for s_idx, scenario in enumerate(SCENARIOS):
                outcomes = []
                for _ in range(n_reps):
                    ids = attack.inject(scenario, n_copies=t_adv, variant=variant)
                    run_batches = defence.retrieve(store, scenario.question)
                    run_responses = []
                    for batch in run_batches:
                        ctx = _format_memories(batch)
                        msg = client.messages.create(
                            model="claude-haiku-4-5-20251001", max_tokens=200,
                            system=SYSTEM_PROMPT,
                            messages=[{"role":"user","content":
                                       f"Retrieved context:\n{ctx}\n\nQuestion: {scenario.question}"}])
                        run_responses.append(msg.content[0].text.strip())
                    v, _ = majority_verdict_from_runs(run_responses, scenario.question,
                                                      scenario.correct_answer,
                                                      scenario.malicious_answer, client)
                    outcomes.append(1 if v == "malicious" else 0)
                    attack.cleanup(ids)

                mean = sum(outcomes)/n_reps
                ci   = wilson_ci(sum(outcomes), n_reps)
                per_scenario[scenario.question[:50]] = {"successes": sum(outcomes), "n": n_reps, "mean_asr": mean, "ci": ci}
                print(f"    [{s_idx+1:2d}/15] {scenario.question[:45]:<45} "
                      f"ASR={mean*100:5.1f}% CI=[{ci[0]*100:.1f},{ci[1]*100:.1f}]", flush=True)

            total = sum(v["successes"] for v in per_scenario.values())
            n_tot = n_reps * len(SCENARIOS)
            pool  = total / n_tot
            pci   = wilson_ci(total, n_tot)
            print(f"\n  → {label}: ASR={pool*100:.1f}% CI=[{pci[0]*100:.1f}%,{pci[1]*100:.1f}%] n={n_tot}", flush=True)
            results[label] = {"pooled_asr": pool, "ci_lo": pci[0], "ci_hi": pci[1],
                              "total_success": total, "total_n": n_tot, "per_scenario": per_scenario}
            with open(e1_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [saved] {e1_path}", flush=True)

    return results


# ── Tier 1: large-store (20 seeds, production m=20 regime) ───────────────────
def run_tier1_large_store(client, n_reps=30):
    out_path = OUT_DIR / "tier1_large_store.json"
    if out_path.exists():
        print("\nTier 1 already done — loading from disk", flush=True)
        return json.loads(out_path.read_text())

    print(f"\n{'='*72}", flush=True)
    print("Tier 1: Large-store validation (20 seeds, t=1, m=20, 30 reps)", flush=True)
    print(f"{'='*72}", flush=True)

    provenance = HMACProvenance()
    store      = build_seed_store(provenance, n_seeds=20)
    attack     = AuthenticatedAttack(store, provenance)
    defence    = SMSRDefence(provenance, mode="c1c2", m=20, k=5, n_runs=7)

    print(f"  Store size: {store.count()} verified (target: 21 after t=1 inject)", flush=True)
    d_theory = delta_bound(20, 5, 7, 1)
    print(f"  Certificate δ (m=20, t=1): {d_theory*100:.1f}%", flush=True)

    per_scenario = {}
    for s_idx, scenario in enumerate(SCENARIOS):
        outcomes = []
        for _ in range(n_reps):
            ids = attack.inject(scenario, n_copies=1, variant="direct")
            run_batches = defence.retrieve(store, scenario.question)
            run_responses = []
            for batch in run_batches:
                ctx = _format_memories(batch)
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=200,
                    system=SYSTEM_PROMPT,
                    messages=[{"role":"user","content":
                               f"Retrieved context:\n{ctx}\n\nQuestion: {scenario.question}"}])
                run_responses.append(msg.content[0].text.strip())
            v, _ = majority_verdict_from_runs(run_responses, scenario.question,
                                              scenario.correct_answer,
                                              scenario.malicious_answer, client)
            outcomes.append(1 if v == "malicious" else 0)
            attack.cleanup(ids)

        mean = sum(outcomes)/n_reps
        ci   = wilson_ci(sum(outcomes), n_reps)
        per_scenario[scenario.question[:50]] = {"successes": sum(outcomes), "n": n_reps, "mean_asr": mean, "ci": ci}
        print(f"  [{s_idx+1:2d}/15] {scenario.question[:45]:<45} "
              f"ASR={mean*100:5.1f}% CI=[{ci[0]*100:.1f},{ci[1]*100:.1f}]", flush=True)

    total = sum(v["successes"] for v in per_scenario.values())
    n_tot = n_reps * len(SCENARIOS)
    pool  = total / n_tot
    pci   = wilson_ci(total, n_tot)

    print(f"\n{'='*72}", flush=True)
    print(f"TIER 1 RESULT: ASR={pool*100:.1f}% CI=[{pci[0]*100:.1f}%,{pci[1]*100:.1f}%] n={n_tot}", flush=True)
    print(f"Certificate δ(m=20,t=1) = {d_theory*100:.1f}%", flush=True)
    print(f"Certificate holds: {pool <= d_theory}  ({pool*100:.1f}% <= {d_theory*100:.1f}%)", flush=True)

    result = {"pooled_asr": pool, "ci_lo": pci[0], "ci_hi": pci[1],
              "delta": d_theory, "n_seeds": 20, "n_total": n_tot,
              "per_scenario": per_scenario}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [saved] {out_path}", flush=True)
    return result


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-e4",  action="store_true")
    ap.add_argument("--skip-e6",  action="store_true")
    ap.add_argument("--skip-e7",  action="store_true")
    ap.add_argument("--skip-e1",  action="store_true")
    ap.add_argument("--skip-t1",  action="store_true")
    ap.add_argument("--n-reps",   type=int, default=30)
    args = ap.parse_args()

    client = _get_client()
    t0     = time.time()

    if not args.skip_e4: run_e4(client, scenarios=SCENARIOS)
    if not args.skip_e6: run_e6(client)
    if not args.skip_e7: run_e7()
    if not args.skip_e1: run_e1_remaining(client, n_reps=args.n_reps)
    if not args.skip_t1: run_tier1_large_store(client, n_reps=args.n_reps)

    print(f"\n[ALL DONE in {time.time()-t0:.0f}s]", flush=True)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for jf in ["e4_amemguard.json", "e6_judge_reliability.json",
               "e7_second_llm.json", "e1_repetitions.json",
               "tier1_large_store.json"]:
        p = OUT_DIR / jf
        if p.exists():
            d = json.loads(p.read_text())
            if jf == "tier1_large_store.json":
                print(f"  Tier 1 large-store: ASR={d['pooled_asr']*100:.1f}% CI=[{d['ci_lo']*100:.1f},{d['ci_hi']*100:.1f}]"
                      f"  delta={d['delta']*100:.1f}%  holds={d['pooled_asr']<=d['delta']}")
            elif jf == "e6_judge_reliability.json":
                print(f"  E6 judge: kappa={d.get('kappa',0):.3f} agree={d.get('agreement',0)*100:.1f}%")
            elif jf == "e1_repetitions.json":
                for k, v in d.items():
                    if isinstance(v, dict) and "pooled_asr" in v:
                        print(f"  E1 {k}: ASR={v['pooled_asr']*100:.1f}% CI=[{v['ci_lo']*100:.1f},{v['ci_hi']*100:.1f}]")
            else:
                print(f"  {jf}: {list(d.keys())}")


if __name__ == "__main__":
    main()
