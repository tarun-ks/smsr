#!/usr/bin/env python3
"""
resume_experiments.py
=====================
Picks up where run_final_experiments.py left off after the credit
exhaustion at auth/direct/t=1/c1c2 completion.

Runs:
  E1 remaining:  auth/flooding/t=1, auth/direct/t=2, auth/direct/t=3,
                 auth/flooding/t=2, auth/flooding/t=3
  E4:  A-MemGuard baseline
  E6:  Judge reliability (Haiku vs Sonnet)
  E7:  Second-LLM generality (Sonnet as agent)

The t=1/direct pooled result is already saved in e1_repetitions.json.
This script appends to that file.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(override=False)

from agent.agent import _get_client
from attacks.minja import ATTACK_SCENARIOS
from run_final_experiments import run_e1, run_e4, run_e6, run_e7, OUT_DIR

SCENARIOS = ATTACK_SCENARIOS[:15]
client    = _get_client()

# ── E1 remaining configs ─────────────────────────────────────────────────────
# Load what's already done
existing_e1 = {}
e1_path = OUT_DIR / "e1_repetitions.json"
if e1_path.exists():
    existing_e1 = json.loads(e1_path.read_text())
    print(f"Loaded {len(existing_e1)} already-completed E1 configs from {e1_path}")

import time
from agent.memory_store   import MemoryStore
from agent.agent          import _format_memories, SYSTEM_PROMPT
from security.provenance  import HMACProvenance
from security.smsr        import SMSRDefence
from attacks.minja        import AuthenticatedAttack
from eval.llm_judge       import judge as llm_judge
from run_experiments      import build_seed_store
from run_final_experiments import majority_verdict_from_runs, wilson_ci
from collections import Counter

def run_e1_remaining(client, n_reps=30, scenarios=None):
    if scenarios is None:
        scenarios = ATTACK_SCENARIOS[:15]
    results = dict(existing_e1)

    for t_adv in [1, 2, 3]:
        for variant in ["direct", "flooding"]:
            label = f"auth/{variant}/t={t_adv}/c1c2"
            if label in results:
                print(f"  Skipping {label} (already done)")
                continue

            print(f"\n  [{label}]", flush=True)
            provenance = HMACProvenance()
            store      = build_seed_store(provenance)
            attack     = AuthenticatedAttack(store, provenance)
            defence    = SMSRDefence(provenance, mode="c1c2", m=20, k=5, n_runs=7)
            per_scenario = {}

            for s_idx, scenario in enumerate(scenarios):
                outcomes = []
                for rep in range(n_reps):
                    ids = attack.inject(scenario, n_copies=t_adv, variant=variant)
                    run_batches = defence.retrieve(store, scenario.question)
                    run_responses = []
                    for batch in run_batches:
                        ctx = _format_memories(batch)
                        msg = client.messages.create(
                            model="claude-haiku-4-5-20251001", max_tokens=200,
                            system=SYSTEM_PROMPT,
                            messages=[{"role": "user",
                                       "content": f"Retrieved context:\n{ctx}\n\nQuestion: {scenario.question}"}],
                        )
                        run_responses.append(msg.content[0].text.strip())
                    final_verdict, _ = majority_verdict_from_runs(
                        run_responses, scenario.question,
                        scenario.correct_answer, scenario.malicious_answer, client)
                    outcomes.append(1 if final_verdict == "malicious" else 0)
                    attack.cleanup(ids)

                mean_asr = sum(outcomes) / n_reps
                ci = wilson_ci(sum(outcomes), n_reps)
                per_scenario[scenario.question[:50]] = {
                    "successes": sum(outcomes), "n": n_reps,
                    "mean_asr": mean_asr, "ci": ci,
                }
                print(f"    [{s_idx+1:2d}/15] {scenario.question[:45]:<45} "
                      f"ASR={mean_asr*100:5.1f}% CI=[{ci[0]*100:.1f},{ci[1]*100:.1f}]",
                      flush=True)

            total_success = sum(v["successes"] for v in per_scenario.values())
            total_n       = n_reps * len(scenarios)
            pool_asr      = total_success / total_n
            pool_ci       = wilson_ci(total_success, total_n)
            print(f"\n  → {label}: pooled ASR={pool_asr*100:.1f}% "
                  f"CI=[{pool_ci[0]*100:.1f}%,{pool_ci[1]*100:.1f}%]  (n={total_n})",
                  flush=True)

            results[label] = {
                "pooled_asr": pool_asr, "ci_lo": pool_ci[0], "ci_hi": pool_ci[1],
                "total_success": total_success, "total_n": total_n,
                "per_scenario": per_scenario,
            }
            # save after each config to preserve progress
            with open(e1_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [saved] {e1_path}")

    return results


if __name__ == "__main__":
    print("Resuming experiments after credit refill...", flush=True)
    e1_res = run_e1_remaining(client, n_reps=30, scenarios=SCENARIOS)
    e4_res = run_e4(client, scenarios=SCENARIOS)
    e6_res = run_e6(client)
    e7_res = run_e7()
    print("\n[all experiments complete]")
