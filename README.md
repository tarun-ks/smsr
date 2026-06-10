# SMSR: Signed Memory with Smoothed Retrieval

Reference implementation, attack scenarios, and experiments for SMSR, a certified
defence against runtime memory poisoning in persistent LLM agent systems.

SMSR has two components: (1) HMAC-SHA256 provenance tagging at write time, and
(2) randomised memory ablation at query time with a verdict-based majority
aggregator. This repository contains the code and the recorded results behind
the paper's tables.

## Requirements

- Python 3.11+
- `pip install -r requirements.txt`
- An Anthropic API key: `export ANTHROPIC_API_KEY=...` (or put it in a `.env`
  file at the repository root).

The agent and the judge use Claude Haiku 4.5 through the Anthropic API; the
embedding model is `all-MiniLM-L6-v2`, downloaded automatically on first use.

## Layout

```
agent/              memory-augmented agent + SQLite/FAISS memory store
security/           SMSR (HMAC provenance + randomised ablation) and a heuristic baseline
attacks/            the 15 enterprise attack scenarios and the injection classes (the dataset)
eval/               evaluation harness and the LLM judge
smsr_certificate.py hypergeometric certificate (Theorem 2 / Table I)
run_*.py            experiment drivers
results/            recorded outputs from our runs (JSON) + summary notes
```

The attack scenarios (the dataset) live in `attacks/minja.py` as
`ATTACK_SCENARIOS`: 15 enterprise-policy questions, each with a correct answer,
a malicious answer, and the injected claim text.

## Reproducing the numbers

All commands are run from the repository root.

| Result | Command | Output |
|---|---|---|
| Certificate values (Table I) | `python smsr_certificate.py` | printed table |
| Contamination check, E2 (no API calls) | `python e2_contamination_check.py` | `e2_contamination.json` |
| Main sweep (Table II) | `python run_experiments.py` | `smsr_results*.json` |
| Repetitions + CIs, E1 (Tables I, IV) | `python run_final_experiments.py` | `e1_repetitions.json` |
| Tier-1 store, A-MemGuard, second model (T1/E4/E6/E7) | `python run_all_experiments.py` | `tier1_large_store.json`, `e4_*.json`, `e6_*.json`, `e7*_*.json` |
| End-to-end query-only attack, E10 | `python run_e10_pilot.py --reps 10` | `e10_pilot.json` |

`results/` holds the outputs from our own runs; `results/RESULTS.md` summarises
them. Re-running a driver writes fresh output to the repository root.

## Cost

The full suite is a few US dollars at Claude Haiku pricing. E10 at 10 repetitions
is roughly $3-5; the contamination check (E2) makes no API calls.

## Citation

If you use this code or the scenarios, please cite the SMSR paper.
