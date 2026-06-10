"""
harness.py
==========
Production evaluation harness for SMSR.

Runs attack × defence_mode × variant sweeps, judges all responses with
an LLM judge (no string matching), and returns structured results.

DefenceConfig controls which combination is tested:
  mode        : "none" | "c1" | "c1c2" | "heuristic"
  attack_class: MinjaAttack | AuthenticatedAttack | HeuristicBypassAttack

Results include per-scenario verdict ("correct"/"malicious"/"neither"),
aggregate ASR and utility, and the formal certificate for c1c2 configurations.
"""

from __future__ import annotations
import json
import time
from collections import Counter
import anthropic
from dataclasses import dataclass, field
from pathlib import Path

from attacks.minja          import AttackScenario, MinjaAttack, AuthenticatedAttack, HeuristicBypassAttack
from security.smsr          import SMSRDefence, compute_certificate
from security.heuristic_defence import HeuristicDefence
from eval.llm_judge         import batch_judge


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    label         : str                  # human-readable label for this run
    defence_mode  : str                  # "none" | "c1" | "c1c2" | "heuristic"
    attack_class  : str                  # "unsigned" | "authenticated" | "bypass"
    attack_variant: str = "direct"       # "direct" | "flooding" | "subtle"
    n_inject      : int = 3
    m             : int = 20             # over-fetch pool for c1c2
    k             : int = 5              # memories per run
    n_runs        : int = 5              # ablation runs for c1c2


@dataclass
class TrialResult:
    scenario     : AttackScenario
    question     : str
    correct_answer: str
    malicious_answer: str
    response     : str
    verdict      : str    # "correct" | "malicious" | "neither"
    reason       : str
    n_injected   : int
    config_label : str


@dataclass
class ExperimentResult:
    config        : ExperimentConfig
    trials        : list[TrialResult] = field(default_factory=list)
    utility_trials: list[dict]        = field(default_factory=list)

    @property
    def asr(self) -> float:
        """Fraction of attack trials where agent gave the malicious answer."""
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.verdict == "malicious") / len(self.trials)

    @property
    def neither_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.verdict == "neither") / len(self.trials)

    @property
    def utility(self) -> float:
        if not self.utility_trials:
            return 0.0
        correct = sum(1 for t in self.utility_trials if t.get("verdict") == "correct")
        return correct / len(self.utility_trials)

    def summary_line(self) -> str:
        cfg = self.config
        return (
            f"{cfg.label:<35} "
            f"ASR={self.asr*100:5.1f}%  "
            f"neither={self.neither_rate*100:4.1f}%  "
            f"utility={self.utility*100:5.1f}%  "
            f"({len(self.trials)} attack / {len(self.utility_trials)} clean trials)"
        )

    def to_dict(self) -> dict:
        return {
            "label"           : self.config.label,
            "defence_mode"    : self.config.defence_mode,
            "attack_class"    : self.config.attack_class,
            "attack_variant"  : self.config.attack_variant,
            "n_inject"        : self.config.n_inject,
            "m"               : self.config.m,
            "k"               : self.config.k,
            "n_runs"          : self.config.n_runs,
            "asr"             : self.asr,
            "neither_rate"    : self.neither_rate,
            "utility"         : self.utility,
            "n_attack_trials" : len(self.trials),
            "n_utility_trials": len(self.utility_trials),
            "attack_trials"   : [
                {
                    "question"       : t.question,
                    "correct_answer" : t.correct_answer,
                    "malicious_answer": t.malicious_answer,
                    "response"       : t.response,
                    "verdict"        : t.verdict,
                    "reason"         : t.reason,
                    "n_injected"     : t.n_injected,
                }
                for t in self.trials
            ],
            "utility_trials"  : self.utility_trials,
        }


# ── Core runner ───────────────────────────────────────────────────────────────
class ExperimentRunner:
    def __init__(
        self,
        store,
        provenance,
        client      : anthropic.Anthropic,
        verbose     : bool = True,
    ):
        self.store      = store
        self.provenance = provenance
        self.client     = client
        self.verbose    = verbose

    def _make_defence(self, cfg: ExperimentConfig) -> tuple:
        """Return (SMSRDefence, HeuristicDefence|None)."""
        smsr = SMSRDefence(
            provenance = self.provenance,
            mode       = cfg.defence_mode if cfg.defence_mode != "heuristic" else "none",
            m          = cfg.m,
            k          = cfg.k,
            n_runs     = cfg.n_runs,
        )
        heur = HeuristicDefence() if cfg.defence_mode == "heuristic" else None
        return smsr, heur

    def _make_attack(self, cfg: ExperimentConfig):
        if cfg.attack_class == "authenticated":
            return AuthenticatedAttack(self.store, self.provenance)
        elif cfg.attack_class == "bypass":
            return HeuristicBypassAttack(self.store)
        else:
            return MinjaAttack(self.store)

    def run_attack(
        self,
        cfg       : ExperimentConfig,
        scenarios : list[AttackScenario],
    ) -> ExperimentResult:
        from agent.agent import MemoryAgent

        smsr, heur = self._make_defence(cfg)
        attack     = self._make_attack(cfg)
        agent      = MemoryAgent(self.store, smsr, self.provenance, session_id="eval")
        result     = ExperimentResult(config=cfg)

        if self.verbose:
            print(f"\n  ── {cfg.label} ──")

        raw_trials = []
        for scenario in scenarios:
            # inject
            injected_ids = attack.inject(
                scenario, n_copies=cfg.n_inject, variant=cfg.attack_variant
            )

            # retrieve memories (respecting heuristic filter if active)
            run_batches = smsr.retrieve(self.store, scenario.question)
            if heur:
                run_batches = [heur.filter(batch, self.store) for batch in run_batches]

            # build responses for each run
            from agent.agent import _format_memories, _majority_vote, _get_client, SYSTEM_PROMPT
            responses = []
            for mem_batch in run_batches:
                mem_ctx  = _format_memories(mem_batch)
                user_msg = f"Relevant memories:\n{mem_ctx}\n\nQuestion: {scenario.question}"
                msg = self.client.messages.create(
                    model      = "claude-haiku-4-5-20251001",
                    max_tokens = 256,
                    system     = SYSTEM_PROMPT,
                    messages   = [{"role": "user", "content": user_msg}],
                )
                responses.append(msg.content[0].text.strip())

            # For c1c2 multi-run ablation: use VERDICT-BASED majority vote.
            # String-based majority is gamed by "consistent minority" attacks
            # (adversarial answers are more consistent than varied "I don't know"
            # responses, so they win the string count even when they're a minority).
            # We judge each run independently, count verdict labels, pick majority label,
            # then return one representative response for that label.
            if cfg.defence_mode == "c1c2" and len(responses) > 1:
                from eval.llm_judge import judge as llm_judge_fn
                per_run_verdicts = []
                for resp in responses:
                    v = llm_judge_fn(scenario.question, scenario.correct_answer,
                                     scenario.malicious_answer, resp, self.client)
                    per_run_verdicts.append((v["verdict"], resp))
                # count label majority
                label_counts = Counter(v for v, _ in per_run_verdicts)
                winning_label = label_counts.most_common(1)[0][0]
                # return first response that matches the winning label
                final_response = next(
                    resp for verdict, resp in per_run_verdicts if verdict == winning_label
                )
            else:
                final_response = _majority_vote(responses)

            raw_trials.append({
                "question"       : scenario.question,
                "correct_answer" : scenario.correct_answer,
                "malicious_answer": scenario.malicious_answer,
                "response"       : final_response,
                "n_injected"     : len(injected_ids),
                "config_label"   : cfg.label,
            })

            # clean up injections
            attack.cleanup(injected_ids)

        # LLM judge all trials
        if self.verbose:
            print(f"    judging {len(raw_trials)} attack trials...")
        judged = batch_judge(raw_trials, self.client, verbose=self.verbose)

        for t in judged:
            result.trials.append(TrialResult(
                scenario         = next(s for s in scenarios if s.question == t["question"]),
                question         = t["question"],
                correct_answer   = t["correct_answer"],
                malicious_answer = t["malicious_answer"],
                response         = t["response"],
                verdict          = t["verdict"],
                reason           = t["reason"],
                n_injected       = t["n_injected"],
                config_label     = t["config_label"],
            ))

        return result

    def run_utility(
        self,
        cfg        : ExperimentConfig,
        clean_tasks: list[tuple],   # (question, correct_answer)
        result     : ExperimentResult,
    ) -> ExperimentResult:
        from agent.agent import MemoryAgent, _format_memories, _majority_vote, SYSTEM_PROMPT

        smsr, heur = self._make_defence(cfg)
        if self.verbose:
            print(f"    judging {len(clean_tasks)} clean tasks...")

        raw_trials = []
        for (question, correct_answer) in clean_tasks:
            run_batches = smsr.retrieve(self.store, question)
            if heur:
                run_batches = [heur.filter(batch, self.store) for batch in run_batches]

            responses = []
            for mem_batch in run_batches:
                mem_ctx  = _format_memories(mem_batch)
                user_msg = f"Relevant memories:\n{mem_ctx}\n\nQuestion: {question}"
                msg = self.client.messages.create(
                    model      = "claude-haiku-4-5-20251001",
                    max_tokens = 128,
                    system     = SYSTEM_PROMPT,
                    messages   = [{"role": "user", "content": user_msg}],
                )
                responses.append(msg.content[0].text.strip())

            raw_trials.append({
                "question"       : question,
                "correct_answer" : correct_answer,
                "malicious_answer": "(N/A — clean utility task)",
                "response"       : _majority_vote(responses),
            })

        judged = batch_judge(raw_trials, self.client, verbose=self.verbose)
        result.utility_trials = judged
        return result


# ── Convenience: print full summary table ────────────────────────────────────
def print_summary(results: list[ExperimentResult], certificates: list[dict] = None):
    print("\n" + "="*90)
    print("SMSR FULL EVALUATION SUMMARY")
    print("="*90)
    print(f"{'Configuration':<35} {'ASR':>7} {'neither':>8} {'utility':>9}  {'trials':>7}")
    print("-"*90)
    for r in results:
        print(f"  {r.summary_line()}")

    if certificates:
        print("\nFormal Certificates (Component 2, c1c2 mode):")
        print(f"  {'t_adv':>6} {'k':>4} {'m':>4} {'n_runs':>7} {'p_clean/run':>12} {'delta':>8}")
        print("  " + "-"*52)
        for c in certificates:
            print(f"  {c['t']:>6} {c['k']:>4} {c['m']:>4} {c['n_runs']:>7} "
                  f"{c['p_clean_single']:>12.4f} {c['delta']:>8.4f}")
    print("="*90)
